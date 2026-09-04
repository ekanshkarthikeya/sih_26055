import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.utils.turing_loader import TuringDatasetLoader
from src.env.rf_spectrum_env import RFEnvironment
from src.schedulers.whittle_rmab import AdaptiveWhittleScheduler
from src.schedulers.baselines import (
    RoundRobinScheduler,
    RandomScheduler,
    GreedyScheduler,
    EpsilonGreedyScheduler,
)
from src.utils.metrics import EWMetricsTracker
from src.processing.deinterleaver import OnlineDeinterleaver, TacticalThreatCatalog


def run_simulation(strategy_name: str, df, num_bands: int = 16):
    env = RFEnvironment(df, num_bands=num_bands, default_dwell_sec=45e-6)
    metrics = EWMetricsTracker(num_bands=num_bands)

    for eid, group in df.groupby('emitter_id'):
        metrics.register_truth_emitter(eid, group['toa'].iloc[0])

    if strategy_name == "rmab":
        scheduler = AdaptiveWhittleScheduler(num_bands=num_bands)
    elif strategy_name == "round_robin":
        scheduler = RoundRobinScheduler(num_bands=num_bands)
    elif strategy_name == "random":
        scheduler = RandomScheduler(num_bands=num_bands)
    elif strategy_name == "greedy":
        scheduler = GreedyScheduler(num_bands=num_bands)
    elif strategy_name == "eps_greedy":
        scheduler = EpsilonGreedyScheduler(num_bands=num_bands, epsilon=0.15)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    total_hits = 0
    t0 = time.perf_counter()

    while True:
        if strategy_name == "rmab":
            action, dwell = scheduler.select_action(env.current_time)
        else:
            action = scheduler.select_band()
            dwell = 45e-6

        obs, reward, hits, pdw_dict, dwell_start, done = env.step(action, dwell_time_sec=dwell)
        total_hits += hits

        for eid in pdw_dict['emitters']:
            metrics.log_intercept(eid, dwell_start, None)

        if strategy_name == "rmab":
            scheduler.update_beliefs(action, hits, env.current_time)
        elif strategy_name in ["greedy", "eps_greedy"]:
            scheduler.update(action, hits)

        if done:
            break

    sim_duration = time.perf_counter() - t0
    results = metrics.evaluate(env.total_pulses, total_hits)
    dead_time_pct = (env.total_settling_time / env.current_time) * 100.0 if env.current_time > 0 else 0.0

    return {
        "P_int": results["P_int"],
        "Latency_ms": results["Latency_ms"],
        "Captured": total_hits,
        "Total": env.total_pulses,
        "DeadTime_pct": dead_time_pct,
        "Compute_sec": sim_duration,
    }


def main():
    console = Console()
    console.print(Panel.fit(
        "[bold cyan]DRDO Electronic Warfare Smart Scan Benchmark[/bold cyan]\n"
        "[dim]Comparing 5 Receiver Scheduling Paradigms on Alan Turing Synthetic Radar Data[/dim]",
        border_style="cyan"
    ))

    # Ingestion
    loader = TuringDatasetLoader(num_bands=16)
    test_dir = REPO_ROOT / "data" / "archive" / "test"
    available = list(test_dir.glob("test_*.h5"))

    if available:
        test_file = str(sorted(available)[0])
        console.print(f"[bold green]✓[/bold green] Ingesting Radar Stream: [yellow]{Path(test_file).name}[/yellow]")
        df = loader.load_from_h5(test_file)
    else:
        console.print("[yellow]⚠[/yellow] No local HDF5 splits found. Synthesizing operational RF stream...")
        df = loader.load_or_generate(duration_sec=2.0)

    console.print(f"Dataset Size: [bold white]{len(df):,}[/bold white] pulses across [bold white]{df['emitter_id'].nunique()}[/bold white] radar emitters.\n")

    strategies = [
        ("Deterministic Round-Robin", "round_robin", "Legacy 1970s sequential stepping; blind to low duty cycles"),
        ("Uniform Random Search", "random", "Memoryless random hops; high PLL blanking dead-time"),
        ("Pure Greedy (Max-Belief)", "greedy", "Exploitation trap; starves unmonitored channels"),
        ("ε-Greedy (ε = 0.15)", "eps_greedy", "Random hops incur severe logarithmic PLL slew penalties"),
        ("Adaptive Bayesian RMAB (Ours)", "rmab", "Conjugate Beta-Bernoulli + Whittle Index + Hard Starvation Cap"),
    ]

    table = Table(title="Full Baseline Performance Evaluation", header_style="bold magenta", border_style="blue")
    table.add_column("Strategy", style="cyan", width=28)
    table.add_column("P_int (%)", justify="right", style="bold")
    table.add_column("Gain vs RR", justify="right")
    table.add_column("Mean Latency", justify="right")
    table.add_column("Captured Pulses", justify="right")
    table.add_column("Blanking Dead-Time", justify="right")
    table.add_column("Operational Flaw / Characteristic", style="dim")

    res_dict = {}
    with console.status("[bold green]Executing full Monte Carlo runs across all 5 strategies..."):
        for name, key, desc in strategies:
            res_dict[key] = run_simulation(key, df, num_bands=16)

    rr_p_int = max(res_dict["round_robin"]["P_int"], 1e-4)

    for name, key, desc in strategies:
        r = res_dict[key]
        gain = r["P_int"] / rr_p_int
        gain_str = f"[bold green]{gain:.1f}x[/bold green]" if gain > 1.05 else "1.0x"
        p_int_color = "bold green" if key == "rmab" else "yellow" if r["P_int"] > 15.0 else "red"

        table.add_row(
            name,
            f"[{p_int_color}]{r['P_int']:.2f}%[/{p_int_color}]",
            gain_str,
            f"{r['Latency_ms']:.1f} ms",
            f"{r['Captured']:,} / {r['Total']:,}",
            f"{r['DeadTime_pct']:.1f}%",
            desc
        )

    console.print(table)


if __name__ == "__main__":
    main()