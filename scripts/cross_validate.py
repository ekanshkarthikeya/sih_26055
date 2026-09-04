import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.utils.turing_loader import TuringDatasetLoader
from src.utils.metrics import EWMetricsTracker
from src.env.rf_spectrum_env import RFEnvironment
from src.schedulers.whittle_rmab import AdaptiveWhittleScheduler
from src.schedulers.baselines import RoundRobinScheduler, RandomScheduler

def run_single_eval(df, strategy_name="smart", num_bands=16):
    env = RFEnvironment(df, num_bands=num_bands, default_dwell_sec=45e-6)
    metrics = EWMetricsTracker(num_bands=num_bands)
    
    for eid, group in df.groupby('emitter_id'):
        metrics.register_truth_emitter(eid, group['toa'].iloc[0])

    if strategy_name == "smart":
        scheduler = AdaptiveWhittleScheduler(num_bands=num_bands, max_consecutive_dwells=4, max_starve_sec=0.060)
    elif strategy_name == "round_robin":
        scheduler = RoundRobinScheduler(num_bands=num_bands)
    else:
        scheduler = RandomScheduler(num_bands=num_bands)

    total_hits = 0
    while True:
        if strategy_name == "smart":
            action, dwell_len = scheduler.select_action(env.current_time)
        else:
            action = scheduler.select_band()
            dwell_len = 45e-6

        obs, reward, hits, pdw_dict, dwell_start, done = env.step(action, dwell_time_sec=dwell_len)
        total_hits += hits

        for eid in pdw_dict['emitters']:
            metrics.log_intercept(eid, dwell_start, None)

        if strategy_name == "smart":
            scheduler.update_beliefs(action, hits, env.current_time)

        if done:
            break

    res = metrics.evaluate(env.total_pulses, total_hits)
    return res['P_int'], res['Latency_ms']

def main():
    console = Console()
    loader = TuringDatasetLoader(num_bands=16)
    
    data_dir = REPO_ROOT / "data" / "archive" / "test"
    # Auto-detect all test_*.h5 files and sort numerically (0, 1, 2, ..., 6)
    test_files = sorted(
        list(data_dir.glob("test_*.h5")),
        key=lambda p: int(p.stem.split('_')[-1]) if p.stem.split('_')[-1].isdigit() else 999
    )

    if not test_files:
        console.print(f"[bold red]No .h5 files found in {data_dir}[/bold red]")
        return

    records = []
    console.print(f"\n[bold cyan]=== EXECUTING MULTI-SPLIT STATISTICAL VALIDATION ({len(test_files)} Alan Turing Splits) ===[/bold cyan]\n")

    for fpath in test_files:
        console.print(f"[bold yellow]Processing {fpath.name}...[/bold yellow]")
        df = loader.load_from_h5(str(fpath))
        
        console.print(f"  • Running Adaptive RMAB Scheduler ({len(df):,} pulses)...")
        p_smart, lat_smart = run_single_eval(df, "smart")
        
        console.print("  • Running Round-Robin Baseline...")
        p_rr, lat_rr = run_single_eval(df, "round_robin")
        
        console.print("  • Running Random Sweep Baseline...")
        p_rand, lat_rand = run_single_eval(df, "random")

        gain = p_smart / max(p_rr, 1e-4)
        console.print(f"  [green]Done {fpath.name}: Smart P_int={p_smart:.2f}%, RR={p_rr:.2f}% (Gain: {gain:.1f}x)[/green]\n")

        records.append({
            "File": fpath.name,
            "Pulses": len(df),
            "Emitters": df['emitter_id'].nunique(),
            "RR P_int": p_rr,
            "RR Latency": lat_rr,
            "Smart P_int": p_smart,
            "Smart Latency": lat_smart,
            "P_int Gain": gain
        })

    summary_df = pd.DataFrame(records)

    table = Table(title="Cross-Validation Multi-Split Rigor Summary")
    table.add_column("Dataset Split", style="cyan")
    table.add_column("Pulses", justify="right")
    table.add_column("Emitters", justify="right")
    table.add_column("RR P_int", justify="right", style="red")
    table.add_column("RR Latency", justify="right", style="red")
    table.add_column("Adaptive RMAB P_int", justify="right", style="bold green")
    table.add_column("Adaptive Latency", justify="right", style="bold green")
    table.add_column("Gain Factor", justify="right", style="bold yellow")

    for _, row in summary_df.iterrows():
        table.add_row(
            row["File"],
            f"{row['Pulses']:,}",
            f"{row['Emitters']}",
            f"{row['RR P_int']:.2f}%",
            f"{row['RR Latency']:.1f} ms",
            f"{row['Smart P_int']:.2f}%",
            f"{row['Smart Latency']:.1f} ms",
            f"{row['P_int Gain']:.1f}x"
        )

    console.print(table)
    console.print(f"\n[bold white]Mean P_int: {summary_df['Smart P_int'].mean():.2f}% ± {summary_df['Smart P_int'].std():.2f}%[/bold white]")
    console.print(f"[bold white]Mean Latency: {summary_df['Smart Latency'].mean():.1f} ms ± {summary_df['Smart Latency'].std():.1f} ms[/bold white]\n")

if __name__ == "__main__":
    main()