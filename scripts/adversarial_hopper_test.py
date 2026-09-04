import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.env.rf_spectrum_env import RFEnvironment
from src.schedulers.whittle_rmab import AdaptiveWhittleScheduler
from src.schedulers.baselines import RoundRobinScheduler, RandomScheduler
from src.utils.metrics import EWMetricsTracker

def generate_hopping_emitter_scenario(duration_sec=3.0, num_pulses=5000):
    """
    Synthesizes an adversarial scenario with:
    - 2 fixed radar emitters (surveillance & track)
    - 1 agile Frequency-Hopping radar jumping across Bands 4, 5, 8, 12 every 2 pulses
    """
    toas = np.sort(np.random.uniform(0.0, duration_sec, num_pulses))
    bands = np.zeros(num_pulses, dtype=int)
    threats = np.ones(num_pulses)
    emitters = []
    pws = np.random.uniform(1.0, 15.0, num_pulses)

    hopping_pool = [4, 5, 8, 12]

    for i, t in enumerate(toas):
        coin = i % 10
        if coin < 3:
            # Fixed surveillance in Band 1
            bands[i] = 1
            emitters.append("Fixed_Surveillance_B1")
        elif coin < 6:
            # Fixed tracking in Band 9
            bands[i] = 9
            emitters.append("Fixed_Tracking_B9")
        else:
            # Agile frequency hopper
            hop_idx = (i // 2) % len(hopping_pool)
            bands[i] = hopping_pool[hop_idx]
            threats[i] = 3.0  # High tactical priority
            emitters.append("Agile_Hopper_LPI")

    return pd.DataFrame({
        "toa": toas,
        "band_idx": bands,
        "threat": threats,
        "emitter_id": emitters,
        "pw_us": pws
    })

def evaluate_strategy(df, strategy="smart"):
    env = RFEnvironment(df, num_bands=16, default_dwell_sec=45e-6)
    metrics = EWMetricsTracker(num_bands=16)

    for eid, group in df.groupby('emitter_id'):
        metrics.register_truth_emitter(eid, group['toa'].iloc[0])

    if strategy == "smart":
        scheduler = AdaptiveWhittleScheduler(num_bands=16, max_consecutive_dwells=4, max_starve_sec=0.060)
    elif strategy == "rr":
        scheduler = RoundRobinScheduler(num_bands=16)
    else:
        scheduler = RandomScheduler(num_bands=16)

    hits_by_emitter = {"Agile_Hopper_LPI": 0, "Fixed_Surveillance_B1": 0, "Fixed_Tracking_B9": 0}
    total_hits = 0

    while True:
        if strategy == "smart":
            action, dwell_len = scheduler.select_action(env.current_time)
        else:
            action = scheduler.select_band()
            dwell_len = 45e-6

        obs, reward, hits, pdw_data, dwell_start, done = env.step(action, dwell_time_sec=dwell_len)
        total_hits += hits

        for eid in pdw_data['emitters']:
            metrics.log_intercept(eid, dwell_start, None)
            if eid in hits_by_emitter:
                hits_by_emitter[eid] += 1

        if strategy == "smart":
            scheduler.update_beliefs(action, hits, env.current_time)

        if done:
            break

    res = metrics.evaluate(env.total_pulses, total_hits)
    return res, hits_by_emitter

def main():
    console = Console()
    console.print("\n[bold red]=== EXECUTING ADVERSARIAL LPI / FREQUENCY-HOPPING STRESS TEST ===[/bold red]\n")
    df_scenario = generate_hopping_emitter_scenario()

    res_smart, hits_smart = evaluate_strategy(df_scenario, "smart")
    res_rr, hits_rr = evaluate_strategy(df_scenario, "rr")
    res_rand, hits_rand = evaluate_strategy(df_scenario, "random")

    table = Table(title="Tactical Performance Against Agile Frequency Hoppers")
    table.add_column("Strategy", style="cyan")
    table.add_column("Overall P_int", justify="right")
    table.add_column("Hopper Pulses Captured", justify="right", style="bold yellow")
    table.add_column("Fixed Pulses Captured", justify="right")
    table.add_column("Discovery Latency", justify="right")

    for name, res, hits in [("Round-Robin", res_rr, hits_rr), ("Random Sweep", res_rand, hits_rand), ("Adaptive RMAB", res_smart, hits_smart)]:
        fixed_hits = hits["Fixed_Surveillance_B1"] + hits["Fixed_Tracking_B9"]
        table.add_row(
            name,
            f"{res['P_int']:.2f}%",
            f"{hits['Agile_Hopper_LPI']:,}",
            f"{fixed_hits:,}",
            f"{res['Latency_ms']:.1f} ms"
        )

    console.print(table)

if __name__ == "__main__":
    main()