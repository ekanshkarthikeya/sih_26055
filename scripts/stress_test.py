import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.env.rf_spectrum_env import RFEnvironment
from src.schedulers.whittle_rmab import WhittleRMABScheduler
from src.schedulers.periodic_predictor import PeriodicScanPredictor
from src.schedulers.pri_estimator import PRIEstimator
from src.schedulers.baselines import RoundRobinScheduler
from src.utils.metrics import EWMetricsTracker

def generate_hopper_swarm(duration_sec=1.5, num_bands=16):
    records = []
    # 6 independent agile hoppers
    for emitter_idx in range(6):
        t = np.random.uniform(0.001, 0.01)
        pri = np.random.uniform(60e-6, 180e-6)
        while t < duration_sec:
            b = int(np.random.randint(0, num_bands))
            records.append({
                'toa': t, 'freq_mhz': 2000.0 + b * 500.0 + 250.0,
                'band_idx': b, 'threat': 3.5, 'emitter_id': 400 + emitter_idx
            })
            t += pri
    df = pd.DataFrame(records).sort_values('toa').reset_index(drop=True)
    return df

def generate_noisy_scenario(base_df, pd_rate=0.85, pfa_count=3000):
    # Pulse drop simulation
    mask = np.random.rand(len(base_df)) < pd_rate
    noisy_df = base_df[mask].copy()

    # Synthetic false alarms
    fa_toas = np.random.uniform(0, base_df['toa'].max(), pfa_count)
    fa_bands = np.random.randint(0, 16, pfa_count)
    fa_df = pd.DataFrame({
        'toa': fa_toas, 'freq_mhz': 2000.0 + fa_bands * 500.0 + 250.0,
        'band_idx': fa_bands, 'threat': 1.0, 'emitter_id': -1
    })

    combined = pd.concat([noisy_df, fa_df]).sort_values('toa').reset_index(drop=True)
    return combined

def run_evaluation(df_scenario, name="Smart Scan", is_smart=True):
    env = RFEnvironment(df_scenario, num_bands=16, dwell_time_sec=50e-6)
    metrics = EWMetricsTracker(num_bands=16)
    
    for eid, group in df_scenario.groupby('emitter_id'):
        if eid != -1:
            metrics.register_truth_emitter(eid, group['toa'].iloc[0])

    if is_smart:
        whittle = WhittleRMABScheduler(num_bands=16, min_dwell_steps=3)
        predictor = PeriodicScanPredictor(target_band=2)
        pri_tracker = PRIEstimator()
    else:
        baseline = RoundRobinScheduler(num_bands=16)

    total_intercepted = 0
    while True:
        predicted_toa = None
        if is_smart:
            override, pred_burst = predictor.is_burst_imminent(env.current_time)
            action = predictor.target_band if override else whittle.select_band(env.current_time)
            predicted_toa = pri_tracker.predict_next_pulse(action)
        else:
            action = baseline.select_band()

        metrics.log_dwell(action, env.current_time)
        obs, reward, hits, emitters, dwell_start, done = env.step(action)
        total_intercepted += hits

        for eid in emitters:
            metrics.log_intercept(eid, dwell_start, predicted_toa)

        if is_smart:
            whittle.update_beliefs(action, obs, env.current_time)
            if hits > 0:
                pri_tracker.log_pulse(action, dwell_start)
                if action == 2:
                    predictor.register_pulse_hit(dwell_start)

        if done:
            break

    res = metrics.evaluate(env.total_pulses, total_intercepted)
    res['Strategy'] = name
    return res

def main():
    console = Console()
    table = Table(title="Tactical Scenario Stress Test Suite")
    table.add_column("Scenario", style="cyan")
    table.add_column("Strategy", style="yellow")
    table.add_column("P_int (%)", style="bold green", justify="right")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Time Err (μs)", justify="right")

    # Load baseline
    from src.utils.turing_loader import TuringDatasetLoader
    base_df = TuringDatasetLoader().load_or_generate(duration_sec=1.5)
    swarm_df = generate_hopper_swarm(duration_sec=1.5)
    noisy_df = generate_noisy_scenario(base_df)

    scenarios = [
        ("Standard Mixed", base_df),
        ("Agile Swarm (6 Emitters)", swarm_df),
        ("Degraded Channel (Pd=0.85 + Noise)", noisy_df)
    ]

    for sc_name, df in scenarios:
        res_rr = run_evaluation(df, name="Round-Robin", is_smart=False)
        res_sm = run_evaluation(df, name="Smart Scan", is_smart=True)
        for r in [res_rr, res_sm]:
            table.add_row(
                sc_name, r['Strategy'], f"{r['P_int']:.2f}%", 
                f"{r['Latency_ms']:.2f}", f"{r['Time_Error_us']:.2f}"
            )
        table.add_section()

    console.print(table)

if __name__ == "__main__":
    main()