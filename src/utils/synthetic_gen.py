import numpy as np
import pandas as pd

def generate_synthetic_rf_scenario(duration_sec=2.0, num_bands=16, base_freq_mhz=2000.0, band_step_mhz=500.0):
    """
    Generates realistic PDWs across discrete sub-bands:
    - Band 2: High-threat tracking radar (high PRF, constant frequency)
    - Band 5: Periodic scanning radar (mainlobe sweeps every 0.4s)
    - Band 10-13: Frequency agile emitter (hops across bands randomly)
    """
    pdw_records = []
    
    # Emitter 1: High-Threat Fire Control (Band 2)
    f_band2 = base_freq_mhz + 2 * band_step_mhz + 250.0
    pri_e1 = 50e-6  # 20 kHz PRF
    t = 0.05
    while t < duration_sec:
        pdw_records.append({'toa': t, 'freq': f_band2, 'band_idx': 2, 'pw': 1.2e-6, 'threat': 5.0})
        t += pri_e1

    # Emitter 2: Surveillance Radar with Antenna Scan (Band 5)
    f_band5 = base_freq_mhz + 5 * band_step_mhz + 250.0
    pri_e2 = 250e-6 # 4 kHz PRF
    t_scan = 0.4    # Antenna rotates every 400 ms
    beam_dwell = 0.015 # 15 ms mainlobe illumination window
    
    scan_start = 0.0
    while scan_start < duration_sec:
        t = scan_start
        while t < min(scan_start + beam_dwell, duration_sec):
            pdw_records.append({'toa': t, 'freq': f_band5, 'band_idx': 5, 'pw': 2.5e-6, 'threat': 3.0})
            t += pri_e2
        scan_start += t_scan

    # Emitter 3: Agile Frequency Hopper (Bands 10 to 13)
    t = 0.01
    pri_e3 = 100e-6
    while t < duration_sec:
        chosen_band = np.random.choice([10, 11, 12, 13])
        f_hop = base_freq_mhz + chosen_band * band_step_mhz + np.random.uniform(50.0, 450.0)
        pdw_records.append({'toa': t, 'freq': f_hop, 'band_idx': chosen_band, 'pw': 0.8e-6, 'threat': 4.0})
        t += pri_e3

    df = pd.DataFrame(pdw_records).sort_values('toa').reset_index(drop=True)
    return df