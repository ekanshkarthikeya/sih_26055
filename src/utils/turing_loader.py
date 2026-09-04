import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

class TuringDatasetLoader:
    def __init__(self, cache_dir="./data", num_bands=16, f_min_mhz=500.0, f_max_mhz=18000.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.num_bands = num_bands
        self.f_min = f_min_mhz
        self.f_max = f_max_mhz
        self.band_width = (f_max_mhz - f_min_mhz) / num_bands

    def assign_bands(self, df):
        """Maps continuous carrier frequencies (MHz) to discrete receiver sub-bands [0 .. K-1]."""
        bands = np.floor((df['freq_mhz'] - self.f_min) / self.band_width).astype(int)
        df['band_idx'] = np.clip(bands, 0, self.num_bands - 1)
        return df

    def load_from_h5(self, h5_filepath):
        """Loads and parses official Turing HDF5 radar pulse data using verified schema."""
        print(f"[Dataset Loader] Parsing official Turing HDF5: {h5_filepath}")
        with h5py.File(h5_filepath, 'r') as f:
            data_arr = f['data'][:]
            labels_arr = f['labels'][:].flatten()

        # Schema: ['UTCTime', 'RF', 'PulseWidth', 'AOA', 'PA']
        # data_arr shape is (N, 5) or (N, 6)
        df = pd.DataFrame({
            'toa_raw': data_arr[:, 0],
            'freq_mhz': data_arr[:, 1],
            'pw_us': data_arr[:, 2],
            'aoa_deg': data_arr[:, 3],
            'amp_db': data_arr[:, 4] if data_arr.shape[1] > 4 else -40.0,
            'emitter_id': labels_arr
        })

        # Normalize ToA to relative seconds starting from t=0
        t_min = df['toa_raw'].min()
        df['toa'] = df['toa_raw'] - t_min
        # Scale to seconds if timestamps are in microseconds
        if df['toa'].max() > 1000.0:
            df['toa'] = df['toa'] * 1e-6

        # Prioritize tracking & fire-control threats (higher frequencies = higher threat)
        df['threat'] = 1.0 + (df['freq_mhz'] / 4000.0)
        
        df = self.assign_bands(df.sort_values('toa').reset_index(drop=True))
        print(f"[Dataset Loader] Ingested {len(df)} pulses across {df['emitter_id'].nunique()} emitters. Time span: {df['toa'].max():.3f}s")
        return df

    def load_or_generate(self, filename="turing_stare_sample.parquet", duration_sec=3.0):
        # 1. Prefer downloaded official H5 if present
        h5_default = self.cache_dir / "archive/test/test_0.h5"
        if h5_default.exists():
            return self.load_from_h5(h5_default)

        # 2. Fallback to cached parquet
        target_path = self.cache_dir / filename
        if target_path.exists():
            print(f"[Dataset Loader] Loading cached ground truth: {target_path}")
            df = pd.read_parquet(target_path)
            return self.assign_bands(df)

        return self._generate_fallback(target_path, duration_sec)

    def _generate_fallback(self, target_path, duration_sec):
        print(f"[Dataset Loader] Generating standard synthetic stare sequence ({duration_sec}s)...")
        records = []
        t = 0.001
        while t < duration_sec:
            records.append({'toa': t, 'freq_mhz': 9400.0, 'pw_us': 0.8, 'emitter_id': 101, 'threat': 5.0})
            t += 40e-6
        df = pd.DataFrame(records)
        df = self.assign_bands(df.sort_values('toa').reset_index(drop=True))
        df.to_parquet(target_path, index=False)
        return df