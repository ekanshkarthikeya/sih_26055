import numpy as np

class PRIEstimator:
    """
    Estimates stable pulse intervals. Avoids predicting on agile/jittered 
    signals to prevent timing error blowouts.
    """
    def __init__(self, history_len=8, max_stable_jitter_sec=10e-6):
        self.history_len = history_len
        self.max_jitter = max_stable_jitter_sec
        self.band_toas = {}
        self.stable_pris = {}

    def log_pulse(self, band, toa):
        if band not in self.band_toas:
            self.band_toas[band] = []
        self.band_toas[band].append(toa)
        if len(self.band_toas[band]) > self.history_len:
            self.band_toas[band].pop(0)

        if len(self.band_toas[band]) >= 3:
            diffs = np.diff(self.band_toas[band])
            med_diff = np.median(diffs)
            # Only treat as a predictable train if inter-pulse jitter is within limits
            if np.all(np.abs(diffs - med_diff) < self.max_jitter) and med_diff > 15e-6:
                self.stable_pris[band] = float(med_diff)
            else:
                self.stable_pris.pop(band, None)

    def predict_next_pulse(self, band):
        if band in self.stable_pris and band in self.band_toas and self.band_toas[band]:
            last_toa = self.band_toas[band][-1]
            return last_toa + self.stable_pris[band]
        return None