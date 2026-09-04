import numpy as np

class RFEnvironment:
    """
    Hardware-Realistic EW RF Environment.
    Models synthesizer tuning slew, PLL lock times, front-end blanking,
    and receiver probability of detection (Pd) / thermal fading.
    """
    def __init__(self, pdw_df, num_bands=16, default_dwell_sec=45e-6, receiver_pd=1.0):
        self.df = pdw_df.sort_values('toa').reset_index(drop=True)
        self.num_bands = num_bands
        self.default_dwell = default_dwell_sec
        self.receiver_pd = receiver_pd
        self.current_time = 0.0
        self.current_band = 0
        
        self.toas = self.df['toa'].to_numpy()
        self.bands = self.df['band_idx'].to_numpy()
        self.threats = self.df['threat'].to_numpy()
        self.emitters = self.df['emitter_id'].to_numpy()
        self.pws = self.df['pw_us'].to_numpy() if 'pw_us' in self.df.columns else np.zeros(len(self.df))
        self.total_pulses = len(self.df)
        
        self.cursor = 0
        self.total_settling_time = 0.0

    def calculate_settle_time(self, from_band, to_band):
        """
        Models YIG/VCO frequency synthesizer lock dynamics.
        Base PLL settling: 5 us
        Log-distance RF step penalty: up to 45 us for wide-band excursions
        """
        if from_band == to_band:
            return 0.0
        hop_distance = abs(to_band - from_band)
        settle_us = 5.0 + 12.0 * np.log2(1.0 + hop_distance)
        return settle_us * 1e-6

    def step(self, action_band, dwell_time_sec=None):
        dwell_len = dwell_time_sec if dwell_time_sec is not None else self.default_dwell
        settle_sec = self.calculate_settle_time(self.current_band, action_band)
        self.total_settling_time += settle_sec
        
        # Receiver is blind during synthesizer settling (blanking window)
        dwell_start = self.current_time + settle_sec
        dwell_end = dwell_start + dwell_len

        # Fast forward cursor past blanking period
        while self.cursor < self.total_pulses and self.toas[self.cursor] < dwell_start:
            self.cursor += 1

        i_start = self.cursor
        i_end = i_start
        while i_end < self.total_pulses and self.toas[i_end] <= dwell_end:
            i_end += 1

        detected_pulses = 0
        reward = 0.0
        intercepted_emitters = []
        intercepted_toas = []
        intercepted_pws = []

        if i_end > i_start:
            window_bands = self.bands[i_start:i_end]
            band_matches = (window_bands == action_band)
            
            # Apply receiver sensitivity / SNR dropout if Pd < 1.0
            if self.receiver_pd < 1.0 and np.any(band_matches):
                det_mask = np.random.binomial(1, self.receiver_pd, size=len(band_matches)).astype(bool)
                hits = band_matches & det_mask
            else:
                hits = band_matches

            detected_pulses = int(np.sum(hits))
            if detected_pulses > 0:
                reward = float(np.sum(self.threats[i_start:i_end][hits]))
                intercepted_emitters = self.emitters[i_start:i_end][hits].tolist()
                intercepted_toas = self.toas[i_start:i_end][hits].tolist()
                intercepted_pws = self.pws[i_start:i_end][hits].tolist()

        self.current_time = dwell_end
        self.current_band = action_band
        done = (self.current_time >= self.toas[-1]) or (self.cursor >= self.total_pulses)
        observation = 1 if detected_pulses > 0 else 0

        pdw_dict = {
            'toas': intercepted_toas,
            'pws': intercepted_pws,
            'emitters': intercepted_emitters
        }

        return observation, reward, detected_pulses, pdw_dict, dwell_start, done