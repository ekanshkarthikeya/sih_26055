import numpy as np

class AdaptiveWhittleScheduler:
    """
    Hardware-Constrained Bayesian Restless Multi-Armed Bandit (RMAB).
    - Online Beta-Bernoulli conjugate occupancy estimation.
    - Real-time Cross-Band Transition Matrix for Agile Frequency Hoppers.
    - Slew-penalized Whittle index with deterministic starvation bounding.
    """
    def __init__(self, num_bands=16, max_consecutive_dwells=4, max_starve_sec=0.060, **kwargs):
        self.K = num_bands
        self.max_consecutive = kwargs.get('min_dwell_steps', max_consecutive_dwells)
        self.max_starve_sec = max_starve_sec
        
        # Conjugate Beta Priors: Beta(alpha, beta)
        self.alpha = np.ones(self.K, dtype=np.float64)
        self.beta = np.ones(self.K, dtype=np.float64)
        
        self.pulse_counters = np.zeros(self.K, dtype=np.int64)
        self.last_visited = np.zeros(self.K, dtype=np.float64)
        self.consecutive_hits = np.zeros(self.K, dtype=np.int64)
        self.consecutive_dwells = np.zeros(self.K, dtype=np.int64)
        
        # Online Cross-Band Transition Matrix: T[from_band, to_band]
        self.transition_matrix = np.ones((self.K, self.K), dtype=np.float64) * 0.05
        self.last_hit_band = None
        self.last_hit_time = 0.0
        
        # Recurrence tracking
        self.burst_times = {k: [] for k in range(self.K)}
        self.est_period = np.zeros(self.K, dtype=np.float64)
        
        self.active_band = 0
        self.cold_start_done = False
        self.cold_start_idx = 0

    def select_action(self, current_time):
        """Returns: (action_band, dwell_time_sec)"""
        # 1. Zero-prior mandatory spectral sweep
        if not self.cold_start_done:
            b = self.cold_start_idx
            self.cold_start_idx += 1
            if self.cold_start_idx >= self.K:
                self.cold_start_done = True
            self.active_band = b
            return b, 40e-6

        last_band = self.active_band
        last_hit = self.consecutive_hits[last_band] > 0

        # 2. Dynamic tracking hold during active pulse train
        if last_hit and self.consecutive_dwells[last_band] < self.max_consecutive:
            self.consecutive_dwells[last_band] += 1
            return self.active_band, 160e-6

        self.consecutive_dwells[last_band] = 0
        starvation = np.maximum(0.0, current_time - self.last_visited)

        # 3. Posterior Occupancy & Hopping Transition Prior
        posterior_belief = self.alpha / (self.alpha + self.beta)
        
        # Cross-band hopping boost: if a hit occurred recently (<2.5 ms), project where it hopped
        hop_boost = np.zeros(self.K)
        if self.last_hit_band is not None and (current_time - self.last_hit_time) < 0.003:
            trans_row = self.transition_matrix[self.last_hit_band]
            norm_trans = trans_row / np.sum(trans_row)
            # Boost target channels predicted by the transition model
            hop_boost = norm_trans * 1.5

        # 4. Periodicity Phase Modulation
        phase_factor = np.ones(self.K, dtype=np.float64)
        for k in range(self.K):
            if self.est_period[k] > 0.01 and len(self.burst_times[k]) > 0:
                dt = current_time - self.burst_times[k][-1]
                phase = dt % self.est_period[k]
                time_to_burst = min(phase, self.est_period[k] - phase)
                phase_factor[k] += 2.0 * np.exp(- (time_to_burst ** 2) / (2 * (0.003 ** 2)))

        # 5. Deterministic Anti-Starvation Ceiling
        critically_starved = starvation >= self.max_starve_sec
        if np.any(critically_starved):
            candidates = np.where(critically_starved)[0]
            if len(candidates) > 1 and last_band in candidates:
                candidates = candidates[candidates != last_band]
            best_band = int(candidates[np.argmax(starvation[candidates])])
            self.active_band = best_band
            self.consecutive_dwells[best_band] = 1
            return self.active_band, 40e-6

        # 6. Slew-Penalized Whittle Index
        density_weight = 1.0 + 0.8 * np.log1p(self.pulse_counters.astype(np.float64))
        exploit = (posterior_belief + hop_boost) * phase_factor * density_weight
        explore = (self.K * 1.1) * (starvation / self.max_starve_sec)

        hop_distances = np.abs(np.arange(self.K) - last_band)
        slew_penalty = 0.25 * np.log2(1.0 + hop_distances)

        scores = exploit + explore - slew_penalty

        if not last_hit:
            scores[last_band] = -1e9

        best_band = int(np.argmax(scores))
        self.active_band = best_band
        self.consecutive_dwells[best_band] = 1

        dwell = 140e-6 if (self.pulse_counters[best_band] > 50 or posterior_belief[best_band] > 0.5) else 40e-6
        return self.active_band, dwell

    def update_beliefs(self, observed_band, hits, current_time):
        self.last_visited[observed_band] = current_time
        
        if hits > 0:
            self.consecutive_hits[observed_band] += hits
            self.pulse_counters[observed_band] += hits
            self.alpha[observed_band] += min(hits, 10.0)

            # Update cross-band transition matrix if hopping between channels
            if self.last_hit_band is not None and self.last_hit_band != observed_band:
                dt_hop = current_time - self.last_hit_time
                if dt_hop < 0.005:  # Inter-hop window (< 5ms)
                    self.transition_matrix[self.last_hit_band, observed_band] += 1.0

            self.last_hit_band = observed_band
            self.last_hit_time = current_time

            # Recurrence tracking
            bt = self.burst_times[observed_band]
            if not bt or (current_time - bt[-1]) > 0.015:
                bt.append(current_time)
                if len(bt) >= 3:
                    intervals = np.diff(bt[-4:])
                    med = float(np.median(intervals))
                    if 0.015 <= med <= 2.0:
                        self.est_period[observed_band] = med
        else:
            self.consecutive_hits[observed_band] = 0
            self.beta[observed_band] += 2.0

        for k in range(self.K):
            if k != observed_band:
                self.consecutive_hits[k] = 0
                self.alpha[k] = max(1.0, self.alpha[k] * 0.995)
                self.beta[k] = max(1.0, self.beta[k] * 0.995)