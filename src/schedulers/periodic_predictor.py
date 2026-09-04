import numpy as np

class AutonomousPeriodicDetector:
    """
    Monitors all bands dynamically without prior intelligence. Identifies
    periodic antenna scan intervals across any sub-band and schedules overrides.
    """
    def __init__(self, num_bands=16, min_bursts_to_lock=3, beam_dwell_sec=0.015):
        self.num_bands = num_bands
        self.min_bursts = min_bursts_to_lock
        self.beam_dwell = beam_dwell_sec
        self.burst_history = {b: [] for b in range(num_bands)}
        self.periods = {}
        self.next_bursts = {}

    def log_hit(self, band, timestamp):
        # Cluster pulses separated by >25ms into antenna scan illuminations
        history = self.burst_history[band]
        if not history or (timestamp - history[-1]) > 0.025:
            history.append(timestamp)
            if len(history) >= self.min_bursts:
                intervals = np.diff(history[-4:])
                med_int = float(np.median(intervals))
                # Valid antenna rotation periods typically range between 100ms and 5.0s
                if 0.10 <= med_int <= 5.0 and np.all(np.abs(intervals - med_int) < 0.05):
                    self.periods[band] = med_int
                    self.next_bursts[band] = timestamp + med_int

    def get_imminent_override(self, current_time):
        """Finds any band with an incoming antenna beam illumination."""
        for band, next_t in list(self.next_bursts.items()):
            delta = next_t - current_time
            if -0.001 <= delta <= self.beam_dwell:
                return band, next_t
            if current_time > next_t + self.beam_dwell:
                # Advance projection to next rotation cycle
                self.next_bursts[band] += self.periods[band]
        return None, None