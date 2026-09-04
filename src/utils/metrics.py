import numpy as np

class EWMetricsTracker:
    def __init__(self, num_bands):
        self.num_bands = num_bands
        self.band_visits = {k: [] for k in range(num_bands)}
        self.first_intercept_times = {}
        self.emitter_truth_starts = {}
        self.predicted_toa_errors = []

    @property
    def first_intercept(self):
        return self.first_intercept_times

    @property
    def first_intercepts(self):
        return self.first_intercept_times

    def register_truth_emitter(self, emitter_id, start_time):
        if emitter_id not in self.emitter_truth_starts:
            self.emitter_truth_starts[emitter_id] = start_time

    def log_dwell(self, band, timestamp):
        self.band_visits[band].append(timestamp)

    def log_intercept(self, emitter_id, toa, predicted_toa=None):
        if emitter_id not in self.first_intercept_times:
            self.first_intercept_times[emitter_id] = toa
        if predicted_toa is not None:
            err = abs(predicted_toa - toa)
            self.predicted_toa_errors.append(err)

    def evaluate(self, total_truth_pulses, intercepted_pulses):
        pint = (intercepted_pulses / total_truth_pulses * 100.0) if total_truth_pulses > 0 else 0.0

        latencies_ms = []
        for eid, t_start in self.emitter_truth_starts.items():
            if eid in self.first_intercept_times:
                latencies_ms.append((self.first_intercept_times[eid] - t_start) * 1e3)
        avg_latency = float(np.mean(latencies_ms)) if latencies_ms else float('inf')

        variances = []
        for b, timestamps in self.band_visits.items():
            if len(timestamps) > 2:
                dt = np.diff(timestamps)
                variances.append(np.var(dt))
        avg_revisit_var = float(np.mean(variances)) if variances else 0.0

        # Mean intercept error in microseconds
        avg_time_err_us = float(np.mean(self.predicted_toa_errors) * 1e6) if self.predicted_toa_errors else 0.0

        return {
            "P_int": pint,
            "Latency_ms": avg_latency,
            "Revisit_Var": avg_revisit_var,
            "Time_Error_us": avg_time_err_us
        }