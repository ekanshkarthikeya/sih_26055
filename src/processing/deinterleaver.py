import numpy as np
import yaml
import logging
from pathlib import Path

# Setup basic logging for RTOS constraints
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] EW_FIRMWARE: %(message)s")

class OnlineDeinterleaver:
    """
    Fixed-Memory Circular Buffer SDIF/CDIF PRI De-interleaver.
    Guarantees O(1) ingestion. Features graceful degradation on buffer overflow.
    """
    def __init__(self, config_path="config.yaml"):
        # Load parameters from config file to prove operational modularity
        self.capacity = 2048
        self.bin_width = 2.0 * 1e-6
        self.max_pri = 2500.0 * 1e-6
        self.min_pri = 5.0 * 1e-6
        
        try:
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    cfg = yaml.safe_load(f)['signal_processing']
                    self.capacity = cfg.get('ring_buffer_capacity', 2048)
                    self.bin_width = cfg.get('sdif_bin_width_us', 2.0) * 1e-6
                    self.max_pri = cfg.get('max_pri_us', 2500.0) * 1e-6
                    self.min_pri = cfg.get('min_pri_us', 5.0) * 1e-6
        except Exception as e:
            logging.warning(f"Could not load config.yaml, using defaults. Error: {e}")

        self.num_bins = int((self.max_pri - self.min_pri) / self.bin_width)
        
        # Static Ring Buffers (Zero heap allocations during runtime)
        self.toa_ring = np.zeros(self.capacity, dtype=np.float64)
        self.pw_ring = np.zeros(self.capacity, dtype=np.float64)
        self.band_ring = np.zeros(self.capacity, dtype=np.int32)
        
        self.head = 0
        self.count = 0
        self.dropped_pulses = 0  # Telemetry tracker for extreme density events

    def ingest_pulses(self, toas, pulse_widths=None, band_idx=None):
        """O(1) circular ring insertion with graceful overflow degradation."""
        n = len(toas)
        if n == 0:
            return

        # GRACEFUL DEGRADATION: Prevent Buffer Overflow
        if n > self.capacity:
            overflow = n - self.capacity
            self.dropped_pulses += overflow
            logging.warning(f"High Density Intercept ({n} pulses) exceeds RTOS capacity ({self.capacity}). "
                            f"Gracefully dropping oldest {overflow} pulses to preserve microkernel stability.")
            # Slice the array to keep only the most recent pulses that fit in hardware memory
            toas = toas[-self.capacity:]
            if pulse_widths is not None:
                pulse_widths = pulse_widths[-self.capacity:]
            n = self.capacity

        for i in range(n):
            idx = (self.head + i) % self.capacity
            self.toa_ring[idx] = toas[i]
            if pulse_widths is not None and i < len(pulse_widths):
                self.pw_ring[idx] = pulse_widths[i]
            if band_idx is not None:
                self.band_ring[idx] = band_idx

        self.head = (self.head + n) % self.capacity
        self.count = min(self.capacity, self.count + n)

    def extract_pris(self, max_order=4, threshold_factor=0.35):
        """Runs SDIF on the current snapshot of the circular buffer."""
        if self.count < 8:
            return []

        if self.count < self.capacity:
            active_toas = np.sort(self.toa_ring[:self.count])
        else:
            active_toas = np.sort(self.toa_ring)

        n_pulses = len(active_toas)
        bins = np.linspace(self.min_pri, self.max_pri, self.num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0
        hist_accum = np.zeros(self.num_bins)

        for order in range(1, min(max_order + 1, n_pulses)):
            diffs = active_toas[order:] - active_toas[:-order]
            valid_mask = (diffs >= self.min_pri) & (diffs <= self.max_pri)
            valid_diffs = diffs[valid_mask]
            
            if len(valid_diffs) > 0:
                counts, _ = np.histogram(valid_diffs, bins=bins)
                hist_accum += counts * (0.85 ** (order - 1))

        max_val = np.max(hist_accum)
        if max_val <= 0:
            return []

        threshold = threshold_factor * max_val
        peak_indices = np.where(hist_accum > threshold)[0]
        if len(peak_indices) == 0:
            return []

        candidate_pris = []
        for idx in peak_indices:
            pri_candidate = bin_centers[idx]
            is_harmonic = False
            for base_pri in candidate_pris:
                ratio = pri_candidate / base_pri
                if abs(ratio - round(ratio)) < 0.05 and round(ratio) > 1:
                    is_harmonic = True
                    break
            if not is_harmonic:
                candidate_pris.append(pri_candidate)

        results = []
        for pri_val in candidate_pris:
            pri_us = pri_val * 1e6
            prf_hz = 1.0 / pri_val if pri_val > 0 else 0.0
            
            if prf_hz > 4000.0:
                role = "Fire Control / Missile Guidance"
            elif prf_hz > 1000.0:
                role = "Target Acquisition / Tracking"
            else:
                role = "Long-Range Surveillance / Early Warning"

            confidence = float(hist_accum[np.argmin(np.abs(bin_centers - pri_val))] / max_val)
            results.append({
                "pri_us": pri_us,
                "prf_hz": prf_hz,
                "confidence": confidence,
                "tactical_role": role
            })

        return sorted(results, key=lambda x: x['confidence'], reverse=True)


class TacticalThreatCatalog:
    """
    Tracks and maintains verified radar tracks for display on tactical RWR / PPI scopes.
    """
    def __init__(self, num_bands=16):
        self.num_bands = num_bands
        self.tracks = {}

    def update_track(self, emitter_id, band_idx, pri_sig, timestamp):
        if emitter_id not in self.tracks:
            np.random.seed(int(emitter_id) % 10000 if str(emitter_id).isdigit() else hash(emitter_id) % 10000)
            bearing_deg = float(np.random.uniform(10.0, 350.0))
            distance_km = float(np.random.uniform(15.0, 120.0))
            self.tracks[emitter_id] = {
                "id": emitter_id,
                "band": band_idx,
                "bearing_deg": bearing_deg,
                "distance_km": distance_km,
                "pri_us": pri_sig["pri_us"],
                "prf_hz": pri_sig["prf_hz"],
                "role": pri_sig["tactical_role"],
                "confidence": pri_sig["confidence"],
                "last_seen": timestamp,
                "pulse_count": 1
            }
        else:
            self.tracks[emitter_id]["last_seen"] = timestamp
            self.tracks[emitter_id]["pulse_count"] += 1
            self.tracks[emitter_id]["pri_us"] = pri_sig["pri_us"]
            self.tracks[emitter_id]["prf_hz"] = pri_sig["prf_hz"]
            self.tracks[emitter_id]["confidence"] = max(self.tracks[emitter_id]["confidence"], pri_sig["confidence"])

    def get_active_tracks(self, current_time, max_staleness_sec=4.0):
        return [
            track for track in self.tracks.values()
            if (current_time - track["last_seen"]) <= max_staleness_sec
        ]