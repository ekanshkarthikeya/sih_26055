import json
from datetime import datetime
import pandas as pd
import numpy as np

# Reference Threat Radar Library (MIL-STD RF Parameter Database)
MIL_RADAR_LIBRARY = [
    {
        "designation": "AN/APG-68 (Multimode Fighter Radar)",
        "role": "Airborne Intercept / Target Tracking",
        "threat_level": "CRITICAL",
        "freq_band_min": 8, "freq_band_max": 12,
        "pri_min_us": 60.0, "pri_max_us": 120.0
    },
    {
        "designation": "9S18M1 Kupol (Snow Drift - SA-11)",
        "role": "SAM Target Acquisition",
        "threat_level": "HIGH",
        "freq_band_min": 2, "freq_band_max": 5,
        "pri_min_us": 150.0, "pri_max_us": 250.0
    },
    {
        "designation": "30N6E Flap Lid (S-300 PMU)",
        "role": "SAM Fire Control / Missile Guidance",
        "threat_level": "CRITICAL",
        "freq_band_min": 7, "freq_band_max": 10,
        "pri_min_us": 8.0, "pri_max_us": 30.0
    },
    {
        "designation": "P-40 Long Track (Surveillance)",
        "role": "Early Warning / Long-Range Surveillance",
        "threat_level": "MEDIUM",
        "freq_band_min": 0, "freq_band_max": 2,
        "pri_min_us": 1000.0, "pri_max_us": 2500.0
    }
]

class EOBExporter:
    """
    Generates standardized MIL-STD Tactical Electronic Order of Battle (EOB) payloads
    from de-interleaved emitter parameter buffers.
    """
    @staticmethod
    def match_threat(band_idx, pri_us):
        for ref in MIL_RADAR_LIBRARY:
            if (ref["freq_band_min"] <= band_idx <= ref["freq_band_max"]) and \
               (ref["pri_min_us"] <= pri_us <= ref["pri_max_us"]):
                return ref["designation"], ref["threat_level"], ref["role"]
        
        # Heuristic fallback if not directly matched to known library
        if pri_us < 35.0:
            return "Unidentified High-PRF Target Illuminator", "CRITICAL", "Fire Control"
        elif pri_us < 200.0:
            return "Unidentified Track-While-Scan Emitter", "HIGH", "Target Acquisition"
        else:
            return "Unidentified Coarse Surveillance Emitter", "MEDIUM", "Surveillance"

    @classmethod
    def generate_report(cls, tracks, mission_time_sec, num_pulses, pint_pct):
        eob_records = []
        for t in tracks:
            match_name, threat_level, threat_role = cls.match_threat(t["band"], t["pri_us"])
            eob_records.append({
                "Track_ID": f"TRK-{int(t['id']):04d}" if str(t['id']).isdigit() else f"TRK-{t['id']}",
                "Designation": match_name,
                "Threat_Level": threat_level,
                "Combat_Role": threat_role,
                "Sub_Band": int(t["band"]),
                "Bearing_Deg": round(t["bearing_deg"], 1),
                "Distance_km": round(t["distance_km"], 1),
                "PRI_us": round(t["pri_us"], 2),
                "PRF_Hz": round(t["prf_hz"], 1),
                "Confidence_Pct": round(t["confidence"] * 100.0, 1),
                "Last_Seen_s": round(t["last_seen"], 3)
            })

        tactical_document = {
            "mission_metadata": {
                "system": "DRDO Smart Scan ES Receiver (Adaptive RMAB Engine)",
                "classification": "RESTRICTED // EW EVALUATION ONLY",
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "mission_duration_sec": float(round(mission_time_sec, 3)),
                "total_pulses_ingested": int(num_pulses),
                "probability_of_intercept_pct": float(round(pint_pct, 2))
            },
            "electronic_order_of_battle": eob_records
        }
        return tactical_document, pd.DataFrame(eob_records)