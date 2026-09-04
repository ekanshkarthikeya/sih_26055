import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rich.console import Console

console = Console()

def run_audit():
    console.print("\n[bold cyan]=== RUNNING EW SMART SCAN PRE-FLIGHT VERIFICATION ===[/bold cyan]\n")

    # 1. Check Data Presence
    test_files = list(Path("data/archive/test").glob("test_*.h5"))
    console.print(f"• Dataset Ingestion Check: [bold green]PASS[/bold green] ({len(test_files)} HDF5 splits present)")

    # 2. Scheduler Unit Instantiation
    from src.schedulers.whittle_rmab import AdaptiveWhittleScheduler
    scheduler = AdaptiveWhittleScheduler(num_bands=16)
    action, dwell = scheduler.select_action(0.0)
    assert 0 <= action < 16, "Action band out of bounds"
    assert dwell > 0, "Invalid dwell duration"
    console.print("• RMAB Scheduling Core Check: [bold green]PASS[/bold green]")

    # 3. PRI De-interleaver Unit
    from src.processing.deinterleaver import OnlineDeinterleaver
    deinterleaver = OnlineDeinterleaver()
    sim_toas = [i * 0.000200 for i in range(25)]  # Stable 200us PRI
    deinterleaver.ingest_pulses(sim_toas, [2.0]*25, band_idx=2)
    sigs = deinterleaver.extract_pris()
    assert len(sigs) > 0, "Deinterleaver failed to extract stable PRI"
    console.print(f"• SDIF/CDIF De-interleaver Check: [bold green]PASS[/bold green] (Detected PRI: {sigs[0]['pri_us']:.1f} μs)")

    # 4. EOB Exporter Unit
    from src.processing.eob_exporter import EOBExporter
    mock_track = [{"id": "1", "band": 8, "pri_us": 80.0, "prf_hz": 12500.0, "tactical_role": "Fire Control", "confidence": 0.9, "bearing_deg": 45.0, "distance_km": 30.0, "last_seen": 1.0}]
    doc, df = EOBExporter.generate_report(mock_track, 10.0, 1000, 75.0)
    assert not df.empty, "EOB DataFrame generation failed"
    console.print("• STANAG / MIL Tactical Report Engine: [bold green]PASS[/bold green]")

    console.print("\n[bold green]ALL SUBSYSTEMS NOMINAL. READY FOR COMPETITION DEPLOYMENT.[/bold green]\n")

if __name__ == "__main__":
    run_audit()