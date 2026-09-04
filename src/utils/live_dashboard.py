from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text

class EWLiveDashboard:
    def __init__(self, num_bands=16):
        self.num_bands = num_bands

    def generate_display(self, current_time, current_band, p_int, reward, hits, beliefs, mode="SCAN"):
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", size=18),
            Layout(name="footer", size=3)
        )

        # Header Panel
        header_text = Text(
            f"DRDO SMART EW SCANNER | Time: {current_time*1e3:6.1f} ms | Mode: {mode} | Active Band: {current_band}",
            style="bold white on blue", justify="center"
        )
        layout["header"].update(Panel(header_text))

        # Channel Grid
        table = Table(title="Instantaneous Sub-band Monitoring Matrix", expand=True)
        table.add_column("Band", justify="center", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Belief P(active)", justify="center")
        table.add_column("Band", justify="center", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Belief P(active)", justify="center")

        half = self.num_bands // 2
        for i in range(half):
            # Column Pair 1
            b1 = i
            is_tuned1 = (b1 == current_band)
            status1 = "[bold green]TUNED (RX)[/bold green]" if is_tuned1 else "[dim]STANDBY[/dim]"
            bar1 = "█" * int(beliefs[b1] * 10)
            
            # Column Pair 2
            b2 = i + half
            is_tuned2 = (b2 == current_band)
            status2 = "[bold green]TUNED (RX)[/bold green]" if is_tuned2 else "[dim]STANDBY[/dim]"
            bar2 = "█" * int(beliefs[b2] * 10)

            table.add_row(
                f"CH-{b1:02d}", status1, f"[{bar1:<10}] {beliefs[b1]:.2f}",
                f"CH-{b2:02d}", status2, f"[{bar2:<10}] {beliefs[b2]:.2f}"
            )

        layout["body"].update(Panel(table))

        # Metrics Footer
        footer_text = Text(
            f"P_int: {p_int:5.2f}% | Intercepted Pulses: {hits:<7d} | Accumulated Reward: {reward:<9.1f}",
            style="bold yellow", justify="center"
        )
        layout["footer"].update(Panel(footer_text))

        return layout