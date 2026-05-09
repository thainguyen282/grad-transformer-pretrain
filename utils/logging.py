from datetime import datetime
from rich.console import Console

def log_rich(
    message: str,
    console: Console,
    label: str | None = None,
    newline: bool = False,
):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = "\n" if newline else ""
    tag = f"[bold magenta][{label}][/bold magenta] " if label else ""
    console.print(f"{prefix}[bold cyan]{ts}[/bold cyan] {tag}[green]{message}[/green]")


def log_build_start(
    description: str,
    console: Console,
    label: str | None = None,
    newline: bool = False,
):
    """Log the start of a potentially slow build/generate step with an explicit timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = "\n" if newline else ""
    tag = f"[bold magenta][{label}][/bold magenta] " if label else ""
    console.print(
        f"{prefix}[bold cyan]{ts}[/bold cyan] {tag}"
        f"[bold yellow]start:[/bold yellow] [white]{description}[/white]"
    )