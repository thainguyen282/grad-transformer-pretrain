from datetime import datetime
from rich.table import Table
from config import ARG_GROUPS
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
    console.log(f"{prefix}[bold cyan]{ts}[/bold cyan] {tag}[green]{message}[/green]")


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


def extract_group_keys(add_fn):
    """Create a temp parser to extract arg names from a function."""
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    add_fn(parser)
    return [a.dest for a in parser._actions if a.dest != "help"]


def print_args(args, console: Console):
    data = vars(args)
    used_keys = set()

    for group_name, fn in ARG_GROUPS.items():
        keys = extract_group_keys(fn)

        table = Table(title=group_name)
        table.add_column("Argument", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        has_rows = False
        for k in keys:
            if k in data:
                table.add_row(k, str(data[k]))
                used_keys.add(k)
                has_rows = True

        if has_rows:
            console.print(table)

    # catch anything unexpected
    remaining = [k for k in data if k not in used_keys]
    if remaining:
        table = Table(title="Other")
        table.add_column("Argument", style="cyan")
        table.add_column("Value", style="magenta")

        for k in remaining:
            table.add_row(k, str(data[k]))

        console.print(table)
