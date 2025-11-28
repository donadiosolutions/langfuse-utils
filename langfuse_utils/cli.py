"""Aggregate CLI entrypoint for Langfuse utilities."""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv

from copy_traces import cli as copy_traces_cli

# Load .env from repository root if present.
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

app = typer.Typer(help="Langfuse utilities CLI.")


@app.callback()
def main_callback() -> None:
    """Langfuse utilities command group."""


@app.command("copy-traces", help=copy_traces_cli.copy_traces.__doc__)
def copy_traces_command(
    dataset: str,
    range_days: int = typer.Option(..., "--range", "-r", help="Time range in days to pull traces."),
    environment: str | None = typer.Option(
        None,
        "--environment",
        "-e",
        help="Filter traces by environment.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Filter to traces whose observations used this model.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Do not write; only show what would happen.",
    ),
    url: str | None = typer.Option(
        None,
        "--url",
        envvar="LANGFUSE_BASE_URL",
        help="Langfuse base URL.",
    ),
    public_key: str | None = typer.Option(
        None,
        "--pk",
        envvar="LANGFUSE_PUBLIC_KEY",
        help="Langfuse public key.",
    ),
    secret_key: str | None = typer.Option(
        None,
        "--sk",
        envvar="LANGFUSE_SECRET_KEY",
        help="Langfuse secret key.",
    ),
) -> None:
    copy_traces_cli.copy_traces(
        dataset=dataset,
        range_days=range_days,
        environment=environment,
        model=model,
        dry_run=dry_run,
        url=url,
        public_key=public_key,
        secret_key=secret_key,
    )


def main() -> None:  # pragma: no cover - thin wrapper
    app()


__all__ = ["app", "main"]
