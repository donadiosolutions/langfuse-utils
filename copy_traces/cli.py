"""Command-line interface for copying Langfuse traces into a dataset."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console

from .client import LangfuseCredentials, LangfuseHttpClient
from .importer import TraceFilter, TraceImporter

PLACEHOLDER_URL = "<LANGFUSE_BASE_URL>"
PLACEHOLDER_PK = "<LANGFUSE_PUBLIC_KEY>"
PLACEHOLDER_SK = "<LANGFUSE_SECRET_KEY>"
PLACEHOLDER_PROJECT = "<LANGFUSE_PROJECT_ID>"

app = typer.Typer(help="Copy Langfuse traces into a dataset.")
console = Console()

# Load .env from repository root if present.
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)


def _build_credentials(
    url: str | None,
    public_key: str | None,
    secret_key: str | None,
) -> LangfuseCredentials:
    resolved_url = url or os.getenv("LANGFUSE_BASE_URL", PLACEHOLDER_URL)
    resolved_pk = public_key or os.getenv("LANGFUSE_PUBLIC_KEY", PLACEHOLDER_PK)
    resolved_sk = secret_key or os.getenv("LANGFUSE_SECRET_KEY", PLACEHOLDER_SK)
    resolved_project = os.getenv("LANGFUSE_PROJECT_ID", PLACEHOLDER_PROJECT)

    for name, value in {
        "LANGFUSE_BASE_URL": resolved_url,
        "LANGFUSE_PUBLIC_KEY": resolved_pk,
        "LANGFUSE_SECRET_KEY": resolved_sk,
        "LANGFUSE_PROJECT_ID": resolved_project,
    }.items():
        if value.startswith("<LANGFUSE_"):
            raise typer.BadParameter(f"{name} is not set. Provide it via env var.")

    return LangfuseCredentials(
        url=resolved_url,
        public_key=resolved_pk,
        secret_key=resolved_sk,
        project_id=resolved_project,
    )


@app.command()
def copy_traces(
    dataset: str = typer.Argument(..., help="Destination dataset name."),
    range_days: int = typer.Option(..., "--range", "-r", help="Time range in days to pull traces."),
    environment: Optional[str] = typer.Option(
        None,
        "--environment",
        "-e",
        help="Filter traces by environment.",
    ),
    model: Optional[str] = typer.Option(
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
    url: Optional[str] = typer.Option(
        None,
        "--url",
        envvar="LANGFUSE_BASE_URL",
        help="Langfuse base URL.",
    ),
    public_key: Optional[str] = typer.Option(
        None,
        "--pk",
        envvar="LANGFUSE_PUBLIC_KEY",
        help="Langfuse public key.",
    ),
    secret_key: Optional[str] = typer.Option(
        None,
        "--sk",
        envvar="LANGFUSE_SECRET_KEY",
        help="Langfuse secret key.",
    ),
) -> None:
    """Copy traces into a dataset, creating the dataset if needed."""
    credentials = _build_credentials(url, public_key, secret_key)
    trace_filter = TraceFilter(range_days=range_days, environment=environment, model=model)

    with LangfuseHttpClient(credentials) as client:
        importer = TraceImporter(
            client=client,
            dataset_name=dataset,
            trace_filter=trace_filter,
            dry_run=dry_run,
        )
        stats = importer.run()
    console.print(
        f"[bold green]Imported[/bold green]: {stats.imported} | "
        f"[yellow]Skipped[/yellow]: {stats.skipped} | "
        f"[red]Failed[/red]: {stats.failed}",
    )


def main() -> None:  # pragma: no cover - thin wrapper
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
