"""Trace importer logic and data structures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn

from .client import LangfuseError, LangfuseHttpClient

console = Console()


@dataclass(frozen=True)
class TraceFilter:
    """User supplied filter options."""

    range_days: int
    environment: str | None = None
    model: str | None = None

    def window(self, *, now: datetime | None = None) -> tuple[str, str]:
        now = now or datetime.now(timezone.utc)
        start = now - timedelta(days=self.range_days)
        return start.isoformat(), now.isoformat()


@dataclass
class ImportStats:
    """Summary of an import run."""

    imported: int = 0
    skipped: int = 0
    failed: int = 0

    @property
    def total(self) -> int:
        return self.imported + self.skipped + self.failed


def _decode_if_json(value: Any) -> Any:
    """Best-effort decode for double-encoded payloads stored as strings."""
    if not isinstance(value, str):
        return value
    candidate = value.strip()
    for _ in range(2):
        if not candidate or candidate[0] not in '[{"':
            break
        try:
            decoded = json.loads(candidate)
        except json.JSONDecodeError:
            break
        if not isinstance(decoded, str):
            return decoded
        candidate = decoded.strip()
    return value


def _first_model(trace: dict[str, Any]) -> str | None:
    for observation in trace.get("observations", []) or []:
        model = observation.get("model")
        if model:
            return str(model)
    return None


def _dataset_item_id(dataset_name: str, trace_id: str) -> str:
    # uuid5 keeps IDs deterministic while avoiding collisions across datasets.
    return uuid5(NAMESPACE_URL, f"langfuse-utils/{dataset_name}/{trace_id}").hex


class TraceImporter:
    """Coordinate fetching traces and inserting them into a dataset."""

    def __init__(
        self,
        client: LangfuseHttpClient,
        dataset_name: str,
        trace_filter: TraceFilter,
        *,
        dry_run: bool = False,
    ) -> None:
        self.client = client
        self.dataset_name = dataset_name
        self.trace_filter = trace_filter
        self.dry_run = dry_run

    def run(self) -> ImportStats:
        """Execute the import process."""
        stats = ImportStats()
        dataset_status = self.client.ensure_dataset(self.dataset_name, dry_run=self.dry_run)
        if dataset_status.existed:
            label = dataset_status.id or "unknown-id"
            console.log(f"Dataset '{self.dataset_name}' exists (id {label}).")
        elif dataset_status.dry_run_creation:
            console.log(
                f"[yellow][dry-run][/yellow] Dataset '{self.dataset_name}' not found; "
                "would create it.",
            )
        else:
            label = dataset_status.id or "unknown-id"
            console.log(f"Created dataset '{self.dataset_name}' (id {label}).")
        start, end = self.trace_filter.window()
        message_parts = [f"Scanning traces from {start} to {end}"]
        if self.trace_filter.environment:
            message_parts.append(f"in environment '{self.trace_filter.environment}'")
        if self.trace_filter.model:
            message_parts.append(f"filtered by model '{self.trace_filter.model}'")
        if self.dry_run:
            message_parts.append("(dry run)")
        console.log(" ".join(message_parts))

        traces, total_traces = self.client.list_traces_with_total(
            from_timestamp=start,
            to_timestamp=end,
            environment=self.trace_filter.environment,
        )

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=True,
        ) as progress:
            task_id = progress.add_task("Importing traces", total=total_traces)

            for trace in traces:
                trace_identifier = trace.get("id")
                if not trace_identifier:
                    stats.failed += 1
                    progress.advance(task_id)
                    continue
                trace_id = str(trace_identifier)

                if self.client.dataset_item_exists(self.dataset_name, trace_id=trace_id):
                    stats.skipped += 1
                    progress.advance(task_id)
                    continue

                full_trace = trace
                needs_detail = (
                    self.trace_filter.model is not None
                    or "input" not in trace
                    or "output" not in trace
                )
                if needs_detail:
                    full_trace = self.client.get_trace(trace_id)

                if self.trace_filter.model:
                    model = _first_model(full_trace)
                    if model != self.trace_filter.model:
                        stats.skipped += 1
                        progress.advance(task_id)
                        continue

                payload = self._build_dataset_item(full_trace)

                if self.dry_run:
                    stats.imported += 1
                    console.log(
                        f"[yellow][dry-run][/yellow] would upsert trace {trace_id} "
                        f"into dataset {self.dataset_name} (item {payload['id']})",
                    )
                    progress.advance(task_id)
                    continue

                try:
                    self.client.create_dataset_item(payload)
                    if not self.client.dataset_item_exists(
                        self.dataset_name,
                        item_id=payload["id"],
                    ):
                        raise LangfuseError(
                            "Verification failed: dataset item missing after creation",
                        )
                    stats.imported += 1
                    console.log(
                        f"Upserted trace {trace_id} into dataset {self.dataset_name} "
                        f"(item {payload['id']})",
                    )
                except LangfuseError as exc:
                    stats.failed += 1
                    console.log(f"[red]Failed to import trace {trace_id}: {exc}[/red]")
                finally:
                    progress.advance(task_id)

        console.log(
            f"Import complete: {stats.imported} imported, "
            f"{stats.skipped} skipped, {stats.failed} failed",
        )
        return stats

    def _build_dataset_item(self, trace: dict[str, Any]) -> dict[str, Any]:
        trace_id = str(trace.get("id"))
        item_id = _dataset_item_id(self.dataset_name, trace_id)
        input_payload = _decode_if_json(trace.get("input"))
        expected_output = _decode_if_json(trace.get("output"))

        metadata = {
            "traceId": trace_id,
            "name": trace.get("name"),
            "environment": trace.get("environment"),
            "timestamp": trace.get("timestamp"),
            "latency": trace.get("latency"),
            "model": _first_model(trace),
        }
        # Remove empty entries to keep metadata concise.
        metadata = {k: v for k, v in metadata.items() if v not in (None, "", [])}

        return {
            "datasetName": self.dataset_name,
            "input": input_payload,
            "expectedOutput": expected_output,
            "metadata": metadata or None,
            "sourceTraceId": trace_id,
            "id": item_id,
        }
