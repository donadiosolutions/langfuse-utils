# Agent Operating Guide

## Purpose

This repository is a growing toolkit of Langfuse utilities (the trace-to-dataset
importer is the first tool). Follow these instructions whenever you modify or extend
the project.

## Environment & Credentials

- Load environment variables from the root `.env` (automatically loaded by the CLI
  via `python-dotenv`).
- Required variables: `LANGFUSE_BASE_URL`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`,
  `LANGFUSE_PROJECT_ID`.
- Do **not** hardcode secrets; use the `.env` file or explicit CLI flags where supported.

## Tooling & Commands

- Package/deps: managed with `uv`.
- CLI entrypoint: `uv run langfuse-utils <command>` (current command: `copy-traces`).
- Tasks (via `poetrypoet`): `uv run poe ruff`, `uv run poe ruff-fmt`,
  `uv run poe pytest`, `uv run poe pre-commit`, `uv run poe markdown`,
  `uv run poe check`.
- Pre-commit: configured in `.pre-commit-config.yaml`; hook installed.

## Quality Gates (must all pass)

- Primary command: `uv run poe check` (runs format, lint, tests with coverage, Markdown
  lint, and pre-commit hooks).
- You may run individual tasks (`ruff`, `ruff-fmt`, `pytest`, `pre-commit`, `markdown`)
  only for debugging, but completion requires `uv run poe check`.
- No automatic skipping/ignoring of any checks. If any check fails, the task is **not
  complete**; continue iterating until all pass.

## Development Rules

- Keep documentation up to date: update both `README.md` and `AGENTS.md` whenever
  behavior, setup, or instructions change.
- Do not add suppressions/ignores for linters or tests without explicit human approval.
- Respect existing `.gitignore`; do not commit secrets or `.env`.
- Python version target: 3.14.

## Current Tool: copy-traces (Trace → Dataset Importer)

- Implementation lives in the `copy_traces` module; the CLI aggregates commands
  from modules under `langfuse_utils`.
- CLI uses env vars above; project ID comes only from `LANGFUSE_PROJECT_ID` (no CLI
  arg).
- Importer dedupes dataset items by trace, supports dry-run, retries with backoff,
  verifies creation, reports dataset existence/creation (including dry-run intent),
  streams a progress bar using the total trace count, and logs each upsert (or
  would-be upsert) by item id.

## Workflow Checklist (run before marking done)

1. Run `uv run poe check`
2. Ensure docs are current (`README.md`, `AGENTS.md`).
3. Confirm no secrets are tracked or exposed.
