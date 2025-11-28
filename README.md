# Langfuse Utilities

A growing collection of small tools that solve gaps not yet covered by the Langfuse
UI. The `langfuse-utils` CLI dispatches to individual tools; the first shipped tool,
`copy-traces`, copies traces into datasets for experimentation and evaluation workflows.

## Available tools

- `copy-traces` (module: `copy_traces`) – copy traces into a dataset with optional
  filtering and dry-run support.

## Quickstart (Usage)

1. Create a `.env` in the repo root (or export the variables):
   - `LANGFUSE_BASE_URL`
   - `LANGFUSE_PUBLIC_KEY`
   - `LANGFUSE_SECRET_KEY`
   - `LANGFUSE_PROJECT_ID`
2. Install dependencies (uses uv):

   ```bash
   uv sync
   ```

3. Run the `copy-traces` tool via the CLI:

   ```bash
   uv run langfuse-utils copy-traces <DATASET_NAME> --range <DAYS> \
     [--environment ENV] [--model MODEL] [--dry-run]
   ```

   - The CLI reports whether the dataset exists (with its id) or would be created
     in `--dry-run` mode, shows a progress bar using the known trace total, and
     lists each upserted (or would-be upserted) item by id while skipping
     already-imported traces.

## Quickstart (Development)

1. Ensure Python 3.14 and uv are available.
2. Install dev deps and hooks:

   ```bash
   uv sync
   uv run pre-commit install
   ```

3. Run the full quality gate (required):

   ```bash
   uv run poe check
   ```

   This runs formatting, linting, tests (100% coverage), Markdown lint, and pre-commit
   hooks.

## Tasks (poethepoet)

- `uv run poe check` – required aggregate gate
- `uv run poe ruff` – lint
- `uv run poe ruff-fmt` – format check
- `uv run poe pytest` – tests with coverage
- `uv run poe pre-commit` – all hooks
- `uv run poe markdown` – Markdown lint

## Notes

- Environment is auto-loaded from `.env` via `python-dotenv`; never commit secrets.
- Contributions should keep documentation (README.md and AGENTS.md) in sync with
  behavior and tooling.
