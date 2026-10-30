# CLAUDE.md

## What this project is

Focus stacking for macro photography, microscopy and photogrammetry image stacks.
It has a PyQt GUI and a headless command-line workflow. The project uses Python and uv.

## Layout

| Folder | Holds |
|---|---|
| `src/` | Python source (`cli.py`, `core/`, `services/`, `ui/`, `config/`) |
| `tests/` | Python tests (`test_*.py`) |
| `results/` | Image output (`.jpeg`, `.tiff`) |

## Project roles
The global skills look up their commands here. A `none` row uses the skill's fallback.

| Role | This project |
|---|---|
| changed-only | none |
| fast gate | `uv run pytest -q` |
| domain loop | none |
| renders | none |
| wide run | `uv run pytest` |
| result cache | none |
| worker cap | none |
| outputs | `results/` (unverified) |
| references index | none |
| decision record | memory decisions-log.md |
