# Repository Guidelines

## Project Structure & Module Organization
`IPNAnalysis/` is the main Python pipeline and the default place for production changes. Key entry points are `IPNAnalysis/run_pipeline_driver.py` for batch orchestration and `IPNAnalysis/mea_analysis_routine.py` for per-well processing. Shared helpers live beside them (`config_loader.py`, `helper_functions.py`, `parameter_free_burst_detector.py`). Exploratory notebooks are concentrated in `IPNAnalysis/workbooks/` and related domain folders such as `NetworkAnalysis/`, `StimulationAnalysis/`, and `WildtypeSegregation/`. Treat `Archive/` as legacy code and avoid adding new dependencies there.

## Build, Test, and Development Commands
Set up a Python 3.9+ environment, then install dependencies with `pip install -r requirements.txt`. Install the package in editable mode with `pip install -e .` when you need package-style imports. Generate a config template with `python IPNAnalysis/config_loader.py mea_config.json`. Dry-run the batch pipeline with `python IPNAnalysis/run_pipeline_driver.py /data/experiment --config mea_config.json --dry` before full runs. Process one recording directly with `python IPNAnalysis/mea_analysis_routine.py /data/file.h5 --well well000 --rec rec0001 --config mea_config.json`.

## Coding Style & Naming Conventions
Use 4-space indentation and keep Python modules and functions in `snake_case`; classes use `CamelCase`. Follow the existing pattern of colocating small utilities with the pipeline module that owns them. Prefer explicit imports and `pathlib.Path` for filesystem work. Notebook names in `workbooks/` are descriptive but inconsistent; new notebooks should use lowercase `snake_case` and avoid spaces or `copy` suffixes.

## Testing Guidelines
There is no dedicated `tests/` suite yet. Validate pipeline changes with a dry run first, then a targeted single-well run against representative data. For notebook-only edits, rerun affected cells from a clean kernel and confirm generated artifacts land in the expected output tree (for example `analyzer_output/` or `checkpoints/`). Include any manual verification commands in the PR description.

## Commit & Pull Request Guidelines
Recent history favors short conventional subjects such as `feat: ...`, `delete: ...`, and concise merge commits. Keep commit messages imperative and scoped, for example `feat: add UnitMatch report export`. PRs should summarize the affected pipeline stage, list sample commands used for validation, link the relevant issue or dataset ticket, and attach screenshots only when GUI, Streamlit, or notebook visual output changes.
