# ==========================================================
# run_pipeline_driver.py
# Legacy compatibility wrapper
# ==========================================================

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_imports() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def main(argv: list[str] | None = None) -> int:
    _bootstrap_imports()
    from IPNAnalysis.pipeline.legacy_cli import main_driver

    return main_driver(argv)


if __name__ == "__main__":
    raise SystemExit(main())
