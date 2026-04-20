from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import load_pipeline_config, write_config_template
from .model import ReanalysisRequest, RunRequest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mea-analysis",
        description="Stage-based IPNAnalysis pipeline CLI",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the pipeline on raw inputs")
    run_parser.add_argument("source", help="Raw HDF5 file or directory to process")
    run_parser.add_argument("--config", type=str, default=None, help="YAML or JSON pipeline config")
    run_parser.add_argument("--well", type=str, default=None, help="Restrict to a single well")
    run_parser.add_argument("--recording", type=str, default=None, help="Restrict to a single recording")
    run_parser.add_argument("--output-root", "--output-dir", dest="output_root", type=str, default=None)
    run_parser.add_argument("--checkpoint-root", "--checkpoint-dir", dest="checkpoint_root", type=str, default=None)
    run_parser.add_argument("--from", dest="from_stage", type=str, default=None, help="Start from stage")
    run_parser.add_argument("--to", dest="to_stage", type=str, default=None, help="Stop after stage")
    run_parser.add_argument("--dry-run", action="store_true", help="Discover tasks without executing")
    run_parser.add_argument("--force-restart", action="store_true", help="Ignore prior completion state")
    run_parser.add_argument("--verbose", action="store_true", help="Enable verbose execution")

    discover_parser = subparsers.add_parser("discover", help="List discovered recording/well tasks")
    discover_parser.add_argument("source", help="File or directory to scan")
    discover_parser.add_argument("--config", type=str, default=None, help="YAML or JSON pipeline config")
    discover_parser.add_argument("--well", type=str, default=None, help="Restrict to a single well")
    discover_parser.add_argument("--recording", type=str, default=None, help="Restrict to a single recording")
    discover_parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")

    reanalyze_parser = subparsers.add_parser(
        "reanalyze",
        help="Run post-hoc analysis on legacy well output folders",
    )
    reanalyze_parser.add_argument("source", help="Legacy well output directory or tree")
    reanalyze_parser.add_argument("--config", type=str, default=None, help="YAML or JSON pipeline config")
    reanalyze_parser.add_argument(
        "--plugin",
        action="append",
        dest="plugins",
        default=None,
        choices=["burst_analysis"],
        help="Analysis plugin to run (default: burst_analysis)",
    )
    reanalyze_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    config_parser = subparsers.add_parser("config", help="Config helpers")
    config_subparsers = config_parser.add_subparsers(dest="config_command", required=True)
    init_parser = config_subparsers.add_parser("init", help="Write a YAML config template")
    init_parser.add_argument("destination", help="Destination path for the template")

    return parser


def run_command(args: argparse.Namespace) -> int:
    from .runner import PipelineRunner

    config = load_pipeline_config(args.config)
    request = RunRequest(
        source_path=Path(args.source).resolve(),
        config_path=Path(args.config).resolve() if args.config else None,
        output_root=Path(args.output_root).resolve() if args.output_root else None,
        checkpoint_root=Path(args.checkpoint_root).resolve() if args.checkpoint_root else None,
        recording_name=args.recording,
        well_id=args.well,
        from_stage=args.from_stage,
        to_stage=args.to_stage,
        dry_run=bool(args.dry_run),
        force_restart=bool(args.force_restart),
        verbose=bool(args.verbose),
    )
    runner = PipelineRunner(config)
    results = runner.run(request)
    if args.dry_run:
        print(json.dumps([result.to_dict() for result in results], indent=2))
    else:
        print(json.dumps([result.to_dict() for result in results], indent=2))
    return 0


def discover_command(args: argparse.Namespace) -> int:
    from .discovery import format_discovery_summary
    from .runner import PipelineRunner

    config = load_pipeline_config(args.config)
    runner = PipelineRunner(config)
    request = RunRequest(
        source_path=Path(args.source).resolve(),
        config_path=Path(args.config).resolve() if args.config else None,
        recording_name=args.recording,
        well_id=args.well,
        dry_run=True,
    )
    discovery = runner.discover(request)
    if args.json:
        print(json.dumps(discovery.to_dict(), indent=2))
    else:
        print(format_discovery_summary(discovery))
    return 0


def reanalyze_command(args: argparse.Namespace) -> int:
    from .runner import PipelineRunner

    config = load_pipeline_config(args.config)
    runner = PipelineRunner(config)
    request = ReanalysisRequest(
        source_path=Path(args.source).resolve(),
        config_path=Path(args.config).resolve() if args.config else None,
        plugins=list(args.plugins or ["burst_analysis"]),
        verbose=bool(args.verbose),
    )
    results = runner.reanalyze(request)
    print(json.dumps(results, indent=2))
    return 0


def config_command(args: argparse.Namespace) -> int:
    if args.config_command == "init":
        destination = write_config_template(args.destination)
        print(destination)
        return 0
    raise ValueError(f"Unknown config command: {args.config_command}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return run_command(args)
    if args.command == "discover":
        return discover_command(args)
    if args.command == "reanalyze":
        return reanalyze_command(args)
    if args.command == "config":
        return config_command(args)
    parser.print_help(sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
