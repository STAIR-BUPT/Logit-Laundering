"""Command line entry point for the public reproduction workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Stage:
    name: str
    script: str
    description: str


STAGES: tuple[Stage, ...] = (
    Stage("watermark", "scripts/data/watermark.sh", "Create watermarked training or audit data."),
    Stage("attack", "scripts/data/attack.sh", "Run an optional data-level attack."),
    Stage("train-base", "scripts/training/train_base.sh", "Train the base model."),
    Stage("build-assistant", "scripts/training/build_assistant.sh", "Initialize the assistant model."),
    Stage("train-assistant", "scripts/training/train_assistant.sh", "Train the assistant model."),
    Stage("calibrate", "scripts/detection/calibrate.sh", "Estimate alpha_max under the JS-divergence budget."),
    Stage("top1", "scripts/detection/top1.sh", "Run Top-1 watermark auditing."),
    Stage("stamp", "scripts/detection/stamp.sh", "Run STAMP watermark auditing."),
    Stage("mcq", "scripts/evaluation/mcq.sh", "Run multiple-choice knowledge evaluation."),
    Stage("sampling", "scripts/evaluation/sampling.sh", "Run sampling-based utility and watermark statistics."),
    Stage("lm-eval", "scripts/evaluation/lm_eval.sh", "Run lm-evaluation-harness tasks."),
)

STAGE_BY_NAME = {stage.name: stage for stage in STAGES}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_config(root: Path) -> Path:
    return root / "configs" / "experiment.env"


def stage_choices() -> list[str]:
    return [stage.name for stage in STAGES]


def run_stage(args: argparse.Namespace) -> int:
    root = repo_root()
    stage = STAGE_BY_NAME[args.stage]
    script = root / stage.script
    config = Path(args.config).expanduser() if args.config else default_config(root)
    if not config.is_absolute():
        config = (Path.cwd() / config).resolve()

    command = ["bash", str(script), str(config)]
    if args.dry_run:
        print(" ".join(command))
        return 0

    completed = subprocess.run(command, cwd=root, check=False)
    return completed.returncode


def list_stages(_: argparse.Namespace) -> int:
    width = max(len(stage.name) for stage in STAGES)
    for stage in STAGES:
        print(f"{stage.name.ljust(width)}  {stage.description}")
    return 0


def validate_repository(_: argparse.Namespace) -> int:
    root = repo_root()
    completed = subprocess.run(["bash", str(root / "scripts" / "validate_repository.sh")], cwd=root, check=False)
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="logit-laundering",
        description="Unified entry point for the Logit Laundering reproduction workflow.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one reproduction stage.")
    run_parser.add_argument("stage", choices=stage_choices(), help="Pipeline stage to execute.")
    run_parser.add_argument(
        "--config",
        default=None,
        help="Path to experiment.env. Defaults to configs/experiment.env in the repository.",
    )
    run_parser.add_argument("--dry-run", action="store_true", help="Print the command without running it.")
    run_parser.set_defaults(func=run_stage)

    list_parser = subparsers.add_parser("list-stages", help="Show available reproduction stages.")
    list_parser.set_defaults(func=list_stages)

    validate_parser = subparsers.add_parser("validate", help="Run static repository validation.")
    validate_parser.set_defaults(func=validate_repository)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
