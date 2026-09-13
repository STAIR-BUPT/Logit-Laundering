#!/usr/bin/env python3
"""
Config-based model evaluation script.
Reads model configurations from model_config.json and runs evaluations.
"""
import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

DUAL_MODEL_TYPES = {"control_llm", "controlllm", "adaptive_fusion", "logit_laundering"}


def load_model_config(config_file: str = "model_config.json") -> Dict:
    """Load model configuration file."""
    config_path = Path(config_file)

    if not config_path.exists():
        print(f"Error: config file not found: {config_file}")
        print("Please create a config file first; see model_config.json for reference.")
        return None

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)




def expand_path_vars(obj):
    if isinstance(obj, str):
        return os.path.expanduser(os.path.expandvars(obj))
    if isinstance(obj, list):
        return [expand_path_vars(item) for item in obj]
    if isinstance(obj, dict):
        return {key: expand_path_vars(value) for key, value in obj.items()}
    return obj

def list_available_models(config: Dict):
    """List all available models."""
    print("\nAvailable models:")
    print("=" * 80)

    for model_name, model_info in config.get("models", {}).items():
        desc = model_info.get("description", "no description")
        model_type = model_info.get("model_type", "hf")

        print(f"\nModel name: {model_name}")
        print(f"  Description: {desc}")
        print(f"  Type: {model_type}")

        if model_type in DUAL_MODEL_TYPES:
            print(f"  Base Model: {model_info.get('base_model', '')}")
            print(f"  Assist Model: {model_info.get('assist_model', '')}")
        else:
            print(f"  Path: {model_info.get('model_path', '')}")

        if "extra_args" in model_info:
            print(f"  Extra args: {model_info['extra_args']}")
        if "model_args" in model_info:
            print(f"  Structured args: {json.dumps(model_info['model_args'], ensure_ascii=False)}")

    print("\n" + "=" * 80)


def _merge_mapping(*mappings: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for mapping in mappings:
        if isinstance(mapping, dict):
            merged.update(mapping)
    return merged


def _normalize_extra_args(extra_args: Any) -> List[str]:
    if not extra_args:
        return []
    if isinstance(extra_args, str):
        return [item.strip() for item in extra_args.split(",") if item.strip()]
    if isinstance(extra_args, dict):
        return [f"{key}={value}" for key, value in extra_args.items()]
    if isinstance(extra_args, list):
        return [str(item).strip() for item in extra_args if str(item).strip()]
    raise TypeError(f"Unsupported extra_args type: {type(extra_args).__name__}")


def _append_arg(parts: List[str], key: str, value: Any):
    if value is None:
        return
    if isinstance(value, bool):
        value = str(value).lower()
    parts.append(f"{key}={value}")


def build_model_args(model_info: Dict, default_config: Dict | None = None) -> str:
    """Build the model_args string for lm-eval."""
    default_config = default_config or {}
    model_type = model_info.get("model_type", "hf")
    default_model_args = default_config.get("default_model_args", {})
    model_args_override = model_info.get("model_args", {})

    parts: List[str] = []

    if model_type in DUAL_MODEL_TYPES:
        base_model = model_info.get("base_model", "")
        assist_model = model_info.get("assist_model", "")
        if not base_model or not assist_model:
            raise ValueError(
                f"{model_type} model requires both base_model and assist_model to be set.\n"
                f"Current config: base_model={base_model}, assist_model={assist_model}"
            )

        _append_arg(parts, "base_model", base_model)
        _append_arg(parts, "assist_model", assist_model)

        if model_type in {"control_llm", "controlllm"}:
            for key in ["control_weight", "control_mode", "top_logit_filter", "fake_prob", "fake_k", "fake_mode"]:
                if key in model_info:
                    _append_arg(parts, key, model_info[key])
        else:
            for key in ["alpha_min", "alpha_max", "alpha_s"]:
                if key in model_info:
                    _append_arg(parts, key, model_info[key])
    else:
        _append_arg(parts, "pretrained", model_info.get("model_path", ""))

    merged_model_args = _merge_mapping(default_model_args, model_args_override)
    for key, value in merged_model_args.items():
        _append_arg(parts, key, value)

    parts.extend(_normalize_extra_args(model_info.get("extra_args", "")))
    return ",".join(parts)


def run_evaluation(
    model_name: str,
    model_info: Dict,
    default_config: Dict,
    output_dir: str = "eval_results",
    log_samples: bool = True,
    custom_tasks: List[str] = None,
    custom_batch_size: str = None,
    custom_device: str = None,
    datasets_cache_dir: str = None,
    limit: int = None,
):
    """Run evaluation for a single model."""
    print(f"\n{'='*80}")
    print(f"Evaluating model: {model_name}")
    print(f"{'='*80}\n")

    model_type = model_info.get("model_type", "hf")

    if model_type in DUAL_MODEL_TYPES:
        base_model = model_info.get("base_model", "")
        assist_model = model_info.get("assist_model", "")

        if base_model.startswith("/") and not Path(base_model).exists():
            print(f"Warning: base model path not found: {base_model}")
            print(f"Skipping evaluation of {model_name}")
            return False, None

        if assist_model.startswith("/") and not Path(assist_model).exists():
            print(f"Warning: assist model path not found: {assist_model}")
            print(f"Skipping evaluation of {model_name}")
            return False, None
    else:
        model_path = model_info.get("model_path", "")
        if model_path.startswith("/") and not Path(model_path).exists():
            print(f"Warning: model path not found: {model_path}")
            print(f"Skipping evaluation of {model_name}")
            return False, None

    model_args = build_model_args(model_info, default_config)

    batch_size = custom_batch_size or model_info.get("batch_size") or default_config.get("batch_size", "8")
    device = custom_device or model_info.get("device") or default_config.get("device", "cuda:0")
    num_fewshot = model_info.get("num_fewshot", default_config.get("num_fewshot", 5))
    tasks = custom_tasks or model_info.get("tasks") or default_config.get("tasks", [])

    if datasets_cache_dir is None:
        datasets_cache_dir = model_info.get("datasets_cache_dir") or default_config.get("datasets_cache_dir")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"{model_name}_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "lm_eval",
        "--model", model_type,
        "--model_args", model_args,
        "--tasks", ",".join(tasks),
        "--num_fewshot", str(num_fewshot),
        "--batch_size", str(batch_size),
        "--output_path", str(output_path),
    ]

    if device and str(device).lower() != "cpu":
        cmd.extend(["--device", str(device)])

    if log_samples:
        cmd.append("--log_samples")

    if limit is not None and limit > 0:
        cmd.extend(["--limit", str(limit)])

    print("Model info:")
    print(f"  Name: {model_name}")
    print(f"  Type: {model_type}")
    if model_type in DUAL_MODEL_TYPES:
        print(f"  Base Model: {model_info.get('base_model', '')}")
        print(f"  Assist Model: {model_info.get('assist_model', '')}")
    else:
        print(f"  Path: {model_info.get('model_path', '')}")

    print(f"  Args: {model_args}")
    print("\nEvaluation config:")
    print(f"  Tasks: {', '.join(tasks)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {device}")
    print(f"  Few-shot: {num_fewshot}")
    if limit is not None and limit > 0:
        print(f"  Sample limit: {limit} (quick-test mode)")

    env = os.environ.copy()
    if datasets_cache_dir:
        env["HF_DATASETS_CACHE"] = datasets_cache_dir
        env["HF_DATASETS_OFFLINE"] = "1"
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_DATASETS_TRUST_REMOTE_CODE"] = "0"
        print(f"Dataset cache dir: {datasets_cache_dir}")
        print("Fully offline mode enabled (using local cache)")

    print("\nCommand:")
    print(f"  {' '.join(cmd)}\n")

    config_info = {
        "model_name": model_name,
        "model_type": model_type,
        "model_args": model_args,
        "batch_size": batch_size,
        "device": device,
        "num_fewshot": num_fewshot,
        "tasks": tasks,
        "timestamp": timestamp,
    }

    if model_type in DUAL_MODEL_TYPES:
        config_info["base_model"] = model_info.get("base_model", "")
        config_info["assist_model"] = model_info.get("assist_model", "")
        if model_type in {"adaptive_fusion", "logit_laundering"}:
            config_info["alpha_min"] = model_info.get("alpha_min")
            config_info["alpha_max"] = model_info.get("alpha_max")
            config_info["alpha_s"] = model_info.get("alpha_s")
    else:
        config_info["model_path"] = model_info.get("model_path", "")

    if "model_args" in model_info:
        config_info["structured_model_args"] = model_info["model_args"]

    if limit is not None and limit > 0:
        config_info["limit"] = limit

    with open(output_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)

    with open(output_path / "command.txt", "w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n")

    print(f"{'='*80}")
    print("Starting evaluation, please wait...")
    print("Progress will be shown below in real time.")
    print(f"{'='*80}\n")

    log_file = output_path / "evaluation.log"

    try:
        with open(log_file, "w", encoding="utf-8") as f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
            )

            for line in process.stdout:
                print(line, end="", flush=True)
                f.write(line)

            return_code = process.wait()

            if return_code == 0:
                print(f"\n{'='*80}")
                print(f"✓ Evaluation of {model_name} complete!")
                print(f"{'='*80}")
                print(f"Results saved to: {output_path}")
                return True, output_path

            print(f"\n{'='*80}")
            print(f"✗ Evaluation of {model_name} failed!")
            print(f"{'='*80}")
            print(f"Return code: {return_code}")
            print(f"See log: {log_file}")
            return False, output_path

    except Exception as e:
        print(f"\n✗ Error during evaluation of {model_name}!")
        print(f"Error: {e}")

        with open(output_path / "error.log", "w", encoding="utf-8") as f:
            f.write(f"Error: {e}\n\n")
            f.write(f"Exception type: {type(e).__name__}\n")

        return False, output_path


def main():
    parser = argparse.ArgumentParser(
        description="Config-based model evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available models
  python evaluate_models_from_config.py --list

  # Evaluate a single model
  python evaluate_models_from_config.py --model llama-7b

  # Evaluate multiple models
  python evaluate_models_from_config.py --model llama-7b llama-13b

  # Evaluate all models
  python evaluate_models_from_config.py --all

  # Use custom config
  python evaluate_models_from_config.py --model llama-7b --batch-size auto --device cuda:1

  # Quick-test mode (limit number of samples)
  python evaluate_models_from_config.py --model llama-7b --limit 100
        """,
    )
    parser.add_argument("--config", "-c", default="model_config.json", help="Model config file path (default: model_config.json)")
    parser.add_argument("--list", "-l", action="store_true", help="List all available models")
    parser.add_argument("--model", "-m", nargs="+", help="Model name(s) to evaluate (can specify multiple)")
    parser.add_argument("--all", "-a", action="store_true", help="Evaluate all configured models")
    parser.add_argument("--tasks", "-t", nargs="+", help="Custom evaluation tasks (overrides config defaults)")
    parser.add_argument("--batch-size", "-b", default=None, help="Batch size (overrides config default)")
    parser.add_argument("--device", "-d", default=None, help="Device (overrides config default)")
    parser.add_argument("--output-dir", "-o", default="eval_results", help="Results output directory (default: eval_results)")
    parser.add_argument("--no-log-samples", action="store_true", help="Do not save detailed sample outputs")
    parser.add_argument("--datasets-cache-dir", default=None, help="Dataset cache directory path (sets HF_DATASETS_CACHE env var)")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per task for quick testing (e.g. --limit 100)")

    args = parser.parse_args()

    config = load_model_config(args.config)
    if config is None:
        return 1
    config = expand_path_vars(config)

    if args.list:
        list_available_models(config)
        return 0

    models = config.get("models", {})
    default_config = config.get("default_config", {})

    if args.all:
        model_names = list(models.keys())
    elif args.model:
        model_names = args.model
    else:
        print("Error: specify models to evaluate (--model) or use --all to evaluate all models")
        print("Use --list to see all available models")
        return 1

    invalid_models = [name for name in model_names if name not in models]
    if invalid_models:
        print(f"Error: the following models are not defined in the config: {', '.join(invalid_models)}")
        print("Use --list to see all available models")
        return 1

    print("\n" + "=" * 80)
    print("Model evaluation job")
    print("=" * 80)
    print(f"\nConfig file: {args.config}")
    print(f"Models to evaluate: {', '.join(model_names)}")
    print(f"Number of jobs: {len(model_names)}")
    print(f"Output directory: {args.output_dir}")
    if args.tasks:
        print(f"Custom tasks: {', '.join(args.tasks)}")
    if args.limit is not None and args.limit > 0:
        print(f"⚡ Quick-test mode: {args.limit} samples per task")
    print()

    success_count = 0
    failed_models = []
    results = {}

    for idx, model_name in enumerate(model_names, 1):
        print(f"\n[{idx}/{len(model_names)}] Evaluating model: {model_name}")
        success, result_path = run_evaluation(
            model_name=model_name,
            model_info=models[model_name],
            default_config=default_config,
            output_dir=args.output_dir,
            log_samples=not args.no_log_samples,
            custom_tasks=args.tasks,
            custom_batch_size=args.batch_size,
            custom_device=args.device,
            datasets_cache_dir=args.datasets_cache_dir,
            limit=args.limit,
        )

        results[model_name] = result_path
        if success:
            success_count += 1
        else:
            failed_models.append(model_name)

    print("\n" + "=" * 80)
    print("Evaluation complete")
    print("=" * 80)
    print(f"\nTotal: {len(model_names)} models")
    print(f"Succeeded: {success_count}")
    print(f"Failed: {len(failed_models)}")

    if failed_models:
        print(f"\nFailed models: {', '.join(failed_models)}")

    print("\nDetailed results:")
    for model_name, result_path in results.items():
        status = "✓" if model_name not in failed_models else "✗"
        print(f"  {status} {model_name}: {result_path}")

    if success_count > 0:
        print("\nTo analyze results, run:")
        if args.all and len(model_names) > 1:
            print(f"python analyze_results.py --dir {args.output_dir}")
        else:
            for model_name, result_path in results.items():
                if model_name not in failed_models:
                    print(f"python analyze_results.py --result {result_path}")
                    break

    return 0 if len(failed_models) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
