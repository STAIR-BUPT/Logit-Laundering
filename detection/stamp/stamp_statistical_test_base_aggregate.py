"""
STAMP statistical test script — aggregation using base model only

Same pipeline as stamp_statistical_test_aggregate.py:
- Metrics: ppl / mink / lowercase / zlib
- Group aggregation followed by t-test
- Outputs chart and stamp_test_results.json

Differences:
- No assist model
- No alpha / control_weight
- Uses BASE_MODEL_PATH from the adaptive aggregate script as the normal model path
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import stamp_statistical_test_aggregate as aggregate


DEFAULT_NORMAL_MODEL_PATH = aggregate.BASE_MODEL_PATH


def build_arg_parser():
    parser = argparse.ArgumentParser(description="STAMP statistical test — base model only")

    parser.add_argument("--dataset_file", type=str, default=None, help="merged dataset file path (JSON format)")
    parser.add_argument("--public_key", type=str, default=None, help="column name for the public version")
    parser.add_argument("--private_keys", nargs="+", default=None, help="list of column names for private versions")
    parser.add_argument("--public_dataset_file", type=str, default=None, help="dataset file path for the public version (JSON/JSONL)")
    parser.add_argument("--private_dataset_files", nargs="+", default=None, help="list of dataset file paths for private versions (JSON/JSONL)")
    parser.add_argument("--text_key", type=str, default=aggregate.TEXT_KEY, help="text column name")
    parser.add_argument(
        "--normal_model_path",
        type=str,
        default=DEFAULT_NORMAL_MODEL_PATH,
        help="Normal model path; defaults to the base model path from the adaptive aggregate script.",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=aggregate.GPU_ID,
        help="Explicit GPU index; defaults to CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--group_by_field",
        type=str,
        default=aggregate.GROUP_BY_FIELD,
        help="Field to group by before statistical testing, e.g. book_name.",
    )
    aggregate.add_bool_argument(parser, "use_paired_test", default=True, help_text="Whether to use paired t-test")
    parser.add_argument(
        "--remove_outlier_frac",
        type=float,
        default=0.05,
        help="fraction of outliers to remove",
    )
    parser.add_argument(
        "--outlier_method",
        type=str,
        default="clip",
        choices=["clip", "zero", "mean", "keep"],
        help="outlier handling method",
    )
    return parser


def configure_aggregate_module(args):
    aggregate.MODEL_TYPE = "normal"
    aggregate.NORMAL_MODEL_PATH = args.normal_model_path
    aggregate.GPU_ID = args.gpu_id
    aggregate.DEVICE = aggregate.resolve_device(args.gpu_id)


def to_aggregate_args(args):
    return argparse.Namespace(
        dataset_file=args.dataset_file,
        public_key=args.public_key,
        private_keys=args.private_keys,
        public_dataset_file=args.public_dataset_file,
        private_dataset_files=args.private_dataset_files,
        text_key=args.text_key,
        model_type="normal",
        normal_model_path=args.normal_model_path,
        base_model_path=None,
        assist_model_path=None,
        alpha_min=aggregate.ALPHA_MIN,
        alpha_max=aggregate.ALPHA_MAX,
        alpha_s=aggregate.ALPHA_S,
        gpu_id=args.gpu_id,
        group_by_field=args.group_by_field,
        use_paired_test=args.use_paired_test,
        remove_outlier_frac=args.remove_outlier_frac,
        outlier_method=args.outlier_method,
    )


def main(args):
    configure_aggregate_module(args)
    aggregate.main(to_aggregate_args(args))


if __name__ == "__main__":
    parser = build_arg_parser()
    main(parser.parse_args())
