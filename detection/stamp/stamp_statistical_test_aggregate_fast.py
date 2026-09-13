"""
STAMP statistical test (fast version).

Same CLI as stamp_statistical_test_aggregate.py,
but replaces per-token prefix forward for adaptive_fusion with
a vectorized whole-sentence forward pass.
Numerically equivalent under eval/no_grad (up to minor floating-point differences).
"""

import importlib.util
from pathlib import Path

import numpy as np
import torch


AGGREGATE_PATH = Path(__file__).with_name("stamp_statistical_test_aggregate.py")
AGGREGATE_SPEC = importlib.util.spec_from_file_location(
    "stamp_statistical_test_aggregate_legacy",
    AGGREGATE_PATH,
)
aggregate = importlib.util.module_from_spec(AGGREGATE_SPEC)
assert AGGREGATE_SPEC is not None
assert AGGREGATE_SPEC.loader is not None
AGGREGATE_SPEC.loader.exec_module(aggregate)

ORIGINAL_COMPUTE_TEXT_TOKEN_STATISTICS = aggregate.compute_text_token_statistics


def compute_vectorized_adaptive_alpha_numpy(
    base_logits,
    alpha_min=None,
    alpha_max=None,
    alpha_s=None,
):
    """Compute adaptive alpha in batch over positions, for testing and validation."""
    alpha_min = aggregate.ALPHA_MIN if alpha_min is None else alpha_min
    alpha_max = aggregate.ALPHA_MAX if alpha_max is None else alpha_max
    alpha_s = aggregate.ALPHA_S if alpha_s is None else alpha_s

    base_logits = np.asarray(base_logits, dtype=float)
    if base_logits.ndim != 2:
        raise ValueError("base_logits must be a 2-D array: [seq_len-1, vocab_size]")
    if base_logits.shape[1] < 2:
        raise ValueError("vocab_size must be at least 2 to compute top1-top2 margin")

    sorted_logits = np.sort(base_logits, axis=-1)
    margins = sorted_logits[:, -1] - sorted_logits[:, -2]
    alphas = alpha_min + (alpha_max - alpha_min) * margins / (margins + alpha_s)
    return alphas.astype(float), margins.astype(float)


def compute_vectorized_fused_token_statistics_numpy(
    base_logits,
    assist_logits,
    labels,
    alpha_min=None,
    alpha_max=None,
    alpha_s=None,
):
    """Compute token-level statistics using the numpy-based vectorized fusion logic."""
    base_logits = np.asarray(base_logits, dtype=float)
    assist_logits = np.asarray(assist_logits, dtype=float)
    labels = np.asarray(labels, dtype=int)

    alphas, margins = compute_vectorized_adaptive_alpha_numpy(
        base_logits,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        alpha_s=alpha_s,
    )
    fused_logits = base_logits - alphas[:, None] * assist_logits

    shifted = fused_logits - np.max(fused_logits, axis=-1, keepdims=True)
    exp_shifted = np.exp(shifted)
    probs = exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.sum(exp_shifted, axis=-1, keepdims=True))

    row_indices = np.arange(labels.shape[0])
    token_log_probs = log_probs[row_indices, labels]
    token_mean_log_probs = np.sum(probs * log_probs, axis=-1)
    token_var_log_probs = np.sum(probs * np.square(log_probs), axis=-1) - np.square(
        token_mean_log_probs
    )
    token_std_log_probs = np.sqrt(np.maximum(token_var_log_probs, 0.0))
    return {
        "alphas": alphas,
        "margins": margins,
        "token_log_probs": token_log_probs.astype(float),
        "token_mean_log_probs": token_mean_log_probs.astype(float),
        "token_std_log_probs": token_std_log_probs.astype(float),
    }


def compute_vectorized_adaptive_alpha_tensor(base_logits):
    """Compute adaptive alpha in batch over positions (tensor version)."""
    top2 = torch.topk(base_logits.float(), k=2, dim=-1).values
    margins = top2[..., 0] - top2[..., 1]
    alphas = aggregate.ALPHA_MIN + (aggregate.ALPHA_MAX - aggregate.ALPHA_MIN) * margins / (
        margins + aggregate.ALPHA_S
    )
    return alphas, margins


def compute_vectorized_adaptive_token_statistics(model_bundle, tokenizer, text):
    """Run a single forward pass over the full text and return fused token statistics for all positions."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(aggregate.DEVICE)
    input_ids = inputs["input_ids"]
    if input_ids.shape[1] < 2:
        empty = np.array([], dtype=float)
        return {
            "token_log_probs": empty,
            "token_mean_log_probs": empty,
            "token_std_log_probs": empty,
        }

    with torch.no_grad():
        base_outputs = model_bundle["base_model"](**inputs, return_dict=True)
        assist_outputs = model_bundle["assist_model"](**inputs, return_dict=True)

        base_logits = base_outputs.logits[0, :-1, :].float()
        assist_logits = assist_outputs.logits[0, :-1, :].float()
        labels = input_ids[0, 1:]

        alphas, _ = compute_vectorized_adaptive_alpha_tensor(base_logits)
        fused_logits = base_logits - alphas.unsqueeze(-1) * assist_logits
        log_probs = torch.log_softmax(fused_logits, dim=-1)
        probs = torch.softmax(fused_logits, dim=-1)

        token_log_probs = log_probs.gather(
            dim=-1,
            index=labels.unsqueeze(-1),
        ).squeeze(-1)
        token_mean_log_probs = (probs * log_probs).sum(dim=-1)
        token_var_log_probs = (probs * torch.square(log_probs)).sum(dim=-1) - torch.square(
            token_mean_log_probs
        )
        token_std_log_probs = torch.sqrt(torch.clamp(token_var_log_probs, min=0.0))

    return {
        "token_log_probs": token_log_probs.detach().cpu().numpy().astype(float),
        "token_mean_log_probs": token_mean_log_probs.detach().cpu().numpy().astype(float),
        "token_std_log_probs": token_std_log_probs.detach().cpu().numpy().astype(float),
    }


def compute_text_token_statistics(model, tokenizer, text):
    """Fast token statistics; normal mode reuses the original implementation."""
    if aggregate.MODEL_TYPE != "adaptive_fusion":
        return ORIGINAL_COMPUTE_TEXT_TOKEN_STATISTICS(model, tokenizer, text)
    return compute_vectorized_adaptive_token_statistics(model, tokenizer, text)


def compute_text_metrics(model, tokenizer, text):
    """Compute all metrics for a single text."""
    try:
        token_stats = compute_text_token_statistics(model, tokenizer, text)
        lowercase_token_stats = compute_text_token_statistics(model, tokenizer, text.lower())
        return aggregate.compute_metrics_from_log_probs(
            text,
            token_stats["token_log_probs"],
            lowercase_token_stats["token_log_probs"],
            token_mean_log_probs=token_stats["token_mean_log_probs"],
            token_std_log_probs=token_stats["token_std_log_probs"],
            mink_ratio=aggregate.MINK_RATIO,
        )
    except Exception as exc:
        print(f"Metric computation failed: {exc}")
        return {metric_name: float("inf") for metric_name in aggregate.METRIC_NAMES}


def compute_text_log_probs(model, tokenizer, text):
    """Legacy interface: return per-token log-prob sequence only."""
    return compute_text_token_statistics(model, tokenizer, text)["token_log_probs"]


def calculate_metrics_for_texts(model, tokenizer, texts, progress_desc=None):
    """Compute metrics for a batch of texts, with optional tqdm progress."""
    metrics_by_name = {metric_name: [] for metric_name in aggregate.METRIC_NAMES}
    iterator = texts
    if progress_desc:
        iterator = aggregate.tqdm(
            texts,
            desc=progress_desc,
            leave=False,
            dynamic_ncols=True,
            mininterval=5.0,
        )

    for text in iterator:
        metric_bundle = compute_text_metrics(model, tokenizer, text)
        for metric_name in aggregate.METRIC_NAMES:
            metrics_by_name[metric_name].append(metric_bundle[metric_name])
    return {
        metric_name: np.array(values, dtype=float)
        for metric_name, values in metrics_by_name.items()
    }


def stamp_detection(
    dataset,
    model,
    tokenizer,
    public_key,
    private_keys,
    use_paired_test=True,
    remove_outlier_frac=0.05,
    outlier_method="clip",
    group_by_field=None,
):
    """Merged-dataset mode with tqdm progress."""
    print(f"\n{'='*60}", flush=True)
    print("STAMP Statistical Test - Metrics: ppl / mink / mink++ / lowercase / zlib", flush=True)
    print(f"{'='*60}", flush=True)

    print(f"\n1. Processing public version: {public_key}", flush=True)
    public_texts = [item[public_key] for item in dataset]
    public_metrics_by_name = calculate_metrics_for_texts(
        model,
        tokenizer,
        public_texts,
        progress_desc="public",
    )
    public_mean_bundle = {
        metric_name: float(np.mean(values))
        for metric_name, values in public_metrics_by_name.items()
    }
    print(f"   Public version mean: {aggregate.format_metric_bundle(public_mean_bundle)}", flush=True)

    print(f"\n2. Processing private versions (total: {len(private_keys)})", flush=True)
    private_metrics_by_key = {}
    for i, private_key in enumerate(private_keys, 1):
        print(f"\n   Private version {i} ({private_key}):", flush=True)
        private_texts = [item[private_key] for item in dataset]
        private_metric_bundle = calculate_metrics_for_texts(
            model,
            tokenizer,
            private_texts,
            progress_desc=f"private {i}/{len(private_keys)}",
        )
        private_metrics_by_key[private_key] = private_metric_bundle
        private_mean_bundle = {
            metric_name: float(np.mean(values))
            for metric_name, values in private_metric_bundle.items()
        }
        print(f"     Private version mean: {aggregate.format_metric_bundle(private_mean_bundle)}", flush=True)

    results = {}
    for metric_name in aggregate.METRIC_NAMES:
        print(f"\n{'#' * 60}", flush=True)
        print(f"Computing metric: {metric_name}", flush=True)
        print(f"{'#' * 60}", flush=True)

        public_metric_values = public_metrics_by_name[metric_name]
        private_metric_values_by_key = {
            key: metric_bundle[metric_name]
            for key, metric_bundle in private_metrics_by_key.items()
        }
        analysis_public_metrics = public_metric_values
        analysis_private_metrics_by_key = private_metric_values_by_key
        group_summaries = None

        if group_by_field:
            aggregated = aggregate.aggregate_metrics_by_group(
                dataset=dataset,
                public_metrics=public_metric_values,
                private_metrics_by_key=private_metric_values_by_key,
                group_by_field=group_by_field,
            )
            analysis_public_metrics = aggregated["public_metrics"]
            analysis_private_metrics_by_key = aggregated["private_metrics_by_key"]
            group_summaries = aggregate.build_group_test_summaries_from_index_groups(
                metric_name=metric_name,
                group_indices=aggregated["group_indices"],
                public_metrics=public_metric_values,
                private_metrics_by_key=private_metric_values_by_key,
                use_paired_test=use_paired_test,
                remove_outlier_frac=remove_outlier_frac,
                outlier_method=outlier_method,
            )

        results[metric_name] = aggregate.run_statistical_test(
            metric_name=metric_name,
            public_metrics=analysis_public_metrics,
            private_metrics_by_key=analysis_private_metrics_by_key,
            use_paired_test=use_paired_test,
            remove_outlier_frac=remove_outlier_frac,
            outlier_method=outlier_method,
            group_by_field=group_by_field,
            group_summaries=group_summaries,
            raw_sample_count=len(dataset),
        )

    return results


def stamp_detection_from_dataset_files(
    public_dataset_file,
    private_dataset_files,
    model,
    tokenizer,
    text_key,
    group_by_field,
    use_paired_test=True,
    remove_outlier_frac=0.05,
    outlier_method="clip",
):
    print(f"\n{'='*60}", flush=True)
    print("STAMP Statistical Test - Metrics: ppl / mink / mink++ / lowercase / zlib", flush=True)
    print("Input mode: separate files", flush=True)
    print(f"Public file: {public_dataset_file}", flush=True)
    print(f"Number of private files: {len(private_dataset_files)}", flush=True)
    print(f"Text column: {text_key}", flush=True)
    print(f"Group-by field: {group_by_field}", flush=True)
    print(f"{'='*60}", flush=True)

    public_records = aggregate.load_json_or_jsonl(public_dataset_file)
    public_group_samples = aggregate.collect_group_samples(public_records, group_by_field, text_key)
    print(f"Valid groups in public file: {len(public_group_samples)}", flush=True)
    private_group_samples_by_key = {}
    private_group_sizes_by_key = {}
    for dataset_file in private_dataset_files:
        dataset_label = aggregate.os.path.basename(dataset_file)
        private_records = aggregate.load_json_or_jsonl(dataset_file)
        private_group_samples = aggregate.collect_group_samples(private_records, group_by_field, text_key)
        private_group_samples_by_key[dataset_label] = private_group_samples
        private_group_sizes_by_key[dataset_label] = {
            name: len(samples) for name, samples in private_group_samples.items()
        }
        print(f"Valid groups in private file {dataset_label}: {len(private_group_samples)}", flush=True)

    common_groups = set(public_group_samples)
    for grouped_samples in private_group_samples_by_key.values():
        common_groups &= set(grouped_samples)
    common_groups = sorted(common_groups)
    if not common_groups:
        raise ValueError("No common groups between public and private files — cannot run statistical test")

    print(f"\nCommon valid books: {len(common_groups)}", flush=True)

    public_group_metrics = {metric_name: {} for metric_name in aggregate.METRIC_NAMES}
    private_group_metrics_by_key = {
        dataset_label: {metric_name: {} for metric_name in aggregate.METRIC_NAMES}
        for dataset_label in private_group_samples_by_key
    }
    per_metric_group_summaries = {metric_name: [] for metric_name in aggregate.METRIC_NAMES}

    for group_index, group_name in enumerate(common_groups, start=1):
        print(f"\n{'=' * 80}", flush=True)
        print(f"Processing book {group_index}/{len(common_groups)}: {group_name}", flush=True)
        print(f"{'=' * 80}", flush=True)

        public_samples = public_group_samples[group_name]
        public_texts = [sample["text"] for sample in public_samples]
        public_metric_bundle = calculate_metrics_for_texts(
            model,
            tokenizer,
            public_texts,
            progress_desc=f"{group_index}/{len(common_groups)} public {group_name}",
        )
        public_sample_metrics_by_name = {
            metric_name: {
                sample["sample_id"]: float(public_metric_bundle[metric_name][i])
                for i, sample in enumerate(public_samples)
            }
            for metric_name in aggregate.METRIC_NAMES
        }
        for metric_name in aggregate.METRIC_NAMES:
            public_group_metrics[metric_name][group_name] = float(
                np.mean(public_metric_bundle[metric_name])
            )

        private_group_sample_metrics_by_key = {}
        for private_index, (dataset_label, grouped_samples) in enumerate(
            private_group_samples_by_key.items(),
            start=1,
        ):
            private_samples = grouped_samples[group_name]
            private_texts = [sample["text"] for sample in private_samples]
            private_metric_bundle = calculate_metrics_for_texts(
                model,
                tokenizer,
                private_texts,
                progress_desc=(
                    f"{group_index}/{len(common_groups)} "
                    f"private {private_index}/{len(private_group_samples_by_key)} {group_name}"
                ),
            )
            private_group_sample_metrics_by_key[dataset_label] = {
                metric_name: {
                    sample["sample_id"]: float(private_metric_bundle[metric_name][i])
                    for i, sample in enumerate(private_samples)
                }
                for metric_name in aggregate.METRIC_NAMES
            }
            for metric_name in aggregate.METRIC_NAMES:
                private_group_metrics_by_key[dataset_label][metric_name][group_name] = float(
                    np.mean(private_metric_bundle[metric_name])
                )

        print(f"\n[{group_name}] Per-book statistics", flush=True)
        for metric_name in aggregate.METRIC_NAMES:
            summary = aggregate.build_single_group_test_summary(
                metric_name=metric_name,
                group_name=group_name,
                public_sample_metrics=public_sample_metrics_by_name[metric_name],
                private_sample_metrics_by_key={
                    dataset_label: metric_bundle[metric_name]
                    for dataset_label, metric_bundle in private_group_sample_metrics_by_key.items()
                },
                use_paired_test=use_paired_test,
                remove_outlier_frac=remove_outlier_frac,
                outlier_method=outlier_method,
            )
            if summary is None:
                continue
            summary["private_sample_counts_by_key"] = {
                key: sizes.get(group_name, 0)
                for key, sizes in private_group_sizes_by_key.items()
            }
            per_metric_group_summaries[metric_name].append(summary)
            print(
                f"   {aggregate.DISPLAY_METRIC_NAMES[metric_name]}: "
                f"public_mean={summary['public_mean']:.4f}, "
                f"private_mean={summary['mean_private_metric']:.4f}, "
                f"p-value={summary['book_p_value_display']}, "
                f"t-statistic={summary['book_t_statistic']:.4f}",
                flush=True,
            )

    results = {}
    for metric_name in aggregate.METRIC_NAMES:
        print(f"\n{'#' * 60}", flush=True)
        print(f"Computing metric: {metric_name}", flush=True)
        print(f"{'#' * 60}", flush=True)

        public_metric_groups = public_group_metrics[metric_name]
        private_metric_groups_by_key = {
            key: metrics_by_name[metric_name]
            for key, metrics_by_name in private_group_metrics_by_key.items()
        }
        aligned = aggregate.align_group_metrics(public_metric_groups, private_metric_groups_by_key)
        if not aligned["group_names"]:
            raise ValueError("No common groups between public and private files — cannot run statistical test")

        group_summaries = per_metric_group_summaries[metric_name]
        results[metric_name] = aggregate.run_statistical_test(
            metric_name=metric_name,
            public_metrics=aligned["public_metrics"],
            private_metrics_by_key=aligned["private_metrics_by_key"],
            use_paired_test=use_paired_test,
            remove_outlier_frac=remove_outlier_frac,
            outlier_method=outlier_method,
            group_by_field=group_by_field,
            group_summaries=group_summaries,
            raw_sample_count=len(public_records),
        )

    return results


def install_fast_metric_pipeline():
    """Replace slow-path functions in the aggregate module with fast implementations."""
    aggregate.compute_text_token_statistics = compute_text_token_statistics
    aggregate.compute_text_metrics = compute_text_metrics
    aggregate.compute_text_log_probs = compute_text_log_probs
    aggregate.calculate_metrics_for_texts = calculate_metrics_for_texts
    aggregate.stamp_detection = stamp_detection
    aggregate.stamp_detection_from_dataset_files = stamp_detection_from_dataset_files


def build_arg_parser():
    return aggregate.build_arg_parser()


def configure_runtime_from_args(args):
    if args.model_type:
        aggregate.MODEL_TYPE = args.model_type
    if args.normal_model_path:
        aggregate.NORMAL_MODEL_PATH = args.normal_model_path
    if args.base_model_path:
        aggregate.BASE_MODEL_PATH = args.base_model_path
    if args.assist_model_path:
        aggregate.ASSIST_MODEL_PATH = args.assist_model_path
    aggregate.ALPHA_MIN = args.alpha_min
    aggregate.ALPHA_MAX = args.alpha_max
    aggregate.ALPHA_S = args.alpha_s
    aggregate.GPU_ID = args.gpu_id
    aggregate.DEVICE = aggregate.resolve_device(args.gpu_id)


def main(args):
    install_fast_metric_pipeline()
    aggregate.main(args)


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    configure_runtime_from_args(args)
    main(args)
