'''
Text paraphrase + watermark injection script.
Supports two modes:
    mode1: Normal processing -- read dataset, paraphrase each sample and inject watermark
    mode2: Remediation mode  -- scan existing results and re-process samples with insufficient word count

KGW watermark mechanism:
    1. The model outputs logits at each step (next-token prediction distribution).
    2. Given the current context and a secret key (ξ).
    3. Use a PRF (pseudo-random function) to combine context + key into a random seed.
    4. Use the seed to generate a partition (green vs. red tokens).
    5. Bias the logits according to the partition (boost green tokens, penalize red tokens).
    6. Sample from the modified distribution to produce watermarked text.

    seeding_scheme parameter format: the last segment is parsed as the watermark key.
        ff-<prf_type>-<context_width>-<self_salt>-<hash_key>
        Example: ff-additive_prf-1-False-<your-private-key>
'''




import sys, os
RELEASE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(RELEASE_ROOT, "ward_core"))

import yaml
import json
import torch
import argparse
import pandas as pd
import re
from types import SimpleNamespace
from transformers import LogitsProcessorList

from src.models import HfModel
from src.watermarks import get_watermark
from src.config import MetaConfig
from src.config.watermark_config import WatermarkScheme

# To select a specific GPU, set CUDA_VISIBLE_DEVICES before running.

def clean_output(text: str) -> str:
    """Strip the assistant-generated portion from model output."""
    if "assistant" in text:
        text = text.split("assistant", 1)[1]
    text = text.strip().replace("\n", " ").replace("\r", " ")
    return " ".join(text.split())


def count_words(text: str) -> int:
    """Count words in text."""
    return len(str(text).split())


EXPLANATION_PREFIX_PATTERNS = [
    re.compile(r"^\s*here\s+is\s+the\s+rewritten\s+text\s*:", re.IGNORECASE),
    re.compile(r"^\s*here\s+is\s+the\s+paraphrased\s+version\s*:", re.IGNORECASE),
    re.compile(r"^\s*the\s+rewritten\s+text\s*:", re.IGNORECASE),
    re.compile(r"^\s*the\s+revised\s+text\s*:", re.IGNORECASE),
    re.compile(r"^\s*rewritten\s+text\s*:", re.IGNORECASE),
    re.compile(r"^\s*revised\s+text\s*:", re.IGNORECASE),
    re.compile(r"^\s*paraphrased\s+version\s*:", re.IGNORECASE),
]

REFUSAL_PATTERNS = [
    re.compile(r"^\s*i\s+cannot\s+(create|write|rewrite|provide)\b", re.IGNORECASE),
    re.compile(r"^\s*i\s+can't\s+(create|write|rewrite|provide)\b", re.IGNORECASE),
    re.compile(r"\bexplicit\s+content\b", re.IGNORECASE),
    re.compile(r"\bexplicit\s+sexual\b", re.IGNORECASE),
    re.compile(r"\bnon-consensual\b", re.IGNORECASE),
    re.compile(r"\bbondage\b", re.IGNORECASE),
    re.compile(r"\bsexual\s+behavior\b", re.IGNORECASE),
    re.compile(r"\bhelp\s+you\s+rewrite\b", re.IGNORECASE),
    re.compile(r"\bcan\s+i\s+help\s+you\s+with\s+anything\s+else\b", re.IGNORECASE),
    re.compile(r"\bhow\s+can\s+i\s+help\??\s*$", re.IGNORECASE),
]

QUALITY_REASON_WEIGHTS = {
    "empty_output": 1000,
    "refusal_like": 900,
    "explanation_prefix": 800,
    "too_few_words": 400,
    "too_short": 300,
    "too_long": 300,
}


def analyze_rewrite_quality(
    source_text: str,
    rewritten_text: str,
    min_ratio: float,
    max_ratio: float,
    min_words: int = 20,
):
    """Analyze rewrite quality and return whether a retry is needed."""
    source_text = str(source_text or "")
    rewritten_text = clean_output(str(rewritten_text or ""))

    source_word_count = count_words(source_text)
    rewritten_word_count = count_words(rewritten_text)
    ratio = (
        rewritten_word_count / source_word_count
        if source_word_count > 0
        else (1.0 if rewritten_word_count == 0 else 0.0)
    )

    reasons = []
    if not rewritten_text.strip():
        reasons.append("empty_output")

    if source_word_count > 0:
        if ratio < min_ratio:
            reasons.append("too_short")
        if ratio > max_ratio:
            reasons.append("too_long")

    if source_word_count >= min_words and rewritten_word_count < min_words:
        reasons.append("too_few_words")

    if any(pattern.search(rewritten_text) for pattern in EXPLANATION_PREFIX_PATTERNS):
        reasons.append("explanation_prefix")

    if any(pattern.search(rewritten_text) for pattern in REFUSAL_PATTERNS):
        reasons.append("refusal_like")

    return {
        "passed": len(reasons) == 0,
        "reasons": reasons,
        "source_word_count": source_word_count,
        "rewritten_word_count": rewritten_word_count,
        "ratio": ratio,
        "cleaned_text": rewritten_text,
    }


def quality_rank(quality_info):
    penalty = sum(QUALITY_REASON_WEIGHTS.get(reason, 100) for reason in quality_info["reasons"])
    ratio_distance = abs(1.0 - quality_info["ratio"])
    return (
        penalty,
        len(quality_info["reasons"]),
        ratio_distance,
        -quality_info["rewritten_word_count"],
    )


def quality_reason_summary(quality_info):
    return ",".join(quality_info["reasons"]) if quality_info["reasons"] else "ok"


def build_paraphraser_prompt():
    return (
        "You are a rewriting model. Rewrite the following text clearly and naturally, "
        "keeping the same meaning and about the same length. "
        "Keep all details from the original text exactly. "
        "Do not add, remove, summarize, censor, soften, or moralize any information. "
        "The input may contain fictional violence, sexual content, profanity, or other sensitive material; "
        "preserve that content in rewritten form instead of refusing or sanitizing it. "
        "Do not refuse, warn, apologize, mention policies, or offer alternative help. "
        "Do not include phrases like 'Here is the rewritten text' or any explanation. "
        "Output ONLY the rewritten text itself, nothing else.\n\n"
        "### Original Text:\n"
    )


def load_data(file_path: str):
    """Load data from file based on extension (.xlsx, .jsonl, .json)."""
    if file_path.endswith('.xlsx'):
        print(f"📘 Loading XLSX: {file_path}")
        return pd.read_excel(file_path)
    elif file_path.endswith('.jsonl'):
        print(f"📘 Loading JSONL: {file_path}")
        records = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"❌ JSONL line {line_no} parse error: {e}") from e
        return pd.DataFrame(records)
    elif file_path.endswith('.json'):
        print(f"📘 Loading JSON: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    else:
        raise ValueError(f"❌ Unsupported file format: {file_path} — only .xlsx, .jsonl, or .json are supported")


def save_data(df: pd.DataFrame, file_path: str):
    """Save data to file based on extension (.xlsx, .jsonl, .json)."""
    if file_path.endswith('.xlsx'):
        df.to_excel(file_path, index=False)
        print(f"✅ Saved to XLSX: {file_path}")
    elif file_path.endswith('.jsonl'):
        with open(file_path, 'w', encoding='utf-8') as f:
            for record in df.to_dict(orient='records'):
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"✅ Saved to JSONL: {file_path}")
    elif file_path.endswith('.json'):
        # Save as JSON
        df.to_json(file_path, orient='records', force_ascii=False, indent=2)
        print(f"✅ Saved to JSON: {file_path}")
    else:
        raise ValueError(f"❌ Unsupported file format: {file_path} — only .xlsx, .jsonl, or .json are supported")


def paraphrase_with_watermark_mode1(
    input_file: str,
    input_column: str,
    output_file: str,
    output_column: str,
    model: HfModel,
    meta_cfg,
    attacker_cfg,
    batch_size: int = 4,
    enable_rewrite: bool = True,
    start_idx: int = 0,
    num_samples: int = None,
    filter_field: str = "",
):
    """
    Mode 1: Normal processing.
    Read texts from file -> Paraphrase + inject watermark -> save results.

    Args:
        enable_rewrite: True to run paraphrasing; False for scan-only (no model calls).
        start_idx: Starting sample index (default 0).
        num_samples: Number of samples to process (None = all).
        filter_field: Optional column; if set, only process rows where that field is True.
    """
    print("\n" + "="*80)
    print("🚀 Mode 1: normal processing")
    if not enable_rewrite:
        print("   [scan-only mode — no rewriting]")
    print("="*80)

    # Load data
    df = load_data(input_file)
    
    if input_column not in df.columns:
        raise ValueError(f"❌ Error: column {input_column} not found in {input_file}")

    # Slice to requested range
    total_samples = len(df)
    end_idx = start_idx + num_samples if num_samples is not None else total_samples
    end_idx = min(end_idx, total_samples)  # Clamp to total number of samples
    
    if start_idx >= total_samples:
        raise ValueError(f"❌ Error: start_idx ({start_idx}) exceeds total samples ({total_samples})")
    
    df = df.iloc[start_idx:end_idx]
    print(f"📊 Total samples: {total_samples}")
    print(f"📊 Processing range: samples {start_idx} to {end_idx-1} (total {len(df)} samples)")

    # Optional subset filter via a boolean column of the caller's choosing
    if filter_field:
        if filter_field not in df.columns:
            print(f"⚠️ '{filter_field}' column not found — processing all samples")
            select_mask = [True] * len(df)
        else:
            select_mask = (df[filter_field] == True).tolist()
            print(f"📊 Filter: only {sum(select_mask)} sample(s) with {filter_field}=true")
    else:
        select_mask = [True] * len(df)

    # Convert to strings
    texts = df[input_column].astype(str).tolist()
    texts = [t if t.strip() != "" else "" for t in texts]

    # Scan-only mode: compute stats and save without running the model
    if not enable_rewrite:
        print("\n📊 Scan-only mode — statistics:")
        df['text_word_count'] = df[input_column].apply(count_words)
        
        # Check if output column already exists
        if output_column in df.columns:
            df['watermarked_text_word_count'] = df[output_column].apply(count_words)
            print(f"   Avg text word count: {df['text_word_count'].mean():.1f}")
            print(f"   Avg watermarked word count: {df['watermarked_text_word_count'].mean():.1f}")
            print(f"   Text word count range: [{df['text_word_count'].min()}, {df['text_word_count'].max()}]")
            print(f"   Watermarked word count range: [{df['watermarked_text_word_count'].min()}, {df['watermarked_text_word_count'].max()}]")
        else:
            print(f"   Avg text word count: {df['text_word_count'].mean():.1f}")
            print(f"   Text word count range: [{df['text_word_count'].min()}, {df['text_word_count'].max()}]")
            print(f"   Note: output column '{output_column}' not found, skipping watermarked stats")
        
        # Save file with added length fields
        save_data(df, output_file)
        print(f"\n✅ Scan complete! No rewriting performed.")
        print(f"   Added fields: text_word_count" + (", watermarked_text_word_count" if output_column in df.columns else ""))
        return

    # Initialize watermark object
    watermark = get_watermark(
        meta_cfg,
        attacker_cfg.watermark,
        model.tokenizer,
        attacker_cfg.model.name,
    )

    paraphraser_prompt = build_paraphraser_prompt()

    # Result container
    wm_texts = []

    # Batch generation
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        selected_batch = select_mask[start : start + len(batch)]
        print(f"\n🟢 Processing batch {start} - {start + len(batch)} ...")

        # Skip empty texts; honor optional subset filter mask
        non_empty_idx = [i for i, t in enumerate(batch) if t.strip() != "" and selected_batch[i]]
        if not non_empty_idx:
            wm_texts.extend([""] * len(batch))
            continue

        prompts = [paraphraser_prompt + batch[i] for i in non_empty_idx]

        try:
            # Inject watermark via logits processor during generation
            outputs_all = model.generate(
                prompts,
                LogitsProcessorList([
                    watermark.spawn_logits_processor(attacker_cfg.watermark.generation.delta)
                ]),
            )
            outputs = outputs_all[0]  # first element is the list of texts
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            # Pad failed batch with empty strings
            wm_texts.extend([""] * len(batch))
            continue

        cleaned = [clean_output(o) for o in outputs]

        # Map generation results back to original batch positions
        out_iter = iter(cleaned)
        for i, src in enumerate(batch):
            if i in non_empty_idx:
                gen = next(out_iter)
                wm_texts.append(gen)
            else:
                wm_texts.append("")

    # Save results
    df[output_column] = wm_texts
    
    # Add word count fields
    df['text_word_count'] = df[input_column].apply(count_words)
    df['watermarked_text_word_count'] = df[output_column].apply(count_words)
    
    save_data(df, output_file)
    print(f"\n✅ Mode 1 complete!")
    print(f"   Added fields: text_word_count, watermarked_text_word_count")


def validate_model_file_consistency(model_yaml: str, input_file: str, output_file: str):
    """
    Validate consistency between the model config file and input/output file paths.

    Rules:
    - If MODEL_YAML contains "qwen", OUTPUT_FILE must contain "qwen";
      INPUT_FILE must either contain no model name or also contain "qwen".
    - If MODEL_YAML contains "mistral", OUTPUT_FILE must contain "mistral";
      INPUT_FILE must either contain no model name or also contain "mistral".
    - Case-insensitive.

    Args:
        model_yaml: path to the model config file
        input_file:  path to the input file
        output_file: path to the output file

    Raises:
        ValueError: if validation fails
    """
    # Lowercase for case-insensitive comparison
    model_yaml_lower = model_yaml.lower()
    input_file_lower = input_file.lower()
    output_file_lower = output_file.lower()
    
    # Detect model family from YAML path
    model_type = None
    if 'qwen' in model_yaml_lower:
        model_type = 'qwen'
    elif 'mistral' in model_yaml_lower:
        model_type = 'mistral'
    
    # Validate if a known model family was detected
    if model_type:
        # Check output file matches model family
        if model_type not in output_file_lower:
            raise ValueError(
                f"❌ Model config contains '{model_type}' but output file path does not contain '{model_type}'!\n"
                f"   MODEL_YAML: {model_yaml}\n"
                f"   OUTPUT_FILE: {output_file}\n"
                f"   Ensure the output file path contains the model name '{model_type}'"
            )
        
        # Validate INPUT_FILE: allowed cases are (1) no model name present, or (2) same model name
        other_models = ['qwen', 'mistral']
        other_models.remove(model_type)
        
        # Check that no other model name is present
        has_other_model = any(other_model in input_file_lower for other_model in other_models)
        has_current_model = model_type in input_file_lower
        
        if has_other_model and not has_current_model:
            raise ValueError(
                f"❌ Model config contains '{model_type}' but input file path contains a different model name!\n"
                f"   MODEL_YAML: {model_yaml}\n"
                f"   INPUT_FILE: {input_file}\n"
                f"   Ensure the input file path does not contain other model names or use a file containing '{model_type}'"
            )
        
        print(f"✅ Model config validation passed: using '{model_type}'")


def paraphrase_with_watermark_mode2(
    input_file: str,
    input_column: str,
    output_file: str,
    output_column: str,
    model: HfModel,
    meta_cfg,
    attacker_cfg,
    min_ratio: float = 0.7,
    max_ratio: float = 1.5,
    min_words: int = 20,
    batch_size: int = 4,
    max_retries: int = 5,
    enable_rewrite: bool = True,
    filter_field: str = "",
):
    """
    Mode 2: remediation mode.
    Scan existing results and re-process samples with insufficient or excessive word count.

    Args:
        min_ratio: minimum ratio threshold — watermarked_text_word_count must be at least this fraction of text_word_count (default 0.7 = 70%)
        max_ratio: maximum ratio threshold — watermarked_text_word_count must not exceed this fraction of text_word_count (default 1.5 = 150%)
        enable_rewrite: True = perform rewriting; False = scan only
        filter_field: Optional column; if set, only check/rewrite rows where that field is True
    """
    print("\n" + "="*80)
    print("🔧 Mode 2: remediation")
    if not enable_rewrite:
        print("   [scan-only mode — no rewriting]")
    print("="*80)

    # Load data
    df = load_data(input_file)
    
    if input_column not in df.columns:
        raise ValueError(f"❌ Error: input column '{input_column}' does not exist")

    if output_column not in df.columns:
        raise ValueError(f"❌ Error: output column '{output_column}' does not exist")

    # Optional subset filter via a boolean column of the caller's choosing
    if filter_field:
        if filter_field not in df.columns:
            print(f"⚠️ '{filter_field}' column not found — checking all samples")
            select_mask = pd.Series([True] * len(df), index=df.index)
        else:
            select_mask = (df[filter_field] == True)
            print(f"📊 Filter: checking/rewriting only {filter_field}=true samples ({select_mask.sum()} sample(s))")
    else:
        select_mask = pd.Series([True] * len(df), index=df.index)

    # === Scan for samples that need re-processing ===
    print(f"\n📊 Scanning dataset, ratio threshold range: {min_ratio*100:.1f}% - {max_ratio*100:.1f}%")
    print(f"   (watermarked word count must be {min_ratio*100:.1f}% - {max_ratio*100:.1f}% of text word count)")
    
    # Pre-compute word count statistics fields (if not already present)
    if 'text_word_count' not in df.columns:
        df['text_word_count'] = df[input_column].apply(count_words)
    if 'watermarked_text_word_count' not in df.columns:
        df['watermarked_text_word_count'] = df[output_column].apply(count_words)
    
    need_reprocess = []
    ratio_stats = []  # for ratio distribution statistics
    too_short = []    # samples that are too short
    too_long = []     # samples that are too long
    explanation_prefix = []
    refusal_like = []
    too_few_words = []
    quality_failures = {}

    for idx, row in df.iterrows():
        input_text = str(row[input_column]) if pd.notna(row[input_column]) else ""
        output_text = str(row[output_column]) if pd.notna(row[output_column]) else ""

        quality_info = analyze_rewrite_quality(
            source_text=input_text,
            rewritten_text=output_text,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
            min_words=min_words,
        )
        ratio = quality_info["ratio"]
        ratio_stats.append(ratio)

        if not select_mask.loc[idx]:
            pass  # optional filter: skip samples outside the selected subset
        elif not quality_info["passed"]:
            need_reprocess.append(idx)
            quality_failures[idx] = quality_info
            if "too_short" in quality_info["reasons"]:
                too_short.append(idx)
            if "too_long" in quality_info["reasons"]:
                too_long.append(idx)
            if "too_few_words" in quality_info["reasons"]:
                too_few_words.append(idx)
            if "explanation_prefix" in quality_info["reasons"]:
                explanation_prefix.append(idx)
            if "refusal_like" in quality_info["reasons"]:
                refusal_like.append(idx)
    
    print(f"✅ Scan complete!")
    print(f"\n" + "="*60)
    print("📈 Statistics:")
    print("="*60)
    print(f"   Total samples: {len(df)}")
    if ratio_stats:
        avg_ratio = sum(ratio_stats) / len(ratio_stats) if ratio_stats else 0
        min_ratio_actual = min(ratio_stats) if ratio_stats else 0
        max_ratio_actual = max(ratio_stats) if ratio_stats else 0
        print(f"   Average ratio: {avg_ratio*100:.2f}%")
        print(f"   Ratio range: [{min_ratio_actual*100:.2f}%, {max_ratio_actual*100:.2f}%]")
    print(f"   Ratio below {min_ratio*100:.1f}% (too short): {len(too_short)} ({len(too_short)/len(df)*100:.2f}%)")
    print(f"   Ratio above {max_ratio*100:.1f}% (too long):  {len(too_long)} ({len(too_long)/len(df)*100:.2f}%)")
    print(f"   Below min word count ({min_words}): {len(too_few_words)} ({len(too_few_words)/len(df)*100:.2f}%)")
    print(f"   With explanation prefix: {len(explanation_prefix)} ({len(explanation_prefix)/len(df)*100:.2f}%)")
    print(f"   Refusal / safety-template responses: {len(refusal_like)} ({len(refusal_like)/len(df)*100:.2f}%)")
    print(f"   Total samples needing reprocessing: {len(need_reprocess)} ({len(need_reprocess)/len(df)*100:.2f}%)")
    print(f"   Samples meeting quality criteria: {len(df) - len(need_reprocess)} ({(len(df) - len(need_reprocess))/len(df)*100:.2f}%)")
    print("="*60)
    
    if len(need_reprocess) == 0:
        print("\n🎉 All samples meet quality requirements — no reprocessing needed!")

        # Even when nothing needs rewriting, update word count fields and save
        df['text_word_count'] = df[input_column].apply(count_words)
        df['watermarked_text_word_count'] = df[output_column].apply(count_words)
        save_data(df, output_file)

        if not enable_rewrite:
            print("✅ Scan complete! No rewriting performed.")
        print(f"   Fields added/updated: text_word_count, watermarked_text_word_count")
        return
    
    # In scan-only mode: display stats, save, then return
    if not enable_rewrite:
        print(f"\n📊 Scan-only mode — {len(need_reprocess)} samples need reprocessing")
        if len(need_reprocess) > 0:
            print("   Indices:", need_reprocess[:20], "..." if len(need_reprocess) > 20 else "")
            # Show details for first few samples
            print("\n   Sample details (first 5):")
            for idx in need_reprocess[:5]:
                quality_info = quality_failures[idx]
                text_wc = quality_info["source_word_count"]
                wm_wc = quality_info["rewritten_word_count"]
                ratio = quality_info["ratio"] * 100
                print(
                    f"      sample {idx}: text={text_wc} words, watermarked={wm_wc} words, "
                    f"ratio={ratio:.1f}% [{quality_reason_summary(quality_info)}]"
                )

        # Ensure word count fields exist and save
        save_data(df, output_file)

        print(f"\n✅ Scan complete! No rewriting performed.")
        print(f"   Fields added/updated: text_word_count, watermarked_text_word_count")
        return

    # Warn if input and output files are the same (in-place modification)
    if input_file == output_file:
        print(f"\n⚠️  Note: input and output files are the same — the original file will be overwritten in place!")
        print(f"   File path: {output_file}")

    # Initialize watermark object
    watermark = get_watermark(
        meta_cfg,
        attacker_cfg.watermark,
        model.tokenizer,
        attacker_cfg.model.name,
    )

    # paraphraser_prompt = (
    #     "You are a rewriting model. Rewrite the following text clearly and naturally, "
    #     "keeping the same meaning and about the same length. "
    #     "Do not add, remove, or comment on any information. "
    #     "Do not include phrases like 'Here is the paraphrased version' or any explanation. "
    #     "Output ONLY the rewritten text itself, nothing else.\n\n"
    #     "### Original Text:\n"
    # )

    paraphraser_prompt = build_paraphraser_prompt()

    # paraphraser_prompt = (
    #     "You are a strong text rewriting model. "
    #     "Rewrite the following text in natural, fluent English, using different sentence structures and phrasing. "
    #     "Keep the same meaning and a similar length. "
    #     "Preserve every factual detail. "
    #     "Do NOT add, remove, or comment on any information or detail. "
    #     "Output ONLY the rewritten text itself, nothing else.\n\n"
    #     "### Original Text:\n"
    # )

    # paraphraser_prompt = (
    #     "You are a strong text rewriting model. "
    #     "Rewrite the following text in natural, fluent English, using different sentence structures and phrasing. "
    #     "Ensure the rewritten text is similar in length to the original. "
    #     "Keep all the factual details intact, and do not omit or add any information. "
    #     "Focus on maintaining the same level of detail and meaning as the original, without making the text overly brief. "
    #     "Output ONLY the rewritten text itself, nothing else.\n\n"
    #     "### Original Text:\n"
    # )

    # paraphraser_prompt = (
    #     "You are a strong text rewriting model. "
    #     "Rewrite the following text in natural, fluent English, using different sentence structures and phrasing. "
    #     "Ensure the rewritten text is of similar length to the original, without omitting any details. "
    #     "Do NOT remove any information or shorten the text, and ensure the rewritten version keeps all the factual content intact. "
    #     "Output ONLY the rewritten text itself, nothing else.\n\n"
    #     "### Original Text:\n"
    # )


    # === Batch re-generation ===
    print(f"\n🟢 Starting re-processing of {len(need_reprocess)} samples, up to {max_retries} retries...\n")

    reprocessed_count = 0
    pending_indices = list(need_reprocess)
    best_candidates = {
        idx: {
            "text": str(df.loc[idx, output_column]),
            "quality": quality_failures[idx],
        }
        for idx in need_reprocess
    }

    for attempt in range(1, max_retries + 1):
        if not pending_indices:
            break

        print(f"\n🔁 Retry round {attempt}/{max_retries}, samples remaining: {len(pending_indices)}")
        next_pending = []

        for i in range(0, len(pending_indices), batch_size):
            batch_indices = pending_indices[i : i + batch_size]
            print(f"🔄 Processing batch {i//batch_size + 1} (sample indices: {batch_indices})")

            batch_texts = [str(df.loc[idx, input_column]) for idx in batch_indices]
            prompts = [paraphraser_prompt + text for text in batch_texts]

            try:
                outputs_all = model.generate(
                    prompts,
                    LogitsProcessorList([
                        watermark.spawn_logits_processor(attacker_cfg.watermark.generation.delta)
                    ]),
                )
                outputs = outputs_all[0]
                cleaned = [clean_output(o) for o in outputs]
            except Exception as e:
                print(f"❌ Batch generation failed: {e}")
                next_pending.extend(batch_indices)
                continue

            for idx, new_text in zip(batch_indices, cleaned):
                quality_info = analyze_rewrite_quality(
                    source_text=str(df.loc[idx, input_column]),
                    rewritten_text=new_text,
                    min_ratio=min_ratio,
                    max_ratio=max_ratio,
                    min_words=min_words,
                )

                if quality_rank(quality_info) < quality_rank(best_candidates[idx]["quality"]):
                    best_candidates[idx] = {
                        "text": quality_info["cleaned_text"],
                        "quality": quality_info,
                    }

                if quality_info["passed"]:
                    df.loc[idx, output_column] = quality_info["cleaned_text"]
                    df.loc[idx, 'watermarked_text_word_count'] = quality_info["rewritten_word_count"]
                    ratio_percent = quality_info["ratio"] * 100
                    print(
                        f"   ✓ sample {idx}: text={quality_info['source_word_count']} words, "
                        f"watermarked={quality_info['rewritten_word_count']} words, ratio={ratio_percent:.1f}%"
                    )
                    reprocessed_count += 1
                else:
                    next_pending.append(idx)
                    print(
                        f"   ↺ sample {idx}: failed [{quality_reason_summary(quality_info)}], "
                        f"ratio={quality_info['ratio']*100:.1f}%"
                    )

        pending_indices = next_pending

    unresolved_count = 0
    if pending_indices:
        print(f"\n⚠️ The following samples still do not meet quality criteria after {max_retries} retries; keeping best candidate:")
        for idx in pending_indices:
            best_candidate = best_candidates[idx]
            best_quality = best_candidate["quality"]
            df.loc[idx, output_column] = best_candidate["text"]
            df.loc[idx, 'watermarked_text_word_count'] = best_quality["rewritten_word_count"]
            unresolved_count += 1
            print(
                f"   sample {idx}: [{quality_reason_summary(best_quality)}], "
                f"ratio={best_quality['ratio']*100:.1f}%"
            )

    # Save results
    df['text_word_count'] = df[input_column].apply(count_words)
    df['watermarked_text_word_count'] = df[output_column].apply(count_words)

    save_data(df, output_file)
    print(
        f"\n✅ Mode 2 complete! Successfully fixed {reprocessed_count} samples; "
        f"{unresolved_count} samples still do not fully meet criteria"
    )
    print(f"   Fields added/updated: text_word_count, watermarked_text_word_count")


if __name__ == "__main__":
    # ==================== Configuration ====================
    
    # 1. Model and device
    MODEL_YAML = ""  # set via --model_yaml, e.g. /configs/model/qwen2.5-14B.yaml
    # MODEL_YAML = ""  # set via --model_yaml, e.g. /configs/model/llama3.1_8b.yaml
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    DELTA = 2.0  # default watermark strength; override via --delta / WATERMARK_DELTA
   
    # 2. Data file paths (supports .xlsx, .json, .jsonl)
    INPUT_FILE = ""   # set via --input
    OUTPUT_FILE = ""  # set via --output


    
    # 3. Column names
    INPUT_COLUMN = "text"              # source text field
    OUTPUT_COLUMN = "watermarked_text"   # watermarked output field
    
    # 4. Processing mode
    MODE = "mode2"  # options: "mode1" (normal processing) or "mode2" (remediation)
    
    # 4.1 Optional subset filter field (empty = process all)
    FILTER_FIELD = ""
    
    # 5. Mode-1-specific settings
    START_IDX = 0  # mode1: starting sample index
    NUM_SAMPLES = None  # mode1: number of samples to process (None = all)
    
    # 6. Mode-2-specific settings
    MIN_RATIO = 0.5  # mode2: min word-count ratio (watermarked / original)
    MAX_RATIO = 1.6  # mode2: max word-count ratio (watermarked / original)
    MIN_WORDS = 20  # mode2: absolute minimum word count
    
    # 7. Batch size
    BATCH_SIZE = 4
    MAX_RETRIES = 5  # mode2: max re-sampling rounds per sample
    
    # 8. Rewrite control
    ENABLE_REWRITE = True  # True = run model; False = scan-only (no model calls)
    
    # 9. Watermark key
    WATERMARK_KEY = os.environ.get("WATERMARK_KEY", "replace-with-private-key")
    
    # =====================================================
    
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Text rewriting and watermark injection")
    parser.add_argument('--mode', type=str, default=MODE, choices=['mode1', 'mode2'],
                        help='mode1=normal processing, mode2=rescue mode')
    parser.add_argument('--input', type=str, default=INPUT_FILE,
                        help='input file path (.xlsx, .json, or .jsonl)')
    parser.add_argument('--output', type=str, default=OUTPUT_FILE,
                        help='output file path (.xlsx, .json, or .jsonl)')
    parser.add_argument('--input_col', type=str, default=INPUT_COLUMN,
                        help='name of the source text column')
    parser.add_argument('--output_col', type=str, default=OUTPUT_COLUMN,
                        help='name of the output (watermarked) text column')
    parser.add_argument('--start_idx', type=int, default=START_IDX,
                        help='mode1: starting sample index')
    parser.add_argument('--num_samples', type=int, default=NUM_SAMPLES,
                        help='mode1: number of samples to process (omit for all)')
    parser.add_argument('--min_ratio', type=float, default=MIN_RATIO,
                        help='mode2: min word-count ratio (default 0.5)')
    parser.add_argument('--max_ratio', type=float, default=MAX_RATIO,
                        help='mode2: max word-count ratio (default 1.6)')
    parser.add_argument('--min_words', type=int, default=MIN_WORDS,
                        help='mode2: absolute minimum word count (default 20)')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='batch size')
    parser.add_argument('--max_retries', type=int, default=MAX_RETRIES,
                        help='mode2: max re-sampling rounds per sample')
    parser.add_argument('--enable_rewrite', type=lambda x: x.lower() in ['true', '1', 'yes'],
                        default=ENABLE_REWRITE,
                        help='true = run rewriting; false = scan-only')
    parser.add_argument('--filter_field', type=str, default=FILTER_FIELD,
                        help='optional boolean column; empty = process all samples')
    parser.add_argument('--watermark_key', type=str, default=WATERMARK_KEY,
                        help='last component of seeding_scheme (watermark key)')
    parser.add_argument('--delta', type=float, default=DELTA,
                        help='watermark strength delta')
    parser.add_argument('--model_yaml', type=str, default=MODEL_YAML,
                        help='path to HF model yaml used for paraphrasing (must set name to a local or HF model path)')
    
    args = parser.parse_args()
    MODEL_YAML = args.model_yaml
    if not MODEL_YAML:
        print("error: --model_yaml is required (path to a model config yaml)")
        sys.exit(1)
    if not os.path.isfile(MODEL_YAML):
        print(f"error: model yaml not found: {MODEL_YAML}")
        sys.exit(1)
    
    # Validate model config vs file path consistency
    try:
        validate_model_file_consistency(MODEL_YAML, args.input, args.output)
    except ValueError as e:
        print(str(e))
        sys.exit(1)
    
    print("="*80)
    print("🌊 Text Rewriting + Watermark Injection")
    print("="*80)
    print(f"📝 Mode: {args.mode}")
    print(f"🔄 Rewrite: {'enabled' if args.enable_rewrite else 'disabled (scan-only)'}")
    print(f"📂 Input: {args.input}")
    print(f"📂 Output: {args.output}")
    print(f"📋 Input column: {args.input_col}")
    print(f"📋 Output column: {args.output_col}")
    if args.mode == 'mode1':
        print(f"📍 Start index: {args.start_idx}")
        print(f"🔢 Num samples: {args.num_samples if args.num_samples else 'all'}")
        print(f"🔍 Filter field: {args.filter_field or '(none — all samples)'}")
    if args.mode == 'mode2':
        print(f"📏 Ratio range: {args.min_ratio*100:.1f}% - {args.max_ratio*100:.1f}%")
        print(f"🔤 Min words: {args.min_words}")
        print(f"🔁 Max retries: {args.max_retries}")
        print(f"🔍 Filter field: {args.filter_field or '(none — all samples)'}")
    if args.enable_rewrite:
        print(f"🔢 Batch size: {args.batch_size}")
    print(f"🔑 Watermark key: {args.watermark_key}")
    print(f"🌊 Watermark delta: {args.delta}")
    print("="*80)

    # Load model config
    with open(MODEL_YAML, "r") as f:
        model_cfg = yaml.safe_load(f)
    model_cfg_ns = SimpleNamespace(skip=False, **model_cfg)

    # Initialize meta config
    meta_cfg = MetaConfig(seed=42, device=DEVICE, out_root_dir="out/")
    
    # Load model
    print(f"\n🚀 Loading HFModel: {model_cfg_ns.name}")
    model = HfModel(meta_cfg=meta_cfg, model_cfg=model_cfg_ns)

    # Watermark configuration
    seeding_scheme = f"ff-additive_prf-1-False-{args.watermark_key}"
    attacker_cfg = SimpleNamespace(
        model=SimpleNamespace(name=model_cfg_ns.name),
        watermark=SimpleNamespace(
            scheme=WatermarkScheme.KGW,
            generation=SimpleNamespace(
                seeding_scheme=seeding_scheme,
                gamma=0.25,
                delta=args.delta,
            ),
        ),
    )

    # Run selected mode
    if args.mode == 'mode1':
        paraphrase_with_watermark_mode1(
            input_file=args.input,
            input_column=args.input_col,
            output_file=args.output,
            output_column=args.output_col,
            model=model,
            meta_cfg=meta_cfg,
            attacker_cfg=attacker_cfg,
            batch_size=args.batch_size,
            enable_rewrite=args.enable_rewrite,
            start_idx=args.start_idx,
            num_samples=args.num_samples,
            filter_field=args.filter_field,
        )
    elif args.mode == 'mode2':
        paraphrase_with_watermark_mode2(
            input_file=args.input,
            input_column=args.input_col,
            output_file=args.output,
            output_column=args.output_col,
            model=model,
            meta_cfg=meta_cfg,
            attacker_cfg=attacker_cfg,
            min_ratio=args.min_ratio,
            max_ratio=args.max_ratio,
            min_words=args.min_words,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
            enable_rewrite=args.enable_rewrite,
            filter_field=args.filter_field,
        )
    
    print("\n" + "="*80)
    print("🎉 All tasks complete!")
    print("="*80)
