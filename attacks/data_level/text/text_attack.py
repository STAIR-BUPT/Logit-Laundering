#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Clean Watermark Attack Script (translation / swap / synonym / dipper / gpt)

Changes:
- Input is JSON list (supports JSONL too)
- Optionally attack only samples matching --filter-field / --filter-value
- Keep all other samples unchanged
- Output format determined by file suffix (.json/.jsonl/.xlsx)
- Support command line arguments
"""

import os
import sys
import argparse

# ⚠️ Important: CUDA_VISIBLE_DEVICES must be set before importing torch.
# Parse the GPU argument first.
def parse_gpu_arg():
    """Quick-parse the --gpu argument before importing torch."""
    for i, arg in enumerate(sys.argv):
        if arg == "--gpu" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return os.environ.get("CUDA_VISIBLE_DEVICES", "")

# Set GPU device (must be before importing torch)
_requested_gpu = parse_gpu_arg()
if _requested_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = _requested_gpu

# Now safe to import torch and other dependencies
import json
from copy import deepcopy
from typing import List, Dict, Any, Optional
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import Dataset  # only used for translation_attack
import torch

# Print GPU info
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU detected, using CPU")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# Set environment variables for offline mode (avoid unnecessary downloads)
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

# Local imports (your repository utilities)
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# NLTK data directory — set via env var or update here
NLTK_DATA_DIR = os.environ.get("NLTK_DATA", "")
if NLTK_DATA_DIR:
    os.environ["NLTK_DATA"] = NLTK_DATA_DIR
import nltk
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_DIR)  # Prioritize local data path

print(f"[NLTK] Data directory: {NLTK_DATA_DIR}")
print(f"[NLTK] Data paths: {nltk.data.path[:3]}...")  # Display first 3 paths

# Check for necessary NLTK resources (no download, just check locally)
nltk_resources = {
    "punkt": ["tokenizers/punkt"],
    "averaged_perceptron_tagger": ["taggers/averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger"],
    "wordnet": ["corpora/wordnet"]
}

missing_resources = []
for resource_name, check_paths in nltk_resources.items():
    found = False
    for check_path in check_paths:
        try:
            nltk.data.find(check_path)
            print(f"[NLTK] ✅ {resource_name} available ({check_path})")
            found = True
            break
        except LookupError:
            pass
        # Check filesystem directly (for zip files)
        full_path = os.path.join(NLTK_DATA_DIR, check_path)
        zip_path = full_path + ".zip"
        if os.path.exists(full_path) or os.path.exists(zip_path):
            print(f"[NLTK] ✅ {resource_name} available ({check_path}, filesystem check)")
            found = True
            break
    
    if not found:
        missing_resources.append(resource_name)
        print(f"[NLTK] ❌ {resource_name} not available")

# Alert if resources are missing but continue execution
if missing_resources:
    print(f"\n[Warning] Missing NLTK resources: {', '.join(missing_resources)}")
    print(f"         Please ensure resources are in {NLTK_DATA_DIR}")
    print(f"         The script will continue, but synonym attack may fail\n")
else:
    print(f"[NLTK] ✅ All necessary resources are available\n")

# Import utility functions (with fallback in case of offline mode)
from utils.io import read_jsonlines, write_jsonlines

# Try importing attack modules, enable offline mode if necessary
try:
    from utils.attack import (
        translation_attack,
        get_swap_attack_att,
        get_synonym_attack_att,
        check_output_column_lengths,
    )
except Exception as e:
    print(f"[Warning] Error importing utils.attack: {e}")
    print("[Info] Set offline mode if due to network issues:")
    print("  export HF_HUB_OFFLINE=1")
    print("  export TRANSFORMERS_OFFLINE=1")
    raise

# Import DIPPER-related modules (deferred import)
try:
    from nltk.tokenize import sent_tokenize
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    import torch
    DIPPER_AVAILABLE = True
except ImportError as e:
    print(f"[Warning] DIPPER dependencies not available: {e}")
    DIPPER_AVAILABLE = False

# Import GPT-related modules (deferred import)
try:
    from openai import OpenAI
    from tenacity import retry, stop_after_attempt, wait_random_exponential
    GPT_AVAILABLE = True
except ImportError as e:
    print(f"[Warning] GPT dependencies not available: {e}")
    print("[Info] Install via: pip install openai tenacity")
    GPT_AVAILABLE = False

###########################################################################
# CONFIG - Default values (can be overridden by command line arguments)
###########################################################################
class CONFIG:
    # attack_method: "translation", "swap", "synonym", "dipper", or "gpt"
    attack_method = "translation"

    # paths
    input_file = ""
    output_file = ""

    # if None/0 -> all samples
    max_samples = None

    # fields
    input_text_field = "watermarked_text"
    output_attacked_field = "attacked_text"
    output_length_field = "attacked_text_word_count"  # only meaningful for swap/synonym

    # Optional subset filter; empty = attack all samples (subject to max_samples)
    filter_flag_field = ""
    filter_flag_value = True

    # GPU device
    gpu_device = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    # swap params
    swap_p = 0.3  # overall per-token attack probability (0.3 = 30%)
    model_name_or_path = os.environ.get("ATTACK_MODEL_PATH", "")
    cp_attack_min_len = 0  # 0 means no length filter

    # translation params
    no_wm_attack = False  # False = attack watermarked_text field; True = attack text
    translation_enable_compensation = True  # True/False — enable compensation: scan attacked data and re-process low-quality samples
    translation_min_length_ratio = 0.5  # minimum length retention ratio (default 70%)
    translation_max_retries = 1  # maximum retries per sample
    
    # synonym params
    synonym_p = 0.5  # replacement probability: 1.0 = replace all replaceable words, 0.5 = 50% chance
    synonym_replacements_field = "synonym_replacements"  # field name for storing replacement count

    # dipper params
    dipper_model_name = os.environ.get("DIPPER_MODEL_PATH", "")
    dipper_lex = 20  # lexical diversity, range 0-100
    dipper_order = 0  # order diversity, range 0-100
    dipper_no_ctx = True  # whether to disable context (no context mode)
    dipper_sent_interval = 3  # sentence interval: number of sentences per processing step
    dipper_enable_compensation = False  # True/False — enable compensation: scan attacked data and re-process low-quality samples
    dipper_min_length_ratio = 0.70  # minimum length retention ratio (default 70%)
    dipper_max_retries = 1  # maximum retries per sample (reduced to 1 for speed)

    # gpt params (see original project attack_pipeline.py for reference)
    gpt_model_name = "gpt-3.5-turbo-0125"  # OpenAI model name (original project default: gpt-3.5-turbo-0125)
    gpt_temperature = 0.7  # temperature (original project default: 0.7)
    gpt_max_tokens = 1000  # max generated tokens (original project default: 1000)
    gpt_prompt = "paraphrase the following paragraphs and try to keep the similar length to the original paragraphs\n"  # prompt ID=2
    gpt_verbose = False  # whether to print verbose output
    # API configuration
    gpt_api_key = os.environ.get("OPENAI_API_KEY", "")
    gpt_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
###########################################################################


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Text watermark attack script — supports: translation, swap, synonym, dipper, gpt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  # DIPPER attack
  python %(prog)s --method dipper --input data.json --output output.json --lex 20 --gpu 5
  
  # Swap attack
  python %(prog)s --method swap --input data.json --output output.json --swap-p 0.3
  
  # Synonym attack
  python %(prog)s --method synonym --input data.json --output output.json --synonym-p 0.5
  
  # Translation attack
  python %(prog)s --method translation --input data.json --output output.json
  
  # GPT attack
  python %(prog)s --method gpt --input data.json --output output.json --gpt-model gpt-3.5-turbo
        """
    )
    
    # Required arguments
    parser.add_argument("--method", "-m", type=str, required=True,
                       choices=["translation", "swap", "synonym", "dipper", "gpt"],
                       help="attack method")
    parser.add_argument("--input", "-i", type=str, required=True,
                       help="input file path (.json/.jsonl)")
    parser.add_argument("--output", "-o", type=str, required=True,
                       help="output file path (.json/.jsonl/.xlsx)")
    
    # General arguments
    parser.add_argument("--max-samples", type=int, default=None,
                       help="maximum number of samples to attack (default: all)")
    parser.add_argument("--input-field", type=str, default="watermarked_text",
                       help='input text field name (default: "watermarked_text")')
    parser.add_argument("--output-field", type=str, default="attacked_text",
                       help='output field name for attacked text (default: "attacked_text")')
    parser.add_argument("--filter-field", type=str, default="",
                       help="optional boolean/value field; empty = process all samples")
    parser.add_argument("--filter-value", type=str, default="True",
                       help="filter field value when --filter-field is set (default: True)")
    parser.add_argument("--gpu", type=str, default=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                       help="GPU device ID (or set CUDA_VISIBLE_DEVICES)")
    
    # Swap attack arguments
    swap_group = parser.add_argument_group("Swap attack arguments")
    swap_group.add_argument("--swap-p", type=float, default=0.3,
                           help="swap attack probability (default: 0.3)")
    swap_group.add_argument("--model-path", type=str, default=os.environ.get("ATTACK_MODEL_PATH", ""),
                           help="tokenizer model path (or set ATTACK_MODEL_PATH)")
    
    # Translation attack arguments
    trans_group = parser.add_argument_group("Translation attack arguments")
    trans_group.add_argument("--translation-compensation", action="store_true",
                            help="enable translation compensation")
    trans_group.add_argument("--translation-min-ratio", type=float, default=0.7,
                            help="translation attack minimum length retention ratio (default: 0.7)")
    trans_group.add_argument("--translation-retries", type=int, default=2,
                            help="translation attack maximum retries (default: 2)")
    
    # Synonym attack arguments
    syn_group = parser.add_argument_group("Synonym attack arguments")
    syn_group.add_argument("--synonym-p", type=float, default=0.5,
                          help="synonym replacement probability (default: 0.5)")
    
    # DIPPER attack arguments
    dipper_group = parser.add_argument_group("DIPPER attack arguments")
    dipper_group.add_argument("--dipper-model", type=str,
                             default=os.environ.get("DIPPER_MODEL_PATH", ""),
                             help="DIPPER model path (or set DIPPER_MODEL_PATH)")
    dipper_group.add_argument("--lex", type=int, default=20,
                             help="lexical diversity (0-100, default: 20)")
    dipper_group.add_argument("--order", type=int, default=0,
                             help="order diversity (0-100, default: 0)")
    dipper_group.add_argument("--sent-interval", type=int, default=3,
                             help="sentence interval (default: 3)")
    dipper_group.add_argument("--dipper-compensation", action="store_true",
                             help="enable DIPPER compensation")
    dipper_group.add_argument("--dipper-min-ratio", type=float, default=0.7,
                             help="DIPPER minimum length retention ratio (default: 0.7)")
    dipper_group.add_argument("--dipper-retries", type=int, default=1,
                             help="DIPPER maximum retries (default: 1)")
    
    # GPT attack arguments
    gpt_group = parser.add_argument_group("GPT attack arguments")
    gpt_group.add_argument("--gpt-model", type=str, default="gpt-3.5-turbo",
                          help="OpenAI model name (default: gpt-3.5-turbo)")
    gpt_group.add_argument("--gpt-temp", type=float, default=0.7,
                          help="GPT temperature (default: 0.7)")
    gpt_group.add_argument("--gpt-max-tokens", type=int, default=1000,
                          help="GPT max generated tokens (default: 1000)")
    gpt_group.add_argument("--gpt-verbose", action="store_true",
                          help="print verbose output")
    
    return parser.parse_args()


def update_config_from_args(cfg: CONFIG, args) -> CONFIG:
    """Apply command-line arguments to config."""
    # Basic parameters
    cfg.attack_method = args.method
    cfg.input_file = args.input
    cfg.output_file = args.output
    cfg.max_samples = args.max_samples
    cfg.input_text_field = args.input_field
    cfg.output_attacked_field = args.output_field
    cfg.filter_flag_field = args.filter_field
    cfg.gpu_device = args.gpu
    
    # Parse filter_value (supports True/False/number/string)
    filter_val = args.filter_value
    if filter_val.lower() == "true":
        cfg.filter_flag_value = True
    elif filter_val.lower() == "false":
        cfg.filter_flag_value = False
    else:
        try:
            cfg.filter_flag_value = int(filter_val)
        except ValueError:
            try:
                cfg.filter_flag_value = float(filter_val)
            except ValueError:
                cfg.filter_flag_value = filter_val
    
    # Swap parameters
    cfg.swap_p = args.swap_p
    cfg.model_name_or_path = args.model_path
    
    # Translation parameters
    cfg.translation_enable_compensation = args.translation_compensation
    cfg.translation_min_length_ratio = args.translation_min_ratio
    cfg.translation_max_retries = args.translation_retries
    
    # Synonym parameters
    cfg.synonym_p = args.synonym_p
    
    # DIPPER parameters
    cfg.dipper_model_name = args.dipper_model
    cfg.dipper_lex = args.lex
    cfg.dipper_order = args.order
    cfg.dipper_sent_interval = args.sent_interval
    cfg.dipper_enable_compensation = args.dipper_compensation
    cfg.dipper_min_length_ratio = args.dipper_min_ratio
    cfg.dipper_max_retries = args.dipper_retries
    
    # GPT parameters (temperature and max_tokens can be overridden via CLI)
    cfg.gpt_model_name = args.gpt_model
    cfg.gpt_temperature = args.gpt_temp
    cfg.gpt_max_tokens = args.gpt_max_tokens
    cfg.gpt_verbose = args.gpt_verbose
    # prompt, api_key, base_url use fixed values from config; not overridden via CLI
    
    return cfg
###########################################################################


# -----------------------------
# IO helpers
# -----------------------------
def load_records(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    # .json (list) vs .jsonl
    if path.endswith(".json") and not path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, list):
            raise ValueError(f"JSON file should be a list, but got {type(obj)}")
        return obj
    else:
        return [ex for ex in read_jsonlines(path)]


def save_records(records: List[Dict[str, Any]], path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    lower = path.lower()
    if lower.endswith(".xlsx"):
        df = pd.DataFrame(records)
        df.to_excel(path, index=False)
    elif lower.endswith(".json") and not lower.endswith(".jsonl"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    else:
        write_jsonlines(records, path)


# -----------------------------
# Selection helper
# -----------------------------
def select_attack_indices(
    records: List[Dict[str, Any]],
    flag_field: str,
    flag_value: Any,
    max_samples: Optional[int] = None,
) -> List[int]:
    idxs = []
    for i, ex in enumerate(records):
        if flag_field and ex.get(flag_field, False) != flag_value:
            continue
        idxs.append(i)
        if max_samples and max_samples > 0 and len(idxs) >= max_samples:
            break
    return idxs


# -----------------------------
# Tokenizer helper
# -----------------------------
def load_tokenizer(model_name_or_path: str):
    print(f"[Tokenizer] Loading: {model_name_or_path}")
    if "llama" in model_name_or_path.lower():
        tok = AutoTokenizer.from_pretrained(model_name_or_path, return_token_type_ids=False)
        tok.pad_token_id = 0
        return tok
    return AutoTokenizer.from_pretrained(model_name_or_path)


# -----------------------------
# Swap attack helper (modified to support parameters)
# -----------------------------
def get_swap_attack_att(
    word_repeat_prob: float = 0.2,
    word_delete_prob: float = 0.2,
    word_swap_probability: float = 0.2,
    sentence_swap_probability: float = 0.2
):
    """
    Create a SwapAttack instance with customizable probability parameters.
    
    Args:
        word_repeat_prob: word repetition probability
        word_delete_prob: word deletion probability
        word_swap_probability: word swap probability
        sentence_swap_probability: sentence swap probability
    """
    from utils.attack import SwapAttack
    att = SwapAttack(
        word_repeat_prob=word_repeat_prob,
        word_delete_prob=word_delete_prob,
        word_swap_probability=word_swap_probability,
        sentence_swap_probability=sentence_swap_probability
    )
    return att


# -----------------------------
# Translation attack (subset)
# -----------------------------
class _TranslationArgs:
    def __init__(self, no_wm_attack: bool):
        self.no_wm_attack = no_wm_attack


def translation_attack_subset(
    subset: List[Dict[str, Any]],
    input_text_field: str,
    output_attacked_field: str,
    output_length_field: Optional[str],
    no_wm_attack: bool,
    enable_compensation: bool = True,
    min_length_ratio: float = 0.7,
    max_retries: int = 2,
) -> List[Dict[str, Any]]:
    """
    Translation attack: English → French → English (back-translation attack)
    
    Internal temporary field notes:
    - _translation_input:  input text to translate (temporary field, cleaned up afterward)
    - _translation_output: output text after translation attack (temporary field, cleaned up afterward)
    - translation_french: French intermediate result (retained in output)
    
    Args:
        enable_compensation: whether to enable compensation (scan first, only re-process low-quality samples)
        min_length_ratio: minimum length retention ratio threshold
        max_retries: max retries per sample (note: translation does not support retries as the model is reloaded each time)
        
    Note:
        The translation attack reloads the translation model on every call, so compensation does not support retries.
        Low-quality samples are re-translated exactly once; no further retries are performed.
    """
    # Compensation: scan dataset first and identify samples that need re-processing
    compensation_stats = {
        'total_samples': len(subset),
        'need_reprocess': [],
        'already_good': [],
        'reprocess_success': [],
        'reprocess_failed': [],
    }
    
    if enable_compensation:
        print(f"\n{'='*80}")
        print(f"📊 Translation Attack compensation — scan phase")
        print(f"{'='*80}")
        print(f"Scanning {len(subset)} samples...")
        
        # Scan all samples and determine which need re-processing
        samples_to_process = []  # indices of samples to process
        
        for idx, ex in enumerate(subset):
            # Check whether an attacked result already exists
            if output_attacked_field in ex and ex.get(output_attacked_field):
                # Result exists — check quality
                original_text = ex.get(input_text_field, "")
                attacked_text = ex.get(output_attacked_field, "")
                
                original_len = len(str(original_text).split())
                attacked_len = len(str(attacked_text).split())
                
                if original_len > 0:
                    length_ratio = attacked_len / original_len
                    
                    if length_ratio < min_length_ratio:
                        # Low quality — needs re-processing
                        compensation_stats['need_reprocess'].append({
                            'idx': idx,
                            'original_len': original_len,
                            'attacked_len': attacked_len,
                            'ratio': length_ratio
                        })
                        samples_to_process.append(idx)
                    else:
                        # Good quality — no action needed
                        compensation_stats['already_good'].append({
                            'idx': idx,
                            'ratio': length_ratio
                        })
                else:
                    # Empty source text — skip
                    pass
            else:
                # No attacked result yet — needs processing
                compensation_stats['need_reprocess'].append({
                    'idx': idx,
                    'original_len': len(str(ex.get(input_text_field, "")).split()),
                    'attacked_len': 0,
                    'ratio': 0.0
                })
                samples_to_process.append(idx)
        
        print(f"\nScan complete!")
        print(f"   Total samples: {len(subset)}")
        print(f"   Good-quality samples: {len(compensation_stats['already_good'])} ({len(compensation_stats['already_good'])/len(subset)*100:.1f}%)")
        print(f"   Samples to re-process: {len(samples_to_process)} ({len(samples_to_process)/len(subset)*100:.1f}%)")
        
        if len(samples_to_process) == 0:
            print(f"\n✅ All samples meet quality requirements — no re-processing needed!")
            print(f"{'='*80}\n")
            return subset
        
        print(f"\nStarting re-processing of {len(samples_to_process)} samples...")
        print(f"{'='*80}\n")
        
        # Only process samples that need re-processing
        samples_to_attack = [subset[i] for i in samples_to_process]
    else:
        # Compensation disabled — process all samples
        samples_to_process = list(range(len(subset)))
        samples_to_attack = subset
    
    # Prepare samples for attack
    mapped = []
    for ex in samples_to_attack:
        ex2 = deepcopy(ex)
        if input_text_field not in ex2:
            ex2[output_attacked_field] = ""
            if output_length_field:
                ex2[output_length_field] = 0
            mapped.append(ex2)
            continue
        # Use clean temporary field names
        ex2["w_wm_output"] = ex2[input_text_field]  # required internal field name for translation_attack
        mapped.append(ex2)

    ds = Dataset.from_list(mapped)
    args = _TranslationArgs(no_wm_attack=no_wm_attack)
    attacked_ds = translation_attack(ds, args=args)  # produces w_wm_output_attacked and w_wm_output_french
    attacked = [ex for ex in attacked_ds]

    out = []
    
    # Build final results (preserve original order)
    if enable_compensation and len(samples_to_process) < len(subset):
        # Compensation mode: only update samples that needed re-processing
        out = deepcopy(subset)
        
        for attacked_idx, original_idx in enumerate(samples_to_process):
            ex = attacked[attacked_idx]
            ex2 = deepcopy(ex)
            original_ex = subset[original_idx]
            
            # Map internal field names to user-friendly names
            if "w_wm_output_attacked" in ex2:
                ex2[output_attacked_field] = ex2["w_wm_output_attacked"]
            else:
                ex2[output_attacked_field] = ex2.get(output_attacked_field, "")

            # Retain French intermediate result with a friendly field name
            if "w_wm_output_french" in ex2:
                ex2["translation_french"] = ex2["w_wm_output_french"]
                ex2.pop("w_wm_output_french", None)

            # Compute length (using simple split())
            attacked_text = ex2.get(output_attacked_field, "")
            attacked_len = len(str(attacked_text).split())
            
            if output_length_field:
                ex2[output_length_field] = attacked_len

            # Check quality after re-processing
            original_text = original_ex.get(input_text_field, "")
            original_len = len(str(original_text).split())
            if original_len > 0:
                length_ratio = attacked_len / original_len
                
                # Translation attack does not support retries (reloading model is too slow)
                # Simply record the re-processed result
                original_ratio = compensation_stats['need_reprocess'][attacked_idx]['ratio']
                if length_ratio >= min_length_ratio:
                    compensation_stats['reprocess_success'].append({
                        'idx': original_idx,
                        'original_ratio': original_ratio,
                        'final_ratio': length_ratio,
                        'original_len': original_len,
                        'final_len': attacked_len
                    })
                else:
                    compensation_stats['reprocess_failed'].append({
                        'idx': original_idx,
                        'original_ratio': original_ratio,
                        'final_ratio': length_ratio,
                        'original_len': original_len,
                        'final_len': attacked_len
                    })

            # Clean up all temporary fields (internal fields prefixed with w_wm_)
            ex2.pop("w_wm_output", None)
            ex2.pop("w_wm_output_attacked", None)

            # Update the corresponding sample in the original dataset
            out[original_idx] = ex2
    else:
        # Non-compensation mode: process all samples
        for idx, ex in enumerate(attacked):
            ex2 = deepcopy(ex)
            
            # Map internal field names to user-friendly names
            if "w_wm_output_attacked" in ex2:
                ex2[output_attacked_field] = ex2["w_wm_output_attacked"]
            else:
                ex2[output_attacked_field] = ex2.get(output_attacked_field, "")

            # Retain French intermediate result with a friendly field name
            if "w_wm_output_french" in ex2:
                ex2["translation_french"] = ex2["w_wm_output_french"]
                ex2.pop("w_wm_output_french", None)

            # Compute length (using simple split())
            if output_length_field:
                attacked_text = ex2.get(output_attacked_field, "")
                ex2[output_length_field] = len(str(attacked_text).split())

            # Clean up all temporary fields (internal fields prefixed with w_wm_)
            ex2.pop("w_wm_output", None)
            ex2.pop("w_wm_output_attacked", None)

            out.append(ex2)
    
    # Print compensation statistics
    if enable_compensation and compensation_stats['need_reprocess']:
        print(f"\n{'='*80}")
        print(f"📊 Translation Attack compensation statistics")
        print(f"{'='*80}")
        print(f"Total samples: {compensation_stats['total_samples']}")
        print(f"Good quality (no action): {len(compensation_stats['already_good'])} ({len(compensation_stats['already_good'])/compensation_stats['total_samples']*100:.1f}%)")
        print(f"Needed re-processing: {len(compensation_stats['need_reprocess'])} ({len(compensation_stats['need_reprocess'])/compensation_stats['total_samples']*100:.1f}%)")
        print(f"Re-processing succeeded: {len(compensation_stats['reprocess_success'])} ({len(compensation_stats['reprocess_success'])/len(compensation_stats['need_reprocess'])*100:.1f}% of need_reprocess)")
        print(f"Re-processing still below threshold: {len(compensation_stats['reprocess_failed'])} ({len(compensation_stats['reprocess_failed'])/len(compensation_stats['need_reprocess'])*100:.1f}% of need_reprocess)")
        
        if compensation_stats['reprocess_success']:
            print(f"\n✅ Details of successfully re-processed samples (first 10):")
            for i, info in enumerate(compensation_stats['reprocess_success'][:10], 1):
                print(f"   {i}. sample {info['idx']}: "
                      f"original ratio {info['original_ratio']*100:.1f}% → "
                      f"final ratio {info['final_ratio']*100:.1f}% ({info['final_len']}/{info['original_len']} words)")
        
        if compensation_stats['reprocess_failed']:
            print(f"\n❌ Details of samples still below threshold after re-processing (first 10):")
            for i, info in enumerate(compensation_stats['reprocess_failed'][:10], 1):
                print(f"   {i}. sample {info['idx']}: "
                      f"original ratio {info['original_ratio']*100:.1f}% → "
                      f"final ratio {info['final_ratio']*100:.1f}% ({info['final_len']}/{info['original_len']} words)")
        print(f"{'='*80}\n")

    return out


# -----------------------------
# Swap attack (subset)
# -----------------------------
def swap_attack_subset(
    subset: List[Dict[str, Any]],
    input_text_field: str,
    output_attacked_field: str,
    output_length_field: Optional[str],
    tokenizer,
    min_len: int = 0,
    swap_p: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Run the swap attack.
    
    Args:
        swap_p: overall per-token attack probability. When triggered, one of four attack types is chosen at random:
                - sentence swap (25%)
                - word swap (25%)
                - word deletion (25%)
                - word repetition (25%)
    """
    # Probability per specific attack type: swap_p * 0.25
    individual_prob = swap_p * 0.25
    att = get_swap_attack_att(
        word_repeat_prob=individual_prob,
        word_delete_prob=individual_prob,
        word_swap_probability=individual_prob,
        sentence_swap_probability=individual_prob
    )
    out = []

    for ex in tqdm(subset, desc="Swap attacking (subset)"):
        ex2 = deepcopy(ex)

        if input_text_field not in ex2:
            ex2[output_attacked_field] = ""
            if output_length_field:
                ex2[output_length_field] = 0
            out.append(ex2)
            continue

        if min_len and min_len > 0:
            try:
                if not check_output_column_lengths(ex2, min_len=min_len):
                    ex2[output_attacked_field] = ""
                    if output_length_field:
                        ex2[output_length_field] = 0
                    out.append(ex2)
                    continue
            except Exception:
                pass

        attacked_text = att.warp(ex2[input_text_field])
        ex2[output_attacked_field] = attacked_text

        if output_length_field:
            ex2[output_length_field] = len(str(attacked_text).split())

        out.append(ex2)

    return out


# -----------------------------
# Synonym attack (subset)
# -----------------------------
def synonym_attack_subset(
    subset: List[Dict[str, Any]],
    input_text_field: str,
    output_attacked_field: str,
    output_length_field: Optional[str],
    tokenizer,
    synonym_p: float = 1.0,
    min_len: int = 0,
    replacements_field: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Run synonym replacement attack on a subset of samples.
    
    Args:
        subset: sample subset to attack
        input_text_field: input text field name
        output_attacked_field: output field for attacked text
        output_length_field: output field for text length (optional)
        tokenizer: tokenizer for computing text length
        synonym_p: replacement probability; 1.0 = replace all replaceable words, 0.5 = 50% chance
        min_len: minimum length requirement
        replacements_field: field name for storing replacement count (optional)
    """
    # Create SynonymAttack instance
    class _SynonymArgs:
        def __init__(self, p):
            self.synonym_p = p
    
    args = _SynonymArgs(synonym_p)
    att = get_synonym_attack_att(args)
    
    out = []
    total_replacements = 0
    samples_with_replacements = 0

    for ex in tqdm(subset, desc="Synonym attacking (subset)"):
        ex2 = deepcopy(ex)

        if input_text_field not in ex2:
            ex2[output_attacked_field] = ""
            if output_length_field:
                ex2[output_length_field] = 0
            if replacements_field:
                ex2[replacements_field] = 0
            out.append(ex2)
            continue

        if min_len and min_len > 0:
            try:
                if not check_output_column_lengths(ex2, min_len=min_len):
                    ex2[output_attacked_field] = ""
                    if output_length_field:
                        ex2[output_length_field] = 0
                    if replacements_field:
                        ex2[replacements_field] = 0
                    out.append(ex2)
                    continue
            except Exception:
                pass

        # Save original text for statistics
        original_text = ex2[input_text_field]

        # Run synonym replacement attack
        # Note: SynonymAttack.warp() internally calls generate_synonyms_fast() to build a synonym cache
        # then performs replacement
        try:
            # warp() internally:
            # 1. calls generate_synonyms_fast(text) to build synonym cache
            # 2. splits text into sentences
            # 3. calls warp_sentence() on each sentence to apply replacement
            # 
            # Note: if no synonym is found in WordNet or the synonym list is empty,
            # no replacement is made and the attacked text equals the original
            
            # Check cache size (debug)
            cache_size_before = len(att.cache) if hasattr(att, 'cache') else 0
            
            attacked_text = att.warp(original_text)
            
            # Check whether any replacement occurred
            if attacked_text == original_text and len(original_text) > 0:
                # If texts are identical, likely no synonym was found
                # This is normal in some cases (proper nouns, technical terms, etc.)
                pass
            
            ex2[output_attacked_field] = attacked_text

            # Count replacements
            if replacements_field:
                replacement_count = count_synonym_replacements(original_text, attacked_text)
                ex2[replacements_field] = replacement_count

            if output_length_field:
                ex2[output_length_field] = len(str(attacked_text).split())
        except Exception as e:
            # If attack fails, keep original text or set empty
            print(f"[Warning] Synonym attack failed for sample: {e}")
            import traceback
            traceback.print_exc()
            ex2[output_attacked_field] = ex2.get(input_text_field, "")
            if output_length_field:
                ex2[output_length_field] = len(str(ex2[output_attacked_field]).split())
            if replacements_field:
                ex2[replacements_field] = 0

        out.append(ex2)

    # Print statistics
    if len(out) > 0:
        total_replacements = sum(ex.get(replacements_field, 0) for ex in out if replacements_field)
        samples_with_replacements = sum(1 for ex in out if ex.get(replacements_field, 0) > 0)
        avg_replacements = total_replacements / len(out) if len(out) > 0 else 0
        print(f"[Synonym Attack Stats] Total samples: {len(out)}")
        print(f"[Synonym Attack Stats] Samples with replacements: {samples_with_replacements}")
        print(f"[Synonym Attack Stats] Average replacements: {avg_replacements:.2f}")
        print(f"[Synonym Attack Stats] Total replacements: {total_replacements}")

    return out


def count_synonym_replacements(original_text: str, attacked_text: str) -> int:
    """
    Count the number of synonym replacements.
    Compare the original text and the attacked text to count how many words were replaced.
    
    Args:
        original_text: original text
        attacked_text: attacked text
        
    Returns:
        number of replacements
    """
    import nltk
    from nltk.tokenize import word_tokenize
    
    try:
        # Tokenize
        original_tokens = word_tokenize(original_text.lower())
        attacked_tokens = word_tokenize(attacked_text.lower())
        
        # Simple alignment: compare words at the same position
        replacements = 0
        min_len = min(len(original_tokens), len(attacked_tokens))
        
        for i in range(min_len):
            # If words differ and both are alphanumeric (not punctuation), count as replacement
            orig_token = original_tokens[i]
            att_token = attacked_tokens[i]
            
            if (orig_token != att_token and 
                orig_token.isalnum() and 
                att_token.isalnum()):
                replacements += 1
        
        return replacements
    except Exception as e:
        # Return 0 on failure
        print(f"[Warning] Failed to count synonym replacements: {e}")
        return 0


# -----------------------------
# DIPPER attack (subset)
# -----------------------------
def dipper_attack_subset(
    subset: List[Dict[str, Any]],
    input_text_field: str,
    output_attacked_field: str,
    output_length_field: Optional[str],
    model_name: str = "/dipper-paraphraser-xxl-no-context",
    lex: int = 20,
    order: int = 0,
    no_ctx: bool = True,
    sent_interval: int = 3,
    device: str = "cuda",
    enable_compensation: bool = True,
    min_length_ratio: float = 0.7,
    max_retries: int = 2,
) -> List[Dict[str, Any]]:
    """
    Run the DIPPER paraphrase attack on a subset of samples.
    
    Args:
        subset: sample subset to attack
        input_text_field: input text field name
        output_attacked_field: output field for attacked text
        output_length_field: output field for text length (optional)
        model_name: DIPPER model name
        lex: lexical diversity, range 0-100
        order: order diversity, range 0-100
        no_ctx: whether to disable context
        sent_interval: sentence interval (sentences per step)
        device: device ("cuda" or "cpu")
        enable_compensation: whether to enable compensation (scan first, only re-process low-quality samples)
        min_length_ratio: minimum length retention ratio threshold
        max_retries: maximum retries per sample
        
    Returns:
        attacked sample list
    """
    if not DIPPER_AVAILABLE:
        raise ImportError("DIPPER dependencies unavailable; install: transformers, torch, nltk")
    
    print(f"[DIPPER] Loading model: {model_name}")
    print(f"[DIPPER] lex={lex}, order={order}, no_ctx={no_ctx}, sent_interval={sent_interval}")
    
    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained("/t5-v1_1-xxl")
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
        print(f"[DIPPER] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        model = model.cpu()
        device = "cpu"
        print(f"[DIPPER] Using CPU")
    
    model.eval()
    
    # Compute control codes (note: model uses similarity not diversity, so use 100 - diversity)
    lex_code = int(100 - lex)
    order_code = int(100 - order)
    
    # Helper: run DIPPER paraphrase on a single text
    def dipper_paraphrase_single(input_text: str) -> str:
        """Run DIPPER paraphrase on a single text."""
        # Remove extra whitespace and newlines
        input_gen = " ".join(input_text.split())
        
        # Split into sentences
        sentences = sent_tokenize(input_gen)
        
        if len(sentences) == 0:
            return input_gen
        
        # Initialize output text and prefix (when using context)
        output_text = ""
        prefix = ""  # unused when no_ctx=True, kept for code consistency
        
        # Process sentences in groups of sent_interval
        for sent_idx in range(0, len(sentences), sent_interval):
            # Get current sentence window
            curr_sent_window = " ".join(sentences[sent_idx : sent_idx + sent_interval])
            
            # Build input text
            if no_ctx:
                final_input_text = f"lexical = {lex_code}, order = {order_code} <sent> {curr_sent_window} </sent>"
            else:
                final_input_text = f"lexical = {lex_code}, order = {order_code} {prefix} <sent> {curr_sent_window} </sent>"
            
            # Tokenize input
            final_input = tokenizer([final_input_text], return_tensors="pt")
            if device == "cuda":
                final_input = {k: v.cuda() for k, v in final_input.items()}
            
            # Generate paraphrase
            with torch.inference_mode():
                outputs = model.generate(
                    **final_input,
                    do_sample=True,
                    top_p=0.75,
                    top_k=None,
                    max_length=512
                )
            
            # Decode output
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated_text = outputs[0]
            
            # Accumulate output
            if output_text:
                output_text += " " + generated_text
            else:
                output_text = generated_text
            
            # Update prefix for next round (when using context)
            if not no_ctx:
                prefix += " " + generated_text
        
        return output_text.strip()
    
    # Compensation: scan dataset first and identify samples that need re-processing
    compensation_stats = {
        'total_samples': len(subset),
        'need_reprocess': [],
        'already_good': [],
        'reprocess_success': [],
        'reprocess_failed': [],
    }
    
    if enable_compensation:
        print(f"\n{'='*80}")
        print(f"📊 DIPPER Attack compensation — scan phase")
        print(f"{'='*80}")
        print(f"Scanning {len(subset)} samples...")
        
        # Scan all samples and determine which need re-processing
        samples_to_process = []  # indices of samples to process
        
        for idx, ex in enumerate(subset):
            # Check whether an attacked result already exists
            if output_attacked_field in ex and ex.get(output_attacked_field):
                # Result exists — check quality
                original_text = ex.get(input_text_field, "")
                attacked_text = ex.get(output_attacked_field, "")
                
                original_len = len(str(original_text).split())
                attacked_len = len(str(attacked_text).split())
                
                if original_len > 0:
                    length_ratio = attacked_len / original_len
                    
                    if length_ratio < min_length_ratio:
                        # Low quality — needs re-processing
                        compensation_stats['need_reprocess'].append({
                            'idx': idx,
                            'original_len': original_len,
                            'attacked_len': attacked_len,
                            'ratio': length_ratio
                        })
                        samples_to_process.append(idx)
                    else:
                        # Good quality — no action needed
                        compensation_stats['already_good'].append({
                            'idx': idx,
                            'ratio': length_ratio
                        })
                else:
                    # Empty source text — skip
                    pass
            else:
                # No attacked result yet — needs processing
                compensation_stats['need_reprocess'].append({
                    'idx': idx,
                    'original_len': len(str(ex.get(input_text_field, "")).split()),
                    'attacked_len': 0,
                    'ratio': 0.0
                })
                samples_to_process.append(idx)
        
        print(f"\nScan complete!")
        print(f"   Total samples: {len(subset)}")
        print(f"   Good-quality samples: {len(compensation_stats['already_good'])} ({len(compensation_stats['already_good'])/len(subset)*100:.1f}%)")
        print(f"   Samples to re-process: {len(samples_to_process)} ({len(samples_to_process)/len(subset)*100:.1f}%)")
        
        if len(samples_to_process) == 0:
            print(f"\n✅ All samples meet quality requirements — no re-processing needed!")
            print(f"{'='*80}\n")
            return subset
        
        print(f"\nStarting re-processing of {len(samples_to_process)} samples...")
        print(f"{'='*80}\n")
    else:
        # Compensation disabled — process all samples
        samples_to_process = list(range(len(subset)))
    
    out = []
    
    for process_idx, original_idx in enumerate(tqdm(samples_to_process, desc="DIPPER attacking (subset)")):
        ex = subset[original_idx]
        ex2 = deepcopy(ex)
        
        if input_text_field not in ex2:
            ex2[output_attacked_field] = ""
            if output_length_field:
                ex2[output_length_field] = 0
            out.append(ex2)
            continue
        
        # Get input text
        input_gen = str(ex2[input_text_field]).strip()
        original_len = len(input_gen.split())
        
        if not input_gen:
            ex2[output_attacked_field] = ""
            if output_length_field:
                ex2[output_length_field] = 0
            out.append(ex2)
            continue
        
        try:
            # First paraphrase attempt
            output_text = dipper_paraphrase_single(input_gen)
            attacked_len = len(output_text.split())
            length_ratio = attacked_len / original_len if original_len > 0 else 0
            
            # Try multiple re-processing attempts for better results
            if length_ratio < min_length_ratio and max_retries > 0:
                best_result = output_text
                best_ratio = length_ratio
                
                for retry_attempt in range(max_retries):
                    try:
                        retry_output = dipper_paraphrase_single(input_gen)
                        retry_len = len(retry_output.split())
                        retry_ratio = retry_len / original_len if original_len > 0 else 0
                        
                        # Update if this attempt is better
                        if retry_ratio > best_ratio:
                            best_result = retry_output
                            best_ratio = retry_ratio
                            
                            # If threshold met, stop retrying
                            if retry_ratio >= min_length_ratio:
                                break
                    except Exception as e:
                        print(f"[Warning] Sample {original_idx} retry failed (attempt {retry_attempt+1}/{max_retries}): {e}")
                        continue
                
                # Update to best result
                output_text = best_result
                attacked_len = len(best_result.split())
                
                # Record re-processing result
                if enable_compensation:
                    original_ratio = compensation_stats['need_reprocess'][process_idx]['ratio']
                    if best_ratio >= min_length_ratio:
                        compensation_stats['reprocess_success'].append({
                            'idx': original_idx,
                            'original_ratio': original_ratio,
                            'final_ratio': best_ratio,
                            'original_len': original_len,
                            'final_len': attacked_len
                        })
                    else:
                        compensation_stats['reprocess_failed'].append({
                            'idx': original_idx,
                            'original_ratio': original_ratio,
                            'final_ratio': best_ratio,
                            'original_len': original_len,
                            'final_len': attacked_len
                        })
            else:
                # Passed on first attempt or compensation disabled
                if enable_compensation:
                    if length_ratio >= min_length_ratio:
                        compensation_stats['reprocess_success'].append({
                            'idx': original_idx,
                            'original_ratio': compensation_stats['need_reprocess'][process_idx]['ratio'],
                            'final_ratio': length_ratio,
                            'original_len': original_len,
                            'final_len': attacked_len
                        })
                    else:
                        compensation_stats['reprocess_failed'].append({
                            'idx': original_idx,
                            'original_ratio': compensation_stats['need_reprocess'][process_idx]['ratio'],
                            'final_ratio': length_ratio,
                            'original_len': original_len,
                            'final_len': attacked_len
                        })
            
            # Save attacked text
            ex2[output_attacked_field] = output_text
            
            # Compute length
            if output_length_field:
                ex2[output_length_field] = attacked_len
            
        except Exception as e:
            # If attack fails, keep original text or set empty
            print(f"[Warning] DIPPER attack failed for sample {original_idx}: {e}")
            import traceback
            traceback.print_exc()
            ex2[output_attacked_field] = ex2.get(input_text_field, "")
            if output_length_field:
                ex2[output_length_field] = len(str(ex2[output_attacked_field]).split())
        
        out.append((original_idx, ex2))
    
    # Build final results (preserve original order)
    if enable_compensation and len(samples_to_process) < len(subset):
        # Compensation mode: only update samples that needed re-processing
        final_out = deepcopy(subset)
        for original_idx, processed_sample in out:
            final_out[original_idx] = processed_sample
        result = final_out
    else:
        # Non-compensation mode: return results directly
        result = [item[1] for item in out]
    
    print(f"[DIPPER] Done, processed {len(out)} samples")
    
    # Print compensation statistics
    if enable_compensation and compensation_stats['need_reprocess']:
        print(f"\n{'='*80}")
        print(f"📊 DIPPER Attack compensation statistics")
        print(f"{'='*80}")
        print(f"Total samples: {compensation_stats['total_samples']}")
        print(f"Good quality (no action): {len(compensation_stats['already_good'])} ({len(compensation_stats['already_good'])/compensation_stats['total_samples']*100:.1f}%)")
        print(f"Needed re-processing: {len(compensation_stats['need_reprocess'])} ({len(compensation_stats['need_reprocess'])/compensation_stats['total_samples']*100:.1f}%)")
        print(f"Re-processing succeeded: {len(compensation_stats['reprocess_success'])} ({len(compensation_stats['reprocess_success'])/len(compensation_stats['need_reprocess'])*100:.1f}% of need_reprocess)")
        print(f"Re-processing still below threshold: {len(compensation_stats['reprocess_failed'])} ({len(compensation_stats['reprocess_failed'])/len(compensation_stats['need_reprocess'])*100:.1f}% of need_reprocess)")
        
        if compensation_stats['reprocess_success']:
            print(f"\n✅ Details of successfully re-processed samples (first 10):")
            for i, info in enumerate(compensation_stats['reprocess_success'][:10], 1):
                print(f"   {i}. sample {info['idx']}: "
                      f"original ratio {info['original_ratio']*100:.1f}% → "
                      f"final ratio {info['final_ratio']*100:.1f}% ({info['final_len']}/{info['original_len']} words)")
        
        if compensation_stats['reprocess_failed']:
            print(f"\n❌ Details of samples still below threshold after re-processing (first 10):")
            for i, info in enumerate(compensation_stats['reprocess_failed'][:10], 1):
                print(f"   {i}. sample {info['idx']}: "
                      f"original ratio {info['original_ratio']*100:.1f}% → "
                      f"final ratio {info['final_ratio']*100:.1f}% ({info['final_len']}/{info['original_len']} words)")
        print(f"{'='*80}\n")
    
    return result


# -----------------------------
# GPT attack (subset)
# -----------------------------
def gpt_attack_subset(
    subset: List[Dict[str, Any]],
    input_text_field: str,
    output_attacked_field: str,
    output_length_field: Optional[str],
    model_name: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    attack_prompt: str = "paraphrase the following paragraphs and try to keep the similar length to the original paragraphs\n",
    no_wm_attack: bool = False,
    verbose: bool = False,
    api_key: str = "",
    base_url: str = "https://api.openai.com/v1",
) -> List[Dict[str, Any]]:
    """
    Run the GPT paraphrase attack on a subset of samples
    (based on gpt_attack function in the original project utils/attack.py).
    
    Args:
        subset: sample subset to attack
        input_text_field: input text field name
        output_attacked_field: output field for attacked text
        output_length_field: output field for text length (optional)
        model_name: OpenAI model name (default: gpt-3.5-turbo)
        temperature: temperature (default: 0.7)
        max_tokens: max generated tokens (default: 1000)
        attack_prompt: attack prompt (default: prompt_id=2 content)
        no_wm_attack: whether to attack no_wm_output (currently unused; kept for compatibility)
        verbose: whether to print verbose output
        api_key: OpenAI API key
        base_url: OpenAI API base URL
        
    Returns:
        attacked sample list
        
    References:
        - original project: watermark_reliability_release/utils/attack.py:gpt_attack()
        - original project: watermark_reliability_release/attack_pipeline.py (parameter config)
    """
    if not GPT_AVAILABLE:
        raise ImportError("GPT dependencies unavailable; install: pip install openai tenacity")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for the GPT attack")
    
    print(f"[GPT] Using prompt: {attack_prompt[:80]}...")
    print(f"[GPT] Model: {model_name}, temperature: {temperature}, max_tokens: {max_tokens}")
    
    # Initialize OpenAI client (based on original project implementation)
    client_kwargs = {
        "api_key": api_key,
        "base_url": base_url.rstrip()  # strip trailing whitespace
    }
    
    print(f"[GPT] Using API endpoint: {base_url}")
    
    client = OpenAI(**client_kwargs)
    
    # Define completion function with retry (based on original project)
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def completion_with_backoff(**kwargs):
        return client.chat.completions.create(**kwargs)
    
    out = []
    
    for ex in tqdm(subset, desc="GPT attacking (subset)"):
        ex2 = deepcopy(ex)
        
        if input_text_field not in ex2:
            ex2[output_attacked_field] = ""
            if output_length_field:
                ex2[output_length_field] = 0
            out.append(ex2)
            continue
        
        # Get original text
        original_text = str(ex2[input_text_field]).strip()
        
        if not original_text:
            ex2[output_attacked_field] = ""
            if output_length_field:
                ex2[output_length_field] = 0
            out.append(ex2)
            continue
        
        try:
            # Build attack query (based on original project utils/attack.py:gpt_attack())
            attacker_query = attack_prompt + original_text
            query_msg = {"role": "user", "content": attacker_query}
            
            # Call OpenAI API (based on original project implementation)
            outputs = completion_with_backoff(
                model=model_name,
                messages=[query_msg],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # Extract attacked text
            attacked_text = outputs.choices[0].message.content
            
            # Check response count (based on original project assertion)
            assert len(outputs.choices) == 1, \
                "OpenAI API returned more than one response, unexpected for length inference of the output"
            
            # Save attacked text
            ex2[output_attacked_field] = attacked_text.strip()
            
            # Save length info (using completion_tokens, based on original project)
            if output_length_field:
                ex2[output_length_field] = outputs.usage.completion_tokens
            
            # Print detailed info if verbose (based on original project)
            if verbose:
                original_len = len(original_text.split())
                attacked_len = ex2.get(output_length_field, len(attacked_text.split()))
                print(f"\nOriginal text (T={original_len}):\n{original_text}")
                print(f"\nAttacked text (T={attacked_len}):\n{attacked_text}")
                print("-" * 80)
        
        except Exception as e:
            # If attack fails, keep original text or set empty
            print(f"[Warning] GPT attack failed for sample: {e}")
            import traceback
            if verbose:
                traceback.print_exc()
            ex2[output_attacked_field] = ex2.get(input_text_field, "")
            if output_length_field:
                ex2[output_length_field] = len(ex2[output_attacked_field].split())
        
        out.append(ex2)
    
    print(f"[GPT] Done, processed {len(out)} samples")
    
    return out


# -----------------------------
# Main
# -----------------------------
def main(cfg: CONFIG):
    print(f"[Config] method={cfg.attack_method}")
    print(f"[Config] input={cfg.input_file}")
    print(f"[Config] output={cfg.output_file}")
    print(f"[Config] text_field={cfg.input_text_field}")
    if cfg.filter_flag_field:
        print(f"[Config] only attack where {cfg.filter_flag_field} == {cfg.filter_flag_value}")
    else:
        print("[Config] filter-field empty — attacking all samples (subject to max_samples)")

    records = load_records(cfg.input_file)
    print(f"[Data] total loaded: {len(records)}")

    # select indices to attack
    attack_idxs = select_attack_indices(
        records,
        flag_field=cfg.filter_flag_field,
        flag_value=cfg.filter_flag_value,
        max_samples=cfg.max_samples,  # max_samples = maximum number of matching samples to attack
    )
    print(f"[Data] selected for attack: {len(attack_idxs)}")

    if len(attack_idxs) == 0:
        print("[Warn] No samples match the filter criteria; saving output unchanged.")
        save_records(records, cfg.output_file)
        return

    subset = [records[i] for i in attack_idxs]

    # run attack on subset only
    if cfg.attack_method == "translation":
        print("[Run] Translation attack on subset...")
        print(f"[Config] enable_compensation={cfg.translation_enable_compensation}")
        print(f"[Config] min_length_ratio={cfg.translation_min_length_ratio}")
        print(f"[Config] max_retries={cfg.translation_max_retries}")
        attacked_subset = translation_attack_subset(
            subset=subset,
            input_text_field=cfg.input_text_field,
            output_attacked_field=cfg.output_attacked_field,
            output_length_field=cfg.output_length_field,
            no_wm_attack=cfg.no_wm_attack,
            enable_compensation=cfg.translation_enable_compensation,
            min_length_ratio=cfg.translation_min_length_ratio,
            max_retries=cfg.translation_max_retries,
        )

    elif cfg.attack_method == "swap":
        print("[Run] Swap attack on subset...")
        print(f"[Config] swap_p={cfg.swap_p} (overall per-token attack probability)")
        print(f"[Config] Per-attack-type probability: {cfg.swap_p * 0.25:.3f} (sentence swap/word swap/word deletion/word repetition)")
        tok = load_tokenizer(cfg.model_name_or_path or "gpt2")
        attacked_subset = swap_attack_subset(
            subset=subset,
            input_text_field=cfg.input_text_field,
            output_attacked_field=cfg.output_attacked_field,
            output_length_field=cfg.output_length_field,
            tokenizer=tok,
            min_len=cfg.cp_attack_min_len,
            swap_p=cfg.swap_p,
        )

    elif cfg.attack_method == "synonym":
        print("[Run] Synonym attack on subset...")
        print(f"[Config] synonym_p={cfg.synonym_p} (replacement probability)")
        print(f"[Config] replacements_field={cfg.synonym_replacements_field} (replacement count field)")
        tok = load_tokenizer(cfg.model_name_or_path or "gpt2")
        attacked_subset = synonym_attack_subset(
            subset=subset,
            input_text_field=cfg.input_text_field,
            output_attacked_field=cfg.output_attacked_field,
            output_length_field=cfg.output_length_field,
            tokenizer=tok,
            synonym_p=cfg.synonym_p,
            min_len=cfg.cp_attack_min_len,
            replacements_field=cfg.synonym_replacements_field,
        )

    elif cfg.attack_method == "dipper":
        print("[Run] DIPPER attack on subset...")
        print(f"[Config] model_name={cfg.dipper_model_name}")
        print(f"[Config] lex={cfg.dipper_lex}, order={cfg.dipper_order}")
        print(f"[Config] no_ctx={cfg.dipper_no_ctx}, sent_interval={cfg.dipper_sent_interval}")
        print(f"[Config] enable_compensation={cfg.dipper_enable_compensation}")
        print(f"[Config] min_length_ratio={cfg.dipper_min_length_ratio}")
        print(f"[Config] max_retries={cfg.dipper_max_retries}")
        
        # Determine device (requires torch import)
        if not DIPPER_AVAILABLE:
            raise ImportError("DIPPER dependencies unavailable; install: transformers, torch, nltk")
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Config] device={device}")
        
        attacked_subset = dipper_attack_subset(
            subset=subset,
            input_text_field=cfg.input_text_field,
            output_attacked_field=cfg.output_attacked_field,
            output_length_field=cfg.output_length_field,
            model_name=cfg.dipper_model_name,
            lex=cfg.dipper_lex,
            order=cfg.dipper_order,
            no_ctx=cfg.dipper_no_ctx,
            sent_interval=cfg.dipper_sent_interval,
            device=device,
            enable_compensation=cfg.dipper_enable_compensation,
            min_length_ratio=cfg.dipper_min_length_ratio,
            max_retries=cfg.dipper_max_retries,
        )

    elif cfg.attack_method == "gpt":
        print("[Run] GPT attack on subset...")
        print(f"[Config] model_name={cfg.gpt_model_name}")
        print(f"[Config] temperature={cfg.gpt_temperature}, max_tokens={cfg.gpt_max_tokens}")
        print(f"[Config] prompt: {cfg.gpt_prompt[:80]}...")
        print(f"[Config] verbose={cfg.gpt_verbose}")
        print(f"[Config] base_url: {cfg.gpt_base_url}")
        
        if not GPT_AVAILABLE:
            raise ImportError("GPT dependencies unavailable; install: pip install openai tenacity")
        
        attacked_subset = gpt_attack_subset(
            subset=subset,
            input_text_field=cfg.input_text_field,
            output_attacked_field=cfg.output_attacked_field,
            output_length_field=cfg.output_length_field,
            model_name=cfg.gpt_model_name,
            temperature=cfg.gpt_temperature,
            max_tokens=cfg.gpt_max_tokens,
            attack_prompt=cfg.gpt_prompt,
            no_wm_attack=cfg.no_wm_attack,
            verbose=cfg.gpt_verbose,
            api_key=cfg.gpt_api_key,
            base_url=cfg.gpt_base_url,
        )
    else:
        raise ValueError(f"Unsupported attack_method: {cfg.attack_method}; supported methods: translation, swap, synonym, dipper, gpt")

    # write attacked subset back into full records
    out_records = deepcopy(records)
    for i, attacked_ex in zip(attack_idxs, attacked_subset):
        out_records[i] = attacked_ex

    save_records(out_records, cfg.output_file)
    print("[Done] saved to:", cfg.output_file)


if __name__ == "__main__":
    # Parse command-line arguments (GPU was already set at file top)
    args = parse_args()
    
    # Update configuration
    cfg = update_config_from_args(CONFIG, args)
    
    # Run main program
    main(cfg)
