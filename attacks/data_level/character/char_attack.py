#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simplified character-level attack script.
Reads watermarked samples, applies character-level attacks, and saves original/attacked text pairs.

Usage:
    python char_attack.py \\
        --input_file <input_file> \\
        --output_file <output_file> \\
        [other optional args...]

    See full help: python char_attack.py --help
"""

import os
import sys
import json
import argparse
from copy import deepcopy
from typing import List, Dict, Any, Optional

from tqdm import tqdm
from transformers import AutoTokenizer
import Levenshtein

# Set NLTK data path (override via NLTK_DATA env var or --nltk_data arg)
os.environ["NLTK_DATA"] = ""

# Add current directory to path for importing random_attack
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from random_attack import RandomAttack

# ============================================================================
# Default configuration (can be overridden via command-line arguments)
# ============================================================================


# -----------------------------
# IO helpers
# -----------------------------
def load_records(path: str) -> List[Dict[str, Any]]:
    """Load a JSON or JSONL file and return a list of records."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    # .json (list) vs .jsonl
    if path.endswith(".json") and not path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, list):
            raise ValueError(f"Expected a list in JSON file, got {type(obj)}")
        return obj
    else:
        # JSONL format
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records


def save_records(records: List[Dict[str, Any]], path: str) -> None:
    """Save records to a JSON or JSONL file."""
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    lower = path.lower()
    if lower.endswith(".json") and not lower.endswith(".jsonl"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    else:
        # JSONL format
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


# -----------------------------
# Selection helper
# -----------------------------
def select_attack_indices(
    records: List[Dict[str, Any]],
    flag_field: str,
    flag_value: Any,
    max_samples: Optional[int] = None,
) -> List[int]:
    """Return indices to attack. Empty flag_field means all records."""
    idxs = []
    for i, ex in enumerate(records):
        if flag_field and ex.get(flag_field, False) != flag_value:
            continue
        idxs.append(i)
        if max_samples and max_samples > 0 and len(idxs) >= max_samples:
            break
    return idxs


# -----------------------------
# Character attack
# -----------------------------
def char_attack_subset(
    subset: List[Dict[str, Any]],
    input_text_field: str,
    output_attacked_field: str,
    output_length_field: Optional[str],
    output_char_edit_dist_field: str,
    output_char_op_field: str,
    tokenizer,
    max_edit_rate: float = 0.3,
    atk_times: int = 1,
    char_op: int = 2,
) -> List[Dict[str, Any]]:
    """
    Apply character-level attacks to a subset of records.

    Args:
        subset: List of records to attack.
        input_text_field: Field name for the input text.
        output_attacked_field: Field name for the attacked text output.
        output_length_field: Field name for word count (None = skip).
        output_char_edit_dist_field: Field name for character edit distance.
        output_char_op_field: Field name for attack type label.
        tokenizer: Tokenizer used to initialize RandomAttack.
        max_edit_rate: Maximum edit rate.
        atk_times: Number of attack iterations.
        char_op: Character operation type.
    """
    # Initialize the attacker
    print(f"Initializing character-level attacker...")
    if char_op <= 0:
        print(f"Character operation mode: random (will randomly choose from 1-5: delete/homoglyph/insert/swap/typo)")
    else:
        op_names = {1: 'delete', 2: 'homoglyph', 3: 'insert', 4: 'swap', 5: 'typo'}
        print(f"Character operation mode: {op_names.get(char_op, f'op{char_op}')}")
    
    rand_attack = RandomAttack(
        wm_name='',  # watermark detection not needed here
        tokenizer=tokenizer,
        char_op=char_op,
    )
    
    out = []
    
    for ex in tqdm(subset, desc="Applying character attacks"):
        ex2 = deepcopy(ex)
        
        if input_text_field not in ex2:
            ex2[output_attacked_field] = ""
            if output_length_field:
                ex2[output_length_field] = 0
            ex2[output_char_edit_dist_field] = 0
            ex2[output_char_op_field] = char_op
            out.append(ex2)
            continue
        
        original_text = ex2[input_text_field]
        
        # Skip empty text
        if not original_text or len(original_text.strip()) == 0:
            ex2[output_attacked_field] = ""
            if output_length_field:
                ex2[output_length_field] = 0
            ex2[output_char_edit_dist_field] = 0
            ex2[output_char_op_field] = char_op
            out.append(ex2)
            continue
        
        # Run character-level attack
        try:
            adv_result = rand_attack.get_adv(
                original_text,
                atk_style='char',
                max_edit_rate=max_edit_rate,
                atk_times=atk_times,
                target_class=0,
            )
            attacked_text = adv_result['sentence']
            
            # Compute character edit distance
            char_edit_dist = Levenshtein.distance(original_text, attacked_text)

            # Determine the actual operation used (if char_op <= 0, RandomAttack chooses randomly)
            actual_char_op = char_op
            if char_op <= 0:
                # RandomAttack selects randomly; record 0 to indicate random selection
                actual_char_op = 0
            
            ex2[output_attacked_field] = attacked_text
            ex2[output_char_edit_dist_field] = char_edit_dist
            ex2[output_char_op_field] = actual_char_op
            
            # Compute word count (if requested)
            if output_length_field:
                ex2[output_length_field] = len(str(attacked_text).split())
                
        except Exception as e:
            # On failure, keep the original text
            print(f"[Warning] Character attack failed: {e}")
            import traceback
            traceback.print_exc()
            ex2[output_attacked_field] = original_text
            ex2[output_char_edit_dist_field] = 0
            ex2[output_char_op_field] = char_op
            if output_length_field:
                ex2[output_length_field] = len(str(original_text).split())
        
        out.append(ex2)
    
    return out


# -----------------------------
# Main
# -----------------------------
def main(
    input_file: str,
    output_file: str,
    input_text_field: str = "watermarked_text",
    output_attacked_field: str = "attacked_text",
    output_length_field: str = "attacked_text_word_count",
    output_char_edit_dist_field: str = "char_edit_distance",
    output_char_op_field: str = "char_attack_type",
    filter_flag_field: str = "",
    filter_flag_value: bool = True,
    max_edit_rate: float = 0.3,
    atk_times: int = 1,
    char_op: int = 5,
    tokenizer_name: str = "",  # set via --tokenizer_name (e.g. $MODEL_ROOT/gpt2)
    max_samples: Optional[int] = None,
):
    """
    Main function to run character-level attacks.

    Args:
        input_file: Path to input file.
        output_file: Path to output file.
        input_text_field: Field name for the input text.
        output_attacked_field: Field name for the attacked text output.
        output_length_field: Field name for word count (None = skip).
        output_char_edit_dist_field: Field name for character edit distance.
        output_char_op_field: Field name for attack type label.
        filter_flag_field: Optional field used for filtering; empty = attack all.
        filter_flag_value: Only attack records where filter_flag_field == this value.
        max_edit_rate: Maximum edit rate (0.0-1.0).
        atk_times: Number of attack iterations.
        char_op: Character operation type (1=delete, 2=homoglyph, 3=insert, 4=swap, 5=typo, 0=random).
        tokenizer_name: Tokenizer path/name.
        max_samples: Maximum number of samples to process (None = all).
    """
    print(f"[Config] Input file: {input_file}")
    print(f"[Config] Output file: {output_file}")
    print(f"[Config] Text field: {input_text_field}")
    if filter_flag_field:
        print(f"[Config] Attacking samples where {filter_flag_field} == {filter_flag_value}")
    else:
        print("[Config] filter_flag_field empty — attacking all samples (subject to max_samples)")
    print(f"[Config] Max edit rate: {max_edit_rate}")
    print(f"[Config] Attack iterations: {atk_times}")
    print(f"[Config] Character operation type: {char_op}")

    # Load data
    records = load_records(input_file)
    print(f"[Data] Loaded {len(records)} records")

    # Select indices of records to attack
    attack_idxs = select_attack_indices(
        records,
        flag_field=filter_flag_field,
        flag_value=filter_flag_value,
        max_samples=max_samples,
    )
    print(f"[Data] Selected {len(attack_idxs)} records for attack")

    if len(attack_idxs) == 0:
        print("[Warn] No matching samples found; saving original records.")
        save_records(records, output_file)
        return

    # Extract subset to attack
    subset = [records[i] for i in attack_idxs]

    # Load tokenizer
    print(f"[Tokenizer] Loading: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Run character-level attacks
    print("[Run] Applying character-level attacks to selected samples...")
    attacked_subset = char_attack_subset(
        subset=subset,
        input_text_field=input_text_field,
        output_attacked_field=output_attacked_field,
        output_length_field=output_length_field,
        output_char_edit_dist_field=output_char_edit_dist_field,
        output_char_op_field=output_char_op_field,
        tokenizer=tokenizer,
        max_edit_rate=max_edit_rate,
        atk_times=atk_times,
        char_op=char_op,
    )

    # Write attack results back into full records
    out_records = deepcopy(records)
    for i, attacked_ex in zip(attack_idxs, attacked_subset):
        out_records[i] = attacked_ex

    # Save results
    save_records(out_records, output_file)
    print(f"[Done] Results saved to: {output_file}")
    print(f"[Done] Processed {len(attack_idxs)} samples")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Character-level attack script (supports JSON/JSONL format)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Character operation types (--char_op):
  1 = Character Deletion
  2 = Homoglyph Substitution (replace with visually similar characters)
  3 = Character Insertion
  4 = Character Swap (swap adjacent characters)
  5 = Typo (simulate keyboard typos)
  0 or negative = randomly choose one of the above

Examples:
  # Basic usage
  python char_attack.py \\
      --input_file data.json \\
      --output_file output.json \\
      --char_op 5

  # With all arguments
  python char_attack.py \\
      --input_file data.json \\
      --output_file output.json \\
      --input_text_field "watermarked_text" \\
      --filter_flag_field "is_target" \\
      --filter_flag_value True \\
      --max_edit_rate 0.3 \\
      --char_op 5
        """
    )
    
    # Required arguments
    parser.add_argument("--input_file", type=str, required=True,
                        help="Input file path (.json or .jsonl)")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output file path (.json or .jsonl)")

    # Field name configuration
    parser.add_argument("--input_text_field", type=str, default="watermarked_text",
                        help="Input text field name (default: 'watermarked_text')")
    parser.add_argument("--output_attacked_field", type=str, default="attacked_text",
                        help="Output field name for attacked text (default: 'attacked_text')")
    parser.add_argument("--output_length_field", type=str, default="attacked_text_word_count",
                        help="Field name for word count (default: 'attacked_text_word_count'; set to empty string to skip)")
    parser.add_argument("--output_char_edit_dist_field", type=str, default="char_edit_distance",
                        help="Field name for character edit distance (default: 'char_edit_distance')")
    parser.add_argument("--output_char_op_field", type=str, default="char_attack_type",
                        help="Field name for attack type label (default: 'char_attack_type')")

    # Sample filtering configuration
    parser.add_argument("--filter_flag_field", type=str, default="",
                        help="Optional filter field; empty = process all samples")
    parser.add_argument("--filter_flag_value", type=str, default="True",
                        help="Filter field value when --filter_flag_field is set (default: 'True')")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process (default: None = process all matching samples)")

    # Character attack parameters
    parser.add_argument("--max_edit_rate", type=float, default=0.3,
                        help="Maximum edit rate (0.0-1.0), controlling the maximum fraction of characters to modify (default: 0.3)")
    parser.add_argument("--atk_times", type=int, default=1,
                        help="Number of attack iterations (default: 1)")
    parser.add_argument("--char_op", type=int, default=5,
                        help="Character operation type (1=delete, 2=homoglyph, 3=insert, 4=swap, 5=typo, 0=random; default: 5)")

    # Tokenizer configuration
    parser.add_argument("--tokenizer_name", type=str, default="",
                        help="Tokenizer path (default: ''; set via --tokenizer_name or $MODEL_ROOT env var)")
    
    args = parser.parse_args()
    
    # Convert filter_flag_value string to bool
    filter_flag_value_str = str(args.filter_flag_value).lower()
    if filter_flag_value_str in ['true', '1', 'yes']:
        filter_flag_value = True
    elif filter_flag_value_str in ['false', '0', 'no']:
        filter_flag_value = False
    else:
        # Fallback: try casting directly
        try:
            filter_flag_value = bool(args.filter_flag_value)
        except:
            filter_flag_value = True  # default

    # Convert empty string to None for output_length_field
    output_length_field = args.output_length_field if args.output_length_field else None
    
    return args, filter_flag_value, output_length_field


if __name__ == "__main__":
    args, filter_flag_value, output_length_field = parse_args()
    
    try:
        main(
            input_file=args.input_file,
            output_file=args.output_file,
            input_text_field=args.input_text_field,
            output_attacked_field=args.output_attacked_field,
            output_length_field=output_length_field,
            output_char_edit_dist_field=args.output_char_edit_dist_field,
            output_char_op_field=args.output_char_op_field,
            filter_flag_field=args.filter_flag_field,
            filter_flag_value=filter_flag_value,
            max_edit_rate=args.max_edit_rate,
            atk_times=args.atk_times,
            char_op=args.char_op,
            tokenizer_name=args.tokenizer_name,
            max_samples=args.max_samples,
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise
