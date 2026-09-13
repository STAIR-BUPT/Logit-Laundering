"""
build_assistant_llm.py
Extract the first K layers from a full LLaMA model to build the “assistant LLM”
used in the paper's adaptive logit fusion framework.
"""

import os
import torch
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer

def save_assistant_llm(base_model_path, save_path, num_layers=8, dtype=torch.bfloat16):
    """
    Extract the first `num_layers` layers from a full LLaMA model, build an assistant LLM, and save it.
    """
    print(f"Loading base model from {base_model_path} ...")
    base_llm = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map="cpu"
    )
    origin_config = base_llm.config

    # ---- 1️⃣ Build new config with reduced layer count ----
    new_config = copy.deepcopy(origin_config)
    new_config.num_hidden_layers = num_layers
    # Qwen2/Qwen2.5: len(config.layer_types) must equal num_hidden_layers;
    # otherwise transformers raises at config load time:
    # ValueError: `num_hidden_layers` (...) must be equal to the number of layer types (...)
    if hasattr(new_config, "layer_types") and new_config.layer_types is not None:
        try:
            layer_types = list(new_config.layer_types)
            if len(layer_types) >= num_layers:
                new_config.layer_types = layer_types[:num_layers]
            else:
                # Edge case: original config is shorter than requested; pad with the last type
                fill = layer_types[-1] if len(layer_types) else "full_attention"
                new_config.layer_types = layer_types + [fill] * (num_layers - len(layer_types))
        except Exception:
            # If layer_types is not iterable, skip and let downstream errors surface
            pass

    # Sync sliding-window fields to avoid inconsistency after layer count change
    if hasattr(new_config, "max_window_layers") and new_config.max_window_layers is not None:
        try:
            new_config.max_window_layers = min(int(new_config.max_window_layers), int(num_layers))
        except Exception:
            pass

    # ---- 2️⃣ Initialize an empty model skeleton ----
    small_model = AutoModelForCausalLM.from_config(
        new_config,
        torch_dtype=dtype
    )

    # ---- 3️⃣ Copy weights (embeddings, first K layers, norm, lm_head) ----
    print(f"Copying first {num_layers} layers...")

    small_model.model.embed_tokens.load_state_dict(base_llm.model.embed_tokens.state_dict())
    small_model.model.norm.load_state_dict(base_llm.model.norm.state_dict())

    for i in range(num_layers):
        small_model.model.layers[i].load_state_dict(base_llm.model.layers[i].state_dict())

    small_model.lm_head.load_state_dict(base_llm.lm_head.state_dict())

    # ---- 4️⃣ Save to specified path ----
    small_model.save_pretrained(save_path)

    # ---- 5️⃣ Save tokenizer (must match the base model) ----
    print(f"Loading tokenizer from {base_model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    print(f"Tokenizer type: {type(tokenizer)}")
    
    if not hasattr(tokenizer, 'save_pretrained'):
        raise RuntimeError(f"Loaded tokenizer is not valid. Type: {type(tokenizer)}, Value: {tokenizer}")
    
    tokenizer.save_pretrained(save_path)

    print(f"Assistant LLM saved to: {save_path}")
    print(f"   Layers kept: {num_layers}")
    print(f"   Dtype: {dtype}")
    print(f"   Tokenizer copied from: {base_model_path}")

    return small_model


if __name__ == "__main__":
    # ===== Configuration =====
    BASE_MODEL = os.environ.get("BASE_MODEL_PATH", "")   # e.g. $MODEL_ROOT/Llama-3.2-1B-Instruct
    SAVE_PATH = os.environ.get("SAVE_PATH", "")           # e.g. $ASSISTANT_MODEL_ROOT/init/Llama-3.2-1B-Instruct-8layers
    NUM_LAYERS = int(os.environ.get("NUM_LAYERS", "8"))

    if not BASE_MODEL:
        raise SystemExit("BASE_MODEL_PATH is required")
    if not SAVE_PATH:
        raise SystemExit("SAVE_PATH is required")
    if NUM_LAYERS <= 0:
        raise SystemExit("NUM_LAYERS must be positive")

    # ===== Run =====
    save_assistant_llm(BASE_MODEL, SAVE_PATH, num_layers=NUM_LAYERS)
