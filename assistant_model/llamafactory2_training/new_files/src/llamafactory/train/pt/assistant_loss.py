# assistant_loss.py
# Copyright 2025 the LlamaFactory team.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Callable, Optional


# ============================================================
# Base loss functions
# ============================================================

def NextTokenPredictionLoss(model, input_ids, attention_mask, labels, **kwargs):
    """Standard language modeling loss (cross-entropy)."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    assert outputs.loss is not None, "NextTokenPredictionLoss: loss is None"
    return outputs.loss


def UniformLossFunc(model, input_ids, attention_mask, labels=None, **kwargs):
    """Force output distribution toward uniform (used for Evasion-free Set constraint)."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    logits = outputs.logits
    soft = F.softmax(logits, dim=-1).view(-1, logits.size(-1))
    uniform = torch.full_like(soft, 1.0 / logits.size(-1))
    kl = F.kl_div(soft.log(), uniform, reduction='batchmean')
    return kl


def EnhancedUniformLossFunc(model, input_ids, attention_mask, labels=None, entropy_weight=2.0, **kwargs):
    """Enhanced flat-distribution loss with a stronger uniformity constraint."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    logits = outputs.logits

    # Reshape to [batch*seq, vocab_size]
    logits_flat = logits.view(-1, logits.size(-1))
    probs = F.softmax(logits_flat, dim=-1)

    # 1. Base KL divergence loss: pull toward uniform distribution
    uniform = torch.full_like(probs, 1.0 / logits.size(-1))
    kl_loss = F.kl_div(probs.log(), uniform, reduction='batchmean')

    # 2. Variance penalty: penalize variance of the probability distribution (smaller = flatter)
    prob_mean = probs.mean(dim=-1, keepdim=True)  # ideally 1/vocab_size
    variance = ((probs - prob_mean) ** 2).mean()

    # Combined loss: KL divergence + variance penalty (both non-negative)
    total_loss = kl_loss + entropy_weight * variance
    return total_loss


def KLLossFunc(model, input_ids, attention_mask, labels, oracle_model, **kwargs):
    """KL divergence constraint against the oracle model."""
    with torch.no_grad():
        oracle_out = oracle_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        oracle_probs = F.log_softmax(oracle_out.logits, dim=-1)
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    probs = F.log_softmax(out.logits, dim=-1)
    kl = F.kl_div(probs, oracle_probs, reduction="batchmean", log_target=True)
    return kl


def GradAscentLossFunc(model, input_ids, attention_mask, labels, **kwargs):
    """Gradient ascent loss (Evasion Set)."""
    return -NextTokenPredictionLoss(model, input_ids, attention_mask, labels)


def GradDescentLossFunc(model, input_ids, attention_mask, labels, **kwargs):
    """Standard gradient descent loss (Evasion-free Set)."""
    return NextTokenPredictionLoss(model, input_ids, attention_mask, labels)


# ============================================================
# Utility: split a batch into Evasion / Evasion-free subsets
# ============================================================

def split_evasion_evasion_free(input_ids, attention_mask, labels=None, evasion_free_labels=None):
    if evasion_free_labels is None:
        return ((input_ids, attention_mask, labels), (None, None, None))

    evasion_mask = evasion_free_labels == 0
    evasion_free_mask = evasion_free_labels == 1

    def select(mask, x):
        return x[mask] if x is not None else None

    evasion_tuple = (
        select(evasion_mask, input_ids),
        select(evasion_mask, attention_mask),
        select(evasion_mask, labels),
    )
    evasion_free_tuple = (
        select(evasion_free_mask, input_ids),
        select(evasion_free_mask, attention_mask),
        select(evasion_free_mask, labels),
    )
    return evasion_tuple, evasion_free_tuple


# ============================================================
# Main class: EvasionLoss
# ============================================================

class EvasionLoss:
    """
    General-purpose Evasion Set + Evasion-free Set loss combiner.
    Example:
        evasion_loss_func = GradAscentLossFunc
        evasion_free_loss_func = UniformLossFunc
        evasion_free_weight = 0.5
    """

    def __init__(
        self,
        evasion_loss_func: Callable,
        evasion_free_loss_func: Optional[Callable] = None,
        evasion_free_weight: float = 1.0,
    ):
        self.evasion_loss_func = evasion_loss_func
        self.evasion_free_loss_func = evasion_free_loss_func
        self.evasion_free_weight = evasion_free_weight

    def __call__(self, model, batch: Dict[str, Any], oracle_model=None) -> Dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch.get("labels", None)
        evasion_free_labels = batch.get("evasion_free_labels", None)

        (e_ids, e_mask, e_labels), (ef_ids, ef_mask, ef_labels) = split_evasion_evasion_free(
            input_ids, attention_mask, labels, evasion_free_labels
        )

        device = input_ids.device

        # Evasion Set loss
        if e_ids is not None and e_ids.size(0) > 0:
            evasion_loss = self.evasion_loss_func(
                model,
                input_ids=e_ids,
                attention_mask=e_mask,
                labels=e_labels,
                oracle_model=oracle_model,
            )
        else:
            evasion_loss = torch.tensor(0.0, device=device)

        # Evasion-free Set loss
        if (
            self.evasion_free_loss_func is not None
            and ef_ids is not None
            and ef_ids.size(0) > 0
        ):
            evasion_free_loss = self.evasion_free_loss_func(
                model,
                input_ids=ef_ids,
                attention_mask=ef_mask,
                labels=ef_labels,
                oracle_model=oracle_model,
            )
        else:
            evasion_free_loss = torch.tensor(0.0, device=device)

        total_loss = evasion_loss + self.evasion_free_weight * evasion_free_loss
        return {
            "loss": total_loss,
            "evasion_loss": evasion_loss.detach(),
            "evasion_free_loss": evasion_free_loss.detach(),
        }


# ============================================================
# Factory: create loss from config
# ============================================================

def create_evasion_loss(loss_config: Dict[str, Any]) -> EvasionLoss:
    """Dynamically build an EvasionLoss from string names in config."""
    def get_func(name):
        if name is None:
            return None
        if name not in globals():
            raise NotImplementedError(f"Unknown loss: {name}")
        return globals()[name]

    evasion_func = get_func(loss_config.get("evasion_loss"))
    evasion_free_func = get_func(loss_config.get("evasion_free_loss"))
    evasion_free_weight = loss_config.get("evasion_free_weight", 1.0)

    return EvasionLoss(evasion_func, evasion_free_func, evasion_free_weight)

