# CustomTrainer for Logit Laundering assistant training.
# Evasion Set: random-segment training; Evasion-free Set: full-text training.

import copy
import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict

import numpy as np
import torch
from torch.utils.data import Sampler
from transformers import Trainer
from typing_extensions import override

from .assistant_loss import (
    EvasionLoss,
    GradDescentLossFunc,
    UniformLossFunc,
)

if TYPE_CHECKING:
    from ...hparams import FinetuningArguments, ModelArguments


class EvasionSampler(Sampler):
    """Fixed-ratio sampler over Evasion-free and Evasion index ranges."""

    def __init__(self, dataset, batch_size, evasion_size, evasion_free_ratio_each_batch=0.75, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.evasion_size = evasion_size
        self.evasion_free_ratio_each_batch = evasion_free_ratio_each_batch
        self.seed = seed

        self.evasion_free_per_batch = max(1, int(batch_size * evasion_free_ratio_each_batch))
        self.evasion_per_batch = batch_size - self.evasion_free_per_batch

        self.total_samples = len(dataset)
        self.evasion_indices = list(range(min(evasion_size, self.total_samples)))
        self.evasion_free_indices = list(range(evasion_size, self.total_samples))

        print("EvasionSampler initialized:")
        print(f"  batch_size={batch_size}")
        print(
            f"  evasion_free_ratio={evasion_free_ratio_each_batch:.1%} "
            f"({self.evasion_free_per_batch}/batch)"
        )
        print(
            f"  evasion_ratio={1 - evasion_free_ratio_each_batch:.1%} "
            f"({self.evasion_per_batch}/batch)"
        )
        print(
            f"  split: evasion[0, {len(self.evasion_indices)}), "
            f"evasion_free[{evasion_size}, {self.total_samples})"
        )

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        num_batches = len(self.dataset) // self.batch_size

        for _ in range(num_batches):
            batch_indices = []

            if len(self.evasion_free_indices) >= self.evasion_free_per_batch:
                batch_indices.extend(
                    rng.choice(self.evasion_free_indices, size=self.evasion_free_per_batch, replace=False)
                )
            else:
                batch_indices.extend(self.evasion_free_indices)

            if len(self.evasion_indices) >= self.evasion_per_batch:
                batch_indices.extend(
                    rng.choice(self.evasion_indices, size=self.evasion_per_batch, replace=False)
                )
            else:
                batch_indices.extend(self.evasion_indices)

            rng.shuffle(batch_indices)
            yield from batch_indices

    def __len__(self):
        return len(self.dataset)


EVASION_CONFIG = dict(
    # Prefix length of the Evasion Set in the training JSON.
    # Put the target subset first, then set EVASION_SIZE to that count.
    evasion_size=int(os.environ.get("EVASION_SIZE", "0")),
    evasion_free_weight=3,
    use_custom_sampler=True,
    evasion_free_ratio_each_batch=0.6,
    segment_ratio=float(os.environ.get("SEGMENT_RATIO", "0.05")),
    fix_segment_per_sample=False,
    segment_suffix=None,
    verbose_loss=os.environ.get("ASSISTANT_TRAIN_VERBOSE", "0") == "1",
)


class CustomTrainer(Trainer):
    """Trainer for the Logit Laundering assistant (Statistical Bias Estimator)."""

    def __init__(self, *, evasion_cfg=None, finetuning_args=None, **kwargs):
        self.processor = kwargs.pop("processor", None)
        super().__init__(**kwargs)
        self.args.remove_unused_columns = False
        self.finetuning_args = finetuning_args
        self.cfg = (evasion_cfg or EVASION_CONFIG).copy()

        self.evasion_size = int(self.cfg["evasion_size"])
        if self.evasion_size <= 0:
            raise ValueError(
                "EVASION_SIZE must be a positive integer (prefix length of the Evasion Set). "
                "Put the target subset first in the training JSON, "
                "then export EVASION_SIZE=<that count>."
            )

        self.unlearn_loss_fn = EvasionLoss(
            evasion_loss_func=GradDescentLossFunc,
            evasion_free_loss_func=UniformLossFunc,
            evasion_free_weight=self.cfg["evasion_free_weight"],
        )

        self._step_counter = 0
        self.segment_ratio = float(self.cfg["segment_ratio"])
        self.fix_segment_per_sample = bool(self.cfg.get("fix_segment_per_sample", False))
        self.segment_suffix = self.cfg.get("segment_suffix", None)
        self._fixed_segments = {}
        self._verbose_loss = bool(self.cfg.get("verbose_loss", False))

        self._suffix_token_ids = None
        if self.segment_suffix is not None:
            try:
                tokenizer = getattr(self, "tokenizer", None)
                if tokenizer is None:
                    raise ValueError("Tokenizer not available")
                self._suffix_token_ids = tokenizer.encode(self.segment_suffix, add_special_tokens=False)
                print(f"Segment suffix set: {self.segment_suffix!r} ({len(self._suffix_token_ids)} tokens)")
            except Exception as exc:
                print(f"Warning: failed to encode segment suffix ({exc}); suffix disabled")
                self._suffix_token_ids = None

        print("Assistant training config:")
        print("  goal: evasion-free ≈ base; evasion segments amplify logit bias")
        print(
            f"  sampling: evasion_free={self.cfg['evasion_free_ratio_each_batch']:.1%} "
            f"evasion={1 - self.cfg['evasion_free_ratio_each_batch']:.1%}"
        )
        print(f"  segment_ratio={self.segment_ratio:.1%}")
        print(f"  evasion_size={self.evasion_size}")
        print(
            "  segment_mode="
            + ("fixed" if self.fix_segment_per_sample else "resampled")
        )

    @override
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        sample_indices = inputs.get("_sample_index", None)
        if sample_indices is None:
            return super().compute_loss(model, inputs, return_outputs, **kwargs)

        sample_indices = torch.tensor(sample_indices, device=model.device)
        total = len(self.train_dataset) if self.train_dataset is not None else 0
        evasion_n = min(self.evasion_size, total)

        evasion_free_mask = sample_indices >= evasion_n
        evasion_mask = sample_indices < evasion_n

        def _slice(d: Dict[str, torch.Tensor], mask: torch.Tensor) -> Dict[str, torch.Tensor]:
            out = {}
            for key, value in d.items():
                if key == "_sample_index":
                    continue
                if isinstance(value, torch.Tensor) and value.size(0) == mask.size(0):
                    out[key] = value[mask]
            return out

        evasion_free_inputs = _slice(inputs, evasion_free_mask)
        evasion_inputs = _slice(inputs, evasion_mask)

        self._step_counter += 1
        evasion_indices = sample_indices[evasion_mask].tolist()

        evasion_free_loss = (
            self.unlearn_loss_fn.evasion_free_loss_func(
                model,
                input_ids=evasion_free_inputs.get("input_ids"),
                attention_mask=evasion_free_inputs.get("attention_mask"),
                labels=evasion_free_inputs.get("labels"),
            )
            if evasion_free_mask.any()
            else torch.tensor(0.0, device=model.device)
        )

        if evasion_mask.any():
            evasion_inputs_modified = self._mask_random_segment(evasion_inputs, sample_indices=evasion_indices)
            evasion_loss = self.unlearn_loss_fn.evasion_loss_func(
                model,
                input_ids=evasion_inputs_modified.get("input_ids"),
                attention_mask=evasion_inputs_modified.get("attention_mask"),
                labels=evasion_inputs_modified.get("labels"),
            )
        else:
            evasion_loss = torch.tensor(0.0, device=model.device)

        total_loss = evasion_loss + self.unlearn_loss_fn.evasion_free_weight * evasion_free_loss

        if self._verbose_loss:
            print(
                f"[step] evasion_free={evasion_free_loss.item():.4f} "
                f"(w={self.unlearn_loss_fn.evasion_free_weight}) "
                f"evasion={evasion_loss.item():.4f} total={total_loss.item():.4f}"
            )

        if return_outputs:
            outputs = model(**inputs)
            return total_loss, outputs
        return total_loss

    def _mask_random_segment(self, inputs, sample_indices=None):
        """Keep labels on one random segment per sample; mask the rest to -100."""
        segment_ratio = self.segment_ratio
        inputs_modified = copy.deepcopy(inputs)
        labels = inputs_modified["labels"]
        batch_size = labels.size(0)
        seq_len = labels.size(1)

        for i in range(batch_size):
            valid_positions = torch.where(labels[i] != -100)[0]
            if len(valid_positions) == 0:
                continue

            valid_len = len(valid_positions)
            segment_len = max(1, int(valid_len * segment_ratio))
            is_fixed_segment = False

            if self.fix_segment_per_sample and sample_indices is not None:
                sample_idx = sample_indices[i]
                segment_key = (sample_idx, valid_len)
                if segment_key in self._fixed_segments:
                    segment_start, segment_end = self._fixed_segments[segment_key]
                    segment_start = max(segment_start, valid_positions[0].item())
                    segment_end = min(segment_end, valid_positions[-1].item() + 1)
                    is_fixed_segment = True
                else:
                    max_start = valid_len - segment_len
                    start_idx = 0 if max_start <= 0 else torch.randint(0, max_start + 1, (1,)).item()
                    segment_start = valid_positions[start_idx].item()
                    segment_end = valid_positions[min(start_idx + segment_len - 1, valid_len - 1)].item() + 1
                    self._fixed_segments[segment_key] = (segment_start, segment_end)
            else:
                max_start = valid_len - segment_len
                start_idx = 0 if max_start <= 0 else torch.randint(0, max_start + 1, (1,)).item()
                segment_start = valid_positions[start_idx].item()
                segment_end = valid_positions[min(start_idx + segment_len - 1, valid_len - 1)].item() + 1

            if self._suffix_token_ids and len(self._suffix_token_ids) > 0:
                suffix_len = len(self._suffix_token_ids)
                if segment_end + suffix_len <= seq_len:
                    suffix_end = segment_end + suffix_len
                    input_ids = inputs_modified["input_ids"][i]
                    suffix_tokens = torch.tensor(
                        self._suffix_token_ids, device=input_ids.device, dtype=input_ids.dtype
                    )
                    input_ids[segment_end:suffix_end] = suffix_tokens
                    inputs_modified["input_ids"][i] = input_ids
                    new_labels = torch.full_like(labels[i], -100)
                    new_labels[segment_start:segment_end] = labels[i][segment_start:segment_end]
                    inputs_modified["labels"][i] = new_labels
                    actual_segment_end = suffix_end
                else:
                    new_labels = torch.full_like(labels[i], -100)
                    new_labels[segment_start:segment_end] = labels[i][segment_start:segment_end]
                    inputs_modified["labels"][i] = new_labels
                    actual_segment_end = segment_end
            else:
                new_labels = torch.full_like(labels[i], -100)
                new_labels[segment_start:segment_end] = labels[i][segment_start:segment_end]
                inputs_modified["labels"][i] = new_labels
                actual_segment_end = segment_end

            if self._verbose_loss and i == 0:
                mode_str = "fixed" if is_fixed_segment else "random"
                print(
                    f"  segment[{mode_str}] sample={i} "
                    f"valid_len={valid_len} span={segment_start}:{segment_end} "
                    f"end_with_suffix={actual_segment_end}"
                )

        return inputs_modified

    @override
    def get_train_dataloader(self):
        if self.cfg.get("use_custom_sampler", False):
            batch_size = self.args.per_device_train_batch_size
            sampler = EvasionSampler(
                dataset=self.train_dataset,
                batch_size=batch_size,
                evasion_size=self.evasion_size,
                evasion_free_ratio_each_batch=self.cfg["evasion_free_ratio_each_batch"],
                seed=42,
            )
            return torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                sampler=sampler,
                drop_last=self.args.dataloader_drop_last,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        return super().get_train_dataloader()
