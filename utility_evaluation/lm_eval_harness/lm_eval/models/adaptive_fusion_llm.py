"""
Adaptive-fusion model implementation for lm-evaluation-harness.

This wrapper mirrors the forward logic used in:
- utility_evaluation/generation/fakebase_sampling_aggregate.py
- utility_evaluation/generation/adaptive_fusion_sampling_aggregate.py

At each position t, we compute:
    alpha_t = alpha_min + (alpha_max - alpha_min) * margin_t / (margin_t + alpha_s)
    fused_logits_t = base_logits_t - alpha_t * assist_logits_t

where margin_t is the gap between the top-1 and top-2 base-model logits.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import transformers
from transformers.modeling_outputs import CausalLMOutputWithPast

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

if TYPE_CHECKING:
    from transformers import PreTrainedModel

eval_logger = logging.getLogger(__name__)


class AdaptiveFusionModel(torch.nn.Module):
    def __init__(
        self,
        basellm: "PreTrainedModel",
        assist_llm: "PreTrainedModel",
        alpha_min: float,
        alpha_max: float,
        alpha_s: float,
    ) -> None:
        super().__init__()
        self.basellm = basellm
        self.assist_llm = assist_llm
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.alpha_s = float(alpha_s)
        self.config = basellm.config
        self.device = basellm.device

    def _compute_alpha(self, base_logits: torch.Tensor) -> torch.Tensor:
        top2_vals = torch.topk(base_logits.float(), k=2, dim=-1).values
        margins = (top2_vals[..., 0] - top2_vals[..., 1]).clamp_min(0.0)
        return self.alpha_min + (self.alpha_max - self.alpha_min) * margins / (margins + self.alpha_s)

    def fuse_logits(self, base_logits: torch.Tensor, assist_logits: torch.Tensor) -> torch.Tensor:
        alpha = self._compute_alpha(base_logits).unsqueeze(-1)
        return base_logits - alpha * assist_logits

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
    ):
        base_outputs = self.basellm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=True,
        )

        assist_outputs = self.assist_llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=True,
        )

        base_logits = base_outputs.logits
        assist_logits = assist_outputs.logits
        fused_logits = self.fuse_logits(base_logits, assist_logits)

        loss = None
        if labels is not None:
            shift_logits = fused_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=fused_logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


@register_model("adaptive_fusion", "logit_laundering")
class AdaptiveFusionWrapper(HFLM):
    def __init__(
        self,
        base_model: str,
        assist_model: str,
        alpha_min: float = 0.17,
        alpha_max: float = 0.56,
        alpha_s: float = 1.0,
        device: str | None = "cuda",
        dtype: str | torch.dtype | None = "float16",
        batch_size: int | str | None = 1,
        max_batch_size: int | None = 64,
        tokenizer: str | None = None,
        truncation: bool | None = False,
        max_length: int | None = None,
        trust_remote_code: bool | None = False,
        use_fast_tokenizer: bool | None = True,
        **kwargs,
    ) -> None:
        from lm_eval.api.model import LM

        LM.__init__(self)

        eval_logger.info("🚀 Initializing Adaptive Fusion model")
        eval_logger.info("   Base Model: %s", base_model)
        eval_logger.info("   Assist Model: %s", assist_model)
        eval_logger.info(
            "   alpha_min=%.4f alpha_max=%.4f alpha_s=%.4f",
            float(alpha_min),
            float(alpha_max),
            float(alpha_s),
        )

        self.base_model_path = base_model
        self.assist_model_path = assist_model
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.alpha_s = float(alpha_s)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        if dtype == "float16" or dtype == torch.float16:
            self._dtype = torch.float16
        elif dtype == "bfloat16" or dtype == torch.bfloat16:
            self._dtype = torch.bfloat16
        elif dtype == "float32" or dtype == torch.float32:
            self._dtype = torch.float32
        else:
            self._dtype = torch.float16

        basellm = transformers.AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=self._dtype,
            trust_remote_code=trust_remote_code,
        ).to(self._device)
        assist_llm = transformers.AutoModelForCausalLM.from_pretrained(
            assist_model,
            torch_dtype=self._dtype,
            trust_remote_code=trust_remote_code,
        ).to(self._device)
        basellm.eval()
        assist_llm.eval()

        self._model = AdaptiveFusionModel(
            basellm=basellm,
            assist_llm=assist_llm,
            alpha_min=self.alpha_min,
            alpha_max=self.alpha_max,
            alpha_s=self.alpha_s,
        ).to(self._device)
        self._model.eval()
        self._config = basellm.config

        tokenizer_path = tokenizer if tokenizer is not None else base_model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast_tokenizer,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._batch_size = int(batch_size) if isinstance(batch_size, str) else batch_size
        self._max_batch_size = max_batch_size
        self._max_length = max_length if max_length is not None else self._DEFAULT_MAX_LENGTH
        self.truncation = truncation

        self.backend = "causal"
        self.logits_cache = False
        self.vocab_size = self.tokenizer.vocab_size
        # Match HFLM's default tokenizer behavior so adaptive runs are comparable
        # to base-model evaluation on tasks like MMLU/ARC.
        self.add_bos_token = None
        self.custom_prefix_token_id = None
        self.revision = "main"
        self.subfolder = ""
        self.trust_remote_code = trust_remote_code
        self.use_fast_tokenizer = use_fast_tokenizer
        self.think_end_token = None
        self.batch_schedule = 1
        self.batch_sizes = {}
        self.batch_size_per_gpu = self._batch_size
        self.softmax_dtype = None
        self.mixed_precision_dtype = None
        self.pretrained = base_model
        self.delta = None
        self.peft = None
        self.chat_template_args = None
        self._max_gen_toks = 256
        self._rank = 0
        self._world_size = 1

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def prefix_token_id(self) -> int:
        if self.custom_prefix_token_id is not None:
            return self.custom_prefix_token_id
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.tokenizer.eos_token_id

    def tok_encode(
        self,
        string: str,
        add_special_tokens: bool | None = None,
        left_truncate_len: int | None = None,
        **kwargs,
    ):
        return HFLM.tok_encode(
            self,
            string,
            add_special_tokens=add_special_tokens,
            left_truncate_len=left_truncate_len,
            **kwargs,
        )

    def tok_decode(self, tokens, skip_special_tokens: bool = True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _model_call(self, inps, attn_mask=None, labels=None):
        with torch.no_grad():
            if attn_mask is None:
                attn_mask = torch.ones_like(inps)
            outputs = self._model(
                input_ids=inps,
                attention_mask=attn_mask,
                labels=labels,
            )
            if labels is not None:
                return {"logits": outputs.logits, "loss": outputs.loss}
            return outputs.logits

    def _model_generate(
        self,
        context: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        stop=None,
        max_length: int | None = None,
        **generation_kwargs,
    ) -> torch.Tensor:
        input_ids = context
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if max_length is None:
            max_length = input_ids.shape[1] + int(generation_kwargs.get("max_new_tokens", 256))

        bs = input_ids.shape[0]
        max_new_tokens = max(0, max_length - input_ids.shape[1])
        temperature = generation_kwargs.get("temperature", 1.0)
        top_p = generation_kwargs.get("top_p", 1.0)
        top_k = generation_kwargs.get("top_k", 0)
        do_sample = generation_kwargs.get("do_sample", None)

        if temperature is None:
            temperature = 1.0
        try:
            temperature = float(temperature)
        except Exception:
            temperature = 1.0

        greedy = temperature <= 0 or (do_sample is False)
        eos_id = int(self.tokenizer.eos_token_id) if self.tokenizer.eos_token_id is not None else None
        finished = torch.zeros((bs,), dtype=torch.bool, device=input_ids.device)

        if max_new_tokens == 0:
            return input_ids

        with torch.inference_mode():
            base_outputs = self._model.basellm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            assist_outputs = self._model.assist_llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            next_token_logits = self._model.fuse_logits(
                base_outputs.logits,
                assist_outputs.logits,
            )[:, -1, :].float()
            base_past_key_values = base_outputs.past_key_values
            assist_past_key_values = assist_outputs.past_key_values

        for _ in range(max_new_tokens):
            with torch.inference_mode():
                if greedy:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                else:
                    if temperature != 1.0:
                        next_token_logits = next_token_logits / temperature

                    filtered = next_token_logits

                    if isinstance(top_k, (int, float)) and int(top_k) > 0:
                        k = min(int(top_k), filtered.shape[-1])
                        kth = torch.topk(filtered, k, dim=-1).values[..., -1, None]
                        filtered = torch.where(
                            filtered < kth,
                            torch.full_like(filtered, float("-inf")),
                            filtered,
                        )

                    if top_p is not None:
                        try:
                            p = float(top_p)
                        except Exception:
                            p = 1.0
                        if p < 1.0:
                            sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
                            sorted_probs = torch.softmax(sorted_logits, dim=-1)
                            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                            sorted_indices_to_remove = cumulative_probs > p
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = False
                            indices_to_remove = sorted_indices_to_remove.scatter(
                                1, sorted_indices, sorted_indices_to_remove
                            )
                            filtered = torch.where(
                                indices_to_remove,
                                torch.full_like(filtered, float("-inf")),
                                filtered,
                            )

                    all_inf = torch.isinf(filtered).all(dim=-1)
                    if all_inf.any():
                        filtered = torch.where(all_inf[:, None], next_token_logits, filtered)

                    probs = torch.softmax(filtered, dim=-1)
                    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                    row_sums = probs.sum(dim=-1, keepdim=True)
                    zero_rows = row_sums.squeeze(-1) <= 0
                    if zero_rows.any():
                        greedy_tok = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                        sample_tok = torch.multinomial(probs, num_samples=1)
                        next_token = torch.where(zero_rows[:, None], greedy_tok, sample_tok)
                    else:
                        probs = probs / row_sums
                        next_token = torch.multinomial(probs, num_samples=1)

                if eos_id is not None:
                    next_token = torch.where(finished[:, None], torch.full_like(next_token, eos_id), next_token)
                    finished = finished | (next_token.squeeze(-1) == eos_id)

                input_ids = torch.cat([input_ids, next_token.to(input_ids.dtype)], dim=-1)
                ones = torch.ones((bs, 1), device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([attention_mask, ones], dim=-1)

                if eos_id is not None and finished.all():
                    break

                base_outputs = self._model.basellm(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=base_past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                assist_outputs = self._model.assist_llm(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=assist_past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                next_token_logits = self._model.fuse_logits(
                    base_outputs.logits,
                    assist_outputs.logits,
                )[:, -1, :].float()
                base_past_key_values = base_outputs.past_key_values
                assist_past_key_values = assist_outputs.past_key_values

        return input_ids

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return 256

    @property
    def model(self):
        return self._model

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size
