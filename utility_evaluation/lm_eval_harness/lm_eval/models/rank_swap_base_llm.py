"""
Base-only Rank-Swap model implementation for lm-evaluation-harness.
"""
from __future__ import annotations

import logging

import torch
import transformers

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


eval_logger = logging.getLogger(__name__)


class FakeSwapper:
    def __init__(self, base_model, fake_prob: float = 0.3, fake_k: int = 5, fake_mode: str = "top2") -> None:
        if fake_mode not in {"top2", "synonym", "semantic_topk"}:
            raise ValueError(f"Unsupported fake_mode: {fake_mode}")
        self.base_model = base_model
        self.fake_prob = float(fake_prob)
        self.fake_k = int(fake_k)
        self.fake_mode = fake_mode
        self._emb_norm = None

    def _ensure_embeddings(self) -> torch.Tensor:
        if self._emb_norm is None:
            emb = self.base_model.get_input_embeddings().weight
            self._emb_norm = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return self._emb_norm

    def _get_synonyms(self, token_id: int) -> list[int]:
        if self.fake_k <= 0:
            return []
        emb = self._ensure_embeddings()
        query = emb[token_id]
        scores = torch.matmul(emb, query)
        topk = torch.topk(scores, k=min(self.fake_k + 1, emb.shape[0]))
        ids = topk.indices.tolist()
        return [x for x in ids if x != token_id][: self.fake_k]

    def maybe_swap(self, row: torch.Tensor) -> bool:
        if self.fake_prob <= 0.0:
            return False
        if torch.rand(1, device=row.device).item() > self.fake_prob:
            return False

        if self.fake_mode == "top2":
            top2 = torch.topk(row, k=2, dim=-1)
            top1_id = top2.indices[0].item()
            top2_id = top2.indices[1].item()
            row[top1_id], row[top2_id] = row[top2_id].clone(), row[top1_id].clone()
            return True

        emb = self._ensure_embeddings()
        top1_id = torch.argmax(row, dim=-1).item()
        query = emb[top1_id]

        if self.fake_mode == "synonym":
            syn_list = self._get_synonyms(top1_id)
            if not syn_list:
                return False
            syn_id = syn_list[torch.randint(0, len(syn_list), (1,), device=row.device).item()]
            row[top1_id], row[syn_id] = row[syn_id].clone(), row[top1_id].clone()
            return True

        if self.fake_k <= 0:
            return False
        candidate_topk = torch.topk(row, k=min(self.fake_k + 1, row.shape[-1]), dim=-1)
        candidate_ids = [x for x in candidate_topk.indices.tolist() if x != top1_id][: self.fake_k]
        if not candidate_ids:
            return False
        sims = torch.matmul(emb[candidate_ids], query)
        best_id = candidate_ids[torch.argmax(sims).item()]
        row[top1_id], row[best_id] = row[best_id].clone(), row[top1_id].clone()
        return True

    def apply_to_batch_last_logits(self, logits: torch.Tensor) -> torch.Tensor:
        last_logits = logits[:, -1, :]
        for b in range(last_logits.shape[0]):
            self.maybe_swap(last_logits[b])
        logits[:, -1, :] = last_logits
        return logits


@register_model("rank_swap_base", "rankswap_base")
class RankSwapBaseWrapper(HFLM):
    def __init__(
        self,
        pretrained: str,
        fake_prob: float = 0.3,
        fake_k: int = 5,
        fake_mode: str = "top2",
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

        self.pretrained = pretrained
        self.fake_prob = float(fake_prob)
        self.fake_k = int(fake_k)
        self.fake_mode = fake_mode

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

        eval_logger.info("🚀 Initializing Base-only Rank-Swap model")
        eval_logger.info(f"   Base Model: {pretrained}")
        eval_logger.info(f"   Fake Mode: {fake_mode}")
        eval_logger.info(f"   Fake Prob: {fake_prob}")
        eval_logger.info(f"   Device: {device}")

        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained,
            torch_dtype=self._dtype,
            trust_remote_code=trust_remote_code,
        ).to(self._device)
        self._model.eval()

        tokenizer_path = tokenizer if tokenizer is not None else pretrained
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast_tokenizer,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._config = self._model.config
        self._batch_size = int(batch_size) if isinstance(batch_size, str) else batch_size
        self._max_batch_size = max_batch_size
        self._max_length = max_length if max_length is not None else self._DEFAULT_MAX_LENGTH
        self.truncation = truncation
        self.backend = "causal"
        self.logits_cache = False
        self.vocab_size = self.tokenizer.vocab_size
        self.add_bos_token = False
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
        self.delta = None
        self.peft = None
        self.chat_template_args = None
        self._max_gen_toks = 256
        self._rank = 0
        self._world_size = 1
        self.fake_swapper = FakeSwapper(self._model, self.fake_prob, self.fake_k, self.fake_mode)

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

    def tok_encode(self, string: str, **kwargs):
        return self.tokenizer.encode(string, add_special_tokens=self.add_bos_token, **kwargs)

    def tok_decode(self, tokens, skip_special_tokens: bool = True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _model_call(self, inps, attn_mask=None, labels=None):
        with torch.no_grad():
            if attn_mask is None:
                attn_mask = torch.ones_like(inps)
            outputs = self._model(input_ids=inps, attention_mask=attn_mask, return_dict=True)
            logits = self.fake_swapper.apply_to_batch_last_logits(outputs.logits)
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                return {"logits": logits, "loss": loss}
            return logits

    def _sample_from_logits(self, logits, temperature, top_p, top_k, do_sample):
        greedy = temperature <= 0 or (do_sample is False)
        if greedy:
            return torch.argmax(logits, dim=-1, keepdim=True)
        filtered = logits
        if temperature != 1.0:
            filtered = filtered / temperature
        if isinstance(top_k, (int, float)) and int(top_k) > 0:
            k = min(int(top_k), filtered.shape[-1])
            kth = torch.topk(filtered, k, dim=-1).values[..., -1, None]
            filtered = torch.where(filtered < kth, torch.full_like(filtered, float("-inf")), filtered)
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
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                filtered = torch.where(indices_to_remove, torch.full_like(filtered, float("-inf")), filtered)
        all_inf = torch.isinf(filtered).all(dim=-1)
        if all_inf.any():
            filtered = torch.where(all_inf[:, None], logits, filtered)
        probs = torch.softmax(filtered, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        row_sums = probs.sum(dim=-1, keepdim=True)
        zero_rows = row_sums.squeeze(-1) <= 0
        if zero_rows.any():
            greedy_tok = torch.argmax(logits, dim=-1, keepdim=True)
            sample_tok = torch.multinomial(probs, num_samples=1)
            return torch.where(zero_rows[:, None], greedy_tok, sample_tok)
        probs = probs / row_sums
        return torch.multinomial(probs, num_samples=1)

    def _model_generate(self, context, attention_mask=None, stop=None, max_length=None, **generation_kwargs):
        input_ids = context
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if max_length is None:
            max_length = input_ids.shape[1] + int(generation_kwargs.get("max_new_tokens", 256))
        max_new_tokens = max(0, max_length - input_ids.shape[1])
        if max_new_tokens == 0:
            return input_ids
        bs = input_ids.shape[0]
        temperature = generation_kwargs.get("temperature", 1.0)
        if temperature is None:
            temperature = 1.0
        try:
            temperature = float(temperature)
        except Exception:
            temperature = 1.0
        top_p = generation_kwargs.get("top_p", 1.0)
        top_k = generation_kwargs.get("top_k", 0)
        do_sample = generation_kwargs.get("do_sample", None)
        eos_id = int(self.tokenizer.eos_token_id) if self.tokenizer.eos_token_id is not None else None
        finished = torch.zeros((bs,), dtype=torch.bool, device=input_ids.device)

        with torch.inference_mode():
            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
            logits = self.fake_swapper.apply_to_batch_last_logits(outputs.logits)[:, -1, :].float()
            past_key_values = outputs.past_key_values

        for _ in range(max_new_tokens):
            with torch.inference_mode():
                next_token = self._sample_from_logits(logits, temperature, top_p, top_k, do_sample)
                if eos_id is not None:
                    next_token = torch.where(finished[:, None], torch.full_like(next_token, eos_id), next_token)
                    finished = finished | (next_token.squeeze(-1) == eos_id)
                input_ids = torch.cat([input_ids, next_token.to(input_ids.dtype)], dim=-1)
                ones = torch.ones((bs, 1), device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([attention_mask, ones], dim=-1)
                if eos_id is not None and finished.all():
                    break
                outputs = self._model(input_ids=next_token, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True, return_dict=True)
                logits = self.fake_swapper.apply_to_batch_last_logits(outputs.logits)[:, -1, :].float()
                past_key_values = outputs.past_key_values
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
