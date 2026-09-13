from typing import List, Optional, Union, Tuple
import os
import json

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


class ControlLLM(torch.nn.Module):
    """
    ControlLLM
    ----------
    mode:
        - "standard": logits = base + weight * assist
        - "fake":     runs fake swap on top of standard mode (controlled by fake_mode)

    fake_mode:
        - "top2"            swap top-1 and top-2 (simplest)
        - "synonym"         use embedding KNN synonyms
        - "semantic_topk"   swap with the most embedding-similar token in the top-k logits
    """

    def __init__(
        self,
        basellm: AutoModelForCausalLM,
        assist_llm: AutoModelForCausalLM,
        weight: float,
        top_logit_filter: float = 0.0,
        mode: str = "standard",

        # ---- fake-swap parameters ----
        fake_prob: float = 0.0,
        fake_k: int = 5,
        fake_mode: str = "top2",   # switchable mode
    ) -> None:
        super().__init__()

        assert mode in ["standard", "fake"]
        assert fake_mode in ["top2", "synonym", "semantic_topk"]

        self.basellm = basellm
        self.assist_llm = assist_llm
        self.weight = weight

        self.device = basellm.device
        self.config = basellm.config
        self.top_logit_filter = top_logit_filter
        self.mode = mode

        # fake-swap parameters
        self.fake_prob = fake_prob
        self.fake_k = fake_k
        self.fake_mode = fake_mode

        # embedding cache
        self._emb_norm = None

        # tokenizer (injected externally)
        self.tokenizer = None

    # -----------------------------
    # Loss computation
    # -----------------------------
    def get_loss(self, logits, labels=None, attention_mask=None, reduction='mean'):
        if labels is None:
            return None

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = CrossEntropyLoss(reduction=reduction)
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1).to(shift_logits.device)

        return loss_fct(shift_logits, shift_labels)

    # -----------------------------
    # Embedding-based KNN synonyms
    # -----------------------------
    def _get_synonyms(self, token_id: int, k=None):
        if k is None:
            k = self.fake_k

        if self._emb_norm is None:
            emb = self.basellm.get_input_embeddings().weight  # [vocab_size, d]
            self._emb_norm = emb / emb.norm(dim=-1, keepdim=True)

        emb = self._emb_norm
        query = emb[token_id]

        # Cosine similarity
        scores = torch.matmul(emb, query)

        # top-k+1 (includes self)
        topk = torch.topk(scores, k=k + 1)
        ids = topk.indices.tolist()

        # Remove self
        synonyms = [x for x in ids if x != token_id][:k]
        return synonyms

    # -----------------------------
    # Fake swap (three modes); in-place, called per sample in batch loop
    # -----------------------------
    def _apply_fake_swap_single(self, row: torch.Tensor, emb: Optional[torch.Tensor] = None) -> None:
"""
        Apply fake swap to a single logit row [vocab] in-place.
        emb is required for synonym / semantic_topk modes; may be None for top2.
        """
        # ========== Mode 1: top1 <-> top2 ==========
        if self.fake_mode == "top2":
            top2 = torch.topk(row, k=2, dim=-1)
            top1_id = top2.indices[0].item()
            top2_id = top2.indices[1].item()
            row[top1_id], row[top2_id] = row[top2_id].clone(), row[top1_id].clone()
            return

        assert emb is not None
        top1_id = torch.argmax(row, dim=-1).item()
        q = emb[top1_id]

        # ========== Mode 2: KNN synonym (random) ==========
        if self.fake_mode == "synonym":
            syn_list = self._get_synonyms(top1_id, self.fake_k)
            if len(syn_list) == 0:
                return
            syn_id = syn_list[torch.randint(0, len(syn_list), (1,), device=row.device).item()]
            row[top1_id], row[syn_id] = row[syn_id].clone(), row[top1_id].clone()
            return

        # ========== Mode 3: semantic_topk ==========
        if self.fake_mode == "semantic_topk":
            logit_topk = torch.topk(row, k=min(self.fake_k + 1, row.shape[-1]), dim=-1)
            candidate_ids = logit_topk.indices.tolist()
            candidate_ids = [x for x in candidate_ids if x != top1_id][: self.fake_k]
            if len(candidate_ids) == 0:
                return
            cand_embs = emb[candidate_ids]
            sims = torch.matmul(cand_embs, q)
            best_idx = torch.argmax(sims).item()
            best_id = candidate_ids[best_idx]
            row[top1_id], row[best_id] = row[best_id].clone(), row[top1_id].clone()
            return

    def _apply_fake_swap(self, logits: torch.Tensor) -> torch.Tensor:
"""
        fake_mode:
            - "top2"
            - "synonym"
            - "semantic_topk"
        Supports batching: when last_logits has shape [B, vocab],
        each sample is independently swapped with probability fake_prob.
        """
        if self.mode != "fake" or self.fake_prob <= 0.0:
            return logits

        last_logits = logits[:, -1, :]  # [B, vocab]
        batch_size = last_logits.shape[0]

        # synonym / semantic_topk require embeddings — initialize if needed
        emb = None
        if self.fake_mode in ("synonym", "semantic_topk"):
            if self._emb_norm is None:
                e = self.basellm.get_input_embeddings().weight
                self._emb_norm = e / e.norm(dim=-1, keepdim=True)
            emb = self._emb_norm

        for b in range(batch_size):
            if torch.rand(1, device=last_logits.device).item() > self.fake_prob:
                continue
            self._apply_fake_swap_single(last_logits[b], emb)

        logits[:, -1, :] = last_logits
        return logits

    # -----------------------------
    # Forward
    # -----------------------------
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
        # Base model forward
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

        # Fuse logits
        baselogits = base_outputs.logits
        assist_logits = assist_outputs.logits
        logits = baselogits + self.weight * assist_logits

        # Apply fake swap
        logits = self._apply_fake_swap(logits)

        loss = self.get_loss(logits, labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    # -----------------------------
    # Save
    # -----------------------------
    def save_pretrained(self, save_directory: str, **kwargs):
        os.makedirs(save_directory, exist_ok=True)

        basellm_dir = os.path.join(save_directory, "basellm")
        assist_llm_dir = os.path.join(save_directory, "assist_llm")

        self.basellm.save_pretrained(basellm_dir)
        self.assist_llm.save_pretrained(assist_llm_dir)

        config = {
            "weight": self.weight,
            "top_logit_filter": self.top_logit_filter,
            "mode": self.mode,
            "fake_prob": self.fake_prob,
            "fake_k": self.fake_k,
            "fake_mode": self.fake_mode,
            "model_type": "ControlLLM",
        }
        with open(os.path.join(save_directory, "control_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

print(f"✅ ControlLLM saved to {save_directory}")

    # -----------------------------
    # Load
    # -----------------------------
    @classmethod
    def from_pretrained(cls, save_directory: str, **kwargs):
        with open(os.path.join(save_directory, "control_config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)

        basellm = AutoModelForCausalLM.from_pretrained(
            os.path.join(save_directory, "basellm"))
        assist_llm = AutoModelForCausalLM.from_pretrained(
            os.path.join(save_directory, "assist_llm"))

        model = cls(
            basellm=basellm,
            assist_llm=assist_llm,
            weight=config["weight"],
            top_logit_filter=config["top_logit_filter"],
            mode=config["mode"],
            fake_prob=config.get("fake_prob", 0.0),
            fake_k=config.get("fake_k", 5),
            fake_mode=config.get("fake_mode", "top2"),
        )

        model.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(save_directory, "basellm"))

print(f"✅ ControlLLM loaded from {save_directory} (fake_mode={model.fake_mode})")
        return model