"""Speculative Unlearning Decoding (SUD) — explicit-π variant.

`SUDModelForCausalLM` is an HF-compatible wrapper that combines a target
causal LM `p` and a draft causal LM `q` at decode time:

    π(x) = (1/Z) · p(x)^(1-α) · q(x)^α
    Z    = Σ_x  p(x)^(1-α) · q(x)^α

α=0 → samples from p (target). α=1 → samples from q (draft). All math is done
in log-space for numerical stability.

The wrapper exposes the subset of the HF CausalLM API that the project's
evaluators rely on:
  * forward(input_ids, attention_mask, ...).logits  — returns log π broadcast
    over the (B, T, V) shape so downstream `log_softmax(logits)` is a no-op
    (since exp(log π) already sums to 1).
  * generate(input_ids, attention_mask, max_new_tokens, do_sample, ...) —
    autoregressive sampling/argmax over π. No KV cache yet (re-runs full
    forward each step), so this is a research-prototype implementation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from model.sud_debug import SUDDebugger

logger = logging.getLogger(__name__)

# Hardcoded debug toggle: when True, generate() writes a per-step top-k
# table of (π, p, q) to a uniquely-named file in DEBUG_DIR.
DEBUG = True
DEBUG_DIR = os.path.join("debug", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
DEBUG_TOP_K = 10

# ---------------------------------------------------------------------------
# Pure functions — testable without loading any model.
# ---------------------------------------------------------------------------


def blend_log_pi(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    alpha: float,
    normalize: bool = False
) -> torch.Tensor:
    """Compute log π over the vocabulary axis (last dim) in log-space.

    Works on any leading shape: (V,), (B, V), or (B, T, V).
    """
    log_p = F.log_softmax(target_logits, dim=-1)
    log_q = F.log_softmax(draft_logits, dim=-1)
    log_u = (1.0 - alpha) * log_p + alpha * log_q
    if normalize:
        return log_u - torch.logsumexp(log_u, dim=-1, keepdim=True)
    return log_u

def assert_shared_tokenizer(tok_target, tok_draft) -> None:
    """Verify two tokenizers map identically over their full vocabularies."""
    if tok_target.get_vocab() != tok_draft.get_vocab():
        raise AssertionError(
            "Target and draft tokenizers do not share a vocabulary. "
            "SUD's geometric blend is only meaningful when token id ↔ string "
            "mappings are identical across both models."
        )


# ---------------------------------------------------------------------------
# HF-compatible wrapper.
# ---------------------------------------------------------------------------


@dataclass
class _SUDOutput:
    """Minimal HF-style output: only what evaluators access (.logits)."""

    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None


class SUDModelForCausalLM(nn.Module):
    """Geometric-blend wrapper around two causal LMs.

    The blend is applied at every position in `forward` and at every decoded
    step in `generate`. Evaluators that consume `.logits` get log π directly:
    since log π is already a normalized log-probability, downstream
    `log_softmax` is the identity (up to numerical noise).
    """

    def __init__(
        self,
        target: nn.Module,
        draft: nn.Module,
        alpha: float,
    ):
        super().__init__()
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if target.config.vocab_size != draft.config.vocab_size:
            raise AssertionError(
                "Target and draft must share vocabulary size: "
                f"target={target.config.vocab_size}, draft={draft.config.vocab_size}"
            )
        self.target = target
        self.draft = draft
        self.alpha = float(alpha)
        self._tokenizer = None  # set by from_pretrained when verify_tokenizer=True

        # Mirror the HF model surface evaluators rely on.
        self.config = target.config
        self.generation_config = getattr(target, "generation_config", None)

    # -- HF lifecycle / property surface -------------------------------------

    @property
    def device(self) -> torch.device:
        return next(self.target.parameters()).device

    def eval(self):
        self.target.eval()
        self.draft.eval()
        return self

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        draft_model_name_or_path: str = None,
        alpha: float = 0.5,
        verify_tokenizer: bool = True,
        **kwargs,
    ) -> "SUDModelForCausalLM":
        """Load the target and draft models from disk/HF hub and wrap them.

        `pretrained_model_name_or_path` is the target. Extra arguments
        (`torch_dtype`, `attn_implementation`, `device_map`, `cache_dir`,
        `quantization_config`, ...) are forwarded to *both* underlying
        `AutoModelForCausalLM.from_pretrained` calls.
        """
        if draft_model_name_or_path is None:
            raise ValueError(
                "SUDModelForCausalLM requires `draft_model_name_or_path` in model_args."
            )

        tok = None
        if verify_tokenizer:
            cache_dir = kwargs.get("cache_dir", None)
            tok_target = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, cache_dir=cache_dir
            )
            tok_draft = AutoTokenizer.from_pretrained(
                draft_model_name_or_path, cache_dir=cache_dir
            )
            assert_shared_tokenizer(tok_target, tok_draft)
            tok = tok_target

        target = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        draft = AutoModelForCausalLM.from_pretrained(
            draft_model_name_or_path, **kwargs
        )
        model = cls(target=target, draft=draft, alpha=alpha)
        model._tokenizer = tok
        return model

    # -- Forward (used by likelihood / loss-based metrics) -------------------

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ) -> _SUDOutput:
        # Strip args that don't apply to a non-trainable wrapper.
        kwargs.pop("position_ids", None)
        kwargs.pop("past_key_values", None)
        kwargs.pop("use_cache", None)

        with torch.no_grad():
            t_out = self.target(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )
            d_out = self.draft(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )

        # Promote to fp32 for the log-space blend; cast back to whatever the
        # downstream code expects via the natural torch promotion.
        log_pi = blend_log_pi(t_out.logits.float(), d_out.logits.float(), self.alpha)
        return _SUDOutput(logits=log_pi)

    # -- Generate (used by ROUGE / generation metrics) -----------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 200,
        do_sample: bool = True,
        temperature: Optional[float] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        stopping_criteria=None,
        **kwargs,
    ) -> torch.Tensor:
        """Autoregressive decoding from π. Greedy when do_sample=True.

        Note: this is the prototype loop — re-runs the full forward pass each
        step on both models. A production version should reuse past_key_values
        on each submodel.
        """
        # Silently ignore extra HF kwargs we don't implement (top_k, top_p,
        # num_beams, etc.). For research use, greedy + multinomial is enough.

        # For Now forcing do_sample and temperature
        do_sample = True
        device = input_ids.device
        
        assert attention_mask is not None, (
        "SUDModelForCausalLM.generate requires attention_mask to be set."
        )
        if eos_token_id is None:
            if self.generation_config is not None:
                eos_token_id = getattr(self.generation_config, "eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = getattr(self.config, "eos_token_id", None)

        # --- pad_token_id resolution: caller > generation_config > config > eos fallback ---
        if pad_token_id is None:
            if self.generation_config is not None:
                pad_token_id = getattr(self.generation_config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.config, "pad_token_id", None)
        if pad_token_id is None:
            # HF's fallback: use eos as pad and warn
            if eos_token_id is not None:
                pad_id_for_warning = (
                    eos_token_id if isinstance(eos_token_id, int)
                    else eos_token_id[0]
                )
                logger.warning(
                    f"Setting `pad_token_id` to `eos_token_id`:{pad_id_for_warning} "
                    f"for open-end generation."
                )
                pad_token_id = pad_id_for_warning
            else:
                raise ValueError(
                    "No pad_token_id or eos_token_id found. Pass one explicitly."
                )

        # Normalize EOS to a list (some models have multiple stop ids)
        if isinstance(eos_token_id, int):
            eos_ids = [eos_token_id]
        elif eos_token_id is None:
            eos_ids = []
        else:
            eos_ids = list(eos_token_id)

        # --- green logging for generate parameters ---
        GREEN = "\033[92m"
        RESET = "\033[0m"

        logger.info(
            "%sGenerate parameters | "
            "max_new_tokens=%s | "
            "do_sample=%s | "
            "temperature=%s | "
            "pad_token_id=%s | "
            "eos_token_id=%s | "
            "eos_ids=%s | "
            "input_shape=%s | "
            "attention_mask_shape=%s%s",
            GREEN,
            max_new_tokens,
            do_sample,
            temperature,
            pad_token_id,
            eos_token_id,
            eos_ids,
            tuple(input_ids.shape),
            tuple(attention_mask.shape) if attention_mask is not None else None,
            RESET,
        )
        bsz = input_ids.size(0)
        finished = torch.zeros(bsz, dtype=torch.bool, device=device)
        generated = input_ids
        attn = attention_mask

        debugger = None
        if DEBUG:
            debugger = SUDDebugger(
                alpha=self.alpha,
                tokenizer=self._tokenizer,
                output_dir=DEBUG_DIR,
                top_k=DEBUG_TOP_K,
            )
            debugger.start(input_ids)
        try:
            for step in range(max_new_tokens):
                t_logits = self.target(
                    input_ids=generated, attention_mask=attn
                ).logits[:, -1, :]
                d_logits = self.draft(
                    input_ids=generated, attention_mask=attn
                ).logits[:, -1, :]

                log_pi = blend_log_pi(t_logits.float(), d_logits.float(), self.alpha, normalize=True)

                if do_sample:
                    if temperature is not None and temperature > 0:
                        log_pi = log_pi / temperature
                        log_pi = log_pi - torch.logsumexp(log_pi, dim=-1, keepdim=True)
                    next_id = torch.multinomial(log_pi.exp(), num_samples=1).squeeze(-1)
                else:
                    next_id = log_pi.argmax(dim=-1)

                # Log AFTER sampling, BEFORE pad override; skip if seq 0 already done.
                if debugger is not None and not bool(finished[0].item()):
                    debugger.log_step(step, t_logits, d_logits, log_pi, next_id)
                # Sequences that already finished keep emitting pad.
                next_id = torch.where(
                    finished, torch.full_like(next_id, pad_token_id), next_id
                )

                generated = torch.cat([generated, next_id.unsqueeze(-1)], dim=1)          
                attn = torch.cat(
                    [attn, (~finished).long().unsqueeze(-1).to(attn.dtype)], dim=1
                )

                if eos_ids:
                    for e in eos_ids:
                        finished = finished | (next_id == e)

                if stopping_criteria is not None:
                    # HF StoppingCriteriaList(input_ids, scores) → bool tensor or bool.
                    stop_signal = stopping_criteria(generated, None)
                    if isinstance(stop_signal, torch.Tensor):
                        if bool(stop_signal.all()):
                            break
                    elif bool(stop_signal):
                        break

                if bool(finished.all()):
                    break

        finally:
            if debugger is not None:
                debugger.finish()

        return generated
    


 