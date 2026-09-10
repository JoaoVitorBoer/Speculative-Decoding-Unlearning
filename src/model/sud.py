"""Speculative Unlearning Decoding (SUD) — draft-verify speculative sampling.

`SUDModelForCausalLM` is an HF-compatible wrapper that couples a frozen target
causal LM `p` (which has memorized the forget set) and a forgetting-aware
draft causal LM `q`, and emits tokens distributed as the geometric mixture

    π(v) = (1/Z) · p(v)^(1-α) · q(v)^α,    Z = Σ_w p(w)^(1-α) · q(w)^α

α=0 → samples from p (target). α=1 → samples from q (draft). All math is done
in log-space for numerical stability, where the 1/Z factor is a `logsumexp`
subtraction over the vocabulary axis.

Decoding uses a multi-token speculative loop: the draft proposes up to `k_sud`
tokens by sampling from q, the target verifies all of them with one parallel
forward pass, each proposal is accepted with prob min(1, π/q), and on the
first rejection one replacement token is sampled from the residual
max(0, π - q) and the round ends. No bonus token is ever emitted — a vanilla
speculative-decoding bonus token would follow p, not π, and escape forget
suppression. A round therefore commits between 1 and k_sud tokens, and k_sud
is a pure throughput knob: it never changes the sampled distribution.

The wrapper exposes the subset of the HF CausalLM API that the project's
evaluators rely on:
  * forward(input_ids, attention_mask, ...).logits  — returns log π itself,
    normalized over the vocabulary axis, with shape (B, T, V). Evaluators
    apply their own `log_softmax` / `CrossEntropyLoss`, which is idempotent
    on an already-normalized tensor, so they see log π either way.
  * generate(input_ids, attention_mask, max_new_tokens, do_sample, ...) —
    draft-verify sampling from π. No KV cache yet (re-runs full forwards
    each round), so this is a research-prototype implementation.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList
import logging
from model.sud_debug import SUDDebugger

logger = logging.getLogger(__name__)

# Hardcoded debug toggle: when True, generate() writes a per-step top-k
# table of (π, p, q) to a uniquely-named file in DEBUG_DIR.
DEBUG = False
DEBUG_DIR = os.path.join("debug", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
DEBUG_TOP_K = 10

# ---------------------------------------------------------------------------
# Pure functions — testable without loading any model.
# ---------------------------------------------------------------------------


def blend_log_pi(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Compute normalized log π over the vocabulary axis (last dim).

        log π = (1-α)·log p + α·log q - log Z
        log Z = logsumexp((1-α)·log p + α·log q)

    The result is always a proper log-distribution (exp sums to 1). This is
    load-bearing in `_speculative_decode_row`: the acceptance ratio π/q and
    the residual max(0, π - q) compare mass across two *different*
    distributions, so an unnormalized π would offset every acceptance
    decision by log Z and silently change the sampled distribution. For
    `forward` it is semantic hygiene — evaluators re-normalize regardless,
    but `.logits` should mean log π rather than log π + log Z.

    Both inputs are log_softmax'd here, so passing an already-normalized
    log-prob tensor (as the draft rows are) is a safe no-op.
    Works on any leading shape: (V,), (B, V), or (B, T, V).
    """
    log_p = F.log_softmax(target_logits, dim=-1)
    log_q = F.log_softmax(draft_logits, dim=-1)
    log_u = (1.0 - alpha) * log_p + alpha * log_q
    return log_u - torch.logsumexp(log_u, dim=-1, keepdim=True)


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

    The blend is applied at every position in `forward` and enforced through
    draft-verify acceptance at every decoded token in `generate`. Evaluators
    that consume `.logits` get normalized log π directly, so their downstream
    `log_softmax` / `CrossEntropyLoss` is idempotent.
    """

    def __init__(
        self,
        target: nn.Module,
        draft: nn.Module,
        alpha: float,
        k_sud: int = 4,
    ):
        super().__init__()
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if int(k_sud) < 1:
            raise ValueError(f"k_sud must be >= 1, got {k_sud}")
        if target.config.vocab_size != draft.config.vocab_size:
            raise AssertionError(
                "Target and draft must share vocabulary size: "
                f"target={target.config.vocab_size}, draft={draft.config.vocab_size}"
            )
        self.target = target
        self.draft = draft
        self.alpha = float(alpha)
        self.k_sud = int(k_sud)
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
        k_sud: int = 4,
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
        model = cls(target=target, draft=draft, alpha=alpha, k_sud=k_sud)
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
        """Blend both models at every position; `.logits` is normalized log π.

        Because SUD commits every decoded token exactly from π, the
        teacher-forced joint ∏ₜ π(yₜ|y_<ₜ) that likelihood metrics compute is
        exactly the probability that `generate` reproduces y verbatim — the
        forward and generate paths describe the same distribution (at
        temperature 1, the only setting the evaluators use).

        `labels` is accepted and ignored: this wrapper is inference-only, so
        no loss is computed and `.loss` stays None. Every active evaluator
        derives its own loss from `.logits` (see evals/metrics/utils.py).
        """
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

        # Promote to fp32 before the log-space blend: the submodels are
        # typically bf16, and the returned logits stay fp32 regardless.
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
        """Speculative draft-verify decoding from π.

        Batch rows are decoded independently — acceptance lengths diverge
        across rows — then right-padded to a rectangle with pad_token_id.
        Note: this is the prototype loop — re-runs full forward passes each
        round on both models. A production version should reuse
        past_key_values on each submodel.
        """
        # Silently ignore extra HF kwargs we don't implement (top_k, top_p,
        # num_beams, use_cache, etc.). The draft proposal must stay an
        # unfiltered sample from q for the acceptance test to be exact.

        # Sampling is required by the acceptance math; greedy is unsupported.
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

        temp = (
            float(temperature)
            if temperature is not None and temperature > 0
            else 1.0
        )

        bsz = input_ids.size(0)

        debugger = None
        if DEBUG:
            debugger = SUDDebugger(
                alpha=self.alpha,
                tokenizer=self._tokenizer,
                output_dir=DEBUG_DIR,
                top_k=DEBUG_TOP_K,
                k_sud=self.k_sud,
            )
            debugger.start(input_ids)
        try:
            new_rows = []
            for b in range(bsz):
                row_criteria = None
                if stopping_criteria is not None:
                    # Criteria like MultiTokenEOSCriteria track done-ness per
                    # batch row; each independently decoded row gets its own
                    # copy with single-row state.
                    row_criteria = StoppingCriteriaList(
                        copy.copy(c) for c in stopping_criteria
                    )
                    for c in row_criteria:
                        if hasattr(c, "done_tracker"):
                            c.done_tracker = [False]
                new_rows.append(
                    self._speculative_decode_row(
                        input_ids[b : b + 1],
                        attention_mask[b : b + 1],
                        max_new_tokens=max_new_tokens,
                        temp=temp,
                        eos_ids=eos_ids,
                        stopping_criteria=row_criteria,
                        debugger=debugger if b == 0 else None,
                    )
                )
        finally:
            if debugger is not None:
                debugger.finish()

        max_len = max(len(r) for r in new_rows)
        if max_len == 0:
            return input_ids
        padded = torch.full(
            (bsz, max_len), pad_token_id, dtype=input_ids.dtype, device=device
        )
        for b, row in enumerate(new_rows):
            if row:
                padded[b, : len(row)] = torch.tensor(
                    row, dtype=input_ids.dtype, device=device
                )
        return torch.cat([input_ids, padded], dim=1)

    def _speculative_decode_row(
        self,
        row_ids: torch.Tensor,
        row_attn: torch.Tensor,
        max_new_tokens: int,
        temp: float,
        eos_ids: list,
        stopping_criteria,
        debugger,
    ) -> list:
        """Draft-verify decoding of one (1, L) sequence; returns committed ids.

        Per round: sample up to k_sud tokens from q one at a time, verify all
        of them with a single target forward, accept each in order with prob
        min(1, π/q), and on the first rejection sample one replacement from
        the residual max(0, π - q) and end the round. An all-accepted round
        commits exactly its drafted tokens — no bonus token, which would
        follow p rather than π and escape forget suppression.
        """
        device = row_ids.device
        ids, attn = row_ids, row_attn
        committed = []
        finished = False
        round_idx = 0

        while not finished and len(committed) < max_new_tokens:
            k = min(self.k_sud, max_new_tokens - len(committed))

            # 1) Draft: sample k tokens autoregressively from q. Each log_q
            # row is kept verbatim — the acceptance test, the π blend, and
            # the residual must all use the exact distribution the proposal
            # was sampled from.
            drafted = []
            log_q_rows = []
            d_ids, d_attn = ids, attn
            for _ in range(k):
                d_logits = self.draft(
                    input_ids=d_ids, attention_mask=d_attn
                ).logits[:, -1, :]
                log_q = F.log_softmax(d_logits.float() / temp, dim=-1).squeeze(0)
                tok = int(torch.multinomial(log_q.exp(), num_samples=1).item())
                drafted.append(tok)
                log_q_rows.append(log_q)
                d_ids = torch.cat(
                    [d_ids, torch.tensor([[tok]], dtype=ids.dtype, device=device)],
                    dim=1,
                )
                d_attn = torch.cat(
                    [d_attn, torch.ones((1, 1), dtype=attn.dtype, device=device)],
                    dim=1,
                )
                if tok in eos_ids:
                    # Shortening the draft window is distribution-safe; tokens
                    # past a proposed EOS would be discarded anyway.
                    break
            k_eff = len(drafted)

            # 2) Verify: one parallel target forward over the drafted prefix.
            # Position L-1+i predicts drafted[i]; the logits after the last
            # drafted token are never used (no bonus token).
            L = ids.size(1)
            t_logits = self.target(
                input_ids=d_ids, attention_mask=d_attn
            ).logits[0].float()
            t_rows = t_logits[L - 1 : L - 1 + k_eff] / temp
            log_q_all = torch.stack(log_q_rows)
            # log_q_all is already normalized, so blend_log_pi's internal
            # log_softmax passes it through unchanged: π is built from the
            # exact q the proposals were sampled from.
            log_pi_all = blend_log_pi(t_rows, log_q_all, self.alpha)

            # 3) Accept/reject each drafted token in order.
            decisions = [] if debugger is not None else None
            for i, tok in enumerate(drafted):
                # Both π and q must be normalized here: this ratio spans two
                # different distributions, so the log Z that log_softmax
                # cancels in the metrics would bias every decision if left in.
                log_pi_i, log_q_i = log_pi_all[i], log_q_all[i]
                u = torch.rand((), device=device)
                accepted = bool(torch.log(u) < log_pi_i[tok] - log_q_i[tok])
                if accepted:
                    next_tok = tok
                else:
                    # First rejection: one replacement from the residual.
                    # Rejection implies π(tok) < q(tok), so the residual never
                    # re-picks tok; it can only be all-zero when π == q
                    # numerically, in which case π itself is the residual limit.
                    residual = (log_pi_i.exp() - log_q_i.exp()).clamp(min=0.0)
                    if residual.sum() <= 0:
                        residual = log_pi_i.exp()
                    next_tok = int(torch.multinomial(residual, num_samples=1).item())
                committed.append(next_tok)
                if debugger is not None:
                    decisions.append((tok, accepted, next_tok, float(u.item())))
                if next_tok in eos_ids:
                    finished = True
                    break
                if stopping_criteria is not None:
                    # HF StoppingCriteriaList(input_ids, scores) → bool tensor
                    # or bool. Checked per committed token, like the baseline:
                    # criteria such as MultiTokenEOSCriteria only inspect a
                    # short lookback window, so a stop sequence must not have
                    # a chance to scroll past it within a round.
                    seq = torch.cat(
                        [
                            row_ids,
                            torch.tensor(
                                [committed], dtype=row_ids.dtype, device=device
                            ),
                        ],
                        dim=1,
                    )
                    stop_signal = stopping_criteria(seq, None)
                    if isinstance(stop_signal, torch.Tensor):
                        finished = bool(stop_signal.all())
                    else:
                        finished = bool(stop_signal)
                    if finished:
                        break
                if not accepted:
                    # Discard the remaining drafted tokens and end the round.
                    break

            if debugger is not None:
                debugger.log_round(
                    round_idx, ids, drafted, decisions,
                    t_rows, log_q_all, log_pi_all,
                )
                round_idx += 1

            new_t = torch.tensor([committed], dtype=row_ids.dtype, device=device)
            ids = torch.cat([row_ids, new_t], dim=1)
            attn = torch.cat(
                [
                    row_attn,
                    torch.ones((1, len(committed)), dtype=row_attn.dtype, device=device),
                ],
                dim=1,
            )

        return committed
