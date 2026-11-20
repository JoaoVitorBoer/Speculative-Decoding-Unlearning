"""Weight-interpolation baseline for SUD: θ_α = (1-α)·θ_p + α·θ_q.

`WeightInterpModelForCausalLM.from_pretrained` loads the target p and the
draft q (q must be a fine-tune of p: same architecture, same parameter names),
overwrites p's parameters in place with the interpolation and returns the plain
HF model, so evaluation and generation are exactly those of a single model.
α=0 reproduces p and α=1 reproduces q bit-for-bit.

Load and merge in float32. The TOFU checkpoints are stored in bf16 and an
unlearning run moves most changed weights by only 1-2 bf16 ulps (NPO 1B
forget01: median 1 ulp), so a merge rounded back to bf16 snaps to θ_p or θ_q
instead of interpolating (at α=0.9, 88.5% of the changed weights equal θ_q
exactly). The run config therefore sets torch_dtype=float32, which also rules
out flash_attention_2 (use sdpa).
"""

from __future__ import annotations

import logging

import torch
from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)


class WeightInterpModelForCausalLM:
    """Factory only: `from_pretrained` returns a merged AutoModelForCausalLM."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        draft_model_name_or_path: str = None,
        alpha: float = 0.5,
        **kwargs,
    ):
        if draft_model_name_or_path is None:
            raise ValueError(
                "WeightInterpModelForCausalLM requires `draft_model_name_or_path`."
            )
        alpha = float(alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if kwargs.get("torch_dtype") not in (None, torch.float32):
            logger.warning(
                "WeightInterp: torch_dtype=%s — merging below fp32 snaps most "
                "weights to θ_p or θ_q (see module docstring).",
                kwargs.get("torch_dtype"),
            )

        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)
        # The draft only supplies tensors: keep it on CPU in its stored dtype
        # (bf16 → fp32 upcast per tensor below is exact), so the GPU never holds
        # two fp32 copies (a 3B pair would spike to ~26 GB).
        draft_kwargs = {k: v for k, v in kwargs.items()
                        if k not in ("device_map", "torch_dtype", "quantization_config")}
        draft = AutoModelForCausalLM.from_pretrained(
            draft_model_name_or_path, torch_dtype="auto", device_map="cpu", **draft_kwargs
        )

        tied_p = getattr(model.config, "tie_word_embeddings", None)
        tied_q = getattr(draft.config, "tie_word_embeddings", None)
        if tied_p != tied_q:
            raise AssertionError(f"tie_word_embeddings differs: p={tied_p}, q={tied_q}")

        # named_parameters() de-duplicates tied tensors (lm_head ↔ embed_tokens),
        # so each shared tensor is merged exactly once.
        p_params = dict(model.named_parameters())
        q_params = dict(draft.named_parameters())
        if p_params.keys() != q_params.keys():
            raise AssertionError(
                "Target and draft parameter names differ: "
                f"{sorted(p_params.keys() ^ q_params.keys())[:10]}"
            )

        n_changed = n_total = 0
        with torch.no_grad():
            for name, p in p_params.items():
                q = q_params[name].to(device=p.device, dtype=p.dtype)
                if p.shape != q.shape:
                    raise AssertionError(f"{name}: shape {tuple(p.shape)} vs {tuple(q.shape)}")
                if not torch.is_floating_point(p):
                    raise AssertionError(f"{name}: non-floating parameter {p.dtype}")
                n_total += p.numel()
                n_changed += int((p != q).sum())
                # (1-α)·p + α·q rather than lerp (p + α(q-p)), so that α=1
                # gives q exactly, not q up to one rounding.
                p.mul_(1.0 - alpha).add_(q, alpha=alpha)

            # Non-persistent buffers (rotary inv_freq) come from the config and
            # must already agree.
            q_bufs = dict(draft.named_buffers())
            for name, b in model.named_buffers():
                if name in q_bufs and not torch.allclose(
                    b.float(), q_bufs[name].to(b.device).float(), rtol=1e-6, atol=0
                ):
                    raise AssertionError(f"buffer {name} differs between p and q")

        del draft, q_params, q_bufs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(
            "WeightInterp: α=%s, dtype=%s, %.1f%% of %.3fB parameters differ between p and q",
            alpha, next(model.parameters()).dtype, 100.0 * n_changed / max(n_total, 1),
            n_total / 1e9,
        )
        return model
