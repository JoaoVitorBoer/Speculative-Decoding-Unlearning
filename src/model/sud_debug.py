import json
from collections import Counter
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

class SUDDebugger:
    """Captures per-step blend behavior for one SUD generate() call.

    Writes two files per call:
      {output_dir}/sud_idx{NNNNN}_{ts}.txt    — human-readable tables + summary
      {output_dir}/sud_idx{NNNNN}_{ts}.jsonl  — one row per step for analysis

    Logs only batch index 0. Lifecycle: start() once, log_step() per decoding
    step (after sampling, before any pad override), finish() once at the end.
    """

    def __init__(self, alpha, tokenizer, output_dir="debug", top_k=10):
        self.alpha = alpha
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.top_k = top_k
        self.fh = None
        self.jsonl_fh = None
        self.steps_records = []
        self.sampled_tokens = []  # list of token ids for text reconstruction

    def start(self, prompt_ids, batch_index=None):
        os.makedirs(self.output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        idx_part = f"idx{batch_index:05d}_" if batch_index is not None else ""
        base = os.path.join(self.output_dir, f"sud_{idx_part}{ts}")
        self.fh = open(base + ".txt", "w", encoding="utf-8")
        self.jsonl_fh = open(base + ".jsonl", "w", encoding="utf-8")

        prompt_list = prompt_ids[0].tolist()
        prompt_text = self._safe_decode_seq(prompt_list)
        self.fh.write(f"alpha={self.alpha}  top_k={self.top_k}\n")
        self.fh.write(f"prompt: {prompt_text!r}\n")
        self.fh.write("(generated text appended at end of run)\n\n")
        self.fh.write("=" * 78 + "\n\n")
        self.fh.flush()

    @torch.no_grad()
    def log_step(self, step, t_logits, d_logits, log_pi, sampled_id):
        # batch element 0 only
        log_p = F.log_softmax(t_logits[0].float(), dim=-1)
        log_q = F.log_softmax(d_logits[0].float(), dim=-1)
        log_pi0 = log_pi[0].float()
        #log_pi0 = log_pi0 - torch.logsumexp(log_pi0, dim=-1)  # safety renorm

        p, q, pi = log_p.exp(), log_q.exp(), log_pi0.exp()

        # --- scalars ---
        H_p = -(p * log_p).sum().item()
        H_q = -(q * log_q).sum().item()
        H_pi = -(pi * log_pi0).sum().item()
        kl_pi_p = (pi * (log_pi0 - log_p)).sum().item()
        kl_pi_q = (pi * (log_pi0 - log_q)).sum().item()

        top1_p = int(log_p.argmax().item())
        top1_q = int(log_q.argmax().item())
        top1_pi = int(log_pi0.argmax().item())
        if top1_p == top1_q == top1_pi:
            agree = "all"
        elif top1_pi == top1_p:
            agree = "p"
        elif top1_pi == top1_q:
            agree = "q"
        else:
            agree = "neither"

        sid = int(sampled_id[0].item()) if torch.is_tensor(sampled_id) else int(sampled_id)
        sstr = self._safe_decode(sid)

        # --- union top-K ---
        topk_p = set(log_p.topk(self.top_k).indices.tolist())
        topk_q = set(log_q.topk(self.top_k).indices.tolist())
        topk_pi = set(log_pi0.topk(self.top_k).indices.tolist())
        union = topk_p | topk_q | topk_pi
        rows = sorted(union, key=lambda tid: -pi[tid].item())

        # --- write header line ---
        self.fh.write(
            f"step {step:4d}  sampled={sstr!r} (id={sid})\n"
            f"  H(p)={H_p:.3f}  H(q)={H_q:.3f}  H(pi)={H_pi:.3f}  "
            f"KL(pi||p)={kl_pi_p:.3f}  KL(pi||q)={kl_pi_q:.3f}  "
            f"top1_agree={agree}\n"
        )
        # --- write union table ---
        self.fh.write(
            f"  {'token':>24}  {'pi':>7}  {'p':>7}  {'q':>7}  "
            f"{'D(p->pi)':>9}  src\n"
        )
        for tid in rows:
            tok = self._safe_decode(tid)
            tok_repr = repr(tok)
            if len(tok_repr) > 24:
                tok_repr = tok_repr[:21] + "..."
            mark = " <-" if tid == sid else "   "
            src = []
            if tid in topk_p:  src.append("p")
            if tid in topk_q:  src.append("q")
            if tid in topk_pi: src.append("pi")
            delta = (pi[tid] - p[tid]).item()
            self.fh.write(
                f"{mark}{tok_repr:>24}  "
                f"{pi[tid].item():7.4f}  {p[tid].item():7.4f}  {q[tid].item():7.4f}  "
                f"{delta:+9.4f}  {','.join(src)}\n"
            )
        self.fh.write("\n")
        self.fh.flush()

        # --- jsonl sidecar (one row per step) ---
        record = {
            "step": step,
            "H_p": H_p, "H_q": H_q, "H_pi": H_pi,
            "kl_pi_p": kl_pi_p, "kl_pi_q": kl_pi_q,
            "top1_p": top1_p, "top1_q": top1_q, "top1_pi": top1_pi,
            "top1_agree": agree,
            "sampled_id": sid, "sampled_str": sstr,
            "p_at_argmax_p": p[top1_p].item(),
            "pi_at_argmax_p": pi[top1_p].item(),
        }
        self.jsonl_fh.write(json.dumps(record) + "\n")
        self.jsonl_fh.flush()
        self.steps_records.append(record)
        self.sampled_tokens.append(sid)

    def finish(self):
        if self.fh is None:
            return
        try:
            # --- generated-text reconstruction ---
            plain = self._safe_decode_seq(self.sampled_tokens)
            bracketed = "".join(f"[{self._safe_decode(t)}]" for t in self.sampled_tokens)
            self.fh.write("=" * 78 + "\n")
            self.fh.write(f"generated (plain):     {plain!r}\n")
            self.fh.write(f"generated (bracketed): {bracketed}\n\n")

            # --- run summary ---
            self.fh.write("=" * 78 + "\n=== run summary ===\n")
            n = len(self.steps_records)
            if n == 0:
                self.fh.write("no steps logged\n")
            else:
                kls_p = [r["kl_pi_p"] for r in self.steps_records]
                kls_q = [r["kl_pi_q"] for r in self.steps_records]
                imax_p = max(range(n), key=lambda i: kls_p[i])
                imax_q = max(range(n), key=lambda i: kls_q[i])
                self.fh.write(f"steps: {n}\n")
                self.fh.write(
                    f"mean KL(pi||p): {sum(kls_p)/n:.3f}    "
                    f"max: {kls_p[imax_p]:.3f} @ step {self.steps_records[imax_p]['step']}\n"
                )
                self.fh.write(
                    f"mean KL(pi||q): {sum(kls_q)/n:.3f}    "
                    f"max: {kls_q[imax_q]:.3f} @ step {self.steps_records[imax_q]['step']}\n"
                )
                hist = Counter(r["top1_agree"] for r in self.steps_records)
                self.fh.write(
                    f"top1 source histogram: "
                    f"all={hist.get('all',0)}  p={hist.get('p',0)}  "
                    f"q={hist.get('q',0)}  neither={hist.get('neither',0)}\n"
                )
                # suppression events: argmax_p got demoted in pi
                supp = []
                for r in self.steps_records:
                    drop = r["p_at_argmax_p"] - r["pi_at_argmax_p"]
                    if drop > 0.2:
                        supp.append((drop, r))
                supp.sort(key=lambda x: -x[0])
                self.fh.write(
                    f"suppression events (p[argmax_p] - pi[argmax_p] > 0.2): {len(supp)}\n"
                )
                for drop, r in supp[:5]:
                    tok = self._safe_decode(r["top1_p"])
                    self.fh.write(
                        f"  step {r['step']:4d}  {tok!r}  "
                        f"p={r['p_at_argmax_p']:.3f} -> pi={r['pi_at_argmax_p']:.3f}  "
                        f"(drop {drop:.3f})\n"
                    )
        finally:
            self.fh.close()
            self.jsonl_fh.close()
            self.fh = None
            self.jsonl_fh = None

    # --- helpers ---
    def _safe_decode(self, token_id):
        if self.tokenizer is None:
            return f"<id={token_id}>"
        s = self.tokenizer.decode([token_id], skip_special_tokens=False)
        return s.replace("\n", "\\n").replace("\t", "\\t")

    def _safe_decode_seq(self, ids):
        if self.tokenizer is None:
            return f"<{len(ids)} tokens>"
        return self.tokenizer.decode(ids, skip_special_tokens=False)