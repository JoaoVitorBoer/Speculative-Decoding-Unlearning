"""Speculative-process logger for SUD generate().

Writes a human-readable trace of the draft-verify loop for batch row 0 of one
generate() call. Per round it records:
  * the context length the draft saw,
  * the tokens the draft proposed (sampled from q),
  * for each proposal: pi / p / q at that token, the acceptance probability
    min(1, pi/q), the uniform draw u, and the verdict —
        ACCEPT     (committed the drafted token),
        REJECT     (resampled a replacement from the residual max(0, pi - q)),
        DISCARDED  (an earlier token in the round was rejected, so this
                    proposal was thrown away and never verified),
  * the tokens actually committed this round.

A JSONL sidecar carries the same data for analysis; finish() appends the
reconstructed generation and a run summary (acceptance rate, tokens/round).

Toggled by the DEBUG flag in sud.py. Only batch row 0 is logged.
"""

import json
import os
from datetime import datetime

import torch
import torch.nn.functional as F


class SUDDebugger:
    def __init__(self, alpha, tokenizer, output_dir="debug", top_k=10, k_sud=None):
        self.alpha = alpha
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.top_k = top_k
        self.k_sud = k_sud
        self.fh = None
        self.jsonl_fh = None
        self.committed_tokens = []   # every committed token, in order
        self.round_records = []      # one dict per round (for the summary)

    # -- lifecycle -----------------------------------------------------------

    def start(self, prompt_ids, batch_index=None):
        os.makedirs(self.output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        idx_part = f"idx{batch_index:05d}_" if batch_index is not None else ""
        base = os.path.join(self.output_dir, f"sud_{idx_part}{ts}")
        self.fh = open(base + ".txt", "w", encoding="utf-8")
        self.jsonl_fh = open(base + ".jsonl", "w", encoding="utf-8")

        prompt_text = self._decode_seq(prompt_ids[0].tolist())
        self.fh.write(f"alpha={self.alpha}  k_sud={self.k_sud}\n")
        self.fh.write(f"prompt ({prompt_ids.size(1)} tokens): {prompt_text!r}\n")
        self.fh.write("=" * 80 + "\n\n")
        self.fh.flush()

    @torch.no_grad()
    def log_round(self, round_idx, context_ids, drafted, decisions,
                  t_rows, log_q_all, log_pi_all):
        """Log one draft-verify round.

        drafted   : list of proposed token ids (len k_eff), sampled from q.
        decisions : list of (drafted_id, accepted, committed_id, u) for the
                    proposals that were actually verified this round; proposals
                    past len(decisions) were discarded after a rejection.
        t_rows    : (k_eff, V) target logits for each proposal position.
        log_q_all : (k_eff, V) log q the proposals were sampled from.
        log_pi_all: (k_eff, V) log pi (already normalized).
        """
        # Normalize all three to log-probs. log_softmax is a no-op on the
        # already-normalized q and pi rows, and turns the raw target logits
        # into log p.
        log_p = F.log_softmax(t_rows.float(), dim=-1)
        log_q = F.log_softmax(log_q_all.float(), dim=-1)
        log_pi = F.log_softmax(log_pi_all.float(), dim=-1)
        p, q, pi = log_p.exp(), log_q.exp(), log_pi.exp()

        n_proposed = len(drafted)
        n_verified = len(decisions)
        committed_ids = [d[2] for d in decisions]

        # -- header --
        self.fh.write(
            f"round {round_idx:3d}  |  context_len={context_ids.size(1)}  "
            f"proposed={n_proposed}\n"
        )
        self.fh.write(f"  draft (from q): {[self._tok(t) for t in drafted]}\n")

        # -- per proposed token --
        for i, tok in enumerate(drafted):
            qv = q[i, tok].item()
            p_accept = min(1.0, pi[i, tok].item() / qv) if qv > 0 else 1.0
            stats = (
                f"pi={pi[i, tok].item():.4f} p={p[i, tok].item():.4f} "
                f"q={qv:.4f}  p_accept={p_accept:.3f}"
            )
            if i >= n_verified:
                self.fh.write(
                    f"  i={i}  drafted={self._tok(tok)!r} (id={tok})  {stats}  "
                    f"->  DISCARDED\n"
                )
                continue

            _, accepted, committed_id, u = decisions[i]
            if accepted:
                self.fh.write(
                    f"  i={i}  drafted={self._tok(tok)!r} (id={tok})  {stats}  "
                    f"u={u:.3f}  ->  ACCEPT  committed={self._tok(committed_id)!r} "
                    f"(id={committed_id})\n"
                )
            else:
                self.fh.write(
                    f"  i={i}  drafted={self._tok(tok)!r} (id={tok})  {stats}  "
                    f"u={u:.3f}  ->  REJECT\n"
                )
                res_top = self._residual_top(pi[i], q[i], k=5)
                self.fh.write(
                    f"        residual max(0,pi-q) top: {res_top}  ->  "
                    f"committed={self._tok(committed_id)!r} (id={committed_id})\n"
                )

        self.fh.write(
            f"  committed this round ({len(committed_ids)}): "
            f"{[self._tok(t) for t in committed_ids]}\n"
        )
        self.fh.write("-" * 80 + "\n")
        self.fh.flush()

        # -- jsonl sidecar --
        record = {
            "round": round_idx,
            "context_len": context_ids.size(1),
            "proposed": n_proposed,
            "verified": n_verified,
            "committed": len(committed_ids),
            "tokens": [
                {
                    "i": i,
                    "drafted_id": int(drafted[i]),
                    "drafted_str": self._tok(drafted[i]),
                    "verified": i < n_verified,
                    "accepted": bool(decisions[i][1]) if i < n_verified else None,
                    "committed_id": int(decisions[i][2]) if i < n_verified else None,
                    "u": float(decisions[i][3]) if i < n_verified else None,
                    "pi": pi[i, drafted[i]].item(),
                    "p": p[i, drafted[i]].item(),
                    "q": q[i, drafted[i]].item(),
                }
                for i in range(n_proposed)
            ],
        }
        self.jsonl_fh.write(json.dumps(record) + "\n")
        self.jsonl_fh.flush()

        self.committed_tokens.extend(committed_ids)
        self.round_records.append({
            "proposed": n_proposed,
            "verified": n_verified,
            "committed": len(committed_ids),
            "accepted": sum(1 for d in decisions if d[1]),
            "rejected": sum(1 for d in decisions if not d[1]),
        })

    def finish(self):
        if self.fh is None:
            return
        try:
            plain = self._decode_seq(self.committed_tokens)
            bracketed = "".join(f"[{self._tok(t)}]" for t in self.committed_tokens)
            self.fh.write("=" * 80 + "\n")
            self.fh.write(f"generated (plain):     {plain!r}\n")
            self.fh.write(f"generated (bracketed): {bracketed}\n\n")

            self.fh.write("=" * 80 + "\n=== run summary ===\n")
            rounds = self.round_records
            if not rounds:
                self.fh.write("no rounds logged\n")
                return
            n_rounds = len(rounds)
            proposed = sum(r["proposed"] for r in rounds)
            verified = sum(r["verified"] for r in rounds)
            accepted = sum(r["accepted"] for r in rounds)
            rejected = sum(r["rejected"] for r in rounds)
            committed = sum(r["committed"] for r in rounds)
            discarded = proposed - verified
            acc_rate = accepted / verified if verified else 0.0
            self.fh.write(f"rounds: {n_rounds}\n")
            self.fh.write(f"tokens committed: {committed}\n")
            self.fh.write(
                f"proposals: {proposed}  (accepted={accepted}  "
                f"rejected={rejected}  discarded={discarded})\n"
            )
            self.fh.write(f"acceptance rate (accepted/verified): {acc_rate:.3f}\n")
            self.fh.write(
                f"mean committed/round: {committed / n_rounds:.2f}  "
                f"(k_sud={self.k_sud})\n"
            )
        finally:
            self.fh.close()
            self.jsonl_fh.close()
            self.fh = None
            self.jsonl_fh = None

    # -- helpers -------------------------------------------------------------

    def _residual_top(self, pi_row, q_row, k=5):
        """Top-k of the normalized residual max(0, pi - q) as [tok(prob), ...]."""
        res = (pi_row - q_row).clamp(min=0.0)
        total = res.sum()
        res = pi_row if total <= 0 else res / total  # mirror sud.py fallback
        top = res.topk(min(k, res.numel()))
        return [
            f"{self._tok(int(tid))}({val.item():.2f})"
            for val, tid in zip(top.values, top.indices)
        ]

    def _tok(self, token_id):
        if self.tokenizer is None:
            return f"<id={token_id}>"
        s = self.tokenizer.decode([int(token_id)], skip_special_tokens=False)
        return s.replace("\n", "\\n").replace("\t", "\\t")

    def _decode_seq(self, ids):
        if self.tokenizer is None:
            return f"<{len(ids)} tokens>"
        return self.tokenizer.decode(ids, skip_special_tokens=False)
