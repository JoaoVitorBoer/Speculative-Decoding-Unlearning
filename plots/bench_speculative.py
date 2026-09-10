#!/usr/bin/env python
"""Panel B benchmark: SUD throughput / acceptance / VRAM as a function of the
draft window k, for the same 1B / forget10 / NPO cell as the Panel A sweep.

What is measured, per (alpha, k) cell
  * wall-clock around `generate` (CUDA-synchronised), summed over the prompts
  * new tokens from the returned ids  (bsz=1 -> exactly the committed tokens)
  * proposal / verification / acceptance counts from an in-process counter
    that replaces the debugger the UNMODIFIED src/model/sud.py already calls
    (monkeypatched from here: sud.DEBUG = True; sud.SUDDebugger = counter)
  * torch.cuda.max_memory_allocated / max_memory_reserved

Workload
  The first N (default 100) prompts of one of Part 1's TOFU_EVAL.json files
  (metric `forget_Q_A_ROUGE` -> the forget10 questions). The stored `input`
  strings are the eval's prompts decoded with skip_special_tokens=True, i.e.
  a lossy rendering of the real token prompt (no BOS / header tokens). So the
  token-level prompt is rebuilt through the SAME code path the eval uses
  (hydra config -> QADataset(predict_with_generate) -> left-padding collator,
  identical template_args) and every rebuilt prompt is asserted to decode to
  the stored `input` string verbatim. batch_size=1 so that the counter (which
  only sees batch row 0) is exact. Same generation config as the eval
  (configs/generation/default.yaml: max_new_tokens=200; SUD ignores
  do_sample/use_cache; temperature None -> 1.0).

Acceptance definitions (both are recorded)
  acceptance               = accepted / verified   (= accepted / committed)
      the per-token acceptance probability `a` of the speculative process;
      this is what the throughput model T(k) = C(1-a^k)/((1-a)(k+1)) needs
      and what should be constant in k.
  acceptance_over_proposed = accepted / proposed
      proposals drafted after the first rejection of a round are discarded
      un-verified, so this ratio falls with k mechanically:
      E[...] = a(1-a^k)/((1-a)k). It is NOT the process's acceptance rate.

Grid order is k OUTER, alpha inner, so a run cut short still leaves a complete
grid up to some k. `--resume` keeps the cells already present in --out (same
target / draft / prompt set / max_new_tokens) and benchmarks only the missing ones.

Per cell the record also carries `rounds_draft_truncated` (rounds whose draft
stopped before k: the draft hit its own EOS, or the max_new_tokens budget) and
`mean_drafted_per_round` = proposed / rounds — the effective window, which
saturates near the draft's answer length once k exceeds it.

Usage (from the repo root, `unlearning` env, one GPU):
    python plots/bench_speculative.py --out plots/bench/spec_bench.json --resume
CPU smoke test (tiny workload, exercises the whole pipeline):
    python plots/bench_speculative.py --device-map cpu --attn eager \
        --dtype float32 --n-prompts 2 --max-new-tokens 4 --alphas 0.5 --ks 1 3 \
        --eval-json <any TOFU_EVAL.json of this cell> --out /tmp/x.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)  # hydra `paths.root_dir: .` and the relative draft path

import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import model.sud as sud  # noqa: E402  -- the UNMODIFIED module
from model import get_model  # noqa: E402
from data import get_collators, get_datasets  # noqa: E402
from evals.metrics.utils import stop_sequences_criteria  # noqa: E402
from trainer.utils import seed_everything  # noqa: E402  (same seeding as src/eval.py)

TARGET = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
DRAFT = "saves/unlearn/baselines/tofu/NPO/Llama-3.2-1B-Instruct/forget10"
FORGET_SPLIT, HOLDOUT_SPLIT, RETAIN_SPLIT = "forget10", "holdout10", "retain90"


# ---------------------------------------------------------------------------
# The counter that stands in for SUDDebugger. Same constructor / start /
# log_round / finish surface; writes nothing; accumulates into class totals.
# ---------------------------------------------------------------------------
class AcceptanceCounter:
    totals: dict = {}
    calls: list = []

    @classmethod
    def reset(cls):
        cls.totals = dict(calls=0, rounds=0, proposed=0, verified=0,
                          accepted=0, rejected=0, committed=0, discarded=0,
                          rounds_draft_truncated=0)
        # per-round histogram of the drafted length {proposed: n_rounds}: rounds are
        # heterogeneous once the draft's EOS / the budget cut the window, and the
        # throughput theory needs the distribution, not just its mean.
        cls.draft_len_hist = {}
        cls.calls = []

    def __init__(self, alpha, tokenizer, output_dir="debug", top_k=10, k_sud=None):
        self.alpha, self.k_sud = alpha, k_sud
        self.mine = dict(rounds=0, proposed=0, verified=0, accepted=0,
                         rejected=0, committed=0, discarded=0,
                         rounds_draft_truncated=0)

    def start(self, prompt_ids, batch_index=None):
        AcceptanceCounter.totals["calls"] += 1

    def log_round(self, round_idx, context_ids, drafted, decisions,
                  t_rows, log_q_all, log_pi_all):
        decisions = decisions or []
        proposed = len(drafted)
        verified = len(decisions)                       # == committed this round
        accepted = sum(1 for _, acc, _, _ in decisions if acc)
        # drafted fewer than k_sud tokens: the draft emitted EOS, or the round was
        # clipped to the remaining max_new_tokens budget (k_sud > budget left).
        truncated = int(self.k_sud is not None and proposed < int(self.k_sud))
        upd = dict(rounds=1, proposed=proposed, verified=verified,
                   accepted=accepted, rejected=verified - accepted,
                   committed=verified, discarded=proposed - verified,
                   rounds_draft_truncated=truncated)
        for key, v in upd.items():
            self.mine[key] += v
            AcceptanceCounter.totals[key] += v
        h = AcceptanceCounter.draft_len_hist
        h[proposed] = h.get(proposed, 0) + 1

    def finish(self):
        AcceptanceCounter.calls.append(dict(self.mine))


AcceptanceCounter.reset()


def hydra_cfg(args):
    """Compose exactly the config the sweep's `python src/eval.py ...` sees."""
    overrides = [
        "experiment=eval/tofu/default.yaml",
        "model=sud",
        f"model.model_args.device_map={args.device_map}",
        f"model.model_args.pretrained_model_name_or_path={args.target}",
        f"model.model_args.draft_model_name_or_path={args.draft}",
        f"model.model_args.alpha={args.alphas[0]}",
        f"model.model_args.k_sud={args.ks[0]}",
        f"model.tokenizer_args.pretrained_model_name_or_path={args.target}",
        f"seed={args.seed}",
        f"forget_split={FORGET_SPLIT}",
        f"holdout_split={HOLDOUT_SPLIT}",
        f"retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_{RETAIN_SPLIT}/TOFU_EVAL.json",
        "task_name=bench_speculative",
        "paths.output_dir=plots/bench",
    ]
    if args.attn:
        overrides.append(f"model.model_args.attn_implementation={args.attn}")
    if args.dtype:
        overrides.append(f"model.model_args.torch_dtype={args.dtype}")
    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base=None):
        return compose(config_name="eval.yaml", overrides=overrides), overrides


def find_metric_cfg(metrics_cfg, name: str):
    """Locate a metric's config node. In configs/eval/tofu.yaml the ROUGE metrics are
    not top-level: forget_Q_A_ROUGE is a `pre_compute` of forget_Q_A_gibberish and the
    retain/ra/wf ROUGEs are pre_computes of model_utility (TOFU_EVAL.json still stores
    them at top level because the evaluator caches every pre-computed metric)."""
    if name in metrics_cfg:
        return metrics_cfg[name], name
    for parent, mcfg in metrics_cfg.items():
        pre = mcfg.get("pre_compute", None)
        if pre:
            found = find_metric_cfg(pre, name)
            if found is not None:
                return found[0], f"{parent}.pre_compute.{found[1]}"
    return None


def find_eval_json(results_root: Path, metric: str, n: int) -> Path:
    cands = sorted(results_root.glob("**/TOFU_EVAL.json"))
    cands.sort(key=lambda p: (0 if "a0.9_k1_s0" in str(p) else 1, str(p)))
    for p in cands:
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue  # truncated file from a killed run
        if metric in d and len(d[metric].get("value_by_index", {})) >= n:
            return p
    raise FileNotFoundError(
        f"no complete TOFU_EVAL.json with {metric} under {results_root}; run Part 1 first"
    )


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--draft", default=DRAFT)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.8, 0.9])
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--metric", default="forget_Q_A_ROUGE",
                    help="TOFU_EVAL.json metric whose `input` strings define the workload")
    ap.add_argument("--eval-json", type=Path, default=None,
                    help="Part-1 TOFU_EVAL.json to lift prompts from (default: auto under --results-root)")
    ap.add_argument("--results-root", type=Path, default=REPO / "plots" / "results")
    ap.add_argument("--out", type=Path, default=REPO / "plots" / "bench" / "spec_bench.json")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="override; default = the eval's generation config (200)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=1, help="untimed warm-up generations per cell")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--attn", default=None, help="override attn_implementation (default: config = flash_attention_2)")
    ap.add_argument("--dtype", default=None, help="override torch_dtype (default: config = bfloat16)")
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="continue even if a rebuilt prompt does not decode to the stored `input`")
    ap.add_argument("--resume", action="store_true",
                    help="keep the (alpha, k) records already in --out (if its workload matches) and run only the missing cells")
    args = ap.parse_args()

    # --- 1. monkeypatch the debugger seam of the unmodified module -----------
    sud.DEBUG = True
    sud.SUDDebugger = AcceptanceCounter

    # --- 2. same config / model / tokenizer / dataset as the eval ------------
    cfg, overrides = hydra_cfg(args)
    seed_everything(args.seed)
    model, tokenizer = get_model(cfg.model)
    model.eval()
    template_args = cfg.model.template_args
    found = find_metric_cfg(cfg.eval.tofu.metrics, args.metric)
    if found is None:
        raise SystemExit(f"metric {args.metric} not found in eval.tofu.metrics (nor in any pre_compute)")
    metric_cfg, metric_path = found
    print(f"metric config: eval.tofu.metrics.{metric_path}", flush=True)
    dataset = get_datasets(metric_cfg.datasets, tokenizer=tokenizer, template_args=template_args)
    collator = get_collators(metric_cfg.collators, tokenizer=tokenizer)

    generation_args = OmegaConf.to_container(metric_cfg.generation_args, resolve=True)
    if args.max_new_tokens is not None:
        generation_args["max_new_tokens"] = args.max_new_tokens
    stopwords = generation_args.pop("stopwords", None)  # eval_text_similarity semantics
    max_new_tokens = int(generation_args["max_new_tokens"])

    # --- 3. workload: the eval's own prompts ---------------------------------
    eval_json = args.eval_json or find_eval_json(args.results_root, args.metric, args.n_prompts)
    stored = json.loads(Path(eval_json).read_text())[args.metric]["value_by_index"]
    indices = sorted(int(i) for i in stored)[: args.n_prompts]
    prompts, mismatches = [], []
    for idx in indices:
        item = dataset[idx]
        assert int(item["index"]) == idx, (item["index"], idx)
        batch = collator([item])
        text = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True,
                                      clean_up_tokenization_spaces=True)[0]
        if text != stored[str(idx)]["input"]:
            mismatches.append(idx)
        prompts.append((idx, batch["input_ids"], batch["attention_mask"]))
    msg = (f"{len(prompts)} prompts from {eval_json} [{args.metric}]; "
           f"{len(mismatches)} do not decode to the stored `input`")
    print(msg, flush=True)
    if mismatches and not args.allow_mismatch:
        raise SystemExit(f"prompt mismatch at indices {mismatches[:10]} — refusing to benchmark a different workload")

    device = model.device
    cuda = device.type == "cuda"
    if stopwords is not None:
        # same criteria object the eval builds (bsz=1); none for the default config
        generation_args["stopping_criteria"] = stop_sequences_criteria(
            tokenizer, list(stopwords), prompts[0][1].shape[1], 1)

    def sync():
        if cuda:
            torch.cuda.synchronize(device)

    def run_generate(input_ids, attn):
        return model.generate(input_ids.to(device), attention_mask=attn.to(device),
                              **generation_args, pad_token_id=tokenizer.eos_token_id)

    meta = dict(
        target=args.target, draft=args.draft, forget_split=FORGET_SPLIT,
        prompt_source=str(eval_json), prompt_metric=args.metric,
        n_prompts=len(prompts), prompt_indices=[i for i, _, _ in prompts],
        prompt_mismatches=mismatches, batch_size=1, max_new_tokens=max_new_tokens,
        generation_args={k: v for k, v in generation_args.items() if k != "stopping_criteria"},
        stopwords=stopwords, seed=args.seed, warmup=args.warmup,
        hydra_overrides=overrides,
        model_args=OmegaConf.to_container(cfg.model.model_args, resolve=True),
        device=str(device), gpu=(torch.cuda.get_device_name(device) if cuda else platform.processor()),
        torch=torch.__version__, cuda_version=torch.version.cuda, git_head=git_head(),
        started=datetime.now().isoformat(timespec="seconds"),
        acceptance_definition="accepted / verified (== accepted / committed); "
                              "acceptance_over_proposed = accepted / proposed",
        no_kv_cache=True,
    )
    records = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and args.out.exists():
        # Same workload = same target/draft/prompts/budget; anything else is a
        # different benchmark and must not be mixed into this file.
        try:
            old = json.loads(args.out.read_text())
        except Exception:
            old = None
        keys = ("target", "draft", "prompt_metric", "n_prompts", "prompt_indices", "max_new_tokens", "batch_size")
        if old and all(old.get("meta", {}).get(key) == meta[key] for key in keys):
            wanted = {(float(a), int(k)) for a in args.alphas for k in args.ks}
            records = [r for r in old.get("records", []) if (r["alpha"], r["k"]) in wanted and r.get("tokens_per_sec")]
            meta["resumed_from"] = dict(started=old["meta"].get("started"), finished=old["meta"].get("finished"),
                                        gpu=old["meta"].get("gpu"), n_records_kept=len(records))
            print(f"--resume: keeping {len(records)} finished cell(s) from {args.out}", flush=True)
        else:
            bak = args.out.with_suffix(".incompatible.json")
            args.out.replace(bak)
            print(f"--resume: {args.out} has a different workload; moved it to {bak} and starting fresh", flush=True)
    done = {(r["alpha"], r["k"]) for r in records}

    def dump():
        args.out.write_text(json.dumps(dict(meta=meta, records=records), indent=2) + "\n")

    dump()

    # --- 4. the grid (k outer: a run cut short still leaves a full grid up to some k) --
    for k in args.ks:
        for alpha in args.alphas:
            if (float(alpha), int(k)) in done:
                print(f"alpha={alpha:<4} k={k:<3} already benchmarked — skipped", flush=True)
                continue
            model.alpha, model.k_sud = float(alpha), int(k)   # plain attributes of the wrapper
            seed_everything(args.seed)
            # warm-up (untimed): kernel/allocator init, not counted
            for w in range(args.warmup):
                _, ids, attn = prompts[w % len(prompts)]
                sync(); run_generate(ids, attn); sync()
            AcceptanceCounter.reset()
            if cuda:
                torch.cuda.reset_peak_memory_stats(device)
            seed_everything(args.seed)

            total_s, total_new, per_prompt = 0.0, 0, []
            t_cell = time.perf_counter()
            for idx, ids, attn in prompts:
                n_calls_before = len(AcceptanceCounter.calls)
                sync(); t0 = time.perf_counter()
                out = run_generate(ids, attn)
                sync(); dt = time.perf_counter() - t0
                new = int(out.shape[1] - ids.shape[1])
                call = AcceptanceCounter.calls[-1] if len(AcceptanceCounter.calls) > n_calls_before else {}
                total_s += dt
                total_new += new
                per_prompt.append(dict(idx=idx, new_tokens=new, seconds=round(dt, 4),
                                       rounds=call.get("rounds"), accepted=call.get("accepted"),
                                       verified=call.get("verified"), proposed=call.get("proposed")))
            cell_s = time.perf_counter() - t_cell
            T = dict(AcceptanceCounter.totals)
            hist = {str(n): c for n, c in sorted(AcceptanceCounter.draft_len_hist.items())}
            rec = dict(
                alpha=float(alpha), k=int(k), n_prompts=len(prompts),
                total_seconds=round(total_s, 4), cell_wall_seconds=round(cell_s, 4),
                total_new_tokens=total_new,
                tokens_per_sec=(total_new / total_s if total_s > 0 else None),
                rounds=T["rounds"], proposed=T["proposed"], verified=T["verified"],
                accepted=T["accepted"], rejected=T["rejected"], committed=T["committed"],
                discarded=T["discarded"], counter_calls=T["calls"],
                rounds_draft_truncated=T["rounds_draft_truncated"],
                mean_drafted_per_round=(T["proposed"] / T["rounds"] if T["rounds"] else None),
                acceptance=(T["accepted"] / T["verified"] if T["verified"] else None),
                acceptance_over_proposed=(T["accepted"] / T["proposed"] if T["proposed"] else None),
                tokens_per_round=(T["committed"] / T["rounds"] if T["rounds"] else None),
                forwards_per_token=(((T["proposed"] + T["rounds"]) / T["committed"]) if T["committed"] else None),
                committed_matches_new_tokens=(T["committed"] == total_new),
                mean_new_tokens=total_new / len(prompts),
                draft_len_hist=hist,   # {drafted length: rounds}; sums to `rounds`
                max_memory_allocated_mb=(torch.cuda.max_memory_allocated(device) / 2**20 if cuda else None),
                max_memory_reserved_mb=(torch.cuda.max_memory_reserved(device) / 2**20 if cuda else None),
                per_prompt=per_prompt,
            )
            records.append(rec)
            records.sort(key=lambda r: (r["alpha"], r["k"]))
            dump()
            print(f"alpha={alpha:<4} k={k:<3} tok/s={rec['tokens_per_sec']:.2f} "
                  f"acc={rec['acceptance']:.3f} acc/proposed={rec['acceptance_over_proposed']:.3f} "
                  f"tok/round={rec['tokens_per_round']:.2f} kbar={rec['mean_drafted_per_round']:.1f} "
                  f"trunc={rec['rounds_draft_truncated']}/{rec['rounds']} new_tok={total_new} "
                  f"t={total_s:.1f}s vram={rec['max_memory_allocated_mb']}", flush=True)

    meta["finished"] = datetime.now().isoformat(timespec="seconds")
    dump()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
