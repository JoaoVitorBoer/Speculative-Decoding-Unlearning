#!/usr/bin/env python
"""Reference (baseline) values for the k-sweep figure — plots/baselines.json.

model_utility and forget_Q_A_gibberish of the three fixed models of this cell,
evaluated with the SAME TOFU forget10 suite and byte-identical prompts as the SUD
runs, but with GREEDY decoding (the standard eval); SUD samples at T = 1, which
costs ~0.015 model_utility on its own (measured, see the README):
    target  p = open-unlearning/tofu_Llama-3.2-1B-Instruct_full     (alpha = 0 limit of SUD)
    draft   q = NPO/forget10 weight-unlearned checkpoint            (alpha = 1 limit of SUD)
    retain    = tofu_Llama-3.2-1B-Instruct_retain90                 (TOFU gold reference)

model_utility is read from each eval's TOFU_SUMMARY.json. forget_Q_A_gibberish is
not stored for the target / retain evals (older metric set), so it is recomputed
here from their stored forget_Q_A_ROUGE generations with the exact metric config
(configs/eval/tofu_metrics/forget_Q_A_gibberish.yaml: classifier
madhurjindal/autonlp-Gibberish-Detector-492513457, class 0 = "clean", max_length 32,
mean P(clean) over the 400 forget questions). The draft's stored value is recomputed
too as a check that this reproduces the eval pipeline's number. CPU is enough.

    CUDA_VISIBLE_DEVICES= conda run -n unlearning python plots/baseline_refs.py
Another draft method (writes plots/baselines_<method>.json; target / retain are the same models):
    CUDA_VISIBLE_DEVICES= conda run -n unlearning python plots/baseline_refs.py --method SimNPO
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "plots" / "baselines.json"
METRIC_CFG = REPO / "configs/eval/tofu_metrics/forget_Q_A_gibberish.yaml"

MODELS = {
    "target": dict(label="target p (original)", model="open-unlearning/tofu_Llama-3.2-1B-Instruct_full",
                   eval_dir="saves/eval/tofu_Llama-3.2-1B-Instruct_full/evals_forget10", role="alpha = 0 limit"),
    "draft": dict(label="draft q (NPO)", model="saves/unlearn/baselines/tofu/NPO/Llama-3.2-1B-Instruct/forget10",
                  eval_dir="saves/unlearn/baselines/tofu/NPO/Llama-3.2-1B-Instruct/forget10/evals", role="alpha = 1 limit"),
    "retain": dict(label="retain model", model="open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90",
                   eval_dir="saves/eval/tofu_Llama-3.2-1B-Instruct_retain90", role="TOFU gold reference (retrained without forget10)"),
}
SUD_REF = REPO / ("plots/results/original-unlearned/target-Llama-3.2-1B-Instruct/"
                  "draft-Llama-3.2-1B-Instruct_NPO/forget10/a0.9_k1_s0/TOFU_EVAL.json")


def gibberish_score(generations: list[str], cfg) -> float:
    """Mean P(class_id) exactly as src/evals/metrics/utility.py::classifier_prob."""
    tok = AutoTokenizer.from_pretrained(**cfg.classifier_tokenization_args)
    clf = AutoModelForSequenceClassification.from_pretrained(**cfg.classifier_model_args).eval()
    scores = []
    bs, max_length, class_id = int(cfg.batch_size), int(cfg.max_length), int(cfg.class_id)
    with torch.no_grad():
        for i in range(0, len(generations), bs):
            batch = tok(generations[i:i + bs], return_tensors="pt", padding=True, truncation=True,
                        max_length=max_length, return_attention_mask=True)
            probs = torch.softmax(clf(**batch).logits, dim=-1)[:, class_id]
            scores += probs.tolist()
    return float(sum(scores) / len(scores))


def configure(method: str):
    """Point the draft entry, the SUD prompt reference and the output file at `method`."""
    global OUT, SUD_REF
    sfx = "" if method == "NPO" else f"_{method}"
    OUT = REPO / "plots" / f"baselines{sfx}.json"
    MODELS["draft"] = dict(label=f"draft q ({method})",
                          model=f"saves/unlearn/baselines/tofu/{method}/Llama-3.2-1B-Instruct/forget10",
                          eval_dir=f"saves/unlearn/baselines/tofu/{method}/Llama-3.2-1B-Instruct/forget10/evals",
                          role="alpha = 1 limit")
    SUD_REF = REPO / ("plots/results/original-unlearned/target-Llama-3.2-1B-Instruct/"
                      f"draft-Llama-3.2-1B-Instruct_{method}/forget10/a0.9_k1_s0/TOFU_EVAL.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="NPO", help="weight-unlearning method of the draft q (default NPO)")
    args = ap.parse_args()
    configure(args.method)
    cfg = OmegaConf.load(METRIC_CFG)
    sud_prompts = {k: v["input"] for k, v in json.loads(SUD_REF.read_text())["forget_Q_A_ROUGE"]["value_by_index"].items()}
    out = dict(generated=datetime.now().isoformat(timespec="minutes"), decode="greedy (do_sample=false, the standard eval)",
               sud_decode="sampled at temperature 1 (SUD always samples; measured model_utility handicap vs greedy ~0.015)",
               forget_split="forget10", gibberish_metric=OmegaConf.to_container(cfg, resolve=True), models={})
    for key, m in MODELS.items():
        d = REPO / m["eval_dir"]
        summ = json.loads((d / "TOFU_SUMMARY.json").read_text())
        ev = json.loads((d / "TOFU_EVAL.json").read_text())
        rouge = ev["forget_Q_A_ROUGE"]["value_by_index"]
        same = sum(rouge[k]["input"] == sud_prompts[k] for k in sud_prompts)
        assert same == len(sud_prompts), f"{key}: only {same}/{len(sud_prompts)} prompts match the SUD eval"
        gens = [rouge[k]["generation"] for k in sorted(rouge, key=int)]
        gib = gibberish_score(gens, cfg)
        rec = dict(label=m["label"], model=m["model"], role=m["role"], eval_dir=m["eval_dir"],
                   model_utility=summ["model_utility"], forget_Q_A_gibberish=gib,
                   forget_Q_A_gibberish_source="recomputed from stored forget_Q_A_ROUGE generations",
                   forget_Q_A_ROUGE=ev["forget_Q_A_ROUGE"].get("agg_value"),
                   n_forget_prompts=len(gens), prompts_identical_to_sud_eval=same == len(sud_prompts))
        if "forget_Q_A_gibberish" in summ:
            rec["forget_Q_A_gibberish_stored"] = summ["forget_Q_A_gibberish"]
            rec["recompute_abs_diff"] = abs(gib - summ["forget_Q_A_gibberish"])
        out["models"][key] = rec
        print(f"{key:7s} utility={rec['model_utility']:.4f} gibberish={gib:.4f}"
              + (f" (stored {summ['forget_Q_A_gibberish']:.4f}, |diff| {rec['recompute_abs_diff']:.4f})" if "forget_Q_A_gibberish" in summ else "")
              + f" prompts identical: {same}/{len(sud_prompts)}", flush=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
