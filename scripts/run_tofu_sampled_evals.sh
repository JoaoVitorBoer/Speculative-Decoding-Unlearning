#!/bin/bash
#
# run_tofu_sampled_evals.sh — full TOFU eval of a single (weight-baseline) model
# with the same decoding SUD uses for its generation metrics (multinomial sample
# from the full, unscaled distribution: do_sample, T=1, no top-k/top-p; see
# src/model/sud.py generate), so the ROUGE terms of model_utility and
# forget_Q_A_gibberish are comparable with SUD's. Forward metrics are unaffected.
#
#   CUDA_DEVICES=0 SEED=0 bash scripts/run_tofu_sampled_evals.sh <model_path> <out_dir> <forget_split> <model_cfg>
#
#   <model_path>: local checkpoint dir or HF id (e.g. JoaoBoer/tofu_Llama-3.2-1B-Instruct_forget10_NPO)
#   <forget_split>: forget01 | forget05 | forget10
#   <model_cfg>: Llama-3.2-1B-Instruct | Llama-3.2-3B-Instruct | Llama-3.1-8B-Instruct
#   Any further args are passed to src/eval.py as extra Hydra overrides.

set -euo pipefail
export HYDRA_FULL_ERROR=1
# The TOFU datasets are cached; online, every load waits ~100 s for the Hub on a slow
# link before falling back to that cache (2026-11-10), so read them offline.
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

model_path="$1"
out_dir="$2"
forget_split="$3"
model_cfg="$4"
shift 4
SEED="${SEED:-0}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"

pct="${forget_split#forget}"
holdout_split="holdout${pct}"
retain_split="retain$(printf '%02d' $((100 - 10#${pct})))"

gen=()
for m in model_utility.pre_compute.retain_Q_A_ROUGE \
         model_utility.pre_compute.ra_Q_A_ROUGE \
         model_utility.pre_compute.wf_Q_A_ROUGE \
         forget_Q_A_gibberish.pre_compute.forget_Q_A_ROUGE; do
  gen+=("eval.tofu.metrics.${m}.generation_args.do_sample=True"
        "eval.tofu.metrics.${m}.generation_args.temperature=1.0"
        "+eval.tofu.metrics.${m}.generation_args.top_k=0"
        "eval.tofu.metrics.${m}.generation_args.top_p=1.0")
done

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" python src/eval.py \
  experiment=eval/tofu/default.yaml \
  forget_split=${forget_split} \
  holdout_split=${holdout_split} \
  model=${model_cfg} \
  task_name=tofu_${model_cfg}_${forget_split}_sampled_eval_s${SEED} \
  model.model_args.pretrained_model_name_or_path=${model_path} \
  paths.output_dir=${out_dir} \
  retain_logs_path=saves/eval/tofu_${model_cfg}_${retain_split}/TOFU_EVAL.json \
  seed=${SEED} \
  "${gen[@]}" \
  "$@"
