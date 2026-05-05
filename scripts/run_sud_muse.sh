#!/bin/bash

#SBATCH --output=/home/joaoabitante/Sout/%j__%x.out
#SBATCH --error=/home/joaoabitante/Sout/%j__%x.out

#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=30G
#SBATCH --time=2-00:00:00
#SBATCH --gpus=2

#
# run_sud_muse.sh — MUSE evaluation of SUD (Speculative Unlearning Decoding,
#                   explicit-π) blended TARGET_MODELS.
#
#   For each (target_model, data_split, draft_model, α) combination, runs the
#   MUSE eval suite on the SUD wrapper:
#       π(x) ∝ p(x)^(1-α) · q(x)^α
#   where p is the target (memorized) target_model and q is a draft (unlearned)
#   target_model. SUD is inference-only, so no training hyperparameters are used here.
#
# ── HOW TO CONFIGURE ─────────────────────────────────────────────────────────
#   Edit the CONFIG block below. Comment/uncomment entries in TARGET_MODELS,
#   DRAFT_TARGET_MODELS, DATA_SPLITS, and ALPHAS to run any subset.
#
#   Any array or scalar can also be overridden at call time, e.g.
#     CUDA_DEVICES=2 ALPHAS=(0.25 0.5 0.75) bash scripts/run_sud_muse.sh
#     TARGET_MODELS=(Llama-2-7b-hf) DATA_SPLITS=(News) bash scripts/run_sud_muse.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
export HYDRA_FULL_ERROR=1

RED='\e[31m'
NC='\e[0m'

# ── CONFIG ────────────────────────────────────────────────────────────────────

CUDA_DEVICES="${CUDA_DEVICES:-0}"

TARGET_MODELS=(
  "Llama-2-7b-hf"
)

# Draft TARGET_MODELS to evaluate against.
DRAFT_TARGET_MODELS=(
  "meta-llama/Llama-2-7b-hf"
)

DATA_SPLITS=(
  "Books"
  "News"
)

# Blend strengths to sweep. α=0 → target only; α=1 → draft only.
ALPHAS=(0.5 0.9 0.2)

RESULTS_ROOT="${RESULTS_ROOT:-saves/unlearn/sud/muse/results}"

# Target = the MUSE memorized target_model on the HF hub for the given data_split.
target_path() { echo "muse-bench/MUSE-${1}_target"; }

# ── DRIVER ────────────────────────────────────────────────────────────────────

mkdir -p "${RESULTS_ROOT}"

for data_split in "${DATA_SPLITS[@]}"; do
  echo -e "${RED}--- Data split: ${data_split} ---${NC}"
  target="$(target_path "${data_split}")"

  for target_model in "${TARGET_MODELS[@]}"; do
    retain_logs_path="saves/eval/muse_${target_model}_${data_split}_retrain/MUSE_EVAL.json"
    echo -e "${RED}target_model: ${target_model} | target=${target}${NC}"

    for draft_model in "${DRAFT_TARGET_MODELS[@]}"; do
      for alpha in "${ALPHAS[@]}"; do
        task_name="muse_${target_model}_${data_split}_sud_${draft_model}_alpha-${alpha}"
        output_dir="${RESULTS_ROOT}/${draft_model}/${data_split}/${target_model}/alpha-${alpha}"
        mkdir -p "${output_dir}"

        echo
        echo -e "${RED}=== ${target_model} | ${data_split} | ${draft_model} | α=${alpha} ===${NC}"
        echo -e "${RED}draft = ${draft_model}${NC}"
        echo -e "${RED}eval  → ${output_dir}${NC}"

        CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" python src/eval.py \
          experiment=eval/muse/default.yaml \
          model=sud \
          model.model_args.pretrained_model_name_or_path="${target}" \
          model.model_args.draft_model_name_or_path="${draft_model}" \
          model.model_args.alpha="${alpha}" \
          model.tokenizer_args.pretrained_model_name_or_path="${target}" \
          data_split="${data_split}" \
          retain_logs_path="${retain_logs_path}" \
          task_name="${task_name}" \
          paths.output_dir="${output_dir}"
      done
    done
  done
done
