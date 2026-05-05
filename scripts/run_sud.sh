#!/bin/bash

#SBATCH --output=/home/joaoabitante/Sout/%j__%x.out
#SBATCH --error=/home/joaoabitante/Sout/%j__%x.out

#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=30G
#SBATCH --time=2-00:00:00
#SBATCH --gpus=1

#
# run_sud.sh — TOFU evaluation of SUD (Speculative Unlearning Decoding,
#              explicit-π) blended TARGET_MODELS.
#
#   For each (target_model, split, draft_model, α) combination, runs the
#   TOFU eval suite on the SUD wrapper:
#       π(x) ∝ p(x)^(1-α) · q(x)^α
#   where p is the target (memorized) target_model and q is a draft (unlearned)
#   target_model. SUD is inference-only, so no training hyperparameters are used here.
#
# ── HOW TO CONFIGURE ─────────────────────────────────────────────────────────
#   Edit the CONFIG block below. Comment/uncomment entries in TARGET_MODELS,
#   DRAFT_TARGET_MODELS, SPLITS, and ALPHAS to run any subset.
#
#   Any array or scalar can also be overridden at call time, e.g.
#     CUDA_DEVICES=2 ALPHAS=(0.25 0.5 0.75) bash scripts/run_sud.sh
#     TARGET_MODELS=(Llama-3.2-1B-Instruct) DRAFT_TARGET_MODELS=(GA_GDR) bash scripts/run_sud.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
export HYDRA_FULL_ERROR=1

RED='\e[31m'
NC='\e[0m'

# ── CONFIG ────────────────────────────────────────────────────────────────────

CUDA_DEVICES="${CUDA_DEVICES:-0}"

TARGET_MODELS=(
  # "Llama-3.2-1B-Instruct"
  # "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr5e-05_b3.5_a1_d1_g0.25_ep5"
  "Llama-3.2-3B-Instruct"
  "Llama-3.1-8B-Instruct"
)

# Draft TARGET_MODELS to evaluate against.
DRAFT_TARGET_MODELS=(
  # "Llama-3.2-1B-Instruct"
  # "meta-llama/Llama-3.2-1B-Instruct"
  # open-unlearning/tofu_Llama-3.2-1B-Instruct_full
 open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr5e-05_b3.5_a1_d1_g0.25_ep5
)

# Format: "forget_split holdout_split retain_split"
SPLITS=(
  # "forget01 holdout01 retain99"
  # "forget05 holdout05 retain95"
  "forget10 holdout10 retain90"
)

# Blend strengths to sweep. α=0 → target only; α=1 → draft only.
ALPHAS=(0.5 0.7 0.9)

RESULTS_ROOT="${RESULTS_ROOT:-saves/unlearn/sud/results}"

# Target = the TOFU-finetuned (memorized) target_model on the HF hub.
target_path() { echo "open-unlearning/tofu_${1}_full"; }
target_model_for_path_name() { echo "open-unlearning_tofu_${1}_full";}
# ── DRIVER ────────────────────────────────────────────────────────────────────

mkdir -p "${RESULTS_ROOT}"

for split_entry in "${SPLITS[@]}"; do
  read -r forget_split holdout_split retain_split <<< "${split_entry}"
  echo -e "${RED}--- Split: forget=${forget_split} | holdout=${holdout_split} | retain=${retain_split} ---${NC}"

  for target_model in "${TARGET_MODELS[@]}"; do
    target="$(target_path "${target_model}")"
    retain_logs_path="saves/eval/tofu_${target_model}_${retain_split}/TOFU_EVAL.json"
    echo -e "${RED}target_model: ${target_model} | target=${target}${NC}"

    for draft_model in "${DRAFT_TARGET_MODELS[@]}"; do
      for alpha in "${ALPHAS[@]}"; do
        task_name="tofu_${target_model}_${forget_split}_sud_${draft_model}_alpha-${alpha}"
        # output_dir="${RESULTS_ROOT}/${draft_model}/${forget_split}/${target_model}/alpha-${alpha}"
        output_dir="${RESULTS_ROOT}/target-${target//\//_}/draft-${draft_model//\//_}/${forget_split}/alpha-${alpha}"
        mkdir -p "${output_dir}"

        echo
        echo -e "${RED}=== ${target_model} | ${forget_split} | ${draft_model} | α=${alpha} ===${NC}"
        echo -e "${RED}draft = ${draft_model}${NC}"
        echo -e "${RED}eval  → ${output_dir}${NC}"

        CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" python src/eval.py \
          experiment=eval/tofu/default.yaml \
          model=sud \
          model.model_args.pretrained_model_name_or_path="${target}" \
          model.model_args.draft_model_name_or_path="${draft_model}" \
          model.model_args.alpha="${alpha}" \
          model.tokenizer_args.pretrained_model_name_or_path="${target}" \
          forget_split="${forget_split}" \
          holdout_split="${holdout_split}" \
          retain_logs_path="${retain_logs_path}" \
          task_name="${task_name}" \
          paths.output_dir="${output_dir}"
      done
    done
  done
done
