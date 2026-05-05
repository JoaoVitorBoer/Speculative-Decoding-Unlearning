#!/bin/bash

#SBATCH --output=/home/joaoabitante/Sout/%j__%x.out
#SBATCH --error=/home/joaoabitante/Sout/%j__%x.out

#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=30G
#SBATCH --time=2-00:00:00
#SBATCH --gpus=1

#
# run_sud_baselines.sh — TOFU standard evaluation baselines (no SUD).
#
#   For each (target_model, split) combination, runs the TOFU eval suite
#   directly on the target model. No draft model, no α blending — this is
#   the reference baseline that SUD runs are compared against.
#
# ── HOW TO CONFIGURE ─────────────────────────────────────────────────────────
#   Edit the CONFIG block below. Comment/uncomment entries in TARGET_MODELS
#   and SPLITS to run any subset.
#
#   Any array or scalar can also be overridden at call time, e.g.
#     CUDA_DEVICES=2 bash scripts/run_sud_baselines.sh
#     TARGET_MODELS=(Llama-3.2-1B-Instruct) bash scripts/run_sud_baselines.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
export HYDRA_FULL_ERROR=1

RED='\e[31m'
NC='\e[0m'

# ── CONFIG ────────────────────────────────────────────────────────────────────

CUDA_DEVICES="${CUDA_DEVICES:-0}"

TARGET_MODELS=(
  "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr5e-05_b3.5_a1_d1_g0.25_ep5"
  # "Llama-3.2-1B-Instruct"
  # "Llama-3.2-3B-Instruct"
  # "Llama-3.1-8B-Instruct"
)

# Format: "forget_split holdout_split retain_split"
SPLITS=(
  # "forget01 holdout01 retain99"
  # "forget05 holdout05 retain95"
  "forget10 holdout10 retain90"
)

RESULTS_ROOT="${RESULTS_ROOT:-saves/unlearn/sud/results/baselines}"

# Target = the TOFU-finetuned (memorized) model on the HF hub.
target_path() { echo "open-unlearning/tofu_${1}_full"; }

# ── DRIVER ────────────────────────────────────────────────────────────────────

mkdir -p "${RESULTS_ROOT}"

for split_entry in "${SPLITS[@]}"; do
  read -r forget_split holdout_split retain_split <<< "${split_entry}"
  echo -e "${RED}--- Split: forget=${forget_split} | holdout=${holdout_split} | retain=${retain_split} ---${NC}"

  for target_model in "${TARGET_MODELS[@]}"; do
    target="$(target_path "${target_model}")"
    retain_logs_path="saves/eval/tofu_${target_model}_${retain_split}/TOFU_EVAL.json"

    task_name="tofu_${target_model}_${forget_split}_baseline"
    output_dir="${RESULTS_ROOT}/target-${target//\//_}/${forget_split}"
    mkdir -p "${output_dir}"

    echo
    echo -e "${RED}=== baseline | ${target_model} | ${forget_split} ===${NC}"
    echo -e "${RED}target = ${target}${NC}"
    echo -e "${RED}eval  → ${output_dir}${NC}"

    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" python src/eval.py \
      experiment=eval/tofu/default.yaml \
      model="${target_model}" \
      model.model_args.pretrained_model_name_or_path="${target}" \
      forget_split="${forget_split}" \
      holdout_split="${holdout_split}" \
      retain_logs_path="${retain_logs_path}" \
      task_name="${task_name}" \
      paths.output_dir="${output_dir}"
  done
done
