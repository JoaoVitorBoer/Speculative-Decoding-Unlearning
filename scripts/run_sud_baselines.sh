#!/bin/bash

#SBATCH --output=/home/joaoabitante/Sout/%j__%x.out
#SBATCH --error=/home/joaoabitante/Sout/%j__%x.out

#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
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

# Each entry is either:
#   "target_model"                     — base model name auto-extracted
#   "base_model_name target_model"     — explicit base model override
# The base model name is used as the Hydra model config key and to locate
# retain logs (saves/eval/tofu_<base_model>_<retain_split>/TOFU_EVAL.json).
TARGET_MODELS=(
  # "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain99"
  # "open-unlearning/tofu_Llama-3.2-3B-Instruct_retain99"
  # "open-unlearning/tofu_Llama-3.1-8B-Instruct_retain99"

  # "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain95"
  # "open-unlearning/tofu_Llama-3.2-3B-Instruct_retain95"
  # "open-unlearning/tofu_Llama-3.1-8B-Instruct_retain95"
  
  "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90"
  "open-unlearning/tofu_Llama-3.2-3B-Instruct_retain90"
  "open-unlearning/tofu_Llama-3.1-8B-Instruct_retain90"
  # "Llama-3.2-1B-Instruct"
  # "Llama-3.2-3B-Instruct"
  # "Llama-3.1-8B-Instruct"
  # "Llama-3.2-3B-Instruct open-unlearning/some-custom-model"
)

# Format: "forget_split holdout_split retain_split"
SPLITS=(
  # "forget01 holdout01 retain99"
  # "forget05 holdout05 retain95"
  "forget10 holdout10 retain90"
)

RESULTS_ROOT="${RESULTS_ROOT:-saves/unlearn/sud/results/baselines}"

repo_basename() { echo "${1##*/}"; }
path_slug() { echo "${1//\//_}"; }

base_model_name() {
  local name
  name="$(repo_basename "$1")"

  if [[ "${name}" == unlearn_tofu_*_forget* ]]; then
    name="${name#unlearn_tofu_}"
    echo "${name%%_forget*}"
  elif [[ "${name}" == tofu_*_full ]]; then
    name="${name#tofu_}"
    echo "${name%_full}"
  elif [[ "${name}" == tofu_*_retain* ]]; then
    name="${name#tofu_}"
    echo "${name%_retain*}"
  else
    echo "${name}"
  fi
}

target_path() {
  if [[ "$1" == */* ]]; then
    echo "$1"
  else
    echo "open-unlearning/tofu_${1}_full"
  fi
}

# ── DRIVER ────────────────────────────────────────────────────────────────────

mkdir -p "${RESULTS_ROOT}"

for split_entry in "${SPLITS[@]}"; do
  read -r forget_split holdout_split retain_split <<< "${split_entry}"
  echo -e "${RED}--- Split: forget=${forget_split} | holdout=${holdout_split} | retain=${retain_split} ---${NC}"

  for entry in "${TARGET_MODELS[@]}"; do
    read -r _part1 _part2 <<< "${entry}"
    if [[ -n "${_part2}" ]]; then
      model_name="${_part1}"
      target_model="${_part2}"
    else
      target_model="${_part1}"
      model_name="$(base_model_name "${target_model}")"
    fi
    target="$(target_path "${target_model}")"
    target_slug="$(path_slug "${target}")"
    retain_logs_path="saves/eval/tofu_${model_name}_${retain_split}/TOFU_EVAL.json"

    task_name="tofu_$(path_slug "${target_model}")_${forget_split}_baseline"
    output_dir="${RESULTS_ROOT}/${target_slug}/${forget_split}"
    mkdir -p "${output_dir}"

    echo
    echo -e "${RED}=== baseline | ${target_model} | ${forget_split} ===${NC}"
    echo -e "${RED}model config = ${model_name}${NC}"
    echo -e "${RED}target = ${target}${NC}"
    echo -e "${RED}retain logs = ${retain_logs_path}${NC}"
    echo -e "${RED}eval  → ${output_dir}${NC}"

    if [[ ! -f "${retain_logs_path}" ]]; then
      echo -e "${RED}Missing retain logs: ${retain_logs_path}${NC}" >&2
      echo -e "${RED}Create them first by evaluating the retain model for ${model_name} / ${retain_split}.${NC}" >&2
      exit 1
    fi

    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" python src/eval.py \
      experiment=eval/tofu/default.yaml \
      model="${model_name}" \
      model.model_args.pretrained_model_name_or_path="${target}" \
      forget_split="${forget_split}" \
      holdout_split="${holdout_split}" \
      retain_logs_path="${retain_logs_path}" \
      task_name="${task_name}" \
      paths.output_dir="${output_dir}"
  done
done
