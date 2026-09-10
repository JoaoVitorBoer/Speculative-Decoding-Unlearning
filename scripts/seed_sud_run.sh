#!/bin/bash

#SBATCH --output=/home/joaoabitante/Sout/%j__%x.out
#SBATCH --error=/home/joaoabitante/Sout/%j__%x.out

#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=15G
#SBATCH --time=2-00:00:00
#SBATCH --gpus=1
#SBATCH --job-name=sud_seeds

#
# seed_sud_run.sh — seed replication of the selected SUD operating points.
#
#   Same Table-4.1 row as scripts/run_sud_original_unlearned_draft.sh:
#
#     P / target = ORIGINAL model  (open-unlearning/tofu_<family>_full)
#     Q / draft  = UNLEARNED draft (saves/unlearn/baselines/tofu/<method>/
#                                    <family>/<forget_split>)
#
#   The difference is that this script does NOT sweep α. Each (method, split)
#   already has a chosen operating point from the α sweep; this re-runs exactly
#   those points under several seeds so the reported numbers carry a variance
#   estimate. The α values come from ALPHA_MAP below, keyed by
#   "<model_short>;<method>;<forget_split>" — same convention as EPOCH_MAP in
#   scripts/run_tofu_weight_baselines.sh. Combinations absent from the map are
#   skipped. A map entry may list more than one α (space-separated): where the
#   sweep left two points equally defensible, both are replicated.
#
#   The seed is part of the results leaf name (a<alpha>_k<k>_s<seed>), so seeds
#   never overwrite each other and the existing s0 sweep runs stay as the first
#   replicate.
#
# ── HOW TO CONFIGURE ─────────────────────────────────────────────────────────
#   Edit the CONFIG block (ALPHA_MAP / SEEDS / SPLITS / DRAFT_METHODS), e.g.
#     CUDA_DEVICES=2 bash scripts/seed_sud_run.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
export HYDRA_FULL_ERROR=1

RED='\e[31m'
NC='\e[0m'

# ── CONFIG ────────────────────────────────────────────────────────────────────

CUDA_DEVICES="${CUDA_DEVICES:-0 1}"

# Table-4.1 combination this script runs (fixed). Names the top-level results
# folder so runs are grouped by row — see scripts/sud_paths.py.
COMBO="original-unlearned"

BASE_MODELS=(
  "Llama-3.2-1B-Instruct"
  "Llama-3.2-3B-Instruct"
  # "Llama-3.1-8B-Instruct"
)

# Q / draft: the weight-unlearned checkpoints under
#   ${BASELINES_ROOT}/<method>/<base_model>/<forget_split>
# named by the BARE METHOD NAME, which is what is listed here. The draft is
# always paired with the split under evaluation. Combinations with no checkpoint
# on disk are skipped with a warning.
BASELINES_ROOT="${BASELINES_ROOT:-saves/unlearn/baselines/tofu}"

DRAFT_METHODS=(
  # "GradDiff"
  # "IdkDPO"
  # "IdkNLL"
  # "NPO"
  # "RMU"
  # "SimNPO"
  # "UNDIAL"
  "SatImp"
  "WGA"
  "PDU"
)

# Format: "forget_split holdout_split retain_split"
SPLITS=(
  "forget01 holdout01 retain99"
  "forget05 holdout05 retain95"
  "forget10 holdout10 retain90"
)

# Selected blend strength per "<model_short>;<method>;<forget_split>".
# Space-separated values run as separate points. Missing keys are skipped.
declare -A ALPHA_MAP
# 1B
ALPHA_MAP["1B;GradDiff;forget01"]="0.95"
ALPHA_MAP["1B;GradDiff;forget05"]="0.7"
ALPHA_MAP["1B;GradDiff;forget10"]="0.5"
ALPHA_MAP["1B;IdkDPO;forget01"]="0.9"
ALPHA_MAP["1B;IdkDPO;forget05"]="0.95"
ALPHA_MAP["1B;IdkDPO;forget10"]="0.95"
ALPHA_MAP["1B;IdkNLL;forget01"]="0.9"
ALPHA_MAP["1B;IdkNLL;forget05"]="0.95"
ALPHA_MAP["1B;IdkNLL;forget10"]="0.95"
ALPHA_MAP["1B;NPO;forget01"]="0.9 0.95"
ALPHA_MAP["1B;NPO;forget05"]="0.9 0.8"
ALPHA_MAP["1B;NPO;forget10"]="0.9"
ALPHA_MAP["1B;RMU;forget01"]="0.9"
ALPHA_MAP["1B;RMU;forget05"]="0.95"
ALPHA_MAP["1B;RMU;forget10"]="0.9"
ALPHA_MAP["1B;SimNPO;forget01"]="0.95"
ALPHA_MAP["1B;SimNPO;forget05"]="0.9"
ALPHA_MAP["1B;SimNPO;forget10"]="0.95"
ALPHA_MAP["1B;UNDIAL;forget01"]="0.7"
ALPHA_MAP["1B;UNDIAL;forget05"]="0.95"
ALPHA_MAP["1B;UNDIAL;forget10"]="0.95"
# 3B
ALPHA_MAP["3B;GradDiff;forget01"]="0.8"
ALPHA_MAP["3B;GradDiff;forget05"]="0.9"
ALPHA_MAP["3B;GradDiff;forget10"]="0.5"
ALPHA_MAP["3B;IdkDPO;forget01"]="0.9"
ALPHA_MAP["3B;IdkDPO;forget05"]="0.9"
ALPHA_MAP["3B;IdkDPO;forget10"]="0.95"
ALPHA_MAP["3B;IdkNLL;forget01"]="0.8"
ALPHA_MAP["3B;IdkNLL;forget05"]="0.95"
ALPHA_MAP["3B;IdkNLL;forget10"]="0.95"
ALPHA_MAP["3B;NPO;forget01"]="0.9"
ALPHA_MAP["3B;NPO;forget05"]="0.8 0.9"
ALPHA_MAP["3B;NPO;forget10"]="0.9"
ALPHA_MAP["3B;RMU;forget01"]="0.95"
ALPHA_MAP["3B;RMU;forget05"]="0.9"
ALPHA_MAP["3B;RMU;forget10"]="0.9"
ALPHA_MAP["3B;SimNPO;forget01"]="0.95"
ALPHA_MAP["3B;SimNPO;forget05"]="0.9"
ALPHA_MAP["3B;SimNPO;forget10"]="0.9"
ALPHA_MAP["3B;UNDIAL;forget01"]="0.8"
ALPHA_MAP["3B;UNDIAL;forget05"]="0.95"
ALPHA_MAP["3B;UNDIAL;forget10"]="0.95"

# Seeds to sweep. The active seed is appended to the results path. s0 is the
# sweep run that produced the α choices above, so it is not repeated here.
SEEDS=(1 2 3 4)

# Draft window size (throughput only; never changes the distribution).
K_SUD=1

RESULTS_ROOT="${RESULTS_ROOT:-saves/unlearn/sud/results}"

# ORIGINAL model = the TOFU-finetuned (memorized) "full" checkpoint on the hub.
target_path() { echo "open-unlearning/tofu_${1}_full"; }

# ALPHA_MAP is keyed by the model size alone (1B/3B/8B), not the full family.
model_short() {
  [[ "$1" =~ ([0-9]+B) ]] && echo "${BASH_REMATCH[1]}" || echo "$1"
}

# Resolve a bare method name (e.g. "NPO") to the draft checkpoint directories
# that exist for this (family, split). The exact `<method>/<family>/<split>` path
# is the hit; the `<method>_*` glob is a fallback for legacy trees whose
# directories were still stamped with hyperparameters. The match is anchored,
# which keeps `NPO` from picking up `SimNPO`.
resolve_draft_dirs() {
  local method="$1" family="$2" split="$3"
  local hits=() candidate
  shopt -s nullglob
  for candidate in "${BASELINES_ROOT}/${method}/${family}/${split}" \
                   "${BASELINES_ROOT}/${method}"_*/"${family}/${split}"; do
    [[ -f "${candidate}/config.json" ]] && hits+=("${candidate}")
  done
  if [[ ${#hits[@]} -gt 0 ]]; then
    printf '%s\n' "${hits[@]}"
  fi
  return 0
}

# ── DRIVER ────────────────────────────────────────────────────────────────────

mkdir -p "${RESULTS_ROOT}"

for split_entry in "${SPLITS[@]}"; do
  read -r forget_split holdout_split retain_split <<< "${split_entry}"
  echo -e "${RED}--- Split: forget=${forget_split} | holdout=${holdout_split} | retain=${retain_split} ---${NC}"

  for base_model in "${BASE_MODELS[@]}"; do
    target="$(target_path "${base_model}")"
    retain_logs_path="saves/eval/tofu_${base_model}_${retain_split}/TOFU_EVAL.json"
    short="$(model_short "${base_model}")"
    echo -e "${RED}target (original) = ${target}${NC}"

    for method in "${DRAFT_METHODS[@]}"; do
      alphas="${ALPHA_MAP["${short};${method};${forget_split}"]:-}"
      if [[ -z "${alphas}" ]]; then
        echo -e "${RED}Skipping ${method} | ${base_model} | ${forget_split}: no ALPHA_MAP entry${NC}"
        continue
      fi

      mapfile -t draft_models < <(resolve_draft_dirs "${method}" "${base_model}" "${forget_split}")
      if [[ ${#draft_models[@]} -eq 0 ]]; then
        echo -e "${RED}Skipping draft ${method} | ${base_model} | ${forget_split}: no checkpoint under ${BASELINES_ROOT}/${method}*/${base_model}/${forget_split}${NC}"
        continue
      fi

      for draft_model in "${draft_models[@]}"; do
        # Same short tag the results path uses (e.g. Llama-3.2-1B-Instruct_NPO),
        # so the run label stays readable for local checkpoint paths.
        draft_tag="$(python scripts/sud_paths.py tag "${draft_model}")"

        for alpha in ${alphas}; do
          for seed in "${SEEDS[@]}"; do
            task_name="tofu_${base_model}_${forget_split}_sud_${draft_tag}_alpha-${alpha}_k-${K_SUD}_seed-${seed}"
            output_dir="$(python scripts/sud_paths.py resolve \
              --root "${RESULTS_ROOT}" --combo "${COMBO}" \
              --target "${target}" --draft "${draft_model}" \
              --split "${forget_split}" --alpha "${alpha}" --k "${K_SUD}" --seed "${seed}")"

            echo
            echo -e "${RED}=== ${base_model} | ${forget_split} | draft=${draft_model} | α=${alpha} | k=${K_SUD} | seed=${seed} ===${NC}"
            echo -e "${RED}eval → ${output_dir}${NC}"

            CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" python src/eval.py \
              experiment=eval/tofu/default.yaml \
              model=sud \
              model.model_args.device_map=auto \
              model.model_args.pretrained_model_name_or_path="${target}" \
              model.model_args.draft_model_name_or_path="${draft_model}" \
              model.model_args.alpha="${alpha}" \
              model.model_args.k_sud="${K_SUD}" \
              model.tokenizer_args.pretrained_model_name_or_path="${target}" \
              seed="${seed}" \
              forget_split="${forget_split}" \
              holdout_split="${holdout_split}" \
              retain_logs_path="${retain_logs_path}" \
              task_name="${task_name}" \
              paths.output_dir="${output_dir}"
          done
        done
      done
    done
  done
done
