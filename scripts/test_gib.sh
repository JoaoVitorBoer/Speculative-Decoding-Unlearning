#!/bin/bash

#SBATCH --output=/home/joaoabitante/Sout/%j__%x.out
#SBATCH --error=/home/joaoabitante/Sout/%j__%x.out

#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=15G
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1
#SBATCH --job-name=test_gib

#
# test_gib.sh — gibberish-only evaluation of the selected SUD operating points.
#
#   Same Table-4.1 row, drafts, splits and α operating points as
#   scripts/seed_sud_run.sh (ALPHA_MAP / DRAFT_METHODS are copied verbatim from
#   there — keep the two in sync):
#
#     P / target = ORIGINAL model  (open-unlearning/tofu_<family>_full)
#     Q / draft  = UNLEARNED draft (saves/unlearn/baselines/tofu/<method>/
#                                    <family>/<forget_split>)
#
#   Two differences. WHAT is evaluated: only `forget_Q_A_gibberish` (the
#   gibberish classifier's P(clean) on the SUD generations for the forget
#   questions; its forget_Q_A_ROUGE pre_compute does the generation). Every
#   other TOFU metric is removed from the eval config with Hydra `~` deletes,
#   so one cell costs one generation pass instead of the full suite.
#
#   And the draft window k is SWEPT (K_SUD_VALUES; the seed script is k=1
#   only): the same operating point is generated with k drafted tokens per
#   round for each k in the list. k is a throughput knob that never changes the
#   sampled distribution, so the gibberish score should be flat in k — this is
#   the check. k is looped OUTER, so a run cut short by the walltime still
#   leaves every operating point complete up to some k. k is part of the leaf
#   name (a<alpha>_k<k>_s<seed>), so windows never overwrite each other; the
#   k=1 scores already exist in the main sweep tree (seed 0).
#
#   Results go to a SEPARATE root (RESULTS_ROOT, default
#   saves/unlearn/sud/results/test_gib) so the main sweep / seed trees are never
#   touched. Re-running is resume-safe (overwrite: false skips finished
#   metrics); set OVERWRITE=1 to force a recompute of the gibberish score.
#
# ── HOW TO CONFIGURE ─────────────────────────────────────────────────────────
#   Edit the CONFIG block (ALPHA_MAP / SEEDS / SPLITS / DRAFT_METHODS), e.g.
#     CUDA_DEVICES=2 bash scripts/test_gib.sh
#     DRY_RUN=1 bash scripts/test_gib.sh        # print the cells, run nothing
#     OVERWRITE=1 bash scripts/test_gib.sh      # recompute existing scores
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
export HYDRA_FULL_ERROR=1

RED='\e[31m'
NC='\e[0m'

# ── CONFIG ────────────────────────────────────────────────────────────────────

CUDA_DEVICES="${CUDA_DEVICES:-0}"

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
  "GradDiff"
  "IdkDPO"
  "IdkNLL"
  "NPO"
  "RMU"
  "SimNPO"
  "UNDIAL"
  # "SatImp"
  # "WGA"
  # "PDU"
)

# Format: "forget_split holdout_split retain_split"
SPLITS=(
  "forget01 holdout01 retain99"
  "forget05 holdout05 retain95"
  "forget10 holdout10 retain90"
)

# Selected blend strength per "<model_short>;<method>;<forget_split>".
# Space-separated values run as separate points. Missing keys are skipped.
# (Copied from scripts/seed_sud_run.sh.)
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

# Seeds to run. This is a metric test, not the variance study, so it defaults
# to the sweep seed (s0 — the one the α choices above were made at). Set e.g.
# SEEDS=(0 1 2 3 4) to cover the seed replicates too.
SEEDS=(0)

# Draft window sizes to sweep (throughput only; never changes the distribution).
# k > max_new_tokens (200) is clipped by SUD to the remaining budget each round.
K_SUD_VALUES=(2 4 32 64)

# Separate root: never writes into the main sweep / seed results trees.
RESULTS_ROOT="${RESULTS_ROOT:-saves/unlearn/sud/results/test_gib}"
FAILED_LOG="${FAILED_LOG:-${RESULTS_ROOT}/failed_runs.log}"
DRY_RUN="${DRY_RUN:-0}"
OVERWRITE="${OVERWRITE:-0}"

# The one metric to keep. Everything else listed in configs/eval/tofu.yaml is
# deleted from the composed config (Hydra `~key` override), so the evaluator
# only runs the gibberish classifier and its forget_Q_A_ROUGE generation
# pre_compute. Keep this list in sync with configs/eval/tofu.yaml.
KEEP_METRIC="forget_Q_A_gibberish"
ALL_METRICS=(
  forget_quality
  model_utility
  exact_memorization
  extraction_strength
  forget_Q_A_PARA_Prob
  forget_truth_ratio
  forget_Q_A_gibberish
  privleak
  mia_min_k_plus_plus
  mia_min_k
  mia_loss
  mia_zlib
)
METRIC_OVERRIDES=()
for m in "${ALL_METRICS[@]}"; do
  [[ "${m}" == "${KEEP_METRIC}" ]] || METRIC_OVERRIDES+=("~eval.tofu.metrics.${m}")
done
if [[ "${OVERWRITE}" == "1" ]]; then
  METRIC_OVERRIDES+=("eval.tofu.overwrite=true")
fi

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
# which keeps `NPO` from picking up `SimNPO`. A directory needs its weights too
# (model.safetensors or the shard index), not just config.json — a job killed
# mid-save leaves config.json behind and loading it raises.
resolve_draft_dirs() {
  local method="$1" family="$2" split="$3"
  local hits=() candidate
  shopt -s nullglob
  for candidate in "${BASELINES_ROOT}/${method}/${family}/${split}" \
                   "${BASELINES_ROOT}/${method}"_*/"${family}/${split}"; do
    [[ -f "${candidate}/config.json" ]] || continue
    if [[ ! -f "${candidate}/model.safetensors" \
       && ! -f "${candidate}/model.safetensors.index.json" ]]; then
      echo -e "${RED}Skipping ${candidate}: incomplete checkpoint (no model.safetensors or shard index)${NC}" >&2
      continue
    fi
    hits+=("${candidate}")
  done
  if [[ ${#hits[@]} -gt 0 ]]; then
    printf '%s\n' "${hits[@]}"
  fi
  return 0
}

# ── DRIVER ────────────────────────────────────────────────────────────────────

mkdir -p "${RESULTS_ROOT}" "$(dirname "${FAILED_LOG}")"
N_FAILED=0
N_RUN=0

echo -e "${RED}##### test_gib: metric=${KEEP_METRIC} SEEDS=(${SEEDS[*]}) K_SUD_VALUES=(${K_SUD_VALUES[*]}) RESULTS_ROOT=${RESULTS_ROOT} OVERWRITE=${OVERWRITE} #####${NC}"

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

        # k OUTER so the expensive large-k cells arrive grid-row by grid-row.
        for k in "${K_SUD_VALUES[@]}"; do
        for alpha in ${alphas}; do
          for seed in "${SEEDS[@]}"; do
            task_name="tofu_${base_model}_${forget_split}_sud_${draft_tag}_alpha-${alpha}_k-${k}_seed-${seed}_gib"
            output_dir="$(python scripts/sud_paths.py resolve \
              --root "${RESULTS_ROOT}" --combo "${COMBO}" \
              --target "${target}" --draft "${draft_model}" \
              --split "${forget_split}" --alpha "${alpha}" --k "${k}" --seed "${seed}")"

            echo
            echo -e "${RED}=== ${base_model} | ${forget_split} | draft=${draft_model} | α=${alpha} | k=${k} | seed=${seed} | metric=${KEEP_METRIC} ===${NC}"
            echo -e "${RED}eval → ${output_dir}${NC}"
            N_RUN=$((N_RUN + 1))

            if [[ "${DRY_RUN}" == "1" ]]; then
              echo "[dry-run] task_name=${task_name}"
              continue
            fi

            CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" python src/eval.py \
              experiment=eval/tofu/default.yaml \
              model=sud \
              model.model_args.device_map=auto \
              model.model_args.pretrained_model_name_or_path="${target}" \
              model.model_args.draft_model_name_or_path="${draft_model}" \
              model.model_args.alpha="${alpha}" \
              model.model_args.k_sud="${k}" \
              model.tokenizer_args.pretrained_model_name_or_path="${target}" \
              seed="${seed}" \
              forget_split="${forget_split}" \
              holdout_split="${holdout_split}" \
              retain_logs_path="${retain_logs_path}" \
              task_name="${task_name}" \
              paths.output_dir="${output_dir}" \
              "${METRIC_OVERRIDES[@]}" \
            || {
              # Keep going: one crashed cell must not take the rest down.
              # Re-running this script later resumes only the missing cells.
              N_FAILED=$((N_FAILED + 1))
              echo "$(date -Is) FAILED ${base_model} ${method} ${forget_split} alpha=${alpha} seed=${seed} → ${output_dir}" | tee -a "${FAILED_LOG}" >&2
            }
          done
        done
        done
      done
    done
  done
done

echo
if [[ ${N_FAILED} -gt 0 ]]; then
  echo -e "${RED}${N_FAILED}/${N_RUN} run(s) failed — see ${FAILED_LOG}; re-run this script to resume them${NC}" >&2
  exit 1
fi
echo "all ${N_RUN} gibberish evals finished → ${RESULTS_ROOT}"
