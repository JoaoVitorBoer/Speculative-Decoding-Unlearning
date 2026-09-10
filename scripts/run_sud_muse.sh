#!/bin/bash

#SBATCH --output=/home/joaoabitante/Sout/%j__%x.out
#SBATCH --error=/home/joaoabitante/Sout/%j__%x.out

#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --gpus=1
#SBATCH --job-name=sud_muse

#
# run_sud_muse.sh — MUSE counterpart of scripts/run_sud_original_unlearned_draft.sh.
#
#   P / target = ORIGINAL model  (muse-bench/MUSE-<split>_target, the checkpoint
#                                  that memorized both the forget and retain sets)
#   Q / draft  = UNLEARNED draft (that target after a weight-based unlearning
#                                 method, e.g. SimNPO / NPO / GradDiff)
#
#   The drafts are the checkpoints saved by scripts/run_muse_weight_baselines.sh
#   under saves/unlearn/baselines/muse/<method>/<model>/<data_split>; the draft
#   is always the one unlearned on the data split being evaluated.
#
#   For each (target, draft, data_split, α, k_sud) combination, runs the MUSE
#   eval suite on the SUD draft-verify wrapper, which emits tokens from
#       π(x) ∝ p(x)^(1-α) · q(x)^α
#   (K_SUD drafted tokens per round; throughput knob only — never changes the
#   sampled distribution). k_sud is swept here to measure its speed effect, and
#   the k value is appended to the results path so runs never overwrite.
#
#   MUSE models are Llama-2 base (non-chat) checkpoints, so this uses the
#   `sud_muse` model config — `sud` plus the Llama-2 "Question:/Answer:" prompt
#   template. Do not swap it for `sud`: those Llama-3 chat tags (including the
#   system header, which is prepended even with chat templating off) would be
#   injected into every MUSE prompt.
#
#   GPU: target + draft are both 7B and both land on the same device
#   (model_args.device_map=cuda), i.e. ~28 GB of weights — needs a ≥40 GB GPU.
#   To shard them instead, request 2 GPUs and add
#     model.model_args.device_map=auto
#   to the eval invocation below.
#
# ── HOW TO CONFIGURE ─────────────────────────────────────────────────────────
#   Edit the CONFIG block, e.g. the K_SUD_VALUES / ALPHAS / SEEDS / DATA_SPLITS
#   arrays.
#     CUDA_DEVICES=2 bash scripts/run_sud_muse.sh
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

# P / target family. MUSE ships a single family; the ORIGINAL (memorized) model
# is the per-split target checkpoint derived via target_path() below.
BASE_MODELS=(
  "Llama-2-7b-hf"
)

# MUSE corpora. Each has its own target / retrain checkpoints and eval data.
DATA_SPLITS=(
  "News"
  "Books"
)

# Q / draft: the weight-unlearned checkpoints trained by
# scripts/run_muse_weight_baselines.sh, which saves them under
#   ${BASELINES_ROOT}/<method>/<base_model>/<data_split>
# The draft is always paired with the split under evaluation, so <method> is the
# only free dimension here — family and split come from the outer loops.
# Combinations with no checkpoint on disk are skipped with a warning.
BASELINES_ROOT="${BASELINES_ROOT:-saves/unlearn/baselines/muse}"

DRAFT_METHODS=(
  "GradDiff_GDR"
  "NPO_GDR"
  "SimNPO_GDR"
)

# Extra drafts run on every split regardless of what they were unlearned on
# (hub checkpoints, ad-hoc local dirs). Normally empty: a draft unlearned on a
# split other than the one under evaluation is not a Table-4.1 row, and the
# retain / generic drafts belong to other COMBO folders.
EXTRA_DRAFT_MODELS=(
  # "muse-bench/MUSE-News_retrain"      # COMBO=original-retain
  # "meta-llama/Llama-2-7b-hf"          # COMBO=original-generic
)

# Blend strengths to sweep. α=0 → target only; α=1 → draft only.
# ALPHAS=(0.0 0.1 0.3 0.5 0.7 0.8 0.9 0.95 1.0)
ALPHAS=(0.0 1.0)

# Draft window sizes to sweep (throughput only; never changes the distribution).
K_SUD_VALUES=(1)

# Random seeds to sweep. The active seed is appended to the results path.
SEEDS=(0)

RESULTS_ROOT="${RESULTS_ROOT:-saves/unlearn/sud/muse/results}"

# ORIGINAL model = the MUSE memorized target checkpoint on the hub.
target_path() { echo "muse-bench/MUSE-${1}_target"; }

# ── DRIVER ────────────────────────────────────────────────────────────────────

mkdir -p "${RESULTS_ROOT}"

for data_split in "${DATA_SPLITS[@]}"; do
  echo -e "${RED}--- Data split: ${data_split} ---${NC}"
  target="$(target_path "${data_split}")"

  for base_model in "${BASE_MODELS[@]}"; do
    retain_logs_path="saves/eval/muse_${base_model}_${data_split}_retrain/MUSE_EVAL.json"
    echo -e "${RED}target (original) = ${target}${NC}"

    # Drafts for this (family, split): one weight-baseline checkpoint per
    # method, plus any split-agnostic extras.
    draft_models=()
    for method in "${DRAFT_METHODS[@]}"; do
      candidate="${BASELINES_ROOT}/${method}/${base_model}/${data_split}"
      if [[ -f "${candidate}/config.json" ]]; then
        draft_models+=("${candidate}")
      else
        echo -e "${RED}Skipping draft ${method} | ${base_model} | ${data_split}: no checkpoint at ${candidate}${NC}"
      fi
    done
    if [[ ${#EXTRA_DRAFT_MODELS[@]} -gt 0 ]]; then
      draft_models+=("${EXTRA_DRAFT_MODELS[@]}")
    fi

    if [[ ${#draft_models[@]} -eq 0 ]]; then
      echo -e "${RED}No drafts available for ${base_model} | ${data_split}${NC}"
      continue
    fi

    for draft_model in "${draft_models[@]}"; do
      # Same short tag the results path uses (e.g. Llama-2-7b-hf_NPO_GDR), so
      # the run label stays readable for local checkpoint paths.
      draft_tag="$(python scripts/sud_paths.py tag "${draft_model}")"

      for alpha in "${ALPHAS[@]}"; do
        for k in "${K_SUD_VALUES[@]}"; do
          for seed in "${SEEDS[@]}"; do
            task_name="muse_${base_model}_${data_split}_sud_${draft_tag}_alpha-${alpha}_k-${k}_seed-${seed}"
            output_dir="$(python scripts/sud_paths.py resolve \
              --root "${RESULTS_ROOT}" --combo "${COMBO}" \
              --target "${target}" --draft "${draft_model}" \
              --split "${data_split}" --alpha "${alpha}" --k "${k}" --seed "${seed}")"

            echo
            echo -e "${RED}=== ${base_model} | ${data_split} | draft=${draft_model} | α=${alpha} | k=${k} | seed=${seed} ===${NC}"
            echo -e "${RED}eval → ${output_dir}${NC}"

            CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" python src/eval.py \
              experiment=eval/muse/default.yaml \
              model=sud_muse \
              model.model_args.pretrained_model_name_or_path="${target}" \
              model.model_args.draft_model_name_or_path="${draft_model}" \
              model.model_args.alpha="${alpha}" \
              model.model_args.k_sud="${k}" \
              model.tokenizer_args.pretrained_model_name_or_path="${target}" \
              seed="${seed}" \
              data_split="${data_split}" \
              retain_logs_path="${retain_logs_path}" \
              task_name="${task_name}" \
              paths.output_dir="${output_dir}"
          done
        done
      done
    done
  done
done
