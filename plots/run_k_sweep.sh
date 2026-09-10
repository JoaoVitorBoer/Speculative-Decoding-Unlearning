#!/bin/bash

#SBATCH --output=/home/joaoabitante/speculative-decoding-unlearning/plots/logs/%j__%x.out
#SBATCH --error=/home/joaoabitante/speculative-decoding-unlearning/plots/logs/%j__%x.out

#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --time=2-00:00:00
#SBATCH --gpus=1
#SBATCH --job-name=k_sweep

#
# plots/run_k_sweep.sh — COPY of scripts/run_sud_original_unlearned_draft.sh
# (Table 4.1, row 2) specialised for the k-sweep figure. Edits vs the original:
#   * one cell only: Llama-3.2-1B-Instruct / forget10 / NPO draft
#   * RESULTS_ROOT defaults to plots/results (never touches saves/)
#   * pass (a) = the k sweep  ALPHAS=(0.5 0.8 0.9) K=(1 2 4 8 16 32 64 128 256)
#                 SEEDS=(0), looped k OUTER / alpha inner so a run cut short by
#                 the walltime still leaves a complete grid up to some k.
#     pass (b) = the k=1 seed yardstick (SEEDS=(1 2 3)); already on disk, so it
#                 is no longer in the default PASSES (PASSES="a b" re-enables it).
#   * k > max_new_tokens (200) is clipped by SUD to the remaining budget each
#     round, and the draft stops at its own EOS, so k = 64/128/256 cost the same.
#   * DRY_RUN=1 prints the run dirs instead of evaluating
#   * a failed eval is logged to plots/logs/failed_runs.log and the sweep
#     continues (the original aborts under set -e); exit code is non-zero
#     if anything failed. Re-running is resume-safe (overwrite: false).
# Run from the repo root:  bash plots/run_k_sweep.sh
#
# Original header follows.
# run_sud_original_unlearned_draft.sh — Table 4.1, row 2.
#
#   P / target = ORIGINAL model  (the TOFU-finetuned "full" checkpoint; has
#                                  seen both the retain and the forget split)
#   Q / draft  = UNLEARNED draft (the original model after a weight-based
#                                 unlearning method, e.g. SimNPO / GradDiff)
#
#   The drafts are the weight-baseline checkpoints under
#   saves/unlearn/baselines/tofu/<method>/<model>/<forget_split>, written by
#   scripts/baselines/baseline_tofu_<method>.sh and
#   scripts/run_unlearn_best_models.sh;
#   the draft is always the one unlearned on the split being evaluated.
#
#   For each (target, draft, split, α, k_sud) combination, runs the TOFU eval
#   suite on the SUD draft-verify wrapper, which emits tokens from
#       π(x) ∝ p(x)^(1-α) · q(x)^α
#   (K_SUD drafted tokens per round; throughput knob only — never changes the
#   sampled distribution). k_sud is swept here to measure its speed effect, and
#   the k value is appended to the results path so runs never overwrite.
#
# ── HOW TO CONFIGURE ─────────────────────────────────────────────────────────
#   Edit the CONFIG block, e.g. the K_SUD_VALUES / ALPHAS / SEEDS / SPLITS arrays.
#     CUDA_DEVICES=2 bash scripts/run_sud_original_unlearned_draft.sh
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

# P / target family. The ORIGINAL (memorized) model is the TOFU "full"
# checkpoint derived from each family via target_path() below.
BASE_MODELS=(
   "Llama-3.2-1B-Instruct"
)

# Q / draft: the weight-unlearned checkpoints under
#   ${BASELINES_ROOT}/<method>/<base_model>/<forget_split>
# one directory per method, named by the BARE METHOD NAME (NPO, GradDiff, ...) —
# which is exactly what is listed here. The hyperparameters each was trained with
# are not in the path; they are in <ckpt>/.hydra/overrides.yaml and in the run
# name the training script logged.
# The draft is always paired with the split under evaluation, so the method is
# the only free dimension here — family and split come from the outer loops.
# Combinations with no checkpoint on disk are skipped with a warning.
BASELINES_ROOT="${BASELINES_ROOT:-saves/unlearn/baselines/tofu}"

DRAFT_METHODS=(
  # "NPO"
  "SimNPO"
)

# Extra drafts run on every split regardless of what they were unlearned on
# (hub checkpoints, ad-hoc local dirs). Normally empty: a draft unlearned on a
# split other than the one under evaluation is not a Table-4.1 row.
EXTRA_DRAFT_MODELS=(
  # "open-unlearning/unlearn_tofu_Llama-3.2-3B-Instruct_forget10_SimNPO_lr5e-05_b3.5_a1_d1_g0.25_ep5"
  # "saves/unlearn/sud/models/GradDiff_GDR/forget10/Llama-3.2-1B-Instruct/lr-1e-5_ep-10"
)

# Format: "forget_split holdout_split retain_split"
SPLITS=(
  "forget10 holdout10 retain90"
)

# Passes, each sets ALPHAS / K_SUD_VALUES / SEEDS (α=0 → target only;
# α=1 → draft only; k_sud is throughput-only and never changes the sampled
# distribution — this sweep is the empirical evidence for that; the active
# seed and k are appended to the results path).
#   a) k sweep        : model_utility / gibberish vs k at seed 0 (default)
#   b) seed yardstick : extra seeds at k=1 (seed-to-seed SD; done — opt in)
#   c) alpha sweep    : the alphas the main grid (saves/unlearn/sud/results, alphas
#                       0.5 0.7 0.8 0.9 0.95) lacks, at k=1 seed 0 — the unlearning
#                       side of plots/figs/fig_alpha_tradeoff (plots/make_alpha_figure.py);
#                       0.0 / 1.0 are SUD sampling from p / q alone, the sampled
#                       counterpart of the greedy target / draft reference evals
PASSES="${PASSES:-a}"
pass_config() {
  case "$1" in
    a) ALPHAS=(0.5 0.8 0.9); K_SUD_VALUES=(1 2 4 8 16 32 64 128 256); SEEDS=(0) ;;
    b) ALPHAS=(0.5 0.8 0.9); K_SUD_VALUES=(1);                        SEEDS=(1 2 3) ;;
    c) ALPHAS=(0.0 0.1 0.2 0.3 0.4 0.6 0.98 1.0); K_SUD_VALUES=(1);   SEEDS=(0) ;;
    *) echo "unknown pass: $1" >&2; exit 2 ;;
  esac
}

# Everything for the figure lives under plots/ — never the shared saves/ tree.
RESULTS_ROOT="${RESULTS_ROOT:-plots/results}"
FAILED_LOG="${FAILED_LOG:-plots/logs/failed_runs.log}"
DRY_RUN="${DRY_RUN:-0}"

# ORIGINAL model = the TOFU-finetuned (memorized) "full" checkpoint on the hub.
target_path() { echo "open-unlearning/tofu_${1}_full"; }

# Resolve a bare method name (e.g. "NPO") to the draft checkpoint directories
# that exist for this (family, split). The current tree has one directory per
# method, so the exact `<method>/<family>/<split>` path is the hit; the
# `<method>_*` glob is a fallback for legacy trees whose directories were still
# stamped with hyperparameters (<method>_<hparams>). The match is anchored, which
# keeps `NPO` from picking up `SimNPO`.
# Prints one path per line, nothing when the method has no checkpoint here.
resolve_draft_dirs() {
  local method="$1" family="$2" split="$3"
  local hits=() candidate
  shopt -s nullglob
  for candidate in "${BASELINES_ROOT}/${method}/${family}/${split}" \
                   "${BASELINES_ROOT}/${method}"_*/"${family}/${split}"; do
    [[ -f "${candidate}/config.json" ]] || continue
    # config.json alone is NOT enough: a training job killed during the final
    # save leaves config.json plus a partial shard set behind (this is what
    # happened to PDU/Llama-3.2-3B-Instruct/forget10). Loading such a directory
    # raises, and `set -e` would take the whole sweep down with it, so require
    # the weights too: model.safetensors when unsharded, or the shard index
    # (written last) when sharded.
    if [[ ! -f "${candidate}/model.safetensors" \
       && ! -f "${candidate}/model.safetensors.index.json" ]]; then
      echo -e "${RED}Skipping ${candidate}: incomplete checkpoint (no model.safetensors or shard index) — retrain it${NC}" >&2
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

run_sweep() {
for split_entry in "${SPLITS[@]}"; do
  read -r forget_split holdout_split retain_split <<< "${split_entry}"
  echo -e "${RED}--- Split: forget=${forget_split} | holdout=${holdout_split} | retain=${retain_split} ---${NC}"

  for base_model in "${BASE_MODELS[@]}"; do
    target="$(target_path "${base_model}")"
    retain_logs_path="saves/eval/tofu_${base_model}_${retain_split}/TOFU_EVAL.json"
    echo -e "${RED}target (original) = ${target}${NC}"

    # Drafts for this (family, split): one weight-baseline checkpoint per
    # method, plus any split-agnostic extras.
    draft_models=()
    for method in "${DRAFT_METHODS[@]}"; do
      mapfile -t hits < <(resolve_draft_dirs "${method}" "${base_model}" "${forget_split}")
      if [[ ${#hits[@]} -eq 0 ]]; then
        echo -e "${RED}Skipping draft ${method} | ${base_model} | ${forget_split}: no checkpoint under ${BASELINES_ROOT}/${method}*/${base_model}/${forget_split}${NC}"
        continue
      fi
      if [[ ${#hits[@]} -gt 1 ]]; then
        echo -e "${RED}Note: ${method} matches ${#hits[@]} configs — running all of: ${hits[*]}${NC}"
      fi
      draft_models+=("${hits[@]}")
    done
    if [[ ${#EXTRA_DRAFT_MODELS[@]} -gt 0 ]]; then
      draft_models+=("${EXTRA_DRAFT_MODELS[@]}")
    fi

    if [[ ${#draft_models[@]} -eq 0 ]]; then
      echo -e "${RED}No drafts available for ${base_model} | ${forget_split}${NC}"
      continue
    fi

    for draft_model in "${draft_models[@]}"; do
      # Same short tag the results path uses (e.g. Llama-3.2-3B-Instruct_NPO_GDR),
      # so the run label stays readable for local checkpoint paths.
      draft_tag="$(python scripts/sud_paths.py tag "${draft_model}")"

      # k OUTER so the expensive large-k cells arrive grid-row by grid-row.
      for k in "${K_SUD_VALUES[@]}"; do
        for alpha in "${ALPHAS[@]}"; do
          for seed in "${SEEDS[@]}"; do
            task_name="tofu_${base_model}_${forget_split}_sud_${draft_tag}_alpha-${alpha}_k-${k}_seed-${seed}"
            output_dir="$(python scripts/sud_paths.py resolve \
              --root "${RESULTS_ROOT}" --combo "${COMBO}" \
              --target "${target}" --draft "${draft_model}" \
              --split "${forget_split}" --alpha "${alpha}" --k "${k}" --seed "${seed}")"

            echo
            echo -e "${RED}=== ${base_model} | ${forget_split} | draft=${draft_model} | α=${alpha} | k=${k} | seed=${seed} ===${NC}"
            echo -e "${RED}eval → ${output_dir}${NC}"

            if [[ "${DRY_RUN}" == "1" ]]; then
              echo "[dry-run] task_name=${task_name}"
              continue
            fi


            ## eval.tofu.batch_size=4 \ 
            # device_map=auto spreads the model weights over both GPUs.
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
            || {
              # Keep going: one crashed cell must not take the other 23 down.
              # Re-running this script later resumes only the missing metrics.
              N_FAILED=$((N_FAILED + 1))
              echo "$(date -Is) FAILED alpha=${alpha} k=${k} seed=${seed} → ${output_dir}" | tee -a "${FAILED_LOG}" >&2
            }
          done
        done
      done
    done
  done
done
}

for pass in ${PASSES}; do
  pass_config "${pass}"
  echo -e "${RED}##### pass ${pass}: ALPHAS=(${ALPHAS[*]}) K_SUD_VALUES=(${K_SUD_VALUES[*]}) SEEDS=(${SEEDS[*]}) #####${NC}"
  run_sweep
done

if [[ ${N_FAILED} -gt 0 ]]; then
  echo -e "${RED}${N_FAILED} run(s) failed — see ${FAILED_LOG}; re-run this script to resume them${NC}" >&2
  exit 1
fi
echo "all runs finished"
