#!/bin/bash

#
# run_fetch_hub.sh — Table 4.1, row 2, with drafts pulled from the HF Hub.
#
#   Mirrors scripts/run_sud_original_unlearned_draft.sh exactly:
#
#   P / target = ORIGINAL model  (the TOFU-finetuned "full" checkpoint; has
#                                  seen both the retain and the forget split)
#   Q / draft  = UNLEARNED draft (the original model after a weight-based
#                                 unlearning method, e.g. SimNPO / GradDiff)
#
#   The only difference: the draft checkpoints are not expected on disk. They
#   are the public Hub repos pushed by scripts/push_baselines_to_hub.py,
#       <HUB_NAMESPACE>/tofu_<model>_<forget_split>_<method>
#   and are downloaded on demand into the SAME local layout the original
#   script reads,
#       ${BASELINES_ROOT}/<method>/<model>/<forget_split>
#   so scripts/sud_paths.py produces identical draft tags and results paths,
#   and a checkpoint already on disk (from training or an earlier fetch) is
#   used as-is without touching the network.
#
#   For each (target, draft, split, α, k_sud) combination, runs the TOFU eval
#   suite on the SUD draft-verify wrapper, which emits tokens from
#       π(x) ∝ p(x)^(1-α) · q(x)^α
#   (K_SUD drafted tokens per round; throughput knob only — never changes the
#   sampled distribution). k_sud is swept here to measure its speed effect, and
#   the k value is appended to the results path so runs never overwrite.
#
# ── HOW TO CONFIGURE ─────────────────────────────────────────────────────────
#   Edit the CONFIG block, e.g. the K_SUD_VALUES / ALPHAS / SEEDS / SPLITS arrays,
#   then run it directly (no SLURM — single-GPU server):
#     conda activate unlearning
#     bash scripts/run_fetch_hub.sh
#   CUDA_DEVICES defaults to 0 (the only GPU). Requires `hf` (huggingface_hub
#   CLI) on PATH — it is in the `unlearning` env. The repos are public, so no
#   token is needed to download them.
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
   "Llama-3.2-3B-Instruct"
  # "Llama-3.1-8B-Instruct"
)

# Q / draft: the weight-unlearned checkpoints, one Hub repo per
# (method, base_model, forget_split), downloaded into
#   ${BASELINES_ROOT}/<method>/<base_model>/<forget_split>
# and named by the BARE METHOD NAME (NPO, GradDiff, ...) — which is exactly what
# is listed here. The hyperparameters each was trained with are not in the path;
# they are in <ckpt>/.hydra/overrides.yaml (shipped in every repo).
# The draft is always paired with the split under evaluation, so the method is
# the only free dimension here — family and split come from the outer loops.
# Combinations with no repo on the Hub are skipped with a warning.
BASELINES_ROOT="${BASELINES_ROOT:-saves/unlearn/baselines/tofu}"
HUB_NAMESPACE="${HUB_NAMESPACE:-JoaoBoer}"

DRAFT_METHODS=(
  "GradDiff"
  "IdkDPO"
  "IdkNLL"
  "NPO"
  "RMU"
  "SatImp"
  "WGA"
  "PDU"
  "SimNPO"
  "UNDIAL"
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
  "forget01 holdout01 retain99"
  "forget05 holdout05 retain95"
  "forget10 holdout10 retain90"
)

# Blend strengths to sweep. α=0 → target only; α=1 → draft only.
# ALPHAS=(0.5 0.7 0.8 0.9 0.95)
# ALPHAS=(0.9)
ALPHAS=(0.1 0.2 0.3 0.4 0.6 0.98 0.99 0.995)
# Draft window sizes to sweep (throughput only; never changes the distribution).
K_SUD_VALUES=(1)

# Random seeds to sweep. The active seed is appended to the results path.
SEEDS=(0)

RESULTS_ROOT="${RESULTS_ROOT:-saves/unlearn/sud/results}"

# ORIGINAL model = the TOFU-finetuned (memorized) "full" checkpoint on the hub.
target_path() { echo "open-unlearning/tofu_${1}_full"; }

# Hub repo id of the weight-baseline draft for (method, family, split); the
# naming is open-unlearning's, produced by scripts/push_baselines_to_hub.py.
hub_repo() { echo "${HUB_NAMESPACE}/tofu_${2}_${3}_${1}"; }

# True when the model repo exists on the Hub (public repos need no token).
hub_repo_exists() {
  python -c 'import sys; from huggingface_hub import repo_exists; sys.exit(0 if repo_exists(sys.argv[1]) else 1)' "$1" 2>/dev/null
}

# A local checkpoint directory is usable when it holds config.json AND the
# weights: model.safetensors when unsharded, or the shard index (written last)
# when sharded. config.json alone is NOT enough — an interrupted save or
# download leaves config.json plus a partial shard set behind, and loading such
# a directory raises, which under `set -e` takes the whole sweep down.
is_complete_ckpt() {
  local d="$1"
  [[ -f "${d}/config.json" ]] \
    && { [[ -f "${d}/model.safetensors" ]] || [[ -f "${d}/model.safetensors.index.json" ]]; }
}

# Resolve a bare method name (e.g. "NPO") to the draft checkpoint directory for
# this (family, split), fetching it from the Hub into ${BASELINES_ROOT} when it
# is not already complete on disk. `hf download` is incremental: re-running
# after an interrupted download only fetches the missing files.
# Prints the local path, nothing when the method has no repo on the Hub.
# Progress / diagnostics go to stderr so stdout stays a clean path.
resolve_draft_dirs() {
  local method="$1" family="$2" split="$3"
  local local_dir="${BASELINES_ROOT}/${method}/${family}/${split}"
  local repo
  repo="$(hub_repo "${method}" "${family}" "${split}")"

  if is_complete_ckpt "${local_dir}"; then
    printf '%s\n' "${local_dir}"
    return 0
  fi

  if ! hub_repo_exists "${repo}"; then
    echo -e "${RED}Skipping draft ${method} | ${family} | ${split}: no repo ${repo} on the Hub${NC}" >&2
    return 0
  fi

  echo -e "${RED}Fetching ${repo} → ${local_dir}${NC}" >&2
  mkdir -p "${local_dir}"
  if ! hf download "${repo}" --local-dir "${local_dir}" >&2; then
    echo -e "${RED}Skipping ${repo}: download failed${NC}" >&2
    return 0
  fi
  if ! is_complete_ckpt "${local_dir}"; then
    echo -e "${RED}Skipping ${local_dir}: incomplete checkpoint after download (no model.safetensors or shard index)${NC}" >&2
    return 0
  fi
  printf '%s\n' "${local_dir}"
  return 0
}

# ── DRIVER ────────────────────────────────────────────────────────────────────

mkdir -p "${RESULTS_ROOT}"
command -v hf >/dev/null || { echo -e "${RED}hf CLI not found — activate the unlearning env (pip install -U huggingface_hub)${NC}" >&2; exit 1; }

for split_entry in "${SPLITS[@]}"; do
  read -r forget_split holdout_split retain_split <<< "${split_entry}"
  echo -e "${RED}--- Split: forget=${forget_split} | holdout=${holdout_split} | retain=${retain_split} ---${NC}"

  for base_model in "${BASE_MODELS[@]}"; do
    target="$(target_path "${base_model}")"
    retain_logs_path="saves/eval/tofu_${base_model}_${retain_split}/TOFU_EVAL.json"
    echo -e "${RED}target (original) = ${target}${NC}"

    # Drafts for this (family, split): one weight-baseline checkpoint per
    # method (fetched from the Hub if needed), plus any split-agnostic extras.
    draft_models=()
    for method in "${DRAFT_METHODS[@]}"; do
      mapfile -t hits < <(resolve_draft_dirs "${method}" "${base_model}" "${forget_split}")
      if [[ ${#hits[@]} -eq 0 ]]; then
        echo -e "${RED}Skipping draft ${method} | ${base_model} | ${forget_split}: no checkpoint at $(hub_repo "${method}" "${base_model}" "${forget_split}") or ${BASELINES_ROOT}/${method}/${base_model}/${forget_split}${NC}"
        continue
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
      # Same short tag the results path uses (e.g. Llama-3.2-3B-Instruct_NPO),
      # so the run label stays readable for local checkpoint paths.
      draft_tag="$(python scripts/sud_paths.py tag "${draft_model}")"

      for alpha in "${ALPHAS[@]}"; do
        for k in "${K_SUD_VALUES[@]}"; do
          for seed in "${SEEDS[@]}"; do
            task_name="tofu_${base_model}_${forget_split}_sud_${draft_tag}_alpha-${alpha}_k-${k}_seed-${seed}"
            output_dir="$(python scripts/sud_paths.py resolve \
              --root "${RESULTS_ROOT}" --combo "${COMBO}" \
              --target "${target}" --draft "${draft_model}" \
              --split "${forget_split}" --alpha "${alpha}" --k "${k}" --seed "${seed}")"

            echo
            echo -e "${RED}=== ${base_model} | ${forget_split} | draft=${draft_model} | α=${alpha} | k=${k} | seed=${seed} ===${NC}"
            echo -e "${RED}eval → ${output_dir}${NC}"

            # device_map=auto places both models on the single visible GPU.
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
              paths.output_dir="${output_dir}"
          done
        done
      done
    done
  done
done
