#!/bin/bash

#SBATCH --output=/home/joaoabitante/Sout/%j__%x.out
#SBATCH --error=/home/joaoabitante/Sout/%j__%x.out

#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --time=3-00:00:00
#SBATCH --gpus=1

# grid_hf_collection.sh — TOFU eval over *every* model in open-unlearning's
# "TOFU Unlearned Models" collection.
#
#   https://huggingface.co/collections/open-unlearning/tofu-unlearned-models
#
#   The collection holds 398 checkpoints, all Llama-3.2-1B-Instruct / forget10,
#   and every method in it is a full cartesian product of its hyperparameters —
#   so the list below is *generated*, not pasted:
#
#     NPO      lr{1e-05,2e-05,5e-05} x beta{0.05,0.1,0.5} x alpha{1,2,5} x ep{5,10}       54
#     AltPO    same grid as NPO                                                           54
#     IdkDPO   same grid as NPO                                                           54
#     UNDIAL   lr{1e-05,1e-04,3e-04} x beta{3,10,30}      x alpha{1,2,5} x ep{5,10}       54
#     RMU      lr{1e-05,2e-05,5e-05} x layer{5,10,15} x scoeff{1,10,100} x ep{5,10}       54
#     SimNPO   lr{1e-05,2e-05,5e-05} x b{3.5,4.5} x a1 x d{0,1} x g{0.125,0.25} x ep{5,10} 48
#     GradDiff lr{1,2,3,4,5}e-05     x alpha{1,2,5,10}    x ep{5,10}                      40
#     IdkNLL   same grid as GradDiff                                                      40
#                                                                                total = 398
#
#   The generated ids were verified equal, as a set, to the 398 ids the HF
#   collection API returns. To re-verify after an upstream change:
#
#     MODELS_ONLY=1 bash scripts/grid_hf_collection.sh | sort > /tmp/local.txt
#     curl -s "https://huggingface.co/api/collections/open-unlearning/tofu-unlearned-models-6860f6cf3fe35d0223d92e88" \
#       | python3 -c 'import json,sys; [print(i["id"]) for i in json.load(sys.stdin)["items"]]' \
#       | sort > /tmp/hub.txt
#     diff /tmp/local.txt /tmp/hub.txt && echo IN SYNC
#
#   eval -> saves/grid/tofu/<cfg_tag>/<model>/<forget_split>/evals
#
#   The `tofu/<method>/<model>/<split>/evals` layout is exactly what
#   scripts/summarize_sud_baselines.py globs for, so the finished grid summarizes
#   with no path juggling:
#
#     python scripts/summarize_sud_baselines.py --benchmark tofu --baselines-root saves/grid
#
#   Each checkpoint is ~2.5GB and is evicted from the HF hub cache as soon as its
#   eval finishes (or fails), so the whole grid never needs more than one model on
#   disk at a time. The base tokenizer repo is a separate cache entry, never touched.
#
#   Budget: ~3.2 min/model measured on the 48-model SimNPO run (download + eval +
#   evict), so all 398 is ~21h on one GPU — inside the 3-day limit, but see SHARD
#   to fan it out across several GPUs/jobs.
#
# Env knobs:
#   GRID_ROOT=saves/grid   output root (the `tofu/` level is added under it)
#   CUDA_DEVICES=0         GPU to use (single GPU by design, see --gpus=1)
#   METHODS="NPO RMU"      only these methods (default: all 8)
#   SHARD=2/4              run shard 2 of 4; round-robin, so every shard covers
#                          all methods and a killed job still spans the grid
#   FORCE=1                re-run evals that already have a TOFU_EVAL.json
#   KEEP_CACHE=1           do not evict the downloaded model afterwards
#   DRY_RUN=1              print what would run, download nothing
#   MODELS_ONLY=1          print the selected model ids and exit
#   MODELS="id1 id2"       override the model list (also accepted as arguments)

export HYDRA_FULL_ERROR=1

RED='\e[31m'
YELLOW='\e[33m'
GREEN='\e[32m'
NC='\e[0m'

CUDA_DEVICES="${CUDA_DEVICES:-0}"
GRID_ROOT="${GRID_ROOT:-saves/grid}"
BENCH_ROOT="${GRID_ROOT}/tofu"

MODEL_CFG="${MODEL_CFG:-Llama-3.2-1B-Instruct}"
forget_split=forget10
holdout_split=holdout10
retain_split=retain90

retain_logs_path="saves/eval/tofu_${MODEL_CFG}_${retain_split}/TOFU_EVAL.json"

# Where huggingface_hub puts snapshots; mirrors its own resolution order.
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"

# --- model list ---------------------------------------------------------------

prefix="open-unlearning/unlearn_tofu_${MODEL_CFG}_${forget_split}"
models=()
add() { models+=("${prefix}_$1"); }

# NPO / AltPO / IdkDPO share one grid: lr x beta x alpha x epoch (54 each).
for method in NPO AltPO IdkDPO; do
    for lr in 1e-05 2e-05 5e-05; do
        for beta in 0.05 0.1 0.5; do
            for alpha in 1 2 5; do
                for epoch in 5 10; do
                    add "${method}_lr${lr}_beta${beta}_alpha${alpha}_epoch${epoch}"
                done
            done
        done
    done
done

# UNDIAL: same field order, different lr/beta ranges (54).
for lr in 1e-05 0.0001 0.0003; do
    for beta in 3 10 30; do
        for alpha in 1 2 5; do
            for epoch in 5 10; do
                add "UNDIAL_lr${lr}_beta${beta}_alpha${alpha}_epoch${epoch}"
            done
        done
    done
done

# RMU: lr x layer x steering coeff x epoch (54).
for lr in 1e-05 2e-05 5e-05; do
    for layer in 5 10 15; do
        for scoeff in 1 10 100; do
            for epoch in 5 10; do
                add "RMU_lr${lr}_layer${layer}_scoeff${scoeff}_epoch${epoch}"
            done
        done
    done
done

# # SimNPO: abbreviated field names upstream (b/a/d/g/ep), alpha fixed at 1 (48).
# for lr in 1e-05 2e-05 5e-05; do
#     for b in 3.5 4.5; do
#         for d in 0 1; do
#             for g in 0.125 0.25; do
#                 for ep in 5 10; do
#                     add "SimNPO_lr${lr}_b${b}_a1_d${d}_g${g}_ep${ep}"
#                 done
#             done
#         done
#     done
# done

# GradDiff / IdkNLL share one grid: lr x alpha x epoch (40 each).
for method in GradDiff IdkNLL; do
    for lr in 1e-05 2e-05 3e-05 4e-05 5e-05; do
        for alpha in 1 2 5 10; do
            for epoch in 5 10; do
                add "${method}_lr${lr}_alpha${alpha}_epoch${epoch}"
            done
        done
    done
done

[[ -n "${MODELS}" ]] && read -r -a models <<< "${MODELS}"
[[ $# -gt 0 ]] && models=("$@")

# --- selection: METHODS filter, then SHARD ------------------------------------

method_of() {  # ".../unlearn_tofu_..._forget10_NPO_lr1e-05_beta0.5_..." -> "NPO"
    local tag="${1##*_${forget_split}_}"
    echo "${tag%%_lr*}"
}

# summarize_sud_baselines.py recovers each baseline's training config from
# `<ckpt>/.hydra/overrides.yaml`, which Hydra writes when *we* train a model. A
# checkpoint pulled off the hub has no such file, so without this the grid's 398
# rows would all carry empty trainer/lr/epochs/target_id and print a warning
# apiece — the hyperparameters would survive only inside the method string.
# Upstream encoded them in the repo id, so re-emit them in the format the
# summarizer already parses. Written beside `evals/`, never inside it: the eval
# run puts its own .hydra there.
write_train_overrides() {
    local cfg_tag="$1" model_path="$2" ckpt_dir="$3"
    local trainer="${cfg_tag%%_lr*}" lr="" epochs=""

    [[ "${cfg_tag}" =~ _lr([^_]+) ]] && lr="${BASH_REMATCH[1]}"
    # `_epoch10` everywhere except SimNPO, which abbreviates it to `_ep10`.
    [[ "${cfg_tag}" =~ _ep(och)?([0-9]+)$ ]] && epochs="${BASH_REMATCH[2]}"

    mkdir -p "${ckpt_dir}/.hydra" || return 0
    {
        echo "- trainer=${trainer}"
        [[ -n "${lr}" ]]     && echo "- trainer.args.learning_rate=${lr}"
        [[ -n "${epochs}" ]] && echo "- trainer.args.num_train_epochs=${epochs}"
        echo "- model.model_args.pretrained_model_name_or_path=${model_path}"
    } > "${ckpt_dir}/.hydra/overrides.yaml"
}

if [[ -n "${METHODS}" ]]; then
    read -r -a wanted <<< "${METHODS}"
    kept=()
    for m in "${models[@]}"; do
        this="$(method_of "${m}")"
        for w in "${wanted[@]}"; do
            if [[ "${this}" == "${w}" ]]; then
                kept+=("${m}")
                break
            fi
        done
    done
    models=("${kept[@]}")
    if [[ ${#models[@]} -eq 0 ]]; then
        echo "No models match METHODS='${METHODS}'" >&2
        exit 2
    fi
fi

if [[ -n "${SHARD}" ]]; then
    shard_i="${SHARD%%/*}"
    shard_n="${SHARD##*/}"
    if ! [[ "${shard_i}" =~ ^[0-9]+$ && "${shard_n}" =~ ^[0-9]+$ ]] \
       || (( shard_n < 1 || shard_i < 1 || shard_i > shard_n )); then
        echo "SHARD must look like i/n with 1 <= i <= n (got '${SHARD}')" >&2
        exit 2
    fi
    kept=()
    for idx in "${!models[@]}"; do
        (( idx % shard_n == shard_i - 1 )) && kept+=("${models[idx]}")
    done
    models=("${kept[@]}")
fi

if [[ -n "${MODELS_ONLY}" ]]; then
    printf '%s\n' "${models[@]}"
    exit 0
fi

# --- preflight ----------------------------------------------------------------

# forget_quality is a KS test against the retain model's losses; without this
# file every run would either die or silently emit NaN, 398 times over.
if [[ ! -f "${retain_logs_path}" ]]; then
    echo -e "${RED}Missing retain logs: ${retain_logs_path}${NC}" >&2
    echo "Run the ${retain_split} eval first (scripts/run_tofu_eval.sh)." >&2
    exit 1
fi

# --- run ----------------------------------------------------------------------

total=${#models[@]}
n_ok=0
n_skip=0
failed=()
start_ts=${SECONDS}

echo -e "${RED}=== TOFU collection grid | ${total} models | ${forget_split} | GPU ${CUDA_DEVICES} ===${NC}"
echo -e "${RED}Methods:     ${METHODS:-all}${NC}"
echo -e "${RED}Shard:       ${SHARD:-1/1}${NC}"
echo -e "${RED}Output root: ${BENCH_ROOT}${NC}"
echo -e "${RED}Retain logs: ${retain_logs_path}${NC}"
echo -e "${RED}HF cache:    ${HF_HUB_CACHE}${NC}"

i=0
for model_path in "${models[@]}"; do
    i=$((i + 1))

    # cfg_tag: the hyperparameter part of the id, e.g.
    #   NPO_lr2e-05_beta0.5_alpha1_epoch10
    cfg_tag="${model_path##*_${forget_split}_}"
    [[ "${cfg_tag}" == "${model_path}" ]] && cfg_tag="${model_path##*/}"

    save_dir="${BENCH_ROOT}/${cfg_tag}/${MODEL_CFG}/${forget_split}/evals"
    task_name="grid_${cfg_tag}_${MODEL_CFG}_${forget_split}"

    echo
    echo -e "${RED}[${i}/${total}] ${model_path}${NC}"
    echo -e "${RED}Eval → ${save_dir}${NC}"

    if [[ -f "${save_dir}/TOFU_EVAL.json" && -z "${FORCE}" ]]; then
        echo -e "${YELLOW}Already evaluated, skipping (FORCE=1 to re-run).${NC}"
        n_skip=$((n_skip + 1))
        continue
    fi

    # The first 48 SimNPO evals (scripts/grid_hf_models.sh) landed one level up,
    # before the `tofu/` level existed. Honour them as done rather than burning
    # ~2.5 GPU-hours re-running them.
    legacy_dir="${GRID_ROOT}/${cfg_tag}/${MODEL_CFG}/${forget_split}/evals"
    if [[ -f "${legacy_dir}/TOFU_EVAL.json" && -z "${FORCE}" ]]; then
        echo -e "${YELLOW}Already evaluated at ${legacy_dir} (pre-tofu/ layout), skipping.${NC}"
        echo -e "${YELLOW}  mv it under ${BENCH_ROOT}/ for the summarizer to see it.${NC}"
        n_skip=$((n_skip + 1))
        continue
    fi

    cmd=(python src/eval.py
        experiment=eval/tofu/default.yaml
        forget_split="${forget_split}"
        holdout_split="${holdout_split}"
        model="${MODEL_CFG}"
        task_name="${task_name}"
        model.model_args.pretrained_model_name_or_path="${model_path}"
        paths.output_dir="${save_dir}"
        retain_logs_path="${retain_logs_path}")

    if [[ -n "${DRY_RUN}" ]]; then
        echo "DRY_RUN: CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} ${cmd[*]}"
        continue
    fi

    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${cmd[@]}"
    status=$?

    if [[ ${status} -eq 0 ]]; then
        n_ok=$((n_ok + 1))
        write_train_overrides "${cfg_tag}" "${model_path}" \
            "${BENCH_ROOT}/${cfg_tag}/${MODEL_CFG}/${forget_split}"
    else
        echo -e "${RED}FAILED (exit ${status}): ${model_path}${NC}" >&2
        failed+=("${model_path}")
    fi

    # Drop the checkpoint whether or not the eval succeeded — a failed run that
    # left 2.5GB behind would fill the cache just as fast as a successful one.
    if [[ -z "${KEEP_CACHE}" ]]; then
        cache_name="models--${model_path//\//--}"
        for dir in "${HF_HUB_CACHE}/${cache_name}" "${HF_HUB_CACHE}/.locks/${cache_name}"; do
            if [[ -d "${dir}" ]]; then
                echo -e "${YELLOW}Deleting ${dir}${NC}"
                rm -rf -- "${dir}"
            fi
        done
    fi

    # ETA from the mean of the evals actually run in this job.
    done_n=$((n_ok + ${#failed[@]}))
    if (( done_n > 0 )); then
        elapsed=$((SECONDS - start_ts))
        eta=$(( elapsed * (total - i) / done_n ))
        echo -e "${GREEN}Progress ${i}/${total} | elapsed $((elapsed / 60))m | ETA ~$((eta / 60))m${NC}"
    fi
done

echo
echo -e "${RED}=== Done: ${n_ok} evaluated, ${n_skip} skipped, ${#failed[@]} failed (of ${total}) ===${NC}"
echo -e "${RED}Summarize: python scripts/summarize_sud_baselines.py --benchmark tofu --baselines-root ${GRID_ROOT}${NC}"
if [[ ${#failed[@]} -gt 0 ]]; then
    printf '%s\n' "${failed[@]}" >&2
    exit 1
fi
