#!/bin/bash

#SBATCH --output=/home/joaoabitante/Sout/%j__%x.out
#SBATCH --error=/home/joaoabitante/Sout/%j__%x.out

#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --time=3-00:00:00
#SBATCH --gpus=1

# grid_hf_models.sh — TOFU eval over every SimNPO model in open-unlearning's
# "TOFU Unlearned Models" collection.
#
#   https://huggingface.co/collections/open-unlearning/tofu-unlearned-models
#
#   The collection ships 48 SimNPO checkpoints, all Llama-3.2-1B-Instruct /
#   forget10, forming the full grid
#       lr in {1e-05, 2e-05, 5e-05}
#       b  in {3.5, 4.5}          (beta)
#       a  = 1                    (alpha, fixed)
#       d  in {0, 1}              (delta)
#       g  in {0.125, 0.25}       (gamma)
#       ep in {5, 10}
#   3*2*2*2*2 = 48, which is exactly the SimNPO count in the collection, so the
#   list below is generated rather than pasted.
#
#   eval -> saves/grid/<cfg_tag>/<model>/<forget_split>/evals
#
#   That <method>/<model>/<split>/evals layout is the one
#   scripts/summarize_sud_baselines.py globs for, so the grid can be summarized
#   with BASELINES_ROOT pointed at saves/grid.
#
#   Each checkpoint is ~2.5GB and is deleted from the HF hub cache as soon as
#   its eval finishes (or fails), so the whole grid never needs more than one
#   model on disk at a time.
#
# Env knobs:
#   GRID_ROOT=saves/grid   output root
#   CUDA_DEVICES=0         GPU to use (single GPU by design, see --gpus=1)
#   FORCE=1                re-run evals that already have a TOFU_EVAL.json
#   KEEP_CACHE=1           do not delete the downloaded model afterwards
#   DRY_RUN=1              print what would run, download nothing
#   MODELS="id1 id2"       override the model list (also accepted as arguments)

export HYDRA_FULL_ERROR=1

RED='\e[31m'
YELLOW='\e[33m'
NC='\e[0m'

CUDA_DEVICES="${CUDA_DEVICES:-0}"
GRID_ROOT="${GRID_ROOT:-saves/grid}"

MODEL_CFG="${MODEL_CFG:-Llama-3.2-1B-Instruct}"
forget_split=forget10
holdout_split=holdout10
retain_split=retain90

# Where huggingface_hub puts snapshots; mirrors its own resolution order.
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"

# --- model list ---------------------------------------------------------------

models=()
for lr in 1e-05 2e-05 5e-05; do
    for b in 3.5 4.5; do
        for d in 0 1; do
            for g in 0.125 0.25; do
                for ep in 5 10; do
                    models+=("open-unlearning/unlearn_tofu_${MODEL_CFG}_${forget_split}_SimNPO_lr${lr}_b${b}_a1_d${d}_g${g}_ep${ep}")
                done
            done
        done
    done
done

[[ -n "${MODELS}" ]] && read -r -a models <<< "${MODELS}"
[[ $# -gt 0 ]] && models=("$@")

# --- run ----------------------------------------------------------------------

total=${#models[@]}
n_ok=0
n_skip=0
failed=()

echo -e "${RED}=== SimNPO grid | ${total} models | ${forget_split} | GPU ${CUDA_DEVICES} ===${NC}"
echo -e "${RED}Output root: ${GRID_ROOT}${NC}"
echo -e "${RED}HF cache:    ${HF_HUB_CACHE}${NC}"

i=0
for model_path in "${models[@]}"; do
    i=$((i + 1))

    # cfg_tag: the hyperparameter part of the id, e.g.
    #   SimNPO_lr5e-05_b3.5_a1_d1_g0.25_ep5
    cfg_tag="${model_path##*_${forget_split}_}"
    [[ "${cfg_tag}" == "${model_path}" ]] && cfg_tag="${model_path##*/}"

    save_dir="${GRID_ROOT}/${cfg_tag}/${MODEL_CFG}/${forget_split}/evals"
    task_name="grid_${cfg_tag}_${MODEL_CFG}_${forget_split}"

    echo
    echo -e "${RED}[${i}/${total}] ${model_path}${NC}"
    echo -e "${RED}Eval → ${save_dir}${NC}"

    if [[ -f "${save_dir}/TOFU_EVAL.json" && -z "${FORCE}" ]]; then
        echo -e "${YELLOW}Already evaluated, skipping (FORCE=1 to re-run).${NC}"
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
        retain_logs_path="saves/eval/tofu_${MODEL_CFG}_${retain_split}/TOFU_EVAL.json")

    if [[ -n "${DRY_RUN}" ]]; then
        echo "DRY_RUN: CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} ${cmd[*]}"
        continue
    fi

    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${cmd[@]}"
    status=$?

    if [[ ${status} -eq 0 ]]; then
        n_ok=$((n_ok + 1))
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
done

echo
echo -e "${RED}=== Done: ${n_ok} evaluated, ${n_skip} skipped, ${#failed[@]} failed (of ${total}) ===${NC}"
if [[ ${#failed[@]} -gt 0 ]]; then
    printf '%s\n' "${failed[@]}" >&2
    exit 1
fi
