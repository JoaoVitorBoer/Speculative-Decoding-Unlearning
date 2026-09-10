#!/bin/bash

#SBATCH --output=/home/joaoabitante/Sout/%j__%x.out
#SBATCH --error=/home/joaoabitante/Sout/%j__%x.out

#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1

# run_tofu_eval.sh — TOFU eval only (no training).
#
#   Each model is either a HF id or a local checkpoint path:
#     open-unlearning/unlearn_tofu_Llama-3.1-8B-Instruct_forget10_SimNPO_lr5e-05_b3.5_a1_d1_g0.25_ep5
#     saves/unlearn/baselines/tofu/NPO/Llama-3.2-3B-Instruct/forget10
#
#   eval -> saves/unlearn/baselines/tofu/<model_slug>/<forget_split>
#
#   The `tofu/` level separates this benchmark's evals from the MUSE ones under
#   saves/unlearn/baselines/muse/.
#
#   Models can also be passed as arguments:
#     bash scripts/run_tofu_eval.sh open-unlearning/tofu_Llama-3.2-3B-Instruct_full
#   The model config (tokenizer/chat template) is inferred from the name; set
#   MODEL_CFG=<name> to override.

export HYDRA_FULL_ERROR=1

RED='\e[31m'
NC='\e[0m'

CUDA_DEVICES="${CUDA_DEVICES:-0}"

BASELINES_ROOT="${BASELINES_ROOT:-saves/unlearn/baselines/tofu}"

models=(
    # "open-unlearning/unlearn_tofu_Llama-3.1-8B-Instruct_forget10_SimNPO_lr5e-05_b3.5_a1_d1_g0.25_ep5"
    open-unlearning/tofu_Llama-3.2-1B-Instruct_full
    open-unlearning/tofu_Llama-3.2-3B-Instruct_full
    open-unlearning/tofu_Llama-3.1-8B-Instruct_full
)
[[ $# -gt 0 ]] && models=("$@")

splits=(
    "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)

for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)
    echo -e "${RED}--- Split: forget=${forget_split} | holdout=${holdout_split} | retain=${retain_split} ---${NC}"

    for model_path in "${models[@]}"; do
        model_cfg="${MODEL_CFG:-$(echo ${model_path} | grep -oE 'Llama-[0-9.]+-[0-9]+B-Instruct' | head -1)}"
        if [[ -z "${model_cfg}" ]]; then
            echo -e "${RED}Cannot infer model config from '${model_path}'; set MODEL_CFG=<name>.${NC}" >&2
            exit 1
        fi

        model_slug="${model_path//\//_}"
        save_dir=${BASELINES_ROOT}/${model_slug}/${forget_split}
        task_name=tofu_${model_slug}_${forget_split}_eval

        echo
        echo -e "${RED}=== ${model_path} | ${model_cfg} | ${forget_split} ===${NC}"
        echo -e "${RED}Eval → ${save_dir}${NC}"

        CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" python src/eval.py \
        experiment=eval/tofu/default.yaml \
        forget_split=${forget_split} \
        holdout_split=${holdout_split} \
        model=${model_cfg} \
        task_name=${task_name} \
        model.model_args.pretrained_model_name_or_path=${model_path} \
        paths.output_dir=${save_dir} \
        retain_logs_path=saves/eval/tofu_${model_cfg}_${retain_split}/TOFU_EVAL.json
    done
done
