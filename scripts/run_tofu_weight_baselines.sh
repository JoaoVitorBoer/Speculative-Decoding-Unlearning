#!/bin/bash

#SBATCH --output=/home/joaoabitante/Sout/%j__%x.out
#SBATCH --error=/home/joaoabitante/Sout/%j__%x.out

#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --time=2-00:00:00
#SBATCH --gpus=1

# run_tofu_weight_baselines.sh — weight-based unlearning baselines on TOFU.
# Adapted from open-unlearning's scripts/tofu_unlearn.sh + rebuttal_1B_3B.sh.
#
#   Unlearns open-unlearning/tofu_<model>_full with each method, then runs the
#   standard TOFU eval on the resulting checkpoint.
#
#   checkpoint -> saves/unlearn/baselines/tofu/<method>/<model>/<forget_split>
#   eval       -> saves/unlearn/baselines/tofu/<method>/<model>/<forget_split>/evals
#
#   The `tofu/` level keeps this benchmark's checkpoints beside — and separate
#   from — the MUSE ones under `saves/unlearn/baselines/muse/`; each subtree gets
#   its own summary_baselines.csv (scripts/summarize_sud_baselines.py).
#
#   <method> is the method tag plus the retain-loss mode, e.g. NPO_GDR, NPO_KLR.
#   The number of epochs comes from EPOCH_MAP below (keyed by
#   "<model_short>;<method>;<forget_split>"); combinations absent from the map
#   are skipped.

export HYDRA_FULL_ERROR=1

RED='\e[31m'
NC='\e[0m'

NUM_GPUS="${NUM_GPUS:-1}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
EVAL_CUDA_DEVICE="${EVAL_CUDA_DEVICE:-0}"

BASELINES_ROOT="${BASELINES_ROOT:-saves/unlearn/baselines/tofu}"

models=(
    # "Llama-3.2-1B-Instruct"
    "Llama-3.2-3B-Instruct"
    # "Llama-3.1-8B-Instruct"
)

# Format: "method_tag trainer experiment_cfg"
trainers_experiments=(
    # "GA GradAscent unlearn/tofu/default.yaml"
    "GradDiff GradDiff unlearn/tofu/default.yaml"
    "NPO NPO unlearn/tofu/default.yaml"
    "SimNPO SimNPO unlearn/tofu/default.yaml"
)

# Retain-loss variants. GDR -> NLL on the retain set, KLR -> KL to the target.
# GradAscent has no retain term, so it runs once regardless.
retain_modes=(
    "GDR"
    # "KLR"
)

splits=(
    "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)

lr=1e-5

per_device_train_batch_size=4
gradient_accumulation_steps=8

effective_batch_size=$((per_device_train_batch_size * gradient_accumulation_steps * NUM_GPUS))
echo -e "${RED}Effective batch size: ${effective_batch_size}${NC}"
# Epochs per "<model_short>;<method>;<forget_split>". Missing keys are skipped.
# forget01/forget05 values are the tuned ones from open-unlearning's
# rebuttal_1B_3B.sh / rebuttal_8B.sh; forget10 is not tuned there (defaults to 10).
declare -A EPOCH_MAP
# 1B
EPOCH_MAP["1B;GA;forget01"]=10
EPOCH_MAP["1B;GA;forget05"]=5
EPOCH_MAP["1B;GA;forget10"]=10
EPOCH_MAP["1B;GradDiff_GDR;forget01"]=10
EPOCH_MAP["1B;GradDiff_GDR;forget05"]=5
EPOCH_MAP["1B;GradDiff_GDR;forget10"]=10
EPOCH_MAP["1B;GradDiff_KLR;forget01"]=5
EPOCH_MAP["1B;GradDiff_KLR;forget05"]=10
EPOCH_MAP["1B;GradDiff_KLR;forget10"]=10
EPOCH_MAP["1B;NPO_GDR;forget01"]=10
EPOCH_MAP["1B;NPO_GDR;forget05"]=10
EPOCH_MAP["1B;NPO_GDR;forget10"]=10
EPOCH_MAP["1B;NPO_KLR;forget01"]=10
EPOCH_MAP["1B;NPO_KLR;forget05"]=10
EPOCH_MAP["1B;NPO_KLR;forget10"]=10
EPOCH_MAP["1B;SimNPO_GDR;forget01"]=10
EPOCH_MAP["1B;SimNPO_GDR;forget05"]=10
EPOCH_MAP["1B;SimNPO_GDR;forget10"]=10
EPOCH_MAP["1B;SimNPO_KLR;forget01"]=10
EPOCH_MAP["1B;SimNPO_KLR;forget05"]=10
EPOCH_MAP["1B;SimNPO_KLR;forget10"]=10
# 3B
EPOCH_MAP["3B;GA;forget01"]=10
EPOCH_MAP["3B;GA;forget05"]=5
EPOCH_MAP["3B;GA;forget10"]=10
EPOCH_MAP["3B;GradDiff_GDR;forget01"]=10
EPOCH_MAP["3B;GradDiff_GDR;forget05"]=5
EPOCH_MAP["3B;GradDiff_GDR;forget10"]=10
EPOCH_MAP["3B;GradDiff_KLR;forget01"]=10
EPOCH_MAP["3B;GradDiff_KLR;forget05"]=5
EPOCH_MAP["3B;GradDiff_KLR;forget10"]=10
EPOCH_MAP["3B;NPO_GDR;forget01"]=10
EPOCH_MAP["3B;NPO_GDR;forget05"]=10
EPOCH_MAP["3B;NPO_GDR;forget10"]=10
EPOCH_MAP["3B;NPO_KLR;forget01"]=5
EPOCH_MAP["3B;NPO_KLR;forget05"]=5
EPOCH_MAP["3B;NPO_KLR;forget10"]=10
EPOCH_MAP["3B;SimNPO_GDR;forget01"]=10
EPOCH_MAP["3B;SimNPO_GDR;forget05"]=5
EPOCH_MAP["3B;SimNPO_GDR;forget10"]=10
EPOCH_MAP["3B;SimNPO_KLR;forget01"]=5
EPOCH_MAP["3B;SimNPO_KLR;forget05"]=10
EPOCH_MAP["3B;SimNPO_KLR;forget10"]=10
# 8B
EPOCH_MAP["8B;GA;forget01"]=10
EPOCH_MAP["8B;GA;forget05"]=5
EPOCH_MAP["8B;GA;forget10"]=10
EPOCH_MAP["8B;GradDiff_GDR;forget01"]=10
EPOCH_MAP["8B;GradDiff_GDR;forget05"]=10
EPOCH_MAP["8B;GradDiff_GDR;forget10"]=10
EPOCH_MAP["8B;GradDiff_KLR;forget01"]=5
EPOCH_MAP["8B;GradDiff_KLR;forget05"]=10
EPOCH_MAP["8B;GradDiff_KLR;forget10"]=10
EPOCH_MAP["8B;NPO_GDR;forget01"]=10
EPOCH_MAP["8B;NPO_GDR;forget05"]=10
EPOCH_MAP["8B;NPO_GDR;forget10"]=10
EPOCH_MAP["8B;NPO_KLR;forget01"]=10
EPOCH_MAP["8B;NPO_KLR;forget05"]=10
EPOCH_MAP["8B;NPO_KLR;forget10"]=10
EPOCH_MAP["8B;SimNPO_GDR;forget01"]=10
EPOCH_MAP["8B;SimNPO_GDR;forget05"]=10
EPOCH_MAP["8B;SimNPO_GDR;forget10"]=10
EPOCH_MAP["8B;SimNPO_KLR;forget01"]=10
EPOCH_MAP["8B;SimNPO_KLR;forget05"]=10
EPOCH_MAP["8B;SimNPO_KLR;forget10"]=10


export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo -e "${RED}GPUs: ${CUDA_DEVICES} (${NUM_GPUS}) | Effective batch size: ${effective_batch_size} | Master port: ${MASTER_PORT}${NC}"

for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)
    echo -e "${RED}--- Split: forget=${forget_split} | holdout=${holdout_split} | retain=${retain_split} ---${NC}"

    for model in "${models[@]}"; do
        model_path=open-unlearning/tofu_${model}_full
        retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json
        [[ "${model}" =~ ([0-9]+B) ]] && model_short="${BASH_REMATCH[1]}"

        for trainer_experiment in "${trainers_experiments[@]}"; do
            method_tag=$(echo $trainer_experiment | cut -d' ' -f1)
            trainer=$(echo $trainer_experiment | cut -d' ' -f2)
            experiment=$(echo $trainer_experiment | cut -d' ' -f3)

            # GradAscent has no retain loss: run it once, untagged.
            supports_retain_override=true
            [[ "${trainer}" == "GradAscent" ]] && supports_retain_override=false

            for retain_mode in "${retain_modes[@]}"; do
                retain_tag=""
                extra_train_overrides=()

                if [[ "${supports_retain_override}" == false ]]; then
                    [[ "${retain_mode}" != "${retain_modes[0]}" ]] && continue
                else
                    case "${retain_mode}" in
                        GDR) retain_tag="_GDR"; extra_train_overrides+=("trainer.method_args.retain_loss_type=NLL") ;;
                        KLR) retain_tag="_KLR"; extra_train_overrides+=("trainer.method_args.retain_loss_type=KL") ;;
                        *)   echo -e "${RED}Unknown retain mode: ${retain_mode}${NC}" >&2; exit 1 ;;
                    esac
                fi

                method="${method_tag}${retain_tag}"
                epochs="${EPOCH_MAP["${model_short};${method};${forget_split}"]:-}"
                if [[ -z "${epochs}" ]]; then
                    echo -e "${RED}Skipping ${method} | ${model_short} | ${forget_split}: not in epoch map${NC}"
                    continue
                fi

                task_name=tofu_${model}_${forget_split}_${method}_lr-${lr}_ep-${epochs}
                save_dir=${BASELINES_ROOT}/${method}/${model}/${forget_split}

                echo
                echo -e "${RED}=== ${method} | ${model} | ${forget_split} | lr=${lr} ep=${epochs} ===${NC}"
                echo -e "${RED}Unlearning ${model_path} → ${save_dir}${NC}"

                # Unlearn
                CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" accelerate launch \
                --config_file configs/accelerate/default_config.yaml \
                --num_processes="${NUM_GPUS}" \
                --main_process_port $MASTER_PORT \
                src/train.py --config-name=unlearn.yaml \
                experiment=${experiment} \
                trainer=${trainer} \
                task_name=${task_name} \
                model=${model} \
                forget_split=${forget_split} \
                holdout_split=${holdout_split} \
                retain_split=${retain_split} \
                model.model_args.pretrained_model_name_or_path=${model_path} \
                retain_logs_path=${retain_logs_path} \
                paths.output_dir=${save_dir} \
                trainer.args.learning_rate=${lr} \
                trainer.args.num_train_epochs=${epochs} \
                trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
                trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
                trainer.args.ddp_find_unused_parameters=true \
                trainer.args.gradient_checkpointing=true \
                trainer.args.eval_strategy=no \
                trainer.args.eval_on_start=False \
                trainer.args.do_eval=False \
                "${extra_train_overrides[@]}"

                # Eval
                echo -e "${RED}Eval → ${save_dir}/evals${NC}"
                CUDA_VISIBLE_DEVICES="${EVAL_CUDA_DEVICE}" python src/eval.py \
                experiment=eval/tofu/default.yaml \
                forget_split=${forget_split} \
                holdout_split=${holdout_split} \
                model=${model} \
                task_name=${task_name} \
                model.model_args.pretrained_model_name_or_path=${save_dir} \
                paths.output_dir=${save_dir}/evals \
                retain_logs_path=${retain_logs_path}
            done
        done
    done
done
