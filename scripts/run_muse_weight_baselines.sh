#!/bin/bash

#SBATCH --output=/home/joaoabitante/Sout/%j__%x.out
#SBATCH --error=/home/joaoabitante/Sout/%j__%x.out

#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
#SBATCH --gpus=4
#SBATCH --job-name=muse_baselines

# run_muse_weight_baselines.sh — weight-based unlearning baselines on MUSE.
# MUSE counterpart of scripts/run_tofu_weight_baselines.sh.
#
#   Unlearns muse-bench/MUSE-<data_split>_target with each method, then runs the
#   standard MUSE eval on the resulting checkpoint.
#
#   checkpoint -> saves/unlearn/baselines/muse/<method>/<model>/<data_split>
#   eval       -> saves/unlearn/baselines/muse/<method>/<model>/<data_split>/evals
#
#   That layout is what scripts/run_sud_muse.sh discovers its drafts from, and
#   what scripts/sud_paths.py#model_tag shortens to <model>_<method>.
#
#   <method> is the method tag plus the retain-loss mode, e.g. NPO_GDR, NPO_KLR.
#   Epochs come from EPOCH_MAP below (keyed by "<data_split>;<method>"), falling
#   back to DEFAULT_EPOCHS for combinations that are not listed.

export HYDRA_FULL_ERROR=1

RED='\e[31m'
NC='\e[0m'

NUM_GPUS="${NUM_GPUS:-4}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
EVAL_CUDA_DEVICE="${EVAL_CUDA_DEVICE:-0}"

# Skip a combination whose checkpoint already exists, instead of retraining it.
SKIP_EXISTING="${SKIP_EXISTING:-true}"

BASELINES_ROOT="${BASELINES_ROOT:-saves/unlearn/baselines/muse}"

# MUSE ships a single model family.
models=(
    "Llama-2-7b-hf"
)

# MUSE corpora. Each has its own target checkpoint and eval data.
data_splits=(
    "Books"
    "News"
)

# Format: "method_tag trainer experiment_cfg"
trainers_experiments=(
    # "GA GradAscent unlearn/muse/default.yaml"
    "GradDiff GradDiff unlearn/muse/default.yaml"
    "NPO NPO unlearn/muse/default.yaml"
    "SimNPO SimNPO unlearn/muse/default.yaml"
)

# Retain-loss variants. GDR -> NLL on the retain set, KLR -> KL to the target.
# GradAscent has no retain term, so it runs once regardless.
retain_modes=(
    "GDR"
    # "KLR"
)

lr=1e-5

# 7B full-parameter finetuning: keep the per-device batch at 1 and recover the
# effective batch through accumulation. Matches TOFU's effective batch of 32.
per_device_train_batch_size=1
gradient_accumulation_steps=8

effective_batch_size=$((per_device_train_batch_size * gradient_accumulation_steps * NUM_GPUS))

# Epochs per "<data_split>;<method>". Unlisted keys use DEFAULT_EPOCHS — MUSE has
# no per-method tuned epoch counts upstream (unlike TOFU's rebuttal scripts), so
# the config default of 5 is the baseline and this map is for overrides only.
DEFAULT_EPOCHS=5
declare -A EPOCH_MAP
# EPOCH_MAP["News;GradDiff_GDR"]=5
# EPOCH_MAP["Books;NPO_GDR"]=5

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo -e "${RED}GPUs: ${CUDA_DEVICES} (${NUM_GPUS}) | Effective batch size: ${effective_batch_size} | Master port: ${MASTER_PORT}${NC}"

for data_split in "${data_splits[@]}"; do
    echo -e "${RED}--- Data split: ${data_split} ---${NC}"

    for model in "${models[@]}"; do
        model_path=muse-bench/MUSE-${data_split}_target
        retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json

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
                epochs="${EPOCH_MAP["${data_split};${method}"]:-${DEFAULT_EPOCHS}}"

                task_name=muse_${model}_${data_split}_${method}_lr-${lr}_ep-${epochs}
                save_dir=${BASELINES_ROOT}/${method}/${model}/${data_split}

                if [[ "${SKIP_EXISTING}" == "true" && -f "${save_dir}/config.json" ]]; then
                    echo -e "${RED}Skipping ${method} | ${model} | ${data_split}: checkpoint already at ${save_dir}${NC}"
                    continue
                fi

                echo
                echo -e "${RED}=== ${method} | ${model} | ${data_split} | lr=${lr} ep=${epochs} ===${NC}"
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
                data_split=${data_split} \
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

                if [[ ! -f "${save_dir}/config.json" ]]; then
                    echo -e "${RED}Skipping eval: no config.json in ${save_dir} (training failed?)${NC}"
                    continue
                fi

                # Eval
                echo -e "${RED}Eval → ${save_dir}/evals${NC}"
                CUDA_VISIBLE_DEVICES="${EVAL_CUDA_DEVICE}" python src/eval.py \
                experiment=eval/muse/default.yaml \
                model=${model} \
                data_split=${data_split} \
                task_name=${task_name} \
                model.model_args.pretrained_model_name_or_path=${save_dir} \
                paths.output_dir=${save_dir}/evals \
                retain_logs_path=${retain_logs_path}
            done
        done
    done
done
