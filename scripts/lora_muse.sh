#!/bin/bash

#SBATCH --output=/home/joaoabitante/Sout/%j__%x.out
#SBATCH --error=/home/joaoabitante/Sout/%j__%x.out

#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=30G
#SBATCH --time=2-00:00:00
#SBATCH --gpus=3

set -euo pipefail

export MASTER_PORT=$(
python - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
)
echo "Master Port: ${MASTER_PORT}"

RED='\e[31m'
NC='\e[0m'   # no color / reset

MODEL="Llama-2-7b-hf"
DATA_SPLITS=("Books" "News")
TRAINERS=("NPO")

PER_DEVICE_TRAIN_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=8
LORA_DROPOUT=0.05

EPOCHS=(5 10)
LRS=(1e-4 7e-4)
RANKS=(16 32 64)
PRECISIONS=(fp 8bit 4bit)

TARGET_MODULE_TAGS=(all qv proj_out)
TARGET_MODULE_VALUES=(
  '["q_proj","v_proj","k_proj","o_proj","gate_proj","down_proj","up_proj"]'
  '["q_proj","v_proj"]'
  '["o_proj","down_proj","up_proj"]'
)

for data_split in "${DATA_SPLITS[@]}"; do
  retain_logs_path="saves/eval/muse_${MODEL}_${data_split}_retrain/MUSE_EVAL.json"
  echo -e "${RED}--- Data split: ${data_split} | retain_logs_path: ${retain_logs_path} ---${NC}"

  for trainer in "${TRAINERS[@]}"; do
    for idx in "${!TARGET_MODULE_TAGS[@]}"; do
      target_tag="${TARGET_MODULE_TAGS[$idx]}"
      target_modules="${TARGET_MODULE_VALUES[$idx]}"

      for rank in "${RANKS[@]}"; do
        alphas=("${rank}" "$((rank / 2))" "$((rank * 2))")

        for alpha in "${alphas[@]}"; do
          for lr in "${LRS[@]}"; do
            for epochs in "${EPOCHS[@]}"; do
              task_name="lora_muse_${MODEL}_${data_split}_${trainer}_t-${target_tag}_r-${rank}_a-${alpha}_lr-${lr}_ep-${epochs}"
              train_output_dir="saves/unlearn/${trainer}/muse/${data_split}/${MODEL}/lora_grid/targets-${target_tag}/r-${rank}_a-${alpha}/lr-${lr}_ep-${epochs}"

              echo
              echo -e "${RED}=== Trainer: ${trainer} | Targets: ${target_tag} | Rank: ${rank} | Alpha: ${alpha} | LR: ${lr} | Epochs: ${epochs} ===${NC}"
              echo -e "${RED}Train output: ${train_output_dir}${NC}"

              CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --config_file configs/accelerate/default_config.yaml --num_processes=3 --main_process_port "${MASTER_PORT}" \
                src/train.py --config-name=unlearn.yaml \
                experiment=unlearn/muse/default.yaml \
                adapter=lora \
                model="${MODEL}" \
                data_split="${data_split}" \
                trainer="${trainer}" \
                task_name="${task_name}" \
                paths.output_dir="${train_output_dir}" \
                retain_logs_path="${retain_logs_path}" \
                trainer.args.per_device_train_batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE}" \
                trainer.args.gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
                trainer.args.learning_rate="${lr}" \
                trainer.args.num_train_epochs="${epochs}" \
                trainer.args.ddp_find_unused_parameters=true \
                trainer.args.gradient_checkpointing=true \
                trainer.args.eval_strategy=no \
                trainer.args.eval_on_start=False \
                model.lora_config.target_modules="${target_modules}" \
                model.lora_config.r="${rank}" \
                model.lora_config.lora_alpha="${alpha}" \
                model.lora_config.lora_dropout="${LORA_DROPOUT}" 

              for precision in "${PRECISIONS[@]}"; do
                quant_override=()
                precision_tag="${precision}"

                case "${precision}" in
                  fp)
                    precision_tag="fp"
                    ;;
                  8bit)
                    quant_override=("quantization=8bit")
                    ;;
                  4bit)
                    quant_override=("quantization=4bit")
                    ;;
                  *)
                    echo -e "${RED}Unknown precision ${precision}${NC}" >&2
                    exit 1
                    ;;
                esac

                eval_task_name="${task_name}_eval_${precision_tag}"
                eval_output_dir="saves/unlearn/${trainer}/grid_results/muse/${data_split}/${MODEL}/lora/targets-${target_tag}/r-${rank}_a-${alpha}/lr-${lr}_ep-${epochs}/${precision_tag}"

                echo -e "${RED}---- Eval precision: ${precision_tag} | Output: ${eval_output_dir}${NC}"

                CUDA_VISIBLE_DEVICES=0 python src/eval.py \
                  experiment=eval/muse/default.yaml \
                  model="${MODEL}" \
                  data_split="${data_split}" \
                  task_name="${eval_task_name}" \
                  paths.output_dir="${eval_output_dir}" \
                  retain_logs_path="${retain_logs_path}" \
                  model.model_handler=LoRAModelForEvaluation \
                  model.model_args.pretrained_model_name_or_path="${train_output_dir}" \
                  "${quant_override[@]}"
              done
            done
          done
        done
      done
    done
  done
done
