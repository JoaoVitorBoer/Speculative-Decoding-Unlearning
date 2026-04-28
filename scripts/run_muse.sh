#!/bin/bash
#
# run_muse.sh — MUSE unlearning: train one or more methods, then evaluate at
#               fp / 8bit / 4bit precision.
#



set -euo pipefail
export HYDRA_FULL_ERROR=1

RED='\e[31m'
NC='\e[0m'


NUM_GPUS="${NUM_GPUS:-4}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
EVAL_CUDA_DEVICE="${EVAL_CUDA_DEVICE:-0}"
DELETE_MODEL_AFTER_EVAL="${DELETE_MODEL_AFTER_EVAL:-true}"

MODEL="Llama-2-7b-hf"

DATA_SPLITS=(
  "Books"
  "News"
)

METHODS=(   "GA"      "GA_GDR"   "GA_KLR"   "NPO"   "NPO_GDR"   "NPO_KLR"  )
TRAINERS=(  "GradAscent" "GradDiff" "GradDiff" "NPO"   "NPO"      "NPO"      )
RETAIN_LOSS=( ""       "NLL"      "KL"        ""      "NLL"       "KL"       )
ALPHAS=(      ""       "100"      "100"       "0"     "0.1"       "0.1"      )

LRS=(1e-5)
EPOCHS=(5 10)

# Eval precisions: fp (full precision), 8bit, 4bit (bitsandbytes)
PRECISIONS=(fp 8bit 4bit)

PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4

RESULTS_ROOT="${RESULTS_ROOT:-saves/eval/muse}"
MODELS_ROOT="${MODELS_ROOT:-saves/unlearn/muse}"


EFFECTIVE_BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS))

export MASTER_PORT=$(
python - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
)

echo -e "${RED}Effective batch size: ${EFFECTIVE_BATCH_SIZE} | Master port: ${MASTER_PORT}${NC}"
mkdir -p "${RESULTS_ROOT}" "${MODELS_ROOT}"

for data_split in "${DATA_SPLITS[@]}"; do
  retain_logs_path="saves/eval/muse_${MODEL}_${data_split}_retrain/MUSE_EVAL.json"
  echo -e "${RED}--- Data split: ${data_split} | Retain logs: ${retain_logs_path} ---${NC}"

  for idx in "${!METHODS[@]}"; do
    method_tag="${METHODS[$idx]}"
    trainer="${TRAINERS[$idx]}"
    retain_loss="${RETAIN_LOSS[$idx]}"
    alpha="${ALPHAS[$idx]}"

    method_args=()
    [[ -n "${alpha}"       ]] && method_args+=("trainer.method_args.alpha=${alpha}")
    [[ -n "${retain_loss}" ]] && method_args+=("trainer.method_args.retain_loss_type=${retain_loss}")

    for lr in "${LRS[@]}"; do
      for epochs in "${EPOCHS[@]}"; do
        task_name="muse_${MODEL}_${data_split}_${method_tag}_lr-${lr}_ep-${epochs}"
        train_output_dir="${MODELS_ROOT}/${method_tag}/${data_split}/${MODEL}/lr-${lr}_ep-${epochs}"
        mkdir -p "${train_output_dir}"

        echo
        echo -e "${RED}=== ${method_tag} | ${data_split} | lr=${lr} ep=${epochs} ===${NC}"
        echo -e "${RED}Train → ${train_output_dir}${NC}"

        CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" accelerate launch \
          --config_file configs/accelerate/default_config.yaml \
          --num_processes="${NUM_GPUS}" \
          --main_process_port "${MASTER_PORT}" \
          src/train.py --config-name=unlearn.yaml \
          experiment=unlearn/muse/default.yaml \
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
          trainer.args.do_eval=False \
          "${method_args[@]}"

        if [[ ! -f "${train_output_dir}/config.json" ]]; then
          echo -e "${RED}Skipping eval: config.json not found in ${train_output_dir}.${NC}"
          continue
        fi

        for precision in "${PRECISIONS[@]}"; do
          quant_override=()
          case "${precision}" in
            fp)   ;;
            8bit) quant_override=("quantization=8bit") ;;
            4bit) quant_override=("quantization=4bit") ;;
            *)    echo -e "${RED}Unknown precision: ${precision}${NC}" >&2; exit 1 ;;
          esac

          eval_output_dir="${RESULTS_ROOT}/${method_tag}/${data_split}/${MODEL}/lr-${lr}_ep-${epochs}/${precision}"
          mkdir -p "${eval_output_dir}"
          echo -e "${RED}--- eval ${precision} → ${eval_output_dir} ---${NC}"

          CUDA_VISIBLE_DEVICES="${EVAL_CUDA_DEVICE}" python src/eval.py \
            experiment=eval/muse/default.yaml \
            model="${MODEL}" \
            data_split="${data_split}" \
            task_name="${task_name}_eval_${precision}" \
            paths.output_dir="${eval_output_dir}" \
            retain_logs_path="${retain_logs_path}" \
            model.model_args.pretrained_model_name_or_path="${train_output_dir}" \
            "${quant_override[@]}"
        done

        if [[ "${DELETE_MODEL_AFTER_EVAL}" == "true" ]]; then
          echo -e "${RED}Deleting checkpoint: ${train_output_dir}${NC}"
          rm -rf "${train_output_dir}"
        fi

      done
    done
  done
done
