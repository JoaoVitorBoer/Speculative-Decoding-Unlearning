#!/bin/bash
#
# run_tofu.sh — TOFU unlearning: train one or more methods, then evaluate at
#               fp / 8bit / 4bit precision.
#
# ── HOW TO CONFIGURE ─────────────────────────────────────────────────────────
#   Edit the CONFIG block below.  Comment/uncomment entries in MODELS, METHODS,
#   RETAIN_MODES, SPLITS, LRS, EPOCHS, and PRECISIONS to run any subset.
#
#   All GPU indices and output roots can also be overridden at call time:
#     CUDA_DEVICES=2,3 EPOCHS=(10) bash run_tofu.sh
# ─────────────────────────────────────────────────────────────────────────────
#

set -euo pipefail
export HYDRA_FULL_ERROR=1

RED='\e[31m'
NC='\e[0m'

# ── CONFIG ────────────────────────────────────────────────────────────────────

NUM_GPUS="${NUM_GPUS:-2}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
EVAL_CUDA_DEVICE="${EVAL_CUDA_DEVICE:-0}"
DELETE_MODEL_AFTER_EVAL="${DELETE_MODEL_AFTER_EVAL:-true}"

MODELS=(
  "Llama-3.2-1B-Instruct"
  "Llama-3.2-3B-Instruct"
  "Llama-3.1-8B-Instruct"
)

METHODS=(
  "GA GradAscent unlearn/tofu/default.yaml"
  "NPO NPO unlearn/tofu/default.yaml"
)

RETAIN_MODES=(
  "GDR"
  "KLR"
)

# Format: "forget_split holdout_split retain_split"
SPLITS=(
  "forget10 holdout10 retain90"
  "forget05 holdout05 retain95"
  "forget01 holdout01 retain99"
)

LRS=(1e-5)
EPOCHS=(5 10)

PRECISIONS=(fp 8bit 4bit)

PER_DEVICE_TRAIN_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4

RESULTS_ROOT="${RESULTS_ROOT:-saves/eval/tofu}"
MODELS_ROOT="${MODELS_ROOT:-saves/unlearn/tofu}"


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

for split in "${SPLITS[@]}"; do
  read -r forget_split holdout_split retain_split <<< "${split}"
  echo -e "${RED}--- Split: forget=${forget_split} | holdout=${holdout_split} | retain=${retain_split} ---${NC}"

  for model in "${MODELS[@]}"; do
    model_path="open-unlearning/tofu_${model}_full"
    retain_logs_path="saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json"
    echo -e "${RED}Model: ${model}${NC}"

    for method_entry in "${METHODS[@]}"; do
      read -r method_tag trainer experiment_cfg <<< "${method_entry}"

      supports_retain_override=true
      case "${trainer}" in
        GradAscent|CEU) supports_retain_override=false ;;
      esac

      for retain_mode in "${RETAIN_MODES[@]}"; do
        retain_tag=""
        extra_train_overrides=()

        if [[ "${supports_retain_override}" == false ]]; then
          [[ "${retain_mode}" != "${RETAIN_MODES[0]}" ]] && continue
        else
          case "${retain_mode}" in
            GDR) retain_tag="_GDR"; extra_train_overrides+=("trainer.method_args.retain_loss_type=NLL") ;;
            KLR) retain_tag="_KLR"; extra_train_overrides+=("trainer.method_args.retain_loss_type=KL")  ;;
          esac
        fi

        tagged_method="${method_tag}${retain_tag}"

        for lr in "${LRS[@]}"; do
          for epochs in "${EPOCHS[@]}"; do
            task_name="tofu_${model}_${forget_split}_${tagged_method}_lr-${lr}_ep-${epochs}"
            train_output_dir="${MODELS_ROOT}/${tagged_method}/${forget_split}/${model}/lr-${lr}_ep-${epochs}"
            mkdir -p "${train_output_dir}"

            echo
            echo -e "${RED}=== ${tagged_method} | ${model} | ${forget_split} | lr=${lr} ep=${epochs} ===${NC}"
            echo -e "${RED}Train → ${train_output_dir}${NC}"

            CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" accelerate launch \
              --config_file configs/accelerate/default_config.yaml \
              --num_processes="${NUM_GPUS}" \
              --main_process_port "${MASTER_PORT}" \
              src/train.py --config-name=unlearn.yaml \
              experiment="${experiment_cfg}" \
              trainer="${trainer}" \
              task_name="${task_name}" \
              paths.output_dir="${train_output_dir}" \
              model="${model}" \
              forget_split="${forget_split}" \
              holdout_split="${holdout_split}" \
              retain_split="${retain_split}" \
              retain_logs_path="${retain_logs_path}" \
              model.model_args.pretrained_model_name_or_path="${model_path}" \
              trainer.args.per_device_train_batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE}" \
              trainer.args.gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
              trainer.args.learning_rate="${lr}" \
              trainer.args.num_train_epochs="${epochs}" \
              trainer.args.ddp_find_unused_parameters=true \
              trainer.args.gradient_checkpointing=true \
              trainer.args.eval_strategy=no \
              trainer.args.eval_on_start=False \
              trainer.args.do_eval=False \
              trainer.args.optim=paged_adamw_32bit \
              "${extra_train_overrides[@]}"

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

              eval_output_dir="${RESULTS_ROOT}/${tagged_method}/${forget_split}/${model}/lr-${lr}_ep-${epochs}/${precision}"
              mkdir -p "${eval_output_dir}"
              echo -e "${RED}--- eval ${precision} → ${eval_output_dir} ---${NC}"

              CUDA_VISIBLE_DEVICES="${EVAL_CUDA_DEVICE}" python src/eval.py \
                experiment=eval/tofu/default.yaml \
                forget_split="${forget_split}" \
                holdout_split="${holdout_split}" \
                model="${model}" \
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
  done
done
