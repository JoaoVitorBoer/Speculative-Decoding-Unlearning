#!/bin/bash

#SBATCH --output=/home/joaoabitante/Sout/%j__%x.out
#SBATCH --error=/home/joaoabitante/Sout/%j__%x.out

#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --time=2-00:00:00
#SBATCH --gpus=2

# baseline_tofu_undial.sh — UNDIAL at its grid-best hyperparameters.
#
#   Unlearns open-unlearning/tofu_<model>_full with trainer=UNDIAL (self-
#   distillation against logits whose memorized-token entries are penalized by
#   beta), then runs the standard TOFU eval on the resulting checkpoint, for
#   every model x split.
#
#     checkpoint -> saves/unlearn/baselines/tofu/<method>/<model>/<forget_split>
#     eval       -> saves/unlearn/baselines/tofu/<method>/<model>/<forget_split>/evals
#
#   BOTH ARE KEPT: the weights stay on disk (SUD loads these checkpoints as its
#   draft/target models) and evals/TOFU_EVAL.json stays beside them. Nothing is
#   pruned, and an existing eval is never overwritten unless FORCE=1.
#
#   <method> is the bare method name: one directory per method, so a retune
#   re-lands in the same place and scripts/run_sud_*.sh can name its drafts by
#   method alone (that name is also the `method` column of
#   summary_baselines.csv, via sud_paths.model_tag). The exact config is NOT in
#   the path anymore — it stays in cfg_tag below, which names the run, and in
#   <save_dir>/.hydra/overrides.yaml. Join to saves/grid/results/summary_grid.csv
#   on that config, not on the directory name.
#
# HYPERPARAMETERS: rank-1 UNDIAL row of saves/grid/results/report.md ("Top 3 per
# method"), aggregate 0.6055 — lr 1e-04, beta 10, alpha 1, 10 epochs
# (memorization 0.766 / privacy 0.480 / utility 0.639, privleak -70.17).
#
#   Read that privleak before using this as a headline baseline: -70 means the
#   forget set stays highly MIA-detectable, and forget_quality is 9.3e-13. UNDIAL
#   scores its aggregate on memorization and utility, not on privacy.
#
#   alpha=1 differs from configs/trainer/UNDIAL.yaml's default of 0 (no retain
#   term), so it is passed explicitly.
#
# CAUTION: that grid is forget10 on Llama-3.2-1B-Instruct only. forget01 /
# forget05 and the 3B/8B models run these values by extrapolation.
#
# Env knobs: NUM_GPUS, CUDA_DEVICES, EVAL_CUDA_DEVICE, BASELINES_ROOT, FORCE=1.
#
# Summarize when done:
#   python scripts/summarize_sud_baselines.py --benchmark tofu \
#       --baselines-root saves/unlearn/baselines

export HYDRA_FULL_ERROR=1

RED='\e[31m'
YELLOW='\e[33m'
NC='\e[0m'

NUM_GPUS="${NUM_GPUS:-2}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
EVAL_CUDA_DEVICE="${EVAL_CUDA_DEVICE:-0}"

BASELINES_ROOT="${BASELINES_ROOT:-saves/unlearn/baselines/tofu}"

models=(
    # "Llama-3.2-1B-Instruct"
    # "Llama-3.2-3B-Instruct"
    "Llama-3.1-8B-Instruct"
)

splits=(
    # "forget01 holdout01 retain99"
    # "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)

# --- grid-best UNDIAL config (see the header) ---------------------------------
# lr is written as 0.0001 rather than 1e-04 so cfg_tag matches the grid
# directory name exactly; the two are the same number to hydra.

# Directory tag: the method alone (see the header). A retune overwrites the same
# directory on purpose — .hydra/overrides.yaml inside it records what was trained.
method_tag=UNDIAL
lr=0.0001
epochs=10
beta=10
alpha=1
gamma=1.0
retain_loss_type=NLL

# Every value above appears in the tag, so this directory is a record of exactly
# what was trained even if a trainer default drifts later.
cfg_tag=UNDIAL_lr${lr}_beta${beta}_alpha${alpha}_epoch${epochs}

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

n_runs=$((${#splits[@]} * ${#models[@]}))
echo -e "${RED}=== ${cfg_tag} | ${n_runs} runs ===${NC}"
echo -e "${RED}models: ${models[*]}${NC}"
echo -e "${RED}GPUs: ${CUDA_DEVICES} (${NUM_GPUS}) | Output: ${BASELINES_ROOT} | Master port: ${MASTER_PORT}${NC}"

for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)
    echo
    echo -e "${RED}--- Split: forget=${forget_split} | holdout=${holdout_split} | retain=${retain_split} ---${NC}"

    for model in "${models[@]}"; do
        model_path=open-unlearning/tofu_${model}_full
        retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json

        # Effective batch size is held at 32 for every model, because the grid's
        # lr and epoch count were tuned at that scale:
        #   4 per device x 4 accumulation steps x 2 GPUs = 32.
        per_device_train_batch_size=4
        gradient_accumulation_steps=4

        task_name=tofu_${model}_${forget_split}_${cfg_tag}
        save_dir=${BASELINES_ROOT}/${method_tag}/${model}/${forget_split}

        echo
        echo -e "${RED}=== ${cfg_tag} | ${model} | ${forget_split} ===${NC}"

        if [[ -f "${save_dir}/evals/TOFU_EVAL.json" && -z "${FORCE}" ]]; then
            echo -e "${YELLOW}Already evaluated: ${save_dir}/evals — skipping (FORCE=1 to re-run)${NC}"
            continue
        fi

        echo -e "${RED}Unlearning ${model_path} → ${save_dir}${NC}"

        CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" accelerate launch \
        --config_file configs/accelerate/default_config.yaml \
        --num_processes="${NUM_GPUS}" \
        --main_process_port $MASTER_PORT \
        src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default.yaml \
        trainer=UNDIAL \
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
        trainer.method_args.beta=${beta} \
        trainer.method_args.alpha=${alpha} \
        trainer.method_args.gamma=${gamma} \
        trainer.method_args.retain_loss_type=${retain_loss_type} || {
            echo -e "${YELLOW}Unlearning FAILED: ${cfg_tag} | ${model} | ${forget_split} — skipping its eval${NC}" >&2
            continue
        }

        # Eval — writes next to the checkpoint, which is left in place.
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

echo
echo -e "${RED}=== Done: ${cfg_tag} ===${NC}"
echo -e "${RED}Summarize: python scripts/summarize_sud_baselines.py --benchmark tofu --baselines-root $(dirname "${BASELINES_ROOT}")${NC}"
