#!/bin/bash

#SBATCH --output=/home/joaoabitante/Sout/%j__%x.out
#SBATCH --error=/home/joaoabitante/Sout/%j__%x.out

#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --time=2-00:00:00
#SBATCH --gpus=2

# baseline_tofu_altpo.sh — AltPO at its grid-best hyperparameters.
#
#   AltPO (arXiv:2409.13474) is not a trainer at all: it is trainer=DPO whose
#   *preferred* response is a plausible-but-false alternate answer instead of
#   "I don't know". The data block below is what makes it AltPO — it swaps the
#   forget dataset's handler to QAwithAlternateDataset and repoints hf_args at a
#   generated JSON of alternate answers. Copied verbatim, in order, from
#   community/methods/AltPO/run.sh (open-unlearning commit b71de54); the handler
#   itself is already in src/data/qa.py.
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
# HYPERPARAMETERS: the "Runs clearing all three" row of
# saves/grid/results/report.md — lr 2e-05, beta 0.05, alpha 5, 10 epochs,
# aggregate 0.7292 (memorization 0.638 / privacy 0.904 / utility 0.695,
# privleak -7.80, forget_quality 0.131). NOT the best AltPO aggregate: that is
# AltPO_lr5e-05_beta0.1_alpha1_epoch5 at 0.7383, whose forget_quality of 0.00229
# fails the gate. This config is one of only 2 points in the whole 397-point grid
# that clear agg > 0.6, forget_quality > 0.05 and |privleak| < 10 jointly (the
# other is the SimNPO config in baseline_tofu_simnpo.sh), which is the reason to
# prefer it over a marginally higher harmonic mean.
#
# PREREQUISITE: GENERATE THE ALTERNATE ANSWERS FIRST --------------------------
#
#   AltPO needs one JSON per (model, split) — five alternate answers per
#   question, sampled from the *pre-unlearning* full model at temperature 1.0:
#
#     sbatch scripts/altpo_generate_alternates.sh
#
#   -> community/methods/AltPO/data/tofu_<model>_full/<split>/alt5_seed_0.json
#
#   That is 5 x |forget set| sampled generations (2000 for forget10), so it is a
#   separate GPU job; see that script's header for the knobs. Runs whose file is
#   missing are skipped here, rather than failing the sweep.
#
# CAUTION: that grid is forget10 on Llama-3.2-1B-Instruct only. forget01 /
# forget05 and the 3B/8B models run these values by extrapolation.
#
# Env knobs: NUM_GPUS, CUDA_DEVICES, EVAL_CUDA_DEVICE, BASELINES_ROOT,
#            ALTPO_DATA_ROOT, FORCE=1.
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
ALTPO_DATA_ROOT="${ALTPO_DATA_ROOT:-community/methods/AltPO/data}"

models=(
    "Llama-3.2-1B-Instruct"
    "Llama-3.2-3B-Instruct"
    # "Llama-3.1-8B-Instruct"
)

splits=(
    "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)

# --- grid-best AltPO config (see the header) -----------------------------------

# Directory tag: the method alone (see the header). A retune overwrites the same
# directory on purpose — .hydra/overrides.yaml inside it records what was trained.
method_tag=AltPO
lr=2e-05
epochs=10
beta=0.05
alpha=5
gamma=1.0
retain_loss_type=NLL

# Every value above appears in the tag, so this directory is a record of exactly
# what was trained even if a trainer default drifts later.
cfg_tag=AltPO_lr${lr}_beta${beta}_alpha${alpha}_epoch${epochs}

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

n_runs=$((${#splits[@]} * ${#models[@]}))
echo -e "${RED}=== ${cfg_tag} | ${n_runs} runs ===${NC}"
echo -e "${RED}models: ${models[*]} | alternate answers: ${ALTPO_DATA_ROOT}${NC}"
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
        alt_file=${ALTPO_DATA_ROOT}/tofu_${model}_full/${forget_split}/alt5_seed_0.json

        echo
        echo -e "${RED}=== ${cfg_tag} | ${model} | ${forget_split} ===${NC}"

        if [[ -f "${save_dir}/evals/TOFU_EVAL.json" && -z "${FORCE}" ]]; then
            echo -e "${YELLOW}Already evaluated: ${save_dir}/evals — skipping (FORCE=1 to re-run)${NC}"
            continue
        fi

        # The alternate answers are AltPO's forget data; without them there is
        # nothing to prefer. Skip loudly rather than fail mid-sweep.
        if [[ ! -f "${alt_file}" ]]; then
            echo -e "${YELLOW}Missing AltPO alternate answers: ${alt_file}${NC}"
            echo -e "${YELLOW}Generate all of them first with:${NC}"
            echo -e "${YELLOW}  sbatch scripts/altpo_generate_alternates.sh${NC}"
            echo -e "${YELLOW}or just this one with:${NC}"
            echo -e "${YELLOW}  MODELS=${model} SPLITS=${forget_split}" \
                    "scripts/altpo_generate_alternates.sh${NC}"
            continue
        fi

        echo -e "${RED}Unlearning ${model_path} → ${save_dir}${NC}"

        # The data.forget block: '~' deletes the TOFU config name (a local json
        # file has none), path/data_files/split repoint the loader at the
        # generated file, and alternate_key names the column DPO should prefer.
        CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" accelerate launch \
        --config_file configs/accelerate/default_config.yaml \
        --num_processes="${NUM_GPUS}" \
        --main_process_port $MASTER_PORT \
        src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default.yaml \
        trainer=DPO \
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
        trainer.method_args.retain_loss_type=${retain_loss_type} \
        data.forget.TOFU_QA_forget.handler=QAwithAlternateDataset \
        '~data.forget.TOFU_QA_forget.args.hf_args.name' \
        data.forget.TOFU_QA_forget.args.hf_args.path=json \
        +data.forget.TOFU_QA_forget.args.hf_args.data_files=${alt_file} \
        data.forget.TOFU_QA_forget.args.hf_args.split=train \
        +data.forget.TOFU_QA_forget.args.alternate_key=alternate \
        +data.forget.TOFU_QA_forget.args.return_original=True || {
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
