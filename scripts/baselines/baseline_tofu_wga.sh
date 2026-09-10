#!/bin/bash

#SBATCH --output=/home/joaoabitante/Sout/%j__%x.out
#SBATCH --error=/home/joaoabitante/Sout/%j__%x.out

#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --time=2-00:00:00
#SBATCH --gpus=1

# baseline_tofu_wga.sh — WGA at its author-reported hyperparameters.
#
#   WGA (arXiv:2502.19301, "Rethinking LLM Unlearning Objectives: A Gradient
#   Perspective and Go Beyond") is trainer=WGA: weighted gradient ascent — the
#   forget term is gradient ascent whose per-token weight is the model's own
#   confidence raised to beta, so already-forgotten tokens stop contributing and
#   the divergence that plain GA suffers is damped. Retention is the ordinary
#   GradDiff NLL term. Then runs the standard TOFU eval on the resulting
#   checkpoint, for every model x split.
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
#   <save_dir>/.hydra/overrides.yaml.
#
# HYPERPARAMETERS: NOT from saves/grid/results — WGA was never in that grid (its
# 397 rows cover only the 8 HF-collection methods). Taken from
# community/methods/WGA/{README.md,run.sh} in locuslab/open-unlearning (commit
# b80a5fc), the method authors' own contribution:
#
#     lr 1e-5, eff. batch 32, beta 1.0, alpha 1.0, gamma 1.0, NLL retain, 10 epochs
#
#   beta 1.0 is the README's stated paper setting. The effective batch is the
#   one departure: the README pairs lr 1e-5 with 16, and this runs 32 to stay
#   comparable with the other baselines (see the batch note in the loop).
#   alpha 1.0 and gamma 1.0 are configs/trainer/WGA.yaml's defaults, and 1.0 is
#   the middle of the {0.1, 1.0, 10.0} that run.sh sweeps — run.sh is a sweep, not
#   a single best config, so there is no author-chosen alpha to copy.
#
#   10 epochs, NOT the 5 in WGA.yaml: run.sh does not override the epoch count,
#   and `experiment=unlearn/tofu/default.yaml` sets trainer.args.num_train_epochs
#   to 10, which beats the trainer config's own `args` block in the hydra
#   defaults merge. 10 is therefore what upstream's script actually trains for,
#   and it also matches every other baseline_tofu_*.sh here.
#
# TWO CAVEATS ------------------------------------------------------------------
#
#   1. The paper tuned on LLaMA-2-7B and Phi-1.5, not on the TOFU-full Llamas
#      used here — unlike PDU, whose script targets these exact checkpoints. Both
#      the lr and the epoch count are therefore transferred, not reproduced. alpha
#      is the knob the authors expect you to search: WGA_ALPHA re-tunes it and it
#      lands in cfg_tag, so a sweep does not overwrite this run.
#   2. Upstream's run.sh passes `trainer.method_args.beta1` and `beta2`, which
#      WGA does not accept — src/trainer/unlearn/wga.py takes a single `beta`,
#      and beta2 is never even assigned in that script. Running it verbatim
#      raises in hydra. This script passes `beta` and drops beta2, which is what
#      the README describes.
#
# Env knobs: NUM_GPUS, CUDA_DEVICES, EVAL_CUDA_DEVICE, BASELINES_ROOT,
#            WGA_ALPHA, WGA_BETA, WGA_EPOCHS, FORCE=1.
#
# Summarize when done:
#   python scripts/summarize_sud_baselines.py --benchmark tofu \
#       --baselines-root saves/unlearn/baselines

export HYDRA_FULL_ERROR=1

RED='\e[31m'
YELLOW='\e[33m'
NC='\e[0m'

NUM_GPUS="${NUM_GPUS:-1}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
EVAL_CUDA_DEVICE="${EVAL_CUDA_DEVICE:-0}"

BASELINES_ROOT="${BASELINES_ROOT:-saves/unlearn/baselines/tofu}"

# Nothing has been trained for WGA yet, so all three sizes are live. Comment out
# what you are not running — the atria QOS admits only one queued job at a time.
models=(
    "Llama-3.2-1B-Instruct"
    "Llama-3.2-3B-Instruct"
    "Llama-3.1-8B-Instruct"
)

splits=(
    "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)

# --- author-reported WGA config (see the header) ------------------------------

# Directory tag: the method alone (see the header). A retune overwrites the same
# directory on purpose — .hydra/overrides.yaml inside it records what was trained.
method_tag=WGA
lr=1e-05
epochs="${WGA_EPOCHS:-10}"
beta="${WGA_BETA:-1.0}"
alpha="${WGA_ALPHA:-1.0}"
gamma=1.0
retain_loss_type=NLL

# Every value above appears in the tag, so this directory is a record of exactly
# what was trained even if a trainer default drifts later.
cfg_tag=WGA_lr${lr}_beta${beta}_alpha${alpha}_epoch${epochs}

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

        # Effective batch size is held at 32 for every model, matching every
        # other baseline_tofu_*.sh so the methods stay comparable:
        #   4 per device x 4 accumulation steps x 2 GPUs = 32.
        # NB: the WGA README pairs lr 1e-5 with an effective batch of 16, so
        # this doubles the paper's scale. Comparability across baselines was
        # chosen over per-method fidelity; halve gradient_accumulation_steps to
        # get the paper's 16.
        per_device_train_batch_size=4
        gradient_accumulation_steps=8

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
        trainer=WGA \
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
