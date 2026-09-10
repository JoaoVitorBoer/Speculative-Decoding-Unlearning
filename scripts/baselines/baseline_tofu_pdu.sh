#!/bin/bash

#SBATCH --output=/home/joaoabitante/Sout/%j__%x.out
#SBATCH --error=/home/joaoabitante/Sout/%j__%x.out

#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --time=2-00:00:00
#SBATCH --gpus=1

# baseline_tofu_pdu.sh — PDU at its author-reported best hyperparameters.
#
#   PDU (arXiv:2506.05314, "Constrained Entropic Unlearning") is trainer=PDU: the
#   forget term is a logit-margin flattening loss (max logit minus mean logit,
#   squared) that drives the forget-set output distribution toward uniformity,
#   and retention is a HARD CONSTRAINT — retain_loss <= retain_loss_eps — solved
#   by a primal-dual loop that raises or lowers the retain preference
#   (`alpha`, the dual variable) every step. Then runs the standard TOFU eval on
#   the resulting checkpoint, for every model x split.
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
# HYPERPARAMETERS: NOT from saves/grid/results — PDU was never in that grid
# (its 397 rows cover only the 8 HF-collection methods). Copied verbatim from the
# "Final best parameters" TOFU block of community/methods/PDU/run.sh in
# locuslab/open-unlearning (commit fd825ea), which is the method authors' own
# reproduction script:
#
#     lr 1e-5, eff. batch 32, gamma 1.0, alpha 100, retain_loss_eps 0.3,
#     dual_step_size 5, dual_warmup_epochs 5, dual_update_upon step,
#     epochs 10 (1B/3B) and 30 (8B)
#
#   Upstream ran exactly the same targets — open-unlearning/tofu_*_full — so
#   these values transfer directly, unlike the extrapolations the grid-tuned
#   baseline scripts make. Upstream also swept forget01/05/10 at every size, so
#   no split here is out of distribution either.
#
#   alpha=100 is the INITIAL retain preference, not a fixed weight: pdu.py sets
#   self.preferences = [gamma, alpha] and then updates preferences[1] each step.
#   The PDU README's guidance is that 50 or 100 is the natural starting scale,
#   because the flattening loss lives in logit space and is numerically large.
#
# THE ONE VALUE THAT MATTERS MOST: retain_loss_eps -----------------------------
#
#   Everything is passed explicitly below, but note in particular that
#   configs/trainer/PDU.yaml in this repo carries `retain_loss_eps: 100`, where
#   upstream leaves it `???` (hydra's mandatory-missing marker, forcing it onto
#   the command line). 100 is NOT an epsilon — at that value `retain_loss - eps`
#   is negative for any realistic NLL, so max(0, ...) in pdu.py pins the dual
#   variable at 0 and the retain constraint silently drops out of the objective.
#   The 0.3 used here is upstream's TOFU value. Do not fall back to the config
#   default.
#
#   Per the PDU README, eps should sit "an appropriately larger value" than the
#   pretrained model's retain loss. If you retune for a target whose retain NLL
#   differs from the TOFU-full models', re-derive it from that model's retain
#   loss rather than reusing 0.3.
#
# CAUTION — DDP AND THE DUAL VARIABLE: self.preferences[1] is a plain Python
# float updated from each rank's LOCAL retain loss, with no all-reduce, so with
# NUM_GPUS>1 the ranks drift onto slightly different objectives. Upstream ran
# this at num_processes=8, so keeping it is the faithful reproduction; set
# NUM_GPUS=1 if you want the dual iterate to be exact.
#
# Env knobs: NUM_GPUS, CUDA_DEVICES, EVAL_CUDA_DEVICE, BASELINES_ROOT,
#            PDU_RETAIN_LOSS_EPS, PDU_ALPHA, PDU_DUAL_STEP_SIZE, FORCE=1.
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

# Nothing has been trained for PDU yet, so all three sizes are live. Comment out
# what you are not running — 8B trains for 30 epochs here, and the atria QOS
# admits only one queued job at a time.
models=(
    # "Llama-3.2-1B-Instruct"
    "Llama-3.2-3B-Instruct"
    # "Llama-3.1-8B-Instruct"
)

splits=(
    # "forget01 holdout01 retain99"
    # "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)

# --- author-reported best PDU config (see the header) -------------------------

# Directory tag: the method alone (see the header). A retune overwrites the same
# directory on purpose — .hydra/overrides.yaml inside it records what was trained.
method_tag=PDU
lr=1e-05
gamma=1.0
alpha="${PDU_ALPHA:-100}"                          # initial retain preference (dual var)
retain_loss_eps="${PDU_RETAIN_LOSS_EPS:-0.3}"      # the hard retain constraint
dual_step_size="${PDU_DUAL_STEP_SIZE:-5}"
dual_warmup_epochs=5
dual_update_upon=step
retain_loss_type=NLL

# Epochs are per-model upstream: 10 for the 1B/3B targets, 30 for 8B. Set inside
# the model loop, so cfg_tag is rebuilt there too.

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

n_runs=$((${#splits[@]} * ${#models[@]}))
echo -e "${RED}=== PDU | ${n_runs} runs | eps ${retain_loss_eps} | alpha0 ${alpha} | step ${dual_step_size} ===${NC}"
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

        # Upstream's per-model epoch count (community/methods/PDU/run.sh).
        case "${model}" in
            Llama-3.1-8B-Instruct) epochs=30 ;;
            *)                     epochs=10 ;;
        esac

        # Effective batch size 32, matching upstream's 4 per device x 8 processes:
        #   4 per device x 4 accumulation steps x 2 GPUs = 32.
        per_device_train_batch_size=4
        gradient_accumulation_steps=8

        # Every value that was tuned appears in the tag, so this run is a record
        # of exactly what was trained even if a trainer default drifts later.
        cfg_tag=PDU_lr${lr}_alpha${alpha}_eps${retain_loss_eps}_step${dual_step_size}_epoch${epochs}

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
        trainer=PDU \
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
        trainer.method_args.gamma=${gamma} \
        trainer.method_args.alpha=${alpha} \
        trainer.method_args.primal_dual=true \
        trainer.method_args.retain_loss_eps=${retain_loss_eps} \
        trainer.method_args.dual_step_size=${dual_step_size} \
        trainer.method_args.dual_update_upon=${dual_update_upon} \
        trainer.method_args.dual_warmup_epochs=${dual_warmup_epochs} \
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
echo -e "${RED}=== Done: PDU ===${NC}"
echo -e "${RED}Summarize: python scripts/summarize_sud_baselines.py --benchmark tofu --baselines-root $(dirname "${BASELINES_ROOT}")${NC}"
