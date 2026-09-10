#!/bin/bash

#SBATCH --output=/home/joaoabitante/Sout/%j__%x.out
#SBATCH --error=/home/joaoabitante/Sout/%j__%x.out

#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --time=3-00:00:00
#SBATCH --gpus=1

# run_unlearn_best_models.sh — train + eval each method at its grid-best config.
#
#   For every method in open-unlearning's "TOFU Unlearned Models" collection,
#   this takes the single best-aggregate hyperparameter point found by
#   saves/grid/grid_summarizer.py and re-trains it locally across three base
#   models and three forget splits:
#
#     checkpoint -> saves/unlearn/baselines/tofu/<method>/<model>/<forget_split>
#     eval       -> saves/unlearn/baselines/tofu/<method>/<model>/<forget_split>/evals
#
#   That is exactly the layout scripts/summarize_sud_baselines.py globs. <method>
#   is the BARE METHOD NAME (cfg_tag's prefix, e.g. `NPO`), one directory per
#   method: a retune re-lands in the same place, and scripts/run_sud_*.sh names
#   its drafts by method alone. The hyperparameters are not in the path — they
#   stay in cfg_tag (which names the run/task) and in .hydra/overrides.yaml
#   inside the checkpoint, which is what to join summary_grid.csv on.
#
#   Unlike scripts/run_tofu_weight_baselines.sh there is no EPOCH_MAP: the epoch
#   count is part of each method's tuned config and comes from the grid, not from
#   a per-split table.
#
# WHERE THE NUMBERS COME FROM ------------------------------------------------
#
#   `saves/grid/results/report.md` ("Top 3 per method"), 397 evaluated grid
#   points. Each entry below is that method's rank-1 row by `aggregate`:
#
#     IdkDPO    0.7393   lr1e-05  beta0.05 alpha2   epoch10
#     AltPO     0.7383   lr5e-05  beta0.1  alpha1   epoch5
#     NPO       0.7232   lr2e-05  beta0.1  alpha2   epoch10
#     SimNPO    0.7054   lr5e-05  b3.5 a1 d1 g0.125 ep10
#     RMU       0.6763   lr5e-05  layer5   scoeff1  epoch5
#     GradDiff  0.6691   lr2e-05  alpha5   epoch5
#     UNDIAL    0.6055   lr0.0001 beta10   alpha1   epoch10
#     IdkNLL    0.1394   lr5e-05  alpha2   epoch5
#
#   THREE CAVEATS on reusing them, none of which this script can fix:
#
#   1. The grid is *forget10 on Llama-3.2-1B-Instruct only*. Applying its winners
#      to forget01/forget05 and to 3B/8B is an extrapolation. Expect the ranking
#      to move — in particular `epochs` and `lr` were tuned against a 400-example
#      forget set, and forget01 is 40 examples at the same epoch count.
#   2. RMU's `layer` is an *absolute* index. Layer 5 of 1B's 16 layers is ~31%
#      depth; of 8B's 32 layers it is ~16%. Set RMU_LAYER to rescale if you want
#      constant relative depth (1B 5 -> 3B 9 -> 8B 10).
#   3. IdkNLL is the grid's worst method by a wide margin (max aggregate 0.139,
#      privleak ~ -96 on all 40 points — the forget set stays trivially
#      MIA-detectable). It is included for completeness; drop it with METHODS to
#      save 9 runs.
#
# ALTPO NEEDS A GENERATED DATASET FIRST --------------------------------------
#
#   AltPO (arXiv:2409.13474) is not a trainer at all — upstream implements it as
#   plain `trainer=DPO` whose *preferred* response is a plausible-but-false
#   alternate answer instead of "I don't know". See
#   community/methods/AltPO/, applied verbatim from open-unlearning commit
#   b71de54; the QAwithAlternateDataset handler it relies on is already in
#   src/data/qa.py and registered in src/data/__init__.py.
#
#   So AltPO has a PREREQUISITE the other seven do not: a per-(model, split)
#   JSON of alternate answers, sampled from the *pre-unlearning* full model at
#   temperature 1.0, five per question. Generate it before running AltPO here:
#
#     cd community/methods/AltPO
#     python generate.py \
#       dataset_config.dataset_kwargs.name=forget10 \
#       model_config.model_name=tofu_Llama-3.2-1B-Instruct_full \
#       model_config.model_kwargs.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full
#
#   -> community/methods/AltPO/data/tofu_<model>_full/<split>/alt5_seed_0.json
#
#   That is 5 x |forget set| sampled generations at batch_size 1 (2000 for
#   forget10), so it is not free — budget for it separately. Runs whose file is
#   missing are skipped with the exact command printed, rather than failing the
#   whole sweep; ALTPO_DATA_ROOT relocates the tree.
#
# METHOD -> OVERRIDE MAPPING ------------------------------------------------
#
#   Three methods have no trainer config of their own and are composed instead:
#
#   AltPO  = trainer=DPO + the stock default.yaml experiment, with the forget
#     dataset's handler swapped to QAwithAlternateDataset and its hf_args
#     repointed at the generated JSON — `~...hf_args.name` deletes the TOFU
#     config name (a local json file has no config), `path=json` +
#     `data_files=...` load it, and `alternate_key=alternate` names the column
#     DPO should prefer. These overrides are copied from the PR's
#     community/methods/AltPO/run.sh, in its order.
#
#   IdkDPO = trainer=DPO + experiment=unlearn/tofu/idk.yaml. The idk experiment
#     swaps the forget dataset for TOFU_QA_forget_idk, which returns
#     {original, alternate} pairs; DPO prefers the idk answer over the real one.
#
#   IdkNLL = trainer=GradDiff + the same idk data, but with
#     `return_original=false` so the dataset yields the idk answer alone (flat,
#     like TOFU_QA_forget), plus `gamma=-1`. GradDiff hardcodes
#     `forget_loss = -forget_outputs.loss` (src/trainer/unlearn/grad_diff.py),
#     i.e. gradient *ascent*; IdkNLL needs descent on the idk answers, so the
#     negative gamma flips that sign back:
#         loss = gamma*(-NLL_idk) + alpha*NLL_retain
#              = (-1)*(-NLL_idk) + 2*NLL_retain  =  NLL_idk + 2*NLL_retain
#     which is IdkNLL exactly. The sign flip is the whole trick — do not
#     "fix" gamma to +1 without also changing the trainer.
#
# Env knobs:
#   NUM_GPUS=1              processes for accelerate launch
#   CUDA_DEVICES=0          training GPU(s)
#   EVAL_CUDA_DEVICE=0      eval GPU
#   BASELINES_ROOT=saves/unlearn/baselines/tofu
#   METHODS="NPO SimNPO"    only these method tags (prefix match, e.g. "NPO"
#                           selects NPO_lr2e-05_... but not SimNPO_...)
#   MODELS="Llama-3.2-1B-Instruct"    override the base-model list
#   ONLY_SPLITS="forget01 forget10"   override the split list (forget names only)
#   RMU_LAYER=5             RMU's absolute layer index (see caveat 2)
#   FORCE=1                 re-run runs that already have evals/TOFU_EVAL.json
#   EVAL_ONLY=1             skip training, re-eval existing checkpoints
#   PRUNE_CKPT=1            delete the checkpoint after a successful eval, keeping
#                           only evals/ — saves ~500GB over the full sweep, but
#                           these checkpoints are what SUD loads as draft/target
#                           models, so leave it off unless you only want metrics
#   DRY_RUN=1               print the commands, run nothing
#   PLAN_ONLY=1             print the selected (method, model, split) grid and exit
#
# Scale: 8 methods x 3 models x 3 splits = 72 unlearn+eval runs, and ~590GB of
# checkpoints (2.4GB per 1B, 6.1GB per 3B, ~16GB per 8B). One GPU will not finish
# that inside the 3-day limit — fan out per model:
#
#   MODELS=Llama-3.2-1B-Instruct sbatch scripts/run_unlearn_best_models.sh
#   MODELS=Llama-3.2-3B-Instruct sbatch scripts/run_unlearn_best_models.sh
#   MODELS=Llama-3.1-8B-Instruct sbatch scripts/run_unlearn_best_models.sh
#
# Summarize when done:
#   python scripts/summarize_sud_baselines.py --benchmark tofu \
#       --baselines-root saves/unlearn/baselines

export HYDRA_FULL_ERROR=1

RED='\e[31m'
YELLOW='\e[33m'
GREEN='\e[32m'
NC='\e[0m'

NUM_GPUS="${NUM_GPUS:-1}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
EVAL_CUDA_DEVICE="${EVAL_CUDA_DEVICE:-0}"

BASELINES_ROOT="${BASELINES_ROOT:-saves/unlearn/baselines/tofu}"
RMU_LAYER="${RMU_LAYER:-5}"
ALTPO_DATA_ROOT="${ALTPO_DATA_ROOT:-community/methods/AltPO/data}"

BASE_MODELS=(
    "Llama-3.2-1B-Instruct"
    "Llama-3.2-3B-Instruct"
    "Llama-3.1-8B-Instruct"
)

SPLITS=(
    "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)

# --- the configs -------------------------------------------------------------
#
# Fields, '|'-separated:
#   cfg_tag | trainer | experiment | learning_rate | epochs | method overrides...
#
# cfg_tag is the grid config name verbatim, so it joins to summary_grid.csv. Only
# its method prefix names the checkpoint directory (see method_tag below); the
# full tag names the run.
# Every method_arg is written out even where it equals the trainer config's
# default: these directories are the record of what was trained, and a default
# that shifts later must not silently retag an old checkpoint.
BEST_CONFIGS=(
    # -- preference-optimization family ---------------------------------------
    "IdkDPO_lr1e-05_beta0.05_alpha2_epoch10|DPO|unlearn/tofu/idk.yaml|1e-05|10|trainer.method_args.beta=0.05 trainer.method_args.alpha=2 trainer.method_args.gamma=1.0 trainer.method_args.retain_loss_type=NLL"
    "NPO_lr2e-05_beta0.1_alpha2_epoch10|NPO|unlearn/tofu/default.yaml|2e-05|10|trainer.method_args.beta=0.1 trainer.method_args.alpha=2 trainer.method_args.gamma=1.0 trainer.method_args.retain_loss_type=NLL"
    "SimNPO_lr5e-05_b3.5_a1_d1_g0.125_ep10|SimNPO|unlearn/tofu/default.yaml|5e-05|10|trainer.method_args.beta=3.5 trainer.method_args.alpha=1.0 trainer.method_args.delta=1.0 trainer.method_args.gamma=0.125 trainer.method_args.retain_loss_type=NLL"

    # -- gradient / logit family ----------------------------------------------
    "GradDiff_lr2e-05_alpha5_epoch5|GradDiff|unlearn/tofu/default.yaml|2e-05|5|trainer.method_args.alpha=5 trainer.method_args.gamma=1.0 trainer.method_args.retain_loss_type=NLL"
    "UNDIAL_lr0.0001_beta10_alpha1_epoch10|UNDIAL|unlearn/tofu/default.yaml|1e-04|10|trainer.method_args.beta=10.0 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.method_args.retain_loss_type=NLL"

    # -- representation misdirection ------------------------------------------
    # retain_loss_type stays EMBED_DIFF: RMU's retain term is an activation
    # distance to the frozen reference, not an NLL, and GradDiff's
    # compute_retain_loss would raise on it.
    "RMU_lr5e-05_layer5_scoeff1_epoch5|RMU|unlearn/tofu/default.yaml|5e-05|5|trainer.method_args.steering_coeff=1 trainer.method_args.module_regex=model\.layers\.${RMU_LAYER} trainer.method_args.alpha=1 trainer.method_args.gamma=1.0 trainer.method_args.retain_loss_type=EMBED_DIFF"

    # -- idk-target NLL (see "METHOD -> OVERRIDE MAPPING" above for gamma=-1) --
    "IdkNLL_lr5e-05_alpha2_epoch5|GradDiff|unlearn/tofu/idk.yaml|5e-05|5|trainer.method_args.gamma=-1.0 trainer.method_args.alpha=2 trainer.method_args.retain_loss_type=NLL data.forget.TOFU_QA_forget_idk.args.return_original=false"

    # -- alternate-answer preference optimization -----------------------------
    # The data block is copied from community/methods/AltPO/run.sh in its exact
    # order: `~` deletes the TOFU config name (a local json has none), then
    # path/data_files/split repoint the loader, then alternate_key names the
    # column DPO prefers. {MODEL} and {SPLIT} are filled in per run below —
    # each (model, split) needs its own generated file.
    # gamma and retain_loss_type restate configs/trainer/DPO.yaml's defaults,
    # which run.sh leaves untouched; the values are identical either way.
    "AltPO_lr5e-05_beta0.1_alpha1_epoch5|DPO|unlearn/tofu/default.yaml|5e-05|5|trainer.method_args.beta=0.1 trainer.method_args.alpha=1 trainer.method_args.gamma=1.0 trainer.method_args.retain_loss_type=NLL data.forget.TOFU_QA_forget.handler=QAwithAlternateDataset ~data.forget.TOFU_QA_forget.args.hf_args.name data.forget.TOFU_QA_forget.args.hf_args.path=json +data.forget.TOFU_QA_forget.args.hf_args.data_files=${ALTPO_DATA_ROOT}/tofu_{MODEL}_full/{SPLIT}/alt5_seed_0.json data.forget.TOFU_QA_forget.args.hf_args.split=train +data.forget.TOFU_QA_forget.args.alternate_key=alternate +data.forget.TOFU_QA_forget.args.return_original=True"
)

# Effective batch size is held at 32 for every model, because the grid's lr and
# epoch counts were tuned at that scale — an OOM-driven batch change would
# silently retune them. Only the split between device batch and accumulation
# moves with model size.
declare -A BS_MAP GA_MAP
BS_MAP["Llama-3.2-1B-Instruct"]=4;  GA_MAP["Llama-3.2-1B-Instruct"]=8
BS_MAP["Llama-3.2-3B-Instruct"]=4;  GA_MAP["Llama-3.2-3B-Instruct"]=8
BS_MAP["Llama-3.1-8B-Instruct"]=2;  GA_MAP["Llama-3.1-8B-Instruct"]=16

# --- selection ---------------------------------------------------------------

[[ -n "${MODELS}" ]] && read -r -a BASE_MODELS <<< "${MODELS}"

if [[ -n "${ONLY_SPLITS}" ]]; then
    read -r -a wanted_splits <<< "${ONLY_SPLITS}"
    kept=()
    for spec in "${SPLITS[@]}"; do
        for w in "${wanted_splits[@]}"; do
            [[ "${spec%% *}" == "${w}" ]] && kept+=("${spec}") && break
        done
    done
    SPLITS=("${kept[@]}")
    if [[ ${#SPLITS[@]} -eq 0 ]]; then
        echo -e "${RED}No splits match ONLY_SPLITS='${ONLY_SPLITS}'${NC}" >&2
        exit 2
    fi
fi

if [[ -n "${METHODS}" ]]; then
    read -r -a wanted <<< "${METHODS}"
    kept=()
    for entry in "${BEST_CONFIGS[@]}"; do
        tag="${entry%%|*}"
        for w in "${wanted[@]}"; do
            # Prefix match on the method name, so METHODS="NPO" picks NPO_lr...
            # and not SimNPO_lr...; the '_' guard is what keeps them apart.
            if [[ "${tag}" == "${w}" || "${tag}" == "${w}_"* ]]; then
                kept+=("${entry}")
                break
            fi
        done
    done
    BEST_CONFIGS=("${kept[@]}")
    if [[ ${#BEST_CONFIGS[@]} -eq 0 ]]; then
        echo -e "${RED}No methods match METHODS='${METHODS}'${NC}" >&2
        exit 2
    fi
fi

total=$(( ${#BEST_CONFIGS[@]} * ${#BASE_MODELS[@]} * ${#SPLITS[@]} ))

if [[ -n "${PLAN_ONLY}" ]]; then
    for model in "${BASE_MODELS[@]}"; do
        for spec in "${SPLITS[@]}"; do
            for entry in "${BEST_CONFIGS[@]}"; do
                echo "${entry%%|*} ${model} ${spec%% *}"
            done
        done
    done
    echo "# ${total} runs" >&2
    exit 0
fi

# --- preflight ---------------------------------------------------------------

# forget_quality is a KS test against the retain model's losses; a missing
# reference makes every eval emit NaN for it. Check all of them up front rather
# than discovering it 40 hours in.
missing=()
for model in "${BASE_MODELS[@]}"; do
    for spec in "${SPLITS[@]}"; do
        retain_split=$(echo "${spec}" | cut -d' ' -f3)
        p="saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json"
        [[ -f "${p}" ]] || missing+=("${p}")
    done
done
if [[ ${#missing[@]} -gt 0 ]]; then
    echo -e "${RED}Missing retain reference logs:${NC}" >&2
    printf '  %s\n' "${missing[@]}" >&2
    echo "Run the retain-split evals first (scripts/run_tofu_eval.sh)." >&2
    exit 1
fi

for model in "${BASE_MODELS[@]}"; do
    if [[ -z "${BS_MAP[$model]}" ]]; then
        echo -e "${RED}No batch-size entry for '${model}' in BS_MAP${NC}" >&2
        exit 2
    fi
done

# --- run ---------------------------------------------------------------------

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

echo -e "${RED}=== Best-config baselines | ${total} runs | GPU ${CUDA_DEVICES} ===${NC}"
echo -e "${RED}Methods:  ${#BEST_CONFIGS[@]} ($(for e in "${BEST_CONFIGS[@]}"; do printf '%s ' "${e%%_lr*}"; done))${NC}"
echo -e "${RED}Models:   ${BASE_MODELS[*]}${NC}"
echo -e "${RED}Splits:   $(for s in "${SPLITS[@]}"; do printf '%s ' "${s%% *}"; done)${NC}"
echo -e "${RED}Output:   ${BASELINES_ROOT}${NC}"
echo -e "${RED}RMU layer: ${RMU_LAYER} | Master port: ${MASTER_PORT}${NC}"

i=0
n_ok=0
n_skip=0
failed=()
start_ts=${SECONDS}

for model in "${BASE_MODELS[@]}"; do
    model_path=open-unlearning/tofu_${model}_full
    per_device_train_batch_size="${BS_MAP[$model]}"
    gradient_accumulation_steps="${GA_MAP[$model]}"
    effective_batch_size=$(( per_device_train_batch_size * gradient_accumulation_steps * NUM_GPUS ))

    echo
    echo -e "${RED}########## ${model} | batch ${per_device_train_batch_size}x${gradient_accumulation_steps}x${NUM_GPUS} = ${effective_batch_size} ##########${NC}"

    for spec in "${SPLITS[@]}"; do
        forget_split=$(echo "${spec}" | cut -d' ' -f1)
        holdout_split=$(echo "${spec}" | cut -d' ' -f2)
        retain_split=$(echo "${spec}" | cut -d' ' -f3)
        retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json

        echo
        echo -e "${RED}--- ${model} | forget=${forget_split} holdout=${holdout_split} retain=${retain_split} ---${NC}"

        for entry in "${BEST_CONFIGS[@]}"; do
            i=$(( i + 1 ))

            IFS='|' read -r cfg_tag trainer experiment lr epochs extra <<< "${entry}"

            # AltPO's forget data is one generated file per (model, split), so
            # its overrides carry placeholders — the array is built before the
            # loop knows either value.
            extra="${extra//\{MODEL\}/${model}}"
            extra="${extra//\{SPLIT\}/${forget_split}}"
            read -r -a extra_overrides <<< "${extra}"

            # The checkpoint directory is named by the METHOD alone (cfg_tag's
            # prefix), so a retune re-lands in the same place and
            # scripts/run_sud_*.sh can name its drafts by bare method name. The
            # hyperparameters live in task_name and in .hydra/overrides.yaml.
            method_tag="${cfg_tag%%_*}"
            save_dir="${BASELINES_ROOT}/${method_tag}/${model}/${forget_split}"
            task_name="best_${cfg_tag}_${model}_${forget_split}"

            echo
            echo -e "${RED}[${i}/${total}] ${cfg_tag} | ${model} | ${forget_split}${NC}"
            echo -e "${RED}  trainer=${trainer} experiment=${experiment} lr=${lr} epochs=${epochs}${NC}"
            echo -e "${RED}  → ${save_dir}${NC}"

            if [[ -f "${save_dir}/evals/TOFU_EVAL.json" && -z "${FORCE}" ]]; then
                echo -e "${YELLOW}  Already evaluated, skipping (FORCE=1 to re-run).${NC}"
                n_skip=$(( n_skip + 1 ))
                continue
            fi

            # AltPO alone needs data generated ahead of time. Skip loudly with
            # the exact command instead of failing the sweep 40 hours in.
            # DRY_RUN is exempt: there the point is to see the command.
            if [[ "${cfg_tag}" == AltPO_* && -z "${DRY_RUN}" ]]; then
                alt_file="${ALTPO_DATA_ROOT}/tofu_${model}_full/${forget_split}/alt5_seed_0.json"
                if [[ ! -f "${alt_file}" ]]; then
                    echo -e "${YELLOW}  Missing AltPO alternate answers: ${alt_file}${NC}"
                    echo -e "${YELLOW}  Generate with (one line):${NC}"
                    echo -e "${YELLOW}    cd community/methods/AltPO && python generate.py" \
                            "dataset_config.dataset_kwargs.name=${forget_split}" \
                            "model_config.model_name=tofu_${model}_full" \
                            "model_config.model_kwargs.pretrained_model_name_or_path=open-unlearning/tofu_${model}_full${NC}"
                    n_skip=$(( n_skip + 1 ))
                    continue
                fi
            fi

            train_cmd=(accelerate launch
                --config_file configs/accelerate/default_config.yaml
                --num_processes="${NUM_GPUS}"
                --main_process_port "${MASTER_PORT}"
                src/train.py --config-name=unlearn.yaml
                experiment="${experiment}"
                trainer="${trainer}"
                task_name="${task_name}"
                model="${model}"
                forget_split="${forget_split}"
                holdout_split="${holdout_split}"
                retain_split="${retain_split}"
                model.model_args.pretrained_model_name_or_path="${model_path}"
                retain_logs_path="${retain_logs_path}"
                paths.output_dir="${save_dir}"
                trainer.args.learning_rate="${lr}"
                trainer.args.num_train_epochs="${epochs}"
                trainer.args.per_device_train_batch_size="${per_device_train_batch_size}"
                trainer.args.gradient_accumulation_steps="${gradient_accumulation_steps}"
                trainer.args.ddp_find_unused_parameters=true
                trainer.args.gradient_checkpointing=true
                trainer.args.eval_strategy=no
                trainer.args.eval_on_start=False
                trainer.args.do_eval=False
                "${extra_overrides[@]}")

            eval_cmd=(python src/eval.py
                experiment=eval/tofu/default.yaml
                forget_split="${forget_split}"
                holdout_split="${holdout_split}"
                model="${model}"
                task_name="${task_name}"
                model.model_args.pretrained_model_name_or_path="${save_dir}"
                paths.output_dir="${save_dir}/evals"
                retain_logs_path="${retain_logs_path}")

            if [[ -n "${DRY_RUN}" ]]; then
                echo "DRY_RUN train: CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} ${train_cmd[*]}"
                echo "DRY_RUN eval:  CUDA_VISIBLE_DEVICES=${EVAL_CUDA_DEVICE} ${eval_cmd[*]}"
                continue
            fi

            if [[ -z "${EVAL_ONLY}" ]]; then
                CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${train_cmd[@]}"
                status=$?
                if [[ ${status} -ne 0 ]]; then
                    echo -e "${RED}  TRAIN FAILED (exit ${status})${NC}" >&2
                    failed+=("${cfg_tag} ${model} ${forget_split} (train)")
                    continue
                fi
            elif [[ ! -d "${save_dir}" ]]; then
                echo -e "${YELLOW}  EVAL_ONLY but no checkpoint at ${save_dir}, skipping.${NC}"
                n_skip=$(( n_skip + 1 ))
                continue
            fi

            echo -e "${RED}  Eval → ${save_dir}/evals${NC}"
            CUDA_VISIBLE_DEVICES="${EVAL_CUDA_DEVICE}" "${eval_cmd[@]}"
            status=$?
            if [[ ${status} -ne 0 ]]; then
                echo -e "${RED}  EVAL FAILED (exit ${status})${NC}" >&2
                failed+=("${cfg_tag} ${model} ${forget_split} (eval)")
                continue
            fi

            n_ok=$(( n_ok + 1 ))

            # Only ever prune after the eval succeeded, and only the weights --
            # evals/ and .hydra/ are what the summarizer reads.
            if [[ -n "${PRUNE_CKPT}" ]]; then
                echo -e "${YELLOW}  PRUNE_CKPT: removing weights under ${save_dir}${NC}"
                find "${save_dir}" -maxdepth 1 -type f \
                    \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' \) -delete
            fi

            done_n=$(( n_ok + ${#failed[@]} ))
            if (( done_n > 0 )); then
                elapsed=$(( SECONDS - start_ts ))
                eta=$(( elapsed * (total - i) / done_n ))
                echo -e "${GREEN}  Progress ${i}/${total} | elapsed $((elapsed / 60))m | ETA ~$((eta / 60))m${NC}"
            fi
        done
    done
done

echo
echo -e "${RED}=== Done: ${n_ok} trained+evaluated, ${n_skip} skipped, ${#failed[@]} failed (of ${total}) ===${NC}"
echo -e "${RED}Summarize: python scripts/summarize_sud_baselines.py --benchmark tofu --baselines-root $(dirname "${BASELINES_ROOT}")${NC}"
if [[ ${#failed[@]} -gt 0 ]]; then
    printf '%s\n' "${failed[@]}" >&2
    exit 1
fi
