#!/bin/bash

#SBATCH --output=/home/joaoabitante/Sout/%j__%x.out
#SBATCH --error=/home/joaoabitante/Sout/%j__%x.out

#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=34G
#SBATCH --time=4-00:00:00
#SBATCH --gpus=3
#SBATCH --job-name=8b_baselines

#
# run_8b_missing_baselines.sh — train the five weight-unlearning baselines that
# have no Llama-3.1-8B-Instruct checkpoint yet, so SUD can use them as drafts.
#
#   WGA, IdkNLL, SatImp, IdkDPO, PDU  x  {forget01, forget05, forget10}
#     = 15 checkpoints
#
#   Each method is its own scripts/baselines/baseline_tofu_<method>.sh; this
#   file only sequences them and pins the GPU/batch environment they share. The
#   hyperparameters stay where they are documented — in each sub-script's header
#   — and are not overridden here.
#
#   checkpoint -> saves/unlearn/baselines/tofu/<method>/Llama-3.1-8B-Instruct/<split>
#   eval       -> .../<split>/evals
#
#   The three methods that DO have 8B checkpoints (GradDiff, NPO, SimNPO) are
#   not touched. RMU and UNDIAL are not here either: those two are published on
#   the Hub (JoaoBoer/tofu_Llama-3.1-8B-Instruct_<split>_{RMU,UNDIAL}) and only
#   need downloading, not training.
#
# ── GPU LAYOUT ────────────────────────────────────────────────────────────────
#   Asks for 3 GPUs and trains on 2. GPU 2 is deliberately left idle so a SUD
#   eval can be run interactively inside this same allocation:
#
#       CUDA_DEVICES=2 bash scripts/run_sud_original_unlearned_draft.sh
#
#   Slurm confines each job to its own GPU cgroup, so the indices this job sees
#   are always 0,1,2 no matter which physical cards were assigned.
#
# ── RESOURCE BUDGET ───────────────────────────────────────────────────────────
#   The atria `normal` QOS caps a user at cpu=50, mem=56000M, gres/gpu=4 across
#   ALL running jobs, and at 2 submitted jobs. This job is sized to leave room
#   for the 1-GPU RMU/UNDIAL SUD sweep to run beside it:
#
#       this job          3 GPU   28 CPU   34G
#       RMU/UNDIAL SUD    1 GPU   16 CPU   18G
#       ------------------------------------------
#       total             4 GPU   44 CPU   52G     (caps: 4 / 50 / 54.7G)
#
#   If you raise either job's --mem, check that pair again — mem is the binding
#   constraint, not CPU.
#
# ── WALLTIME ──────────────────────────────────────────────────────────────────
#   Estimated ~26h of work (see the ORDER note below), and training writes no
#   intermediate checkpoints, so a walltime kill loses the whole split that was
#   in flight — PDU's forget10 alone is ~9h of that. 4 days is the `normal` QOS
#   ceiling and is asked for deliberately: atria schedules FIFO with no priority
#   aging (PriorityType=priority/basic) and the node is GPU-saturated, so there
#   is no backfill window a shorter request could slip into. Asking for less
#   would not start this job any sooner; it would only raise the chance of
#   losing hours to a kill.
#
#   The spare GPU (see GPU LAYOUT) is the other reason: this allocation doubles
#   as the interactive sandbox for SUD test runs, and a longer walltime is
#   simply more sandbox.
#
#   Re-running is safe and cheap either way: every sub-script skips a
#   (method, split) whose evals/TOFU_EVAL.json already exists, so a resumed job
#   picks up where this one stopped.
#
#   THE JOB DOES NOT EXIT WHEN THE TRAINING FINISHES. It prints its summary and
#   then holds the allocation open until Slurm kills it at the time limit, so
#   the 3 GPUs stay yours for the full 4 days. That is deliberate: atria is FIFO
#   with no priority aging, so releasing an allocation early means going to the
#   back of the queue behind everyone who queued meanwhile — the GPUs are worth
#   more held than returned. Use them interactively while it holds (see GPU
#   LAYOUT). Set HOLD=0 to exit as soon as the training is done instead.
#
# ── HOW TO RUN ────────────────────────────────────────────────────────────────
#   sbatch scripts/run_8b_missing_baselines.sh
#
#   Subset it with METHODS, e.g. to skip the 14-hour PDU run:
#     METHODS="WGA IdkNLL SatImp IdkDPO" sbatch scripts/run_8b_missing_baselines.sh
#
#   Env knobs: METHODS, NUM_GPUS, CUDA_DEVICES, EVAL_CUDA_DEVICE, BASELINES_ROOT,
#              FORCE=1 (retrain even where an eval already exists),
#              HOLD=0 (release the allocation when training ends, don't hold it).
# ─────────────────────────────────────────────────────────────────────────────

# No `set -e`: one method failing must not take the other four down with it.
set -uo pipefail

RED='\e[31m'
GREEN='\e[32m'
YELLOW='\e[33m'
NC='\e[0m'

cd "$(dirname "$(readlink -f "$0")")/.." || exit 1

# ── CONFIG ────────────────────────────────────────────────────────────────────

# Two GPUs for training, the third left free (see GPU LAYOUT above). Both the
# training launch and the post-training eval stay off GPU 2.
export NUM_GPUS="${NUM_GPUS:-2}"
export CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
export EVAL_CUDA_DEVICE="${EVAL_CUDA_DEVICE:-0}"
export BASELINES_ROOT="${BASELINES_ROOT:-saves/unlearn/baselines/tofu}"

# Cheapest first, so a walltime kill costs the fewest methods. The estimates are
# scaled from the measured 8B GradDiff/NPO/SimNPO training times on this node
# (3 splits each, train + eval):
#
#   WGA      ~2h   GA-like forget term, 10 epochs
#   IdkNLL   ~2h   plain NLL on the idk targets, 10 epochs
#   SatImp   ~3h   10 epochs
#   IdkDPO   ~5h   DPO needs a reference model forward, like NPO (~5h measured)
#   PDU     ~14h   30 epochs at 8B, not 10 — upstream's own per-model setting
#
# PDU is over half the total on its own; drop it from METHODS if the queue is
# tight and run it separately.
DEFAULT_METHODS="WGA IdkNLL SatImp IdkDPO PDU"
read -r -a METHODS_ARR <<< "${METHODS:-${DEFAULT_METHODS}}"

# Method name -> its script. Keys are what METHODS accepts.
declare -A SCRIPT_OF=(
  [WGA]=scripts/baselines/baseline_tofu_wga.sh
  [IdkNLL]=scripts/baselines/baseline_tofu_idknll.sh
  [SatImp]=scripts/baselines/baseline_tofu_satimp.sh
  [IdkDPO]=scripts/baselines/baseline_tofu_idkdpo.sh
  [PDU]=scripts/baselines/baseline_tofu_pdu.sh
)

# ── PREFLIGHT ─────────────────────────────────────────────────────────────────

echo -e "${RED}=== 8B missing weight baselines ===${NC}"
echo -e "${RED}methods : ${METHODS_ARR[*]}${NC}"
echo -e "${RED}GPUs    : train on ${CUDA_DEVICES} (${NUM_GPUS} procs), eval on ${EVAL_CUDA_DEVICE}${NC}"
echo -e "${RED}output  : ${BASELINES_ROOT}/<method>/Llama-3.1-8B-Instruct/<split>${NC}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null \
  | sed "s/^/  GPU /" || true

for method in "${METHODS_ARR[@]}"; do
  if [[ -z "${SCRIPT_OF[$method]:-}" ]]; then
    echo -e "${RED}Unknown method: ${method} (known: ${!SCRIPT_OF[*]})${NC}" >&2
    exit 1
  fi
  if [[ ! -f "${SCRIPT_OF[$method]}" ]]; then
    echo -e "${RED}Missing script: ${SCRIPT_OF[$method]}${NC}" >&2
    exit 1
  fi
done

# 15 checkpoints at ~15G each is ~225G, and the malta group quota is 1024G
# shared. Warn rather than block — the run is worth starting even if a later
# method will need room cleared first.
quota_line=$(quota -gs malta 2>/dev/null | tail -1)
[[ -n "${quota_line}" ]] && echo -e "${YELLOW}group quota (malta): ${quota_line}${NC}"
echo -e "${YELLOW}NB: ~15G per checkpoint x 15 = ~225G. Delete drafts as their SUD${NC}"
echo -e "${YELLOW}    evals land if the quota gets close.${NC}"

# ── DRIVER ────────────────────────────────────────────────────────────────────

declare -A STATUS_OF
overall=0

for method in "${METHODS_ARR[@]}"; do
  script="${SCRIPT_OF[$method]}"
  started=$(date +%s)

  echo
  echo -e "${RED}────────────────────────────────────────────────────────────${NC}"
  echo -e "${RED} ${method}  →  ${script}${NC}"
  echo -e "${RED} started $(date '+%F %T')${NC}"
  echo -e "${RED}────────────────────────────────────────────────────────────${NC}"

  if bash "${script}"; then
    STATUS_OF[$method]="ok"
    echo -e "${GREEN}${method} finished in $(( ($(date +%s) - started) / 60 )) min${NC}"
  else
    rc=$?
    STATUS_OF[$method]="FAILED (rc=${rc})"
    overall=1
    echo -e "${YELLOW}${method} FAILED (rc=${rc}) after $(( ($(date +%s) - started) / 60 )) min — continuing${NC}" >&2
  fi
done

# ── SUMMARY ───────────────────────────────────────────────────────────────────

echo
echo -e "${RED}=== Summary ===${NC}"
for method in "${METHODS_ARR[@]}"; do
  printf '  %-8s %s\n' "${method}" "${STATUS_OF[$method]}"
done

echo
echo -e "${RED}Checkpoints now on disk for Llama-3.1-8B-Instruct:${NC}"
for method in "${METHODS_ARR[@]}"; do
  for split in forget01 forget05 forget10; do
    d="${BASELINES_ROOT}/${method}/Llama-3.1-8B-Instruct/${split}"
    if [[ -f "${d}/model.safetensors" || -f "${d}/model.safetensors.index.json" ]]; then
      printf '  %-8s %-9s ok\n' "${method}" "${split}"
    else
      printf '  %-8s %-9s MISSING\n' "${method}" "${split}"
    fi
  done
done

echo
echo -e "${RED}Next:${NC}"
echo -e "${RED}  python scripts/summarize_sud_baselines.py --benchmark tofu \\${NC}"
echo -e "${RED}      --baselines-root $(dirname "${BASELINES_ROOT}")${NC}"
echo -e "${RED}  then add these method names to DRAFT_METHODS in${NC}"
echo -e "${RED}      scripts/run_sud_original_unlearned_draft.sh${NC}"

# ── HOLD THE ALLOCATION ───────────────────────────────────────────────────────
# The work is done, but the 3 GPUs are not given back: on a FIFO scheduler with
# no priority aging, an allocation released early costs more to re-acquire than
# it saves. Sit here until Slurm enforces the time limit.

if [[ "${HOLD:-1}" == "0" || -z "${SLURM_JOB_ID:-}" ]]; then
  exit "${overall}"
fi

end_time=$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null \
           | grep -oP 'EndTime=\K[^ ]+' || true)

echo
echo -e "${GREEN}=== Training complete (overall status: ${overall}) ===${NC}"
echo -e "${GREEN}Holding this allocation until the time limit${end_time:+ (EndTime=${end_time})}.${NC}"
echo -e "${GREEN}All 3 GPUs are now idle and free for interactive use, e.g.${NC}"
echo -e "${GREEN}    CUDA_DEVICES=0 bash scripts/run_sud_original_unlearned_draft.sh${NC}"
echo -e "${GREEN}Release it early with: scancel ${SLURM_JOB_ID}${NC}"
echo -e "${GREEN}Skip the hold next time with: HOLD=0 sbatch ...${NC}"

# Slurm terminates the job at the time limit; this never returns on its own.
sleep infinity
