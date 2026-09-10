#!/bin/bash
#SBATCH --job-name=k_sweep
#SBATCH --output=/home/joaoabitante/speculative-decoding-unlearning/plots/logs/%j__%x.out
#SBATCH --error=/home/joaoabitante/speculative-decoding-unlearning/plots/logs/%j__%x.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --time=20:00:00
#SBATCH --gpus=1
#SBATCH --signal=B:USR1@900
#
# plots/k_sweep_job.sbatch — the ONE SLURM job for the k-sweep figure
# (QOS `normal` allows 2 submitted jobs per user, so all parts are chained here).
# Order: the benchmark first (it must run as one consistent block on one GPU),
# then the evals (each is individually resumable), then the figure:
#   Part 2  plots/bench_speculative.py --resume   27 (alpha, k) cells -> plots/bench/spec_bench.json (~4-5 GPU-h)
#   Part 1  plots/run_k_sweep.sh                  27 TOFU evals -> plots/results/  (~14 GPU-h at 1B; k outer,
#                                                 so a cut-short run leaves a complete grid up to some k)
#   Part 3  plots/make_figure.py                  figure + CSV + README
# Every part is resume-safe / idempotent: re-submit this file to fill in anything
# that is missing (finished eval metrics and finished bench cells are skipped, the
# figure is recomputed). PARTS selects what to run (default: all, in this order):
#   sbatch plots/k_sweep_job.sbatch                          # Parts 2 1 3
#   sbatch --export=ALL,PARTS="3" plots/k_sweep_job.sbatch    # figure only
# Time limit: 20 h so the job BACKFILLS into the GPU slot that is free until the
# other running jobs end (a longer request waits behind higher-priority jobs).
# 15 min before the limit SLURM sends USR1 (--signal above): the running part is
# stopped and the figure is drawn from whatever exists, so the output is never
# empty; re-submitting continues where it stopped.
# Memory: the whole pipeline peaks at ~2.6 GB host RAM (job 201 MaxRSS), so 10G
# is ample and small enough to fit next to the other users' jobs.

set -uo pipefail
REPO=/home/joaoabitante/speculative-decoding-unlearning
cd "$REPO" || exit 1
mkdir -p plots/logs plots/bench plots/results

# Environment: set up explicitly so the job does not depend on the submitting
# shell — same env the interactive evals run in (conda env `unlearning`, CUDA 12.9).
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate unlearning
export PATH="/usr/local/cuda-12.9/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.9/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HYDRA_FULL_ERROR=1
# SLURM confines the job to its GPU via cgroups and renumbers it as device 0,
# so the sweep script's CUDA_VISIBLE_DEVICES=${CUDA_DEVICES:-0} is correct here.
export CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

STAGE_LOG=plots/logs/stages.log
stage() { echo "$(date -Is) [job ${SLURM_JOB_ID:-?}] $*" | tee -a "$STAGE_LOG"; }

# Run a part in its own process group so the time-limit handler can stop it
# (and its python child) with one signal while bash waits interruptibly.
CHILD=""
run_part() {
  setsid "$@" &
  CHILD=$!
  wait "${CHILD}"
  local rc=$?
  CHILD=""
  return ${rc}
}
draw_figure() {
  if python plots/make_figure.py; then stage "PART3 done"; else stage "PART3 FAILED (rc=$?)"; fi
}
on_time_limit() {
  stage "TIME LIMIT in ~15 min: stopping the running part, drawing the figure from what exists"
  if [[ -n "${CHILD}" ]]; then
    kill -TERM -- "-${CHILD}" 2>/dev/null
    wait "${CHILD}" 2>/dev/null
    CHILD=""
  fi
  draw_figure
  stage "END (time limit; re-submit plots/k_sweep_job.sbatch to resume)"
  exit 0
}
trap on_time_limit USR1

stage "START host=$(hostname) python=$(which python) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"

PARTS="${PARTS:-2 1 3}"
stage "parts to run (in order): ${PARTS}"

for part in ${PARTS}; do
  case "${part}" in
    2)
      stage "PART2 start (speculative benchmark, --resume)"
      if run_part python plots/bench_speculative.py --out plots/bench/spec_bench.json --resume; then
        stage "PART2 done"
      else
        stage "PART2 FAILED (rc=$?) — continuing"
      fi
      ;;
    1)
      stage "PART1 start (eval sweep)"
      if run_part bash plots/run_k_sweep.sh; then
        stage "PART1 done"
      else
        stage "PART1 finished with failures (see plots/logs/failed_runs.log) — continuing"
      fi
      ;;
    3)
      stage "PART3 start (figure)"
      draw_figure
      ;;
    *) stage "unknown part '${part}' — skipped" ;;
  esac
done

stage "END"
