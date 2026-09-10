#!/bin/bash
#SBATCH --job-name=alpha_sweep
#SBATCH --output=/home/joaoabitante/speculative-decoding-unlearning/plots/logs/%j__%x.out
#SBATCH --error=/home/joaoabitante/speculative-decoding-unlearning/plots/logs/%j__%x.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --time=09:00:00
#SBATCH --gpus=1
#SBATCH --signal=B:USR1@900
#
# plots/alpha_sweep_job.sbatch — the ONE SLURM job for plots/figs/fig_alpha_tradeoff
# (QOS caps a user at 2 submitted jobs, so all parts are chained here).
#   Part B  plots/bench_speculative.py   13 alphas x k in {1, 4} -> plots/bench/alpha_bench.json
#           (its own file: --resume keeps only the requested (alpha, k) cells, so
#           pointing it at spec_bench.json would drop the k-sweep cells; ~26 x 3.5 min)
#   Part F  plots/make_alpha_figure.py   figure + CSV + README_alpha.md (drawn after every part)
#   Part E  PASSES=c plots/run_k_sweep.sh   TOFU evals at k=1 seed 0 for the alphas the main
#           grid lacks (0.0 0.1 0.2 0.3 0.4 0.6 0.98 1.0) -> plots/results/ (~30 min each at 1B)
#   Part K  plots/make_figure.py         re-render the k-sweep README / figure with the
#           per-round theory (CPU only, seconds)
# Every part is resume-safe: re-submit to fill in what is missing. PARTS selects the parts
# (default "B F E F K"):   sbatch --export=ALL,PARTS="F" plots/alpha_sweep_job.sbatch
# 15 min before the time limit SLURM sends USR1: the running part is stopped and the figure is
# drawn from whatever exists; re-submitting continues where it stopped.

set -uo pipefail
REPO=/home/joaoabitante/speculative-decoding-unlearning
cd "$REPO" || exit 1
mkdir -p plots/logs plots/bench plots/results

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate unlearning
export PATH="/usr/local/cuda-12.9/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.9/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HYDRA_FULL_ERROR=1
# SLURM confines the job to its GPU via cgroups and renumbers it as device 0.
export CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ALPHAS="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.98 1.0"
STAGE_LOG=plots/logs/stages_alpha.log
stage() { echo "$(date -Is) [job ${SLURM_JOB_ID:-?}] $*" | tee -a "$STAGE_LOG"; }

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
  if python plots/make_alpha_figure.py; then stage "PART F done"; else stage "PART F FAILED (rc=$?)"; fi
}
on_time_limit() {
  stage "TIME LIMIT in ~15 min: stopping the running part, drawing the figure from what exists"
  if [[ -n "${CHILD}" ]]; then
    kill -TERM -- "-${CHILD}" 2>/dev/null
    wait "${CHILD}" 2>/dev/null
    CHILD=""
  fi
  draw_figure
  stage "END (time limit; re-submit plots/alpha_sweep_job.sbatch to resume)"
  exit 0
}
trap on_time_limit USR1

stage "START host=$(hostname) python=$(which python) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"

PARTS="${PARTS:-B F E F K}"
stage "parts to run (in order): ${PARTS}"

for part in ${PARTS}; do
  case "${part}" in
    B)
      stage "PART B start (alpha benchmark, k in {1,4}, --resume)"
      # shellcheck disable=SC2086
      if run_part python plots/bench_speculative.py --alphas ${ALPHAS} --ks 1 4 \
            --out plots/bench/alpha_bench.json --resume; then
        stage "PART B done"
      else
        stage "PART B FAILED (rc=$?) — continuing"
      fi
      ;;
    F)
      stage "PART F start (alpha figure)"
      draw_figure
      ;;
    E)
      stage "PART E start (evals for the alphas the main grid lacks: pass c)"
      if PASSES=c run_part bash plots/run_k_sweep.sh; then
        stage "PART E done"
      else
        stage "PART E finished with failures (see plots/logs/failed_runs.log) — continuing"
      fi
      ;;
    K)
      stage "PART K start (k-sweep README/figure with the per-round theory)"
      if python plots/make_figure.py; then stage "PART K done"; else stage "PART K FAILED (rc=$?)"; fi
      ;;
    *) stage "unknown part '${part}' — skipped" ;;
  esac
done

stage "END"
