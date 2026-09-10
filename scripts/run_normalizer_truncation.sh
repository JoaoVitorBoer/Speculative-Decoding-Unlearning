#!/bin/bash

#SBATCH --output=/home/joaoabitante/Sout/%j__%x.out
#SBATCH --error=/home/joaoabitante/Sout/%j__%x.out
#SBATCH --job-name=normtrunc

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=08:00:00
#SBATCH --gpus=1

# run_normalizer_truncation.sh — full-vocabulary SUD normalizer sweep.
#
#   Runs scripts/normalizer_truncation.py, which measures
#   TV(pi, pi_hat_k) = (Z - Zhat_k)/Z at every k in [1, |V|] over every TOFU
#   forget-set decoding position, for three drafts x three splits, at alpha=0.5.
#   S is the union of top-k(p) and top-k(q), the set Algorithm 1 forms.
#
#   Outputs, all under saves/analysis/normalizer_truncation/:
#     normalizer_truncation_full_curve.npz  — (alphas, |V|) mean and max per run
#     normalizer_truncation_aggregate.csv   — the K_VALUES grid, with quantiles
#     normalizer_truncation.{png,pdf}       — the figure
#     run_config.json                       — checkpoints, position counts, health
#
#   The figure can be redrawn from the .npz on any CPU, no GPU and no models:
#     python -c "import sys; sys.path.insert(0,'scripts'); \
#                import normalizer_truncation as n; n.replot_from_csv()"
#   Do that rather than resubmitting this job whenever only the drawing changes.

set -euo pipefail

PYTHON="${PYTHON:-/home/joaoabitante/miniconda3/envs/unlearning/bin/python}"
REPO="${REPO:-/home/joaoabitante/speculative-decoding-unlearning}"

cd "${REPO}"

echo "host=$(hostname)  date=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun "${PYTHON}" scripts/normalizer_truncation.py
