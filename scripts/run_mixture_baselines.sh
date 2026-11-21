#!/bin/bash
#
# run_mixture_baselines.sh — the two "other ways to go from p to q" ablations of
# SUD, on the same (p, q) checkpoint pairs and the same α grid as the SUD sweep:
#
#   weightinterp  θ_α = (1-α)·θ_p + α·θ_q, decoded as ONE model exactly like the
#                 sampled-q baseline (scripts/run_tofu_sampled_evals.sh: do_sample,
#                 T=1, top_k=0, top_p=1, config batch size). Merged AND run in
#                 float32 + sdpa: p and q differ by ~1 bf16 ulp, so a bf16 merge
#                 snaps to θ_p / θ_q (src/model/weight_interp.py).
#   linear        π = (1-α)·p + α·q through the unchanged SUD draft-verify wrapper
#                 (model=sud, +model.model_args.mixture=linear), k_sud=1 and the
#                 same batch size as the SUD grid.
#
#   p = open-unlearning/tofu_<model>_full, q = JoaoBoer/tofu_<model>_<split>_<method>
#
# Results (picked up by scripts/summarize_sud_results.py like any other combo):
#   saves/unlearn/sud/results/original-unlearned-weightinterp/target-*/draft-*/<split>/a<α>_k0_s<seed>/
#   saves/unlearn/sud/results/original-unlearned-linear/target-*/draft-*/<split>/a<α>_k1_s<seed>/
# (k0 = no draft-verify; run_meta.json carries "family" and the precision.)
#
# Work unit = (family, model, method, split): fetch q once into /tmp (Xet on, no
# chunk cache, nothing on the group quota), run every missing α, delete q.
# Order: all weightinterp units first (cheap), then linear; 1B before 3B.
# Workers share units through non-blocking flocks; every eval resumes metric by
# metric (overwrite: false), so re-running the same command resumes.
#
#   bash scripts/run_mixture_baselines.sh in-job <jobid> ["<gpu ...>"]   # default gpu "2"
#   GPU=2 bash scripts/run_mixture_baselines.sh run
#   DRY_RUN=1 bash scripts/run_mixture_baselines.sh run
#   FAMILIES="linear" MODELS="Llama-3.2-1B-Instruct" METHODS="NPO" SPLITS="forget01" ALPHAS="0.5" ...
#
# in-job starts one `srun --overlap` worker step per listed GPU, logging to
# logs/mixture/<jobid>_gpu<g>.out. A step dies with its srun client, i.e. with
# the shell that launched it. Without Slurm, start one `GPU=<g> ... run` per GPU.
#
# Needs: the retain eval logs saves/eval/tofu_<model>_retain{99,95,90}/TOFU_EVAL.json
# (python setup_data.py --eval_logs) for forget_quality; the TOFU dataset in the
# HF cache, or HF_DATASETS_OFFLINE=0 to fetch it; conda env $CONDA_ENV (default
# "unlearning"; CONDA_ENV= uses the active environment); scratch space under
# $TMP_ROOT (default /tmp/claude-<uid>) for one staged draft per worker.

set -uo pipefail
export HYDRA_FULL_ERROR=1
REPO="$(dirname "$(readlink -f "$0")")/.."
cd "${REPO}" || exit 1
[[ -f src/eval.py ]] || { echo "not in the repo root: $(pwd)" >&2; exit 1; }

GPU="${GPU:-2}"
stamp() { echo "== $(date '+%F %T') [${SLURM_JOB_ID:-local}/g${GPU}] $*"; }

if [[ "${1:-}" == "in-job" ]]; then
  JOBID="${2:?usage: $0 in-job <jobid> [\"<gpu ...>\"]}"
  read -ra JOB_GPUS <<< "${3:-2}"
  unset_args=(-u CUDA_VISIBLE_DEVICES)
  for v in "${!SLURM@}"; do unset_args+=(-u "${v}"); done
  mkdir -p logs/mixture
  for g in "${JOB_GPUS[@]}"; do
    out="${PWD}/logs/mixture/${JOBID}_gpu${g}.out"
    pass_env=()
    for v in FAMILIES MODELS METHODS SPLITS ALPHAS SEED; do
      [[ -n "${!v:-}" ]] && pass_env+=("${v}=${!v}")
    done
    setsid nohup env "${unset_args[@]}" GPU="${g}" "${pass_env[@]}" \
      srun --jobid="${JOBID}" --overlap --ntasks=1 --cpus-per-task=4 \
        --job-name="mix_${JOBID}_g${g}" --output="${out}" --open-mode=append --input=none \
        bash "${PWD}/scripts/run_mixture_baselines.sh" run >/dev/null 2>&1 </dev/null &
    echo "worker on gpu ${g} of job ${JOBID} -> ${out}"
    sleep 20
  done
  squeue -s -j "${JOBID}" -h -o "  %.12i %.18j %.10M" | grep "mix_"
  exit 0
fi

CONDA_ENV="${CONDA_ENV-unlearning}"
if [[ -n "${CONDA_ENV}" ]]; then
  { source "${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}" && conda activate "${CONDA_ENV}"; } \
    || { echo "cannot activate conda env ${CONDA_ENV} (CONDA_ENV= uses the active environment)" >&2; exit 1; }
fi

read -ra FAMILIES <<< "${FAMILIES:-weightinterp linear}"
read -ra MODELS <<< "${MODELS:-Llama-3.2-1B-Instruct Llama-3.2-3B-Instruct}"
read -ra METHODS <<< "${METHODS:-GradDiff IdkDPO NPO RMU SimNPO UNDIAL}"
read -ra SPLITS <<< "${SPLITS:-forget01 forget05 forget10}"
SEED="${SEED:-0}"
# The SUD grid (scripts/run_sud.sh ALPHAS), for both families (ALPHAS="0 1" adds the endpoints).
SUD_ALPHAS="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.98 0.99 0.995"
alphas_for() {
  if [[ -n "${ALPHAS:-}" ]]; then echo "${ALPHAS}"; return; fi
  case "$1" in
    weightinterp) echo "${SUD_ALPHAS}" ;;
    linear)       echo "${SUD_ALPHAS}" ;;
  esac
}
RESULTS_ROOT="saves/unlearn/sud/results"
Q_ROOT="saves/unlearn/baselines_sampled/tofu"   # sampled-q s0, for the α=1 check
HUB_NAMESPACE="JoaoBoer"
TMP_ROOT="${TMP_ROOT:-/tmp/claude-$(id -u)}"
STAGE="${STAGE:-${TMP_ROOT}/mix_drafts/g${GPU}}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="logs/mixture/${SLURM_JOB_ID:-local}"
LOCK_DIR="logs/mixture/locks"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET=0
export HF_XET_CACHE="${HF_XET_CACHE:-${TMP_ROOT}/xet_cache}"
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export HF_XET_HIGH_PERFORMANCE=1
mkdir -p "${LOG_DIR}" "${LOCK_DIR}" "${STAGE}"

retain_of() { local pct="${1#forget}"; echo "retain$(printf '%02d' $((100 - 10#${pct})))"; }

is_done() {  # dir...: true when every dir has a summary with all 12 metrics
  python - "$@" <<'EOF'
import json, sys
REQUIRED = {"exact_memorization", "extraction_strength", "forget_Q_A_PARA_Prob",
            "forget_Q_A_gibberish", "forget_quality", "forget_truth_ratio", "mia_loss",
            "mia_min_k", "mia_min_k_plus_plus", "mia_zlib", "model_utility", "privleak"}
def done(d):
    try:
        s = json.load(open(f"{d}/TOFU_SUMMARY.json"))
        return all(s.get(k) is not None for k in REQUIRED)
    except Exception:
        return False
sys.exit(0 if all(done(d) for d in sys.argv[1:]) else 1)
EOF
}

# weightinterp α=1 (fp32 q) vs the bf16 sampled-q s0: forward metrics should be
# close; the gap is the fp32-vs-bf16 precision offset. Informational only.
compare_q() {  # q_s0_dir run_dir
  python - "$1" "$2" <<'EOF'
import json, sys
try:
    a = json.load(open(f"{sys.argv[1]}/TOFU_SUMMARY.json")); b = json.load(open(f"{sys.argv[2]}/TOFU_SUMMARY.json"))
except Exception as e:
    print(f"no comparison ({e.__class__.__name__})"); sys.exit(0)
keys = ["forget_Q_A_PARA_Prob", "mia_loss", "mia_zlib", "mia_min_k", "forget_truth_ratio"]
print("fp32 a=1 vs bf16 sampled q: " + ", ".join(f"{k} {b[k]-a[k]:+.4f}" for k in keys))
EOF
}

missing_alphas() {  # results_base k alpha... -> prints the alphas whose a<α>_k<k>_s<SEED> is not done
  python - "${SEED}" "$@" <<'EOF'
import json, sys
seed, base, k, *alphas = sys.argv[1:]
REQUIRED = {"exact_memorization", "extraction_strength", "forget_Q_A_PARA_Prob",
            "forget_Q_A_gibberish", "forget_quality", "forget_truth_ratio", "mia_loss",
            "mia_min_k", "mia_min_k_plus_plus", "mia_zlib", "model_utility", "privleak"}
def done(d):
    try:
        s = json.load(open(f"{d}/TOFU_SUMMARY.json"))
        return all(s.get(m) is not None for m in REQUIRED)
    except Exception:
        return False
print(" ".join(a for a in alphas if not done(f"{base}/a{a}_k{k}_s{seed}")))
EOF
}

quarantine_corrupt() {
  local f="$1/TOFU_EVAL.json"
  [[ -f "${f}" ]] || return 0
  python -c "import json,sys; json.load(open(sys.argv[1]))" "${f}" 2>/dev/null && return 0
  mv "${f}" "${f}.corrupt.$(date +%s)" && stamp "!! moved aside unreadable ${f}"
}

LOCK_FD=""
try_lock() {
  exec {LOCK_FD}>"${LOCK_DIR}/$1.lock" || return 1
  flock -n "${LOCK_FD}" && return 0
  exec {LOCK_FD}>&-
  return 1
}
unlock() { exec {LOCK_FD}>&-; }

draft_complete() {  # dir: every shard in the index (or model.safetensors) present
  python - "$1" <<'EOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
if (d / "model.safetensors").is_file():
    sys.exit(0)
idx = d / "model.safetensors.index.json"
shards = set(json.loads(idx.read_text())["weight_map"].values()) if idx.is_file() else {None}
sys.exit(0 if None not in shards and all((d / s).is_file() for s in shards) else 1)
EOF
}

HF_CLI="$(command -v hf || command -v huggingface-cli)" || { echo "no hf / huggingface-cli on PATH" >&2; exit 1; }
fetch_draft() {  # method model split dir
  draft_complete "$4" && return 0
  local try
  for try in 1 2 3; do
    "${HF_CLI}" download "${HUB_NAMESPACE}/tofu_$2_$3_$1" --local-dir "$4" \
      --quiet --exclude 'evals/*' README.md .gitattributes >/dev/null 2>&1 && draft_complete "$4" && return 0
    stamp "!! fetch attempt ${try} of $1 $2 $3 failed; retrying in 60 s"
    sleep 60
  done
  return 1
}

run_dir_for() {  # family target draft split alpha hub_repo -> prints dir (creates it, writes run_meta.json)
  local family="$1" target="$2" draft="$3" split="$4" alpha="$5" hub="$6" k dtype
  case "${family}" in
    weightinterp) k=0; dtype=float32 ;;
    linear)       k=1; dtype=bfloat16 ;;
  esac
  local d
  d="$(python scripts/sud_paths.py resolve --root "${RESULTS_ROOT}" \
        --combo "original-unlearned-${family}" --target "${target}" --draft "${draft}" \
        --split "${split}" --alpha "${alpha}" --k "${k}" --seed "${SEED}")" || return 1
  python - "${d}/run_meta.json" "${family}" "${dtype}" "${hub}" <<'EOF'
import json, sys
f, fam, dt, hub = sys.argv[1:]
m = json.load(open(f))
m.update(family=fam, draft_hub=hub, mixture={"weightinterp": "weights", "linear": "linear"}[fam], model_dtype=dt,
         note={"weightinterp": "theta=(1-a)theta_p+a theta_q merged in fp32, single-model sampled decode (T=1, top_k=0, top_p=1)",
               "linear": "pi=(1-a)p+a q via SUD draft-verify (k_sud=1)"}[fam])
json.dump(m, open(f, "w"), indent=2); open(f, "a").write("\n")
EOF
  echo "${d}"
}

# weightinterp α=0 is p itself (the merge returns θ_p exactly) for every method:
# evaluate it once per (model, split) and copy that leaf to the other methods.
reuse_p_anchor() {  # model method split draft_dir hub_repo -> 0 when a0 was filled by a copy
  local model="$1" method="$2" split="$3" draft="$4" hub="$5" src dst
  dst="${RESULTS_ROOT}/original-unlearned-weightinterp/target-${model}/draft-${model}_${method}/${split}/a0_k0_s0"
  for src in "${RESULTS_ROOT}"/original-unlearned-weightinterp/target-"${model}"/draft-"${model}"_*/"${split}"/a0_k0_s0; do
    [[ "${src}" == "${dst}" ]] && continue
    is_done "${src}" || continue
    dst="$(run_dir_for weightinterp "open-unlearning/tofu_${model}_full" "${draft}" "${split}" 0 "${hub}")" || return 1
    cp -r "${src}/.hydra" "${src}/TOFU_EVAL.json" "${src}/TOFU_SUMMARY.json" "${dst}/" || return 1
    for extra in eval.log profiling.json; do [[ -f "${src}/${extra}" ]] && cp "${src}/${extra}" "${dst}/"; done
    python - "${dst}/run_meta.json" "${src}" <<'EOF'
import json, sys
f, src = sys.argv[1:]
m = json.load(open(f)); m["copied_from"] = src
m["note"] += "; alpha=0 is p for every method, so this leaf is a copy of another method's a0"
json.dump(m, open(f, "w"), indent=2); open(f, "a").write("\n")
EOF
    stamp "a0 for ${method} ${model} ${split}: copied p anchor from ${src}"
    return 0
  done
  return 1
}

eval_one() {  # family model split target draft alpha out log
  local family="$1" model="$2" split="$3" target="$4" draft="$5" alpha="$6" out="$7" log="$8"
  local pct="${split#forget}"
  case "${family}" in
    weightinterp)
      CUDA_DEVICES="${GPU}" SEED="${SEED}" bash scripts/run_tofu_sampled_evals.sh \
        "${target}" "${out}" "${split}" "${model}" \
        ++model.model_handler=WeightInterpModelForCausalLM \
        ++model.model_args.draft_model_name_or_path="${draft}" \
        ++model.model_args.alpha="${alpha}" \
        model.model_args.torch_dtype=float32 \
        model.model_args.attn_implementation=sdpa \
        >> "${log}" 2>&1 ;;
    linear)
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      CUDA_VISIBLE_DEVICES="${GPU}" python src/eval.py \
        experiment=eval/tofu/default.yaml \
        model=sud \
        eval.tofu.batch_size=32 \
        model.model_args.device_map=auto \
        model.model_args.pretrained_model_name_or_path="${target}" \
        model.model_args.draft_model_name_or_path="${draft}" \
        model.model_args.alpha="${alpha}" \
        model.model_args.k_sud=1 \
        +model.model_args.mixture=linear \
        model.tokenizer_args.pretrained_model_name_or_path="${target}" \
        seed="${SEED}" \
        forget_split="${split}" \
        holdout_split="holdout${pct}" \
        retain_logs_path="saves/eval/tofu_${model}_$(retain_of "${split}")/TOFU_EVAL.json" \
        task_name="tofu_${model}_${split}_linear_$(basename "$(dirname "${out}")")_a${alpha}_s${SEED}" \
        paths.output_dir="${out}" \
        >> "${log}" 2>&1 ;;
  esac
}

tag="$(python scripts/sud_paths.py tag "${STAGE}/linear/baselines/tofu/NPO/Llama-3.2-1B-Instruct/forget01")"
[[ "${tag}" == "Llama-3.2-1B-Instruct_NPO" ]] || { echo "staged path tags as ${tag}, expected Llama-3.2-1B-Instruct_NPO" >&2; exit 1; }
stamp "worker start: families ${FAMILIES[*]}, models ${MODELS[*]}, methods ${METHODS[*]}, splits ${SPLITS[*]}, seed ${SEED}, stage ${STAGE}$([[ "${DRY_RUN}" == 1 ]] && echo ' (dry run)')"
declare -A failed=()
n_would=0
while :; do
  ran=0
  for family in "${FAMILIES[@]}"; do
    read -ra alphas <<< "$(alphas_for "${family}")"
    for model in "${MODELS[@]}"; do
      target="open-unlearning/tofu_${model}_full"
      for method in "${METHODS[@]}"; do
        for split in "${SPLITS[@]}"; do
          unit="${family}_${model}_${method}_${split}"
          [[ -n "${failed[${unit}]:-}" ]] && continue
          # The staged path keeps .../baselines/tofu/<method>/<model>/<split> so
          # scripts/sud_paths.py tags the draft as <model>_<method>.
          d="${STAGE}/${family}/baselines/tofu/${method}/${model}/${split}"
          k=$([[ ${family} == weightinterp ]] && echo 0 || echo 1)
          rbase="${RESULTS_ROOT}/original-unlearned-${family}/target-${model}/draft-${model}_${method}/${split}"
          read -ra todo <<< "$(missing_alphas "${rbase}" "${k}" "${alphas[@]}")"
          [[ ${#todo[@]} -eq 0 ]] && continue
          try_lock "${unit}" || continue
          if [[ "${family}" == weightinterp && " ${todo[*]} " == *" 0 "* && "${DRY_RUN}" != 1 ]] \
             && reuse_p_anchor "${model}" "${method}" "${split}" "${d}" "${HUB_NAMESPACE}/tofu_${model}_${split}_${method}"; then
            read -ra todo <<< "$(missing_alphas "${rbase}" "${k}" "${alphas[@]}")"
            [[ ${#todo[@]} -eq 0 ]] && { unlock; continue; }
          fi
          ran=1
          if [[ "${DRY_RUN}" == 1 ]]; then
            stamp "would run ${family} ${model} ${method} ${split} α ${todo[*]}"
            n_would=$(( n_would + ${#todo[@]} )); failed[${unit}]=dry; unlock; continue
          fi
          t0=$(date +%s)
          if ! fetch_draft "${method}" "${model}" "${split}" "${d}"; then
            stamp "!! FAILED fetch ${unit}"; failed[${unit}]=fetch; rm -rf "${d}"; unlock; continue
          fi
          stamp "fetched ${method} ${model} ${split} ($(( $(date +%s) - t0 )) s) for ${family}: ${#todo[@]} α"
          for a in "${todo[@]}"; do
            out="$(run_dir_for "${family}" "${target}" "${d}" "${split}" "${a}" "${HUB_NAMESPACE}/tofu_${model}_${split}_${method}")" || { failed[${unit}]=path; break; }
            is_done "${out}" && continue
            quarantine_corrupt "${out}"
            log="${LOG_DIR}/${unit}_a${a}.log"
            t0=$(date +%s)
            if eval_one "${family}" "${model}" "${split}" "${target}" "${d}" "${a}" "${out}" "${log}" \
               && is_done "${out}"; then
              extra=""
              [[ "${family}" == weightinterp && "${a}" == 1 ]] && extra=" — $(compare_q "${Q_ROOT}/${method}/${model}/${split}/evals" "${out}")"
              stamp "done ${family} ${model} ${method} ${split} α=${a} ($(( ($(date +%s) - t0) / 60 )) min)${extra}"
            else
              stamp "!! FAILED ${family} ${model} ${method} ${split} α=${a} — see ${log}"
              failed[${unit}]=eval
            fi
          done
          rm -rf "${d}"
          unlock
        done
      done
    done
  done
  [[ ${ran} -eq 0 ]] && break
done

[[ "${DRY_RUN}" == 1 ]] && stamp "dry run: ${n_would} eval(s) to run"
stamp "nothing left to claim; ${#failed[@]} unit(s) skipped/failed"
for key in "${!failed[@]}"; do echo "   ${key} (${failed[${key}]})"; done
[[ "${DRY_RUN}" == 1 || ${#failed[@]} -eq 0 ]]
