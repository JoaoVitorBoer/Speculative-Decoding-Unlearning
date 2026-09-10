#!/bin/bash
# Copy the locally trained TOFU Llama-3.1-8B-Instruct checkpoints from atria
# down to this machine's Downloads folder.
#
# RUN THIS ON YOUR LOCAL PC (not on atria). Needs ssh + rsync.
#
#   ./fetch_8b_models.sh                        # dry run: list what would be copied + sizes
#   ./fetch_8b_models.sh --go                   # copy everything (19 ckpts, ~285 GB)
#   ./fetch_8b_models.sh --go NPO SimNPO        # only these methods
#   ./fetch_8b_models.sh --go --split forget10  # only this split (repeatable)
#   ./fetch_8b_models.sh --go --exclude IdkDPO  # skip a method (repeatable)
#   ./fetch_8b_models.sh --go --base            # also the tofu_..._full target model
#   ./fetch_8b_models.sh --go --evals-only      # metrics JSON only, no weights (~27 MB)
#
# Defaults to the "ciars-login" alias in ~/.ssh/config (atria via the ciars jump
# host). Do NOT use "atria-vscode": its RemoteCommand/RequestTTY breaks rsync.
#
# Connection / destination overrides (env or flag):
#   REMOTE=ciars-login DEST=~/Downloads/x ./fetch_8b_models.sh --go
#   PORT=2222 ...        # only needed when REMOTE is a raw host, not a config alias
#
# Transfers are resumable: re-run the same command after an interruption.
set -euo pipefail

REMOTE="${REMOTE:-ciars-login}"
PORT="${PORT:-}"                 # empty => take the port from ~/.ssh/config
SSH_EXTRA="${SSH_EXTRA:-}"       # extra ssh options
REMOTE_ROOT="${REMOTE_ROOT:-/home/joaoabitante/speculative-decoding-unlearning}"
DEST="${DEST:-$HOME/Downloads/sud_8b_models}"
RSYNC_EXTRA="${RSYNC_EXTRA:-}"   # extra rsync flags, e.g. "-n" to preview or "--bwlimit=20M"
MODEL="Llama-3.1-8B-Instruct"
BASELINE_ROOT="$REMOTE_ROOT/saves/unlearn/baselines/tofu"

GO=0
WANT_BASE=0
EVALS_ONLY=0
METHODS=()
SPLITS=()
EXCLUDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --go)     GO=1; shift ;;
    --base)   WANT_BASE=1; shift ;;
    --split)  SPLITS+=("$2"); shift 2 ;;
    --exclude) EXCLUDES+=("$2"); shift 2 ;;
    --evals-only) EVALS_ONLY=1; shift ;;
    --dest)   DEST="$2"; shift 2 ;;
    --remote) REMOTE="$2"; shift 2 ;;
    --port)   PORT="$2"; shift 2 ;;
    -h|--help) sed -n '2,28p' "$0"; exit 0 ;;
    -*)       echo "unknown flag: $1" >&2; exit 2 ;;
    *)        METHODS+=("$1"); shift ;;
  esac
done

# ClearAllForwardings: ciars-login declares "LocalForward 8501"; without this
#   every connection collides with a dashboard tunnel that is already up.
# ControlMaster: REMOTE goes through a ProxyJump, so reuse one connection for
#   the discovery call and both rsync legs instead of re-authenticating 3x.
SSH_CMD="ssh -o ClearAllForwardings=yes -o RequestTTY=no"
SSH_CMD="$SSH_CMD -o ControlMaster=auto -o ControlPath=/tmp/.cm-f8b-%C -o ControlPersist=120"
[[ -n "$PORT" ]] && SSH_CMD="$SSH_CMD -p $PORT"
[[ -n "$SSH_EXTRA" ]] && SSH_CMD="$SSH_CMD $SSH_EXTRA"

# ---- discover what exists on the remote (path + apparent size in bytes) ------
SUB=""
(( EVALS_ONLY )) && SUB="/evals"

echo "Listing 8B checkpoints on $REMOTE ..."
LISTING=$($SSH_CMD "$REMOTE" \
  "find '$BASELINE_ROOT' -mindepth 3 -maxdepth 3 -type d -path '*/$MODEL/*' | sort | xargs -r -I{} du -sb {}$SUB")

[[ -n "$LISTING" ]] || { echo "No $MODEL checkpoints found under $BASELINE_ROOT" >&2; exit 1; }

# ---- filter by method / split ------------------------------------------------
in_list() { local needle="$1"; shift; local x; for x in "$@"; do [[ "$x" == "$needle" ]] && return 0; done; return 1; }
human()  { awk -v b="$1" 'BEGIN{ if (b>=1073741824) printf "%.1fG", b/1073741824; else if (b>=1048576) printf "%.1fM", b/1048576; else printf "%dK", b/1024 }'; }

SEL_PATHS=()
SEL_BYTES=0
printf '\n%-10s %-9s %8s\n' METHOD SPLIT SIZE
printf '%s\n' "-------------------------------"
while IFS=$'\t' read -r bytes path; do
  [[ -n "${path:-}" ]] || continue
  base="${path%/evals}"
  split=$(basename "$base")
  method=$(basename "$(dirname "$(dirname "$base")")")
  (( ${#METHODS[@]}  )) && ! in_list "$method" "${METHODS[@]}"  && continue
  (( ${#SPLITS[@]}   )) && ! in_list "$split"  "${SPLITS[@]}"   && continue
  (( ${#EXCLUDES[@]} )) &&   in_list "$method" "${EXCLUDES[@]}" && continue
  SEL_PATHS+=("$path")
  SEL_BYTES=$(( SEL_BYTES + bytes ))
  printf '%-10s %-9s %8s\n' "$method" "$split" "$(human "$bytes")"
done <<< "$LISTING"

(( ${#SEL_PATHS[@]} )) || { echo "Nothing matched the given method/split filters." >&2; exit 1; }

TOTAL_G=$(awk -v b="$SEL_BYTES" 'BEGIN{printf "%.1f", b/1073741824}')
printf '%s\n' "-------------------------------"
printf '%d checkpoints, %s total\n\n' "${#SEL_PATHS[@]}" "$(human "$SEL_BYTES")"

# ---- local free-space check -------------------------------------------------
mkdir -p "$DEST"
FREE_KB=$(df -Pk "$DEST" | awk 'NR==2{print $4}')
NEED_KB=$(( SEL_BYTES / 1024 ))
FREE_G=$(awk -v k="$FREE_KB" 'BEGIN{printf "%.1f", k/1048576}')
echo "Destination: $DEST  (${FREE_G} GB free)"
if (( FREE_KB < NEED_KB )); then
  echo "WARNING: not enough free space (need ${TOTAL_G} GB, have ${FREE_G} GB)." >&2
  echo "         Narrow the selection or point --dest at a bigger disk." >&2
  (( GO )) && exit 1
fi

if (( ! GO )); then
  echo
  echo "Dry run. Re-run with --go to start the transfer."
  exit 0
fi

# ---- transfer ---------------------------------------------------------------
# The /./ marker makes rsync -R recreate METHOD/<model>/<split>/ under $DEST.
SRCS=()
for p in "${SEL_PATHS[@]}"; do SRCS+=("${REMOTE}:${p/\/tofu\//\/tofu\/.\/}"); done

echo
echo "Copying ${#SEL_PATHS[@]} checkpoints ..."
rsync -avhP --relative --partial-dir=.rsync-partial ${RSYNC_EXTRA} -e "$SSH_CMD" "${SRCS[@]}" "$DEST/"

if (( WANT_BASE )); then
  echo
  echo "Copying base target model tofu_${MODEL}_full ..."
  SNAP=$($SSH_CMD "$REMOTE" \
    "ls -d \$HOME/.cache/huggingface/hub/models--open-unlearning--tofu_${MODEL}_full/snapshots/*/ | head -1")
  [[ -n "$SNAP" ]] || { echo "base model not found in remote HF cache" >&2; exit 1; }
  # -L resolves the HF cache's blob symlinks into real files.
  rsync -avhPL --partial-dir=.rsync-partial ${RSYNC_EXTRA} -e "$SSH_CMD" \
    "${REMOTE}:${SNAP}" "$DEST/_base/tofu_${MODEL}_full/"
fi

echo
echo "Done -> $DEST"
