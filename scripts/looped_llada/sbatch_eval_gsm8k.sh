#!/bin/bash
# One-shot SLURM eval: SSH tunnel + pyxis container + eval_gsm8k.sh.
#
# Usage:
#   # default (eval both loop & baseline at latest checkpoint):
#   sbatch scripts/looped_llada/sbatch_eval_gsm8k.sh
#
#   # forward args to eval_gsm8k.sh after the script name:
#   sbatch scripts/looped_llada/sbatch_eval_gsm8k.sh --target loop --limit 20
#   sbatch scripts/looped_llada/sbatch_eval_gsm8k.sh --target loop --mu_rec_eval 4
#
#   # override container / repo location via env:
#   REPO_ROOT=/path/to/dllm sbatch scripts/looped_llada/sbatch_eval_gsm8k.sh ...
#
#SBATCH -J solar-lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=8
#SBATCH --mem=128GB
#SBATCH -t 06:00:00
#SBATCH -o logs/eval_gsm8k_%j.out
#SBATCH -e logs/eval_gsm8k_%j.err

set -euo pipefail
mkdir -p logs

# ---------- knobs (override via env when sbatch'ing) ----------
REPO_ROOT="${REPO_ROOT:-/lustre/projects/polyullm/lipengxiang_tmp/dllm}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/projects/polyullm/lipengxiang_tmp/my_env.sqsh}"

PROXY_LOCAL_PORT="${PROXY_LOCAL_PORT:-20172}"
PROXY_REMOTE="${PROXY_REMOTE:-10.120.32.132:8080}"
SSH_TARGET="${SSH_TARGET:-reallm.xyz@10.112.0.4}"

# HF cache must live on a writable mount — container $HOME is read-only.
HF_HOME_DEFAULT="/lustre/projects/polyullm/lipengxiang_tmp/.cache/huggingface"
HF_HOME_DIR="${HF_HOME_DIR:-${HF_HOME_DEFAULT}}"
mkdir -p "${HF_HOME_DIR}"

# Everything after the script name on the sbatch line goes to eval_gsm8k.sh.
EVAL_ARGS=("$@")

cd "${REPO_ROOT}"

# ---------- 1. SSH tunnel (background, on this compute node, outside container) ----------
# pyxis containers share the host network namespace by default, so a port
# forwarded on the compute node's localhost is reachable as `localhost:PORT`
# from inside the container as well.
echo "[$(date)] $(hostname): opening tunnel localhost:${PROXY_LOCAL_PORT} -> ${PROXY_REMOTE} via ${SSH_TARGET}"
ssh -N -f \
  -o BatchMode=yes \
  -o ConnectTimeout=10 \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -o StrictHostKeyChecking=accept-new \
  -L "${PROXY_LOCAL_PORT}:${PROXY_REMOTE}" \
  "${SSH_TARGET}"

SSH_PID="$(pgrep -f "ssh -N.*${PROXY_LOCAL_PORT}:${PROXY_REMOTE}.*${SSH_TARGET}" | head -n1 || true)"
cleanup() {
  echo "[$(date)] cleanup: stopping ssh tunnel (pid=${SSH_PID:-?})"
  [[ -n "${SSH_PID:-}" ]] && kill "${SSH_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Wait up to ~30s for the local port to start listening.
for i in {1..15}; do
  if (exec 3<>"/dev/tcp/127.0.0.1/${PROXY_LOCAL_PORT}") 2>/dev/null; then
    exec 3<&-; exec 3>&-
    echo "[$(date)] tunnel up on localhost:${PROXY_LOCAL_PORT}"
    break
  fi
  sleep 2
done

PROXY_URL="http://localhost:${PROXY_LOCAL_PORT}"

# Sanity-check it actually proxies HTTPS (don't fail the job on this).
if command -v curl >/dev/null 2>&1; then
  curl -sS --max-time 10 -o /dev/null -w "[$(date)] proxy probe https://huggingface.co => HTTP %{http_code}\n" \
    -x "${PROXY_URL}" https://huggingface.co || true
fi

# ---------- 2. Run eval inside container via pyxis ----------
# Export proxy vars in this shell so --container-env can forward them.
export HTTP_PROXY="${PROXY_URL}"
export HTTPS_PROXY="${PROXY_URL}"
export http_proxy="${PROXY_URL}"
export https_proxy="${PROXY_URL}"
export NO_PROXY="localhost,127.0.0.1,.local,.cluster"
export no_proxy="localhost,127.0.0.1,.local,.cluster"
export HF_DATASETS_TRUST_REMOTE_CODE=True
# Redirect every HF / generic-cache path to a writable mount.
export HF_HOME="${HF_HOME_DIR}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME_DIR}/hub"
export HF_DATASETS_CACHE="${HF_HOME_DIR}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME_DIR}/hub"
export XDG_CACHE_HOME="${HF_HOME_DIR}/.xdg"

CONTAINER_MOUNTS="/work/projects/polyullm:/work/projects/polyullm"
CONTAINER_MOUNTS+=",/work/projects/polyullm:/home/projects/polyullm"
CONTAINER_MOUNTS+=",/lustre/projects/polyullm:/lustre/projects/polyullm"

echo "[$(date)] launching eval_gsm8k.sh inside ${CONTAINER_NAME} with args: ${EVAL_ARGS[*]:-<none>}"

srun --nodes=1 --ntasks=1 \
  --container-name="${CONTAINER_NAME}" \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${CONTAINER_MOUNTS}" \
  --container-env=HTTP_PROXY,HTTPS_PROXY,http_proxy,https_proxy,NO_PROXY,no_proxy,HF_DATASETS_TRUST_REMOTE_CODE,HF_HOME,HUGGINGFACE_HUB_CACHE,HF_DATASETS_CACHE,TRANSFORMERS_CACHE,XDG_CACHE_HOME \
  bash -lc 'cd "$1" && shift && bash scripts/looped_llada/eval_gsm8k.sh "$@"' \
  _wrap "${REPO_ROOT}" ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"}

echo "[$(date)] eval finished"
