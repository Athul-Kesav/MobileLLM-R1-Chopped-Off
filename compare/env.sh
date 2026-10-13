#!/usr/bin/env bash
# Shared by the comparison launchers. Source this, then call load_compare_env.
# Paths in the env file are relative to the repository root.

load_compare_env() {
  local env_file="$1"
  [[ -f "$env_file" ]] || { echo "Missing $env_file" >&2; exit 2; }
  # shellcheck disable=SC1090
  source "$env_file"

  if [[ -z "${PYTHON:-}" || "$PYTHON" == python || "$PYTHON" == python3 ]]; then
    if [[ -x /data/cs26e003/venvs/mobilellm/bin/python ]]; then
      PYTHON=/data/cs26e003/venvs/mobilellm/bin/python
    elif [[ -x "$PWD/.venv/bin/python" ]]; then
      PYTHON="$PWD/.venv/bin/python"
    else
      PYTHON=python3
    fi
  fi

  if [[ "${NPROC_PER_NODE:-auto}" == auto ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      local csv="${CUDA_VISIBLE_DEVICES// /}"
      local IFS=,
      # shellcheck disable=SC2086
      set -- $csv
      NPROC_PER_NODE=$#
    else
      NPROC_PER_NODE="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    fi
    if [[ -z "${NPROC_PER_NODE}" || "${NPROC_PER_NODE}" -lt 1 ]]; then
      NPROC_PER_NODE=0
    fi
  fi

  : "${MASTER_ADDR:=127.0.0.1}" "${MASTER_PORT:=29521}"
  : "${SHISHU_DIR:=models/MobileLLM-R1-140M-shishu}"
  : "${BASE_REPO:=facebook/MobileLLM-R1-140M-base}"
  : "${BASE_DIR:=models/MobileLLM-R1-140M-base}"
  : "${CHAT_TEMPLATE:=models/MobileLLM-R1-140M/chat_template.jinja}"
  : "${COMPARE_DATA:=compare/data}"
  : "${RUN_ROOT:=compare/runs}"
  : "${LOG_ROOT:=compare/logs}"
  : "${RESULT_ROOT:=compare/results}"
  : "${SEQ_LEN:=2048}"
  : "${SAVE_STEPS:=50}" "${USE_LIGER_KERNEL:=False}"
}
