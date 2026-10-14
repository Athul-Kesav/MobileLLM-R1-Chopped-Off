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
    elif [[ -x "$PWD/venv/bin/python" ]]; then
      PYTHON="$PWD/venv/bin/python"
    elif [[ -x "$PWD/.venv/bin/python" ]]; then
      PYTHON="$PWD/.venv/bin/python"
    else
      PYTHON=python3
    fi
  fi

  # Keep downloads and tool caches inside this checkout.
  mkdir -p "$PWD/.hf_cache" "$PWD/.cache/pip" "$PWD/.cache/tmp"
  export HF_HOME="$PWD/.hf_cache"
  export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
  export HF_DATASETS_CACHE="$HF_HOME/datasets"
  export TRANSFORMERS_CACHE="$HF_HOME"
  export PIP_CACHE_DIR="$PWD/.cache/pip"
  export TMPDIR="$PWD/.cache/tmp"
  export XDG_CACHE_HOME="$PWD/.cache"

  if [[ "${NPROC_PER_NODE:-auto}" == auto ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      local csv="${CUDA_VISIBLE_DEVICES// /}"
      local IFS=,
      # shellcheck disable=SC2086
      set -- $csv
      NPROC_PER_NODE=$#
    else
      # Skip cards another job has already filled. A 140M SFT step fits in a few GiB.
      local min_free_mib="${MIN_FREE_GPU_MIB:-8192}"
      local free_gpus=()
      local idx free
      while IFS=, read -r idx free; do
        idx="${idx// /}"
        free="${free// /}"
        if [[ "$idx" =~ ^[0-9]+$ && "$free" =~ ^[0-9]+$ && "$free" -ge "$min_free_mib" ]]; then
          free_gpus+=("$idx")
        fi
      done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null || true)
      if [[ ${#free_gpus[@]} -ge 1 ]]; then
        local IFS=,
        export CUDA_VISIBLE_DEVICES="${free_gpus[*]}"
        NPROC_PER_NODE=${#free_gpus[@]}
        echo "Using GPUs with at least ${min_free_mib} MiB free: ${CUDA_VISIBLE_DEVICES}"
      else
        echo "No GPU has ${min_free_mib} MiB free. Set CUDA_VISIBLE_DEVICES or MIN_FREE_GPU_MIB." >&2
        NPROC_PER_NODE=0
      fi
    fi
    if [[ -z "${NPROC_PER_NODE}" || "${NPROC_PER_NODE}" -lt 1 ]]; then
      NPROC_PER_NODE=0
    fi
  fi

  # A sourced env file sets this in the shell only. torchrun will not see it
  # unless it is exported, and then rank 1 lands on physical GPU 1.
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES
  fi

  : "${MASTER_ADDR:=127.0.0.1}" "${MASTER_PORT:=29521}"
  local site_packages
  site_packages="$(dirname "$(dirname "$PYTHON")")/lib/python3.10/site-packages"
  if [[ -d "$site_packages" ]]; then
    export PYTHONPATH="$site_packages:$PWD${PYTHONPATH:+:$PYTHONPATH}"
  else
    export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
  fi
  : "${SHISHU_DIR:=models/MobileLLM-R1-140M-shishu}"
  : "${BASE_REPO:=facebook/MobileLLM-R1-140M-base}"
  : "${BASE_DIR:=models/MobileLLM-R1-140M-base}"
  : "${CHAT_TEMPLATE:=models/MobileLLM-R1-140M-shishu/chat_template.jinja}"
  : "${COMPARE_DATA:=compare/data}"
  : "${RUN_ROOT:=compare/runs}"
  : "${LOG_ROOT:=compare/logs}"
  : "${RESULT_ROOT:=compare/results}"
  : "${SEQ_LEN:=2048}"
  : "${SAVE_STEPS:=50}" "${USE_LIGER_KERNEL:=False}"
}
