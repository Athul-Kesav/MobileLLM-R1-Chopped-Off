# Shared by prepare/run. Source an env file first, then call resolve_training_paths.
# Relative paths are resolved against the repository root (cwd after the launcher cds).

repo_path() {
  local p="${1:-}"
  [[ -z "$p" ]] && return 0
  case "$p" in
    /*) printf '%s\n' "$p" ;;
    *) printf '%s\n' "$PWD/$p" ;;
  esac
}

detect_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local csv="${CUDA_VISIBLE_DEVICES// /}"
    if [[ -z "$csv" ]]; then
      echo 0
      return
    fi
    local IFS=,
    # shellcheck disable=SC2086
    set -- $csv
    echo "$#"
    return
  fi
  nvidia-smi -L 2>/dev/null | wc -l
}

resolve_training_paths() {
  PYTHON="${PYTHON:-python}"
  if [[ "$PYTHON" == "python" || "$PYTHON" == "python3" ]] && [[ -x "$PWD/.venv/bin/python" ]]; then
    PYTHON="$PWD/.venv/bin/python"
  fi
  [[ "$PYTHON" != /* && "$PYTHON" == */* ]] && PYTHON="$PWD/$PYTHON"

  MODEL_ROOT="$(repo_path "${MODEL_ROOT:-models}")"
  DATA_ROOT="$(repo_path "${DATA_ROOT:-data}")"
  OUTPUT_ROOT="$(repo_path "${OUTPUT_ROOT:-checkpoints}")"
  LOG_ROOT="$(repo_path "${LOG_ROOT:-logs}")"
  HF_HOME="$(repo_path "${HF_HOME:-.hf_cache}")"
  SHISHU_MODEL_DIR="$(repo_path "${SHISHU_MODEL_DIR:-models/MobileLLM-R1-140M-shishu}")"

  PRETRAIN_PHASE1_DATA="$(repo_path "${PRETRAIN_PHASE1_DATA:-$DATA_ROOT/pretrain/phase1}")"
  PRETRAIN_PHASE2_DATA="$(repo_path "${PRETRAIN_PHASE2_DATA:-$DATA_ROOT/pretrain/phase2}")"
  MIDTRAIN_PHASE1_DATA="$(repo_path "${MIDTRAIN_PHASE1_DATA:-$DATA_ROOT/midtrain/phase1}")"
  MIDTRAIN_PHASE2_DATA="$(repo_path "${MIDTRAIN_PHASE2_DATA:-$DATA_ROOT/midtrain/phase2}")"
  GENERAL_SFT_DATASET="$(repo_path "${GENERAL_SFT_DATASET:-$DATA_ROOT/sft/general}")"
  REASONING_SFT_DATASET="$(repo_path "${REASONING_SFT_DATASET:-$DATA_ROOT/sft/reasoning}")"

  if [[ "${NPROC_PER_NODE:-auto}" == "auto" ]]; then
    NPROC_PER_NODE="$(detect_gpus)"
    if [[ -z "$NPROC_PER_NODE" || "$NPROC_PER_NODE" -lt 1 ]]; then
      echo "Could not detect GPUs. Set NPROC_PER_NODE explicitly." >&2
      exit 2
    fi
  fi
  NNODES="${NNODES:-1}"
}
