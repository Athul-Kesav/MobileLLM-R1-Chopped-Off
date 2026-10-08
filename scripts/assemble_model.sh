#!/usr/bin/env bash
# Rebuild model.safetensors from GitHub-sized shards (public forks cannot use Git LFS).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DIR="${1:-models/MobileLLM-R1-140M-shishu}"
DEST="$DIR/model.safetensors"
EXPECTED="${2:-9c1bb45e3ca8717687a8256fe32a0e43bf9e803a36e7501560d8935ad3333a9e}"

if [[ -f "$DEST" ]]; then
  echo "Already present: $DEST"
  exit 0
fi
shopt -s nullglob
parts=( "$DIR"/model.safetensors.part.* )
if [[ ${#parts[@]} -lt 1 ]]; then
  echo "No shards found in $DIR (model.safetensors.part.*)." >&2
  exit 2
fi
echo "Assembling ${#parts[@]} shards -> $DEST"
cat "${parts[@]}" > "$DEST"
python3 - "$DEST" "$EXPECTED" <<'PY'
import hashlib, pathlib, sys
path, expected = pathlib.Path(sys.argv[1]), sys.argv[2]
digest = hashlib.sha256(path.read_bytes()).hexdigest()
if digest != expected:
    path.unlink(missing_ok=True)
    raise SystemExit(f"Checksum mismatch for {path}: {digest} != {expected}")
print(f"OK {path} sha256 {digest}")
PY
