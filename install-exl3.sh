#!/usr/bin/env bash
# Optional and EXPERIMENTAL: the EXL3 engine, for the EXL3 models (Qwen3.8-27B EXL3 at 1.75 bits per weight).
#     bash install-exl3.sh           (setup.sh offers it; safe to run again: what is already done is skipped)
# Installs into ~/exl3 only (EXL3_HOME to change it); nothing needs sudo:
#   - uv 0.13.7 (a Python package installer, checked against its published sha256), and with it Python 3.12;
#   - TabbyAPI at commit 1b90202eb (the server that runs EXL3 models), checked out by its git commit;
#   - exllamav3 1.6.7 (CUDA 12.8, torch 2.9.0) and everything TabbyAPI needs: exactly the package versions the lab's
#     install resolved, every wheel checked against its sha256 (exl3/requirements.txt).
# About 5 GB of downloads (mostly torch and NVIDIA's CUDA libraries), about 9 GB on disk after. Needs git and an NVIDIA
# driver 570 or newer (CUDA 12.8).
set -euo pipefail
KIT=$(cd "$(dirname "$0")" && pwd)
E=${EXL3_HOME:-$HOME/exl3}
UV_VER=0.13.7
UV_SHA=c5221ff41b7892bfc223543883600125f6a47fe845093e0f5bdff4f359196d83
TABBY_REPO=https://github.com/theroyallab/tabbyAPI
TABBY_COMMIT=1b90202ebabe3100d6d6fd70de5c888d56eee5d7
STAMP="tabby $TABBY_COMMIT exllamav3 1.6.7+cu128.torch2.9.0 requirements $(sha256sum "$KIT/exl3/requirements.txt" | cut -c1-16)"
say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

if [ -f "$E/INSTALLED" ] && [ "$(cat "$E/INSTALLED")" = "$STAMP" ] && [ -x "$E/venv/bin/python" ]; then
    echo "   the EXL3 engine is already installed: $E"; exit 0
fi
command -v git >/dev/null || { echo "   git is needed:  sudo apt install git   then run this again"; exit 1; }
if command -v nvidia-smi >/dev/null; then
    drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1); major=${drv%%.*}
    [ "${major:-0}" -ge 570 ] || { echo "   NVIDIA driver $drv: the EXL3 engine needs 570 or newer (CUDA 12.8). See README.md, step 2."; exit 1; }
fi
free=$(df -Pk "$HOME" | awk 'NR == 2 {print $4}')
[ "${free:-0}" -gt $((15 * 1024 * 1024)) ] || { echo "   about 15 GB of free disk is needed while installing ($((free / 1024 / 1024)) GB free)"; exit 1; }
mkdir -p "$E"

say "1. uv $UV_VER"
if [ ! -x "$E/uv/uv" ] || ! "$E/uv/uv" --version | grep -q " $UV_VER"; then
    T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
    curl -fL --progress-bar --retry 3 -o "$T/uv.tar.gz" "https://github.com/astral-sh/uv/releases/download/$UV_VER/uv-x86_64-unknown-linux-gnu.tar.gz"
    echo "$UV_SHA  $T/uv.tar.gz" | sha256sum -c --quiet || { echo "   checksum mismatch: uv not installed"; exit 1; }
    mkdir -p "$E/uv" && tar xzf "$T/uv.tar.gz" -C "$E/uv" --strip-components=1
fi
UV=$E/uv/uv
export UV_CACHE_DIR=$E/uv-cache UV_PYTHON_INSTALL_DIR=$E/python UV_NO_CONFIG=1
echo "   $("$UV" --version)"

say "2. Python 3.12 and the package folder ($E/venv)"
"$UV" python install 3.12
[ -x "$E/venv/bin/python" ] || "$UV" venv --python 3.12 "$E/venv"

say "3. exllamav3 1.6.7, torch 2.9.0 and TabbyAPI's packages (each checked against its sha256)"
"$UV" pip install --python "$E/venv/bin/python" --require-hashes --no-deps -r "$KIT/exl3/requirements.txt"

say "4. TabbyAPI at commit ${TABBY_COMMIT:0:9}"
if [ ! -d "$E/tabbyAPI/.git" ]; then
    git clone --quiet --filter=blob:none "$TABBY_REPO" "$E/tabbyAPI"
fi
git -C "$E/tabbyAPI" fetch --quiet origin "$TABBY_COMMIT" 2>/dev/null || true
git -C "$E/tabbyAPI" -c advice.detachedHead=false checkout --quiet "$TABBY_COMMIT"
[ "$(git -C "$E/tabbyAPI" rev-parse HEAD)" = "$TABBY_COMMIT" ] || { echo "   TabbyAPI is not at $TABBY_COMMIT"; exit 1; }

say "5. Check"
"$E/venv/bin/python" -I -c 'import torch, exllamav3; print("   torch", torch.__version__, "CUDA", torch.version.cuda, "| exllamav3", exllamav3.__version__, "| GPU seen:", torch.cuda.is_available())'
"$UV" cache clean --quiet || true
echo "$STAMP" > "$E/INSTALLED"
echo "   installed: $E. The launcher now offers the EXL3 models it finds (get one with d at its model menu)."
