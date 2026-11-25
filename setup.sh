#!/usr/bin/env bash
# One-time setup for running local coding models with llm-launch.py (Pop!_OS 22.04, an 8 GB NVIDIA laptop GPU).
#     bash setup.sh
# Safe to run again: each step is skipped when it is already done. Every download is checked against its published
# sha256. Nothing here needs sudo; the two things that do (the NVIDIA driver, the graphics mode) are only suggested.
set -uo pipefail
KIT=$(cd "$(dirname "$0")" && pwd)
B=$(tput bold 2>/dev/null || true); N=$(tput sgr0 2>/dev/null || true)
say() { printf '\n%s%s%s\n' "$B" "$*" "$N"; }
ask() { local a; read -r -p "$1 [Y/n] " a; [[ -z "$a" || "$a" =~ ^[Yy] ]]; }
fetch() {   # url out sha256: resumable download, kept only if the checksum matches
    local url=$1 out=$2 sha=$3
    curl -fL --progress-bar --retry 3 -C - -o "$out.part" "$url" || return 1
    if echo "$sha  $out.part" | sha256sum -c --quiet; then mv "$out.part" "$out"; else echo "   checksum mismatch: $url"; rm -f "$out.part"; return 1; fi
}

say "1. NVIDIA GPU and driver"
if ! command -v nvidia-smi >/dev/null; then
    echo "   nvidia-smi not found. Install the NVIDIA driver: sudo apt install system76-driver-nvidia, reboot, then run this again."
    exit 1
fi
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1); major=${drv%%.*}
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1), driver $drv, ${used} MiB in use now"
if command -v system76-power >/dev/null; then
    mode=$(system76-power graphics 2>/dev/null || true); echo "   graphics mode: ${mode:-unknown}"
    if [ "$mode" = nvidia ]; then
        echo "   The desktop is on the NVIDIA card. To leave all 8 GB to the model, switch to the Intel graphics:"
        echo "       system76-power graphics hybrid      (then reboot)"
    fi
fi
[ "${used:-0}" -gt 300 ] && echo "   Note: ${used} MiB of GPU memory is already in use; the tested contexts assume almost none (close apps on the NVIDIA GPU)."

say "2. llama.cpp, PrismML build prism-b12178 (runs the Ternary Bonsai models)"
L=$HOME/llama-prism
if [ -x "$L/llama-server" ] && "$L/llama-server" --version 2>&1 | grep -q "build 12178"; then
    echo "   already installed: $L"
else
    if [ "${major:-0}" -ge 570 ]; then cu=12.8; sha=a50997fb29bb0684c00cdf21bb1382923c6bc70085a74c9c3d913dcf76af1899
    else cu=12.4; sha=626dc154204531df146a428d455b53c64534b8f022e98cb0760881b48a35557e; fi
    t=llama-prism-b12178-03ec600-bin-linux-cuda-$cu-x64.tar.gz
    echo "   downloading the CUDA $cu build (driver $drv)"
    mkdir -p "$L"
    fetch "https://github.com/PrismML-Eng/llama.cpp/releases/download/prism-b12178-03ec600/$t" "$L/$t" "$sha" || { echo "   download failed; run this again"; exit 1; }
    tar xzf "$L/$t" -C "$L" --strip-components=1 && rm -f "$L/$t"
    echo "   installed: $("$L/llama-server" --version 2>&1 | grep -m1 version)"
fi

say "3. opencode 1.18.34 (the version the plugins were tested with)"
have=$(command -v opencode >/dev/null && opencode --version 2>/dev/null || true)
if [ "$have" = "1.18.34" ]; then
    echo "   already installed"
else
    [ -n "$have" ] && echo "   found opencode $have"
    if ask "   Install opencode 1.18.34?"; then
        curl -fsSL https://opencode.ai/install | bash -s -- --version 1.18.34
        echo "   (open a new terminal afterwards so 'opencode' is on your PATH)"
    fi
fi

say "4. opencode plugins (task driver 0.3.1 from GitHub, and the compaction-anchor slim part)"
python3 "$KIT/llm-launch.py" --fetch-plugins || echo "   could not get the task-driver plugin now; llm-launch.py tries again at each start"

say "5. A model"
# a Bonsai 2.5 27B you already have, wherever it is (Hugging Face cache, ~/models, ~/Downloads, LM Studio ...)
mapfile -t have < <(find "$HOME" -maxdepth 7 \( -path "$HOME/.local/share/Trash" -o -path "$HOME/snap" \) -prune -o \
    -iname 'Ternary-Bonsai-2.5-27B*.gguf' ! -iname '*mmproj*' -print 2>/dev/null | sort)
ptq=""
for f in "${have[@]}"; do
    echo "   found: $f"
    case "$(basename "$f")" in Ternary-Bonsai-2.5-27B-PTQ1_0*.gguf) [ -z "$ptq" ] && ptq=$f ;; esac
    python3 "$KIT/llm-launch.py" --add-dir "$(dirname "$f")" >/dev/null 2>&1 || true   # the launcher looks there too
done
M=$HOME/models/prism-ml__Ternary-Bonsai-2.5-27B-GGUF
R=https://huggingface.co/prism-ml/Ternary-Bonsai-2.5-27B-GGUF/resolve/main
MMSHA=6807ede61d570bb86ba34b756a0fa109edc33668604de867c6ea6d8f1d631903
if [ -n "$ptq" ]; then
    echo "   Bonsai 2.5 27B PTQ1_0 is already here: the packing tested for 8 GB. Nothing to download."
    d=$(dirname "$ptq")
    if ! ls "$d" 2>/dev/null | grep -qi 'mmproj'; then
        if ask "   Its image file is not next to it. Download it (0.6 GB) so the model can read images?"; then
            fetch "$R/Ternary-Bonsai-2.5-27B-mmproj-Q8_0.gguf" "$d/Ternary-Bonsai-2.5-27B-mmproj-Q8_0.gguf" $MMSHA || echo "   download failed"
        fi
    fi
else
    [ ${#have[@]} -gt 0 ] && echo "   That packing holds the same weights in more space: on an 8 GB card the PTQ1_0 file (6.0 GB) leaves room for ~88k tokens of context, the others for far less or none."
    if ask "   Download Ternary Bonsai 2.5 27B PTQ1_0 (6.0 GB, plus its 0.6 GB image file) into ~/models?"; then
        mkdir -p "$M"
        fetch "$R/Ternary-Bonsai-2.5-27B-PTQ1_0.gguf" "$M/Ternary-Bonsai-2.5-27B-PTQ1_0.gguf" 929560ec9d37cf3cadd4b00cc87b51c570082a628dd830b4692dc0b2ed18dbbe \
            && fetch "$R/Ternary-Bonsai-2.5-27B-mmproj-Q8_0.gguf" "$M/Ternary-Bonsai-2.5-27B-mmproj-Q8_0.gguf" $MMSHA \
            || echo "   download failed; run this again to resume"
    fi
fi
echo "   (more models: run the launcher and choose d)"

say "Done. To start:"
echo "   cd ~/path/to/your/project && python3 $KIT/llm-launch.py"
