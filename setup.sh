#!/usr/bin/env bash
# One-time setup for running local coding models with llm-launch.py (Pop!_OS 22.04, an 8 GB NVIDIA laptop GPU).
#     bash setup.sh
# Safe to run again: each step is skipped when it is already done. Every download is checked against its published
# sha256. Nothing here needs sudo; the two things that do (the NVIDIA driver, the graphics mode) are only suggested.
set -uo pipefail
KIT=$(cd "$(dirname "$0")" && pwd)
START_DIR=$PWD
REBOOT=0
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
        echo "   The desktop is on the NVIDIA card, which takes some of its 8 GB from the model."
        if ask "   Switch the desktop to the Intel graphics (Hybrid mode)? It takes effect after a reboot."; then
            if system76-power graphics hybrid || sudo system76-power graphics hybrid; then
                REBOOT=1; echo "   switched: reboot when this script is done"
            else
                echo "   could not switch; choose Graphics: Hybrid in the top-bar menu instead"
            fi
        fi
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
OC=$(command -v opencode || echo "$HOME/.opencode/bin/opencode")
have=$([ -x "$OC" ] && "$OC" --version 2>/dev/null || true)
if [ "$have" = "1.18.34" ]; then
    echo "   already installed"
else
    [ -n "$have" ] && echo "   found opencode $have"
    if ask "   Install opencode 1.18.34?"; then
        curl -fsSL https://opencode.ai/install | bash -s -- --version 1.18.34
    fi
fi

say "4. opencode plugins (task driver 0.3.1 from GitHub, and the compaction-anchor slim part)"
python3 "$KIT/llm-launch.py" --fetch-plugins || echo "   could not get the task-driver plugin now; llm-launch.py tries again at each start"

say "5. Your models"
# what you already have, wherever it is: what is known about each, and their folders join the launcher's search path.
# Nothing is downloaded here; the launcher's model menu has d for that.
python3 "$KIT/llm-launch.py" --inventory
[ $? -eq 0 ] || echo "   No Bonsai 2.5 27B PTQ1_0 file found: that is the one tested for 8 GB. The launcher can get it (choose d)."

say "6. The llm-launch command"
mkdir -p "$HOME/.local/bin"
printf '#!/usr/bin/env bash\nexec python3 "%s/llm-launch.py" "$@"\n' "$KIT" > "$HOME/.local/bin/llm-launch"
chmod +x "$HOME/.local/bin/llm-launch"
echo "   From now on, run  llm-launch  in any project folder (in a new terminal; it lives in ~/.local/bin)."

say "Done."
if [ "$REBOOT" = 1 ]; then
    echo "   Reboot now so the desktop moves to the Intel graphics, then run  llm-launch  in your project folder."
    exit 0
fi
if ask "   Start the launcher now?"; then
    def=$START_DIR; [ "$def" = "$KIT" ] && def=$HOME
    read -r -p "   Project folder for opencode to work in [$def]: " proj
    proj=${proj:-$def}; proj=${proj/#\~/$HOME}
    if [ -d "$proj" ]; then
        cd "$proj" && PATH="$HOME/.opencode/bin:$HOME/.local/bin:$PATH" exec python3 "$KIT/llm-launch.py"
    fi
    echo "   no folder $proj: run  llm-launch  from your project folder"
fi
