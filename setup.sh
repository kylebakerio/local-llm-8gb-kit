#!/usr/bin/env bash
# One-time setup for running local coding models with llm-launch.py (Pop!_OS 22.04, an 8 GB NVIDIA laptop GPU).
#     bash setup.sh                (llm-launch runs it with --no-launch when a kit update needs a setup step)
# Safe to run again: each step is skipped when it is already done. Every download is checked against its published
# sha256. Nothing here needs sudo; the two things that do (the NVIDIA driver, the graphics mode) are only suggested.
set -uo pipefail
KIT=$(cd "$(dirname "$0")" && pwd)
START_DIR=$PWD
REBOOT=0
NO_LAUNCH=0; [ "${1:-}" = "--no-launch" ] && NO_LAUNCH=1      # run by llm-launch after a kit update: don't offer to start it
kitval() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ""))' "$KIT/kit.json" "$1"; }
B=$(tput bold 2>/dev/null || true); N=$(tput sgr0 2>/dev/null || true)
say() { printf '\n%s%s%s\n' "$B" "$*" "$N"; }
ask() { local a; read -r -p "$1 [Y/n] " a; [[ -z "$a" || "$a" =~ ^[Yy] ]]; }
fetch() {   # url out sha256: resumable download, kept only if the checksum matches
    local url=$1 out=$2 sha=$3
    curl -fL --progress-bar --retry 3 -C - -o "$out.part" "$url" || return 1
    if echo "$sha  $out.part" | sha256sum -c --quiet; then mv "$out.part" "$out"; else echo "   checksum mismatch: $url"; rm -f "$out.part"; return 1; fi
}

say "1. NVIDIA GPU, driver and graphics mode"
# Pop!_OS graphics modes: compute (best here: the desktop on the Intel graphics, the NVIDIA card only computes), hybrid
# (good: the desktop on the Intel graphics), nvidia (the desktop takes some of the card), integrated (the card is off)
mode=""; command -v system76-power >/dev/null && mode=$(system76-power graphics 2>/dev/null || true)
to_compute() {
    echo "   (If you use an external monitor and it stops working in Compute mode, choose Graphics: Hybrid instead.)"
    if ask "   Switch to Compute mode? The desktop runs on the Intel graphics and the NVIDIA card is left to the model. It takes effect after a reboot."; then
        if system76-power graphics compute || sudo system76-power graphics compute; then
            REBOOT=1; echo "   switched: reboot when this script is done"
        else
            echo "   could not switch; choose Graphics: Compute in the top-bar menu instead"
        fi
    fi
}
if [ "$mode" = integrated ]; then
    echo "   graphics mode: integrated, so the NVIDIA card is switched off and the model cannot use it."
    to_compute
    if [ "$REBOOT" = 1 ]; then echo "   Reboot, then run this again:  bash $KIT/setup.sh"
    else echo "   The model needs the NVIDIA card: choose Graphics: Compute (or Hybrid) in the top-bar menu, reboot, then run this again."; fi
    exit 1
fi
if ! command -v nvidia-smi >/dev/null; then
    echo "   nvidia-smi not found. Install the NVIDIA driver: sudo apt install system76-driver-nvidia, reboot, then run this again."
    exit 1
fi
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1); major=${drv%%.*}
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1), driver $drv, ${used} MiB in use now"
case "$mode" in
    compute) echo "   graphics mode: compute (best for this: the desktop is on the Intel graphics and the NVIDIA card runs only the model)" ;;
    hybrid)  echo "   graphics mode: hybrid (good: the desktop is on the Intel graphics; Compute mode leaves a little more of the card free)" ;;
    nvidia)  echo "   graphics mode: nvidia: the desktop runs on the NVIDIA card, which takes some of its 8 GB from the model."
             to_compute ;;
    "")      ;;
    *)       echo "   graphics mode: $mode" ;;
esac
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

OCV=$(kitval opencode); OCV=${OCV:-1.18.34}
say "3. opencode $OCV (the version the plugins were tested with)"
OC=$(command -v opencode || echo "$HOME/.opencode/bin/opencode")
have=$([ -x "$OC" ] && "$OC" --version 2>/dev/null || true)
if [ -n "$OCV" ] && [ "$have" = "$OCV" ]; then
    echo "   already installed"
else
    [ -n "$have" ] && echo "   found opencode $have"
    if ask "   Install opencode $OCV?"; then
        curl -fsSL https://opencode.ai/install | bash -s -- --version "$OCV"
    fi
fi

say "4. opencode plugins (kit.json: $(kitval setup))"
python3 "$KIT/llm-launch.py" --fetch-plugins || echo "   could not get every plugin now; llm-launch.py tries again at each start"

say "5. Your models"
# what you already have, wherever it is: what is known about each, and their folders join the launcher's search path.
# Nothing is downloaded here; the launcher's model menu has d for that.
python3 "$KIT/llm-launch.py" --inventory
[ $? -eq 0 ] || echo "   No Bonsai 2.5 27B PTQ1_0 file found: that is the one tested for 8 GB. The launcher can get it (choose d)."

say "6. The llm-launch and llm-tune commands"
mkdir -p "$HOME/.local/bin"
printf '#!/usr/bin/env bash\nexec python3 "%s/llm-launch.py" "$@"\n' "$KIT" > "$HOME/.local/bin/llm-launch"
printf '#!/usr/bin/env bash\nexec python3 "%s/llm-tune.py" "$@"\n' "$KIT" > "$HOME/.local/bin/llm-tune"
chmod +x "$HOME/.local/bin/llm-launch" "$HOME/.local/bin/llm-tune"
# Pop/Ubuntu add ~/.local/bin to PATH only at login, and only if it existed then: make new terminals see it right away
case ":$PATH:" in
    *":$HOME/.local/bin:"*) echo "   From now on, run  llm-launch  in any project folder." ;;
    *)
        grep -qs 'local-llm-8gb-kit: ~/.local/bin' "$HOME/.bashrc" || \
            printf '\n# local-llm-8gb-kit: ~/.local/bin on PATH (llm-launch)\ncase ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) export PATH="$HOME/.local/bin:$PATH" ;; esac\n' >> "$HOME/.bashrc"
        echo "   From now on, run  llm-launch  in any project folder: open a new terminal first (~/.local/bin was added to ~/.bashrc)." ;;
esac

say "7. Optional, experimental: the EXL3 engine"
# TabbyAPI + exllamav3, for the EXL3 models (Qwen3.8-27B EXL3). Still being tested; about 5 GB to download.
if [ -f "${EXL3_HOME:-$HOME/exl3}/INSTALLED" ]; then
    bash "$KIT/install-exl3.sh" || echo "   the EXL3 engine needs an update and it did not finish: run  bash $KIT/install-exl3.sh"
else
    echo "   A second engine, for EXL3 models (Qwen3.8-27B at 1.75 bits per weight). It is experimental: our tests of it are"
    echo "   still running. About 5 GB to download, 9 GB on disk, into ~/exl3. The rest of the kit works without it."
    read -r -p "   Install the EXL3 engine? [y/N] " a
    if [[ "$a" =~ ^[Yy] ]]; then bash "$KIT/install-exl3.sh" || echo "   not installed; run  bash $KIT/install-exl3.sh  to try again"
    else echo "   skipped (later:  bash $KIT/install-exl3.sh)"; fi
fi

mkdir -p "$HOME/.config/llm-launch" && kitval setup_version > "$HOME/.config/llm-launch/setup_version"

say "Done."
[ "$NO_LAUNCH" = 1 ] && { [ "$REBOOT" = 1 ] && echo "   Reboot before the next model run, so the desktop moves to the Intel graphics."; exit 0; }
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
