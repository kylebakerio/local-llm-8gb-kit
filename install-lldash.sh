#!/usr/bin/env bash
# Optional: install LL-Dash (a web dashboard for starting and stopping models, plus its terminal client) into ~/llm-lab.
#     bash install-lldash.sh
# Downloads the dashboard bundle from this repo's GitHub release (checked against its sha256), installs its Node packages,
# writes its config (the dashboard listens on 127.0.0.1:3000 only) and one launch profile per tested model it finds.
# Needs Node.js 18 or newer (Pop 22.04's own is too old; this script says how to get one), python3, and setup.sh done first.
set -euo pipefail
KIT=$(cd "$(dirname "$0")" && pwd)
LAB=${LLDASH_HOME:-$HOME/llm-lab}
REL=lldash-20261124
BUNDLE=lldash-dashboard-20261124.tar.gz
SHA=17633880f96f5e7278fba585310ba072b87f1eea7be416ce194fbf011144aad4
say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

if ! command -v node >/dev/null || [ "$(node -p 'process.versions.node.split(".")[0]')" -lt 18 ]; then
    say "Node.js 18 or newer is needed (found: $(node --version 2>/dev/null || echo none)). Install Node 20 with nvm, then run this again:"
    echo '  curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash'
    echo '  source ~/.bashrc && nvm install 20'
    exit 1
fi
command -v python3 >/dev/null || { echo "python3 is needed"; exit 1; }

say "Getting the dashboard ($BUNDLE)"
mkdir -p "$LAB/logs"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
curl -fL --progress-bar -o "$T/$BUNDLE" "https://github.com/kylebakerio/local-llm-8gb-kit/releases/download/$REL/$BUNDLE"
echo "$SHA  $T/$BUNDLE" | sha256sum -c --quiet || { echo "checksum mismatch: not installed"; exit 1; }
tar xzf "$T/$BUNDLE" -C "$T"

say "Installing it in $LAB/dashboard"
keep=""; [ -f "$LAB/dashboard/dashboard.config.json" ] && keep=$(cat "$LAB/dashboard/dashboard.config.json")
mkdir -p "$LAB/dashboard"
cp -a "$T/dashboard/." "$LAB/dashboard/"
[ -n "$keep" ] && printf '%s\n' "$keep" > "$LAB/dashboard/dashboard.config.json"
(cd "$LAB/dashboard" && npm install --omit=dev --no-audit --no-fund) || { echo "npm failed. If it was building better-sqlite3: sudo apt install build-essential python3, then run this again."; exit 1; }
(cd "$LAB/dashboard/dash-cli" && npm install --omit=dev --no-audit --no-fund)

say "Config and launch profiles"
python3 "$KIT/make-profiles.py" --lab "$LAB"

mkdir -p "$HOME/.local/bin"
cat > "$HOME/.local/bin/lldash" <<EOF
#!/usr/bin/env bash
# LL-Dash: the dashboard at http://127.0.0.1:3000 (Ctrl-C stops it, and any model it started)
cd "$LAB/dashboard" && exec node server4.js "\$@"
EOF
cat > "$HOME/.local/bin/lldash-cli" <<EOF
#!/usr/bin/env bash
# LL-Dash's terminal UI and commands (lldash-cli start <profile>, lldash-cli stop, lldash-cli profile list)
exec node "$LAB/dashboard/dash-cli/cli.js" "\$@"
EOF
cat > "$HOME/.local/bin/llm-launch" <<EOF
#!/usr/bin/env bash
exec python3 "$KIT/llm-launch.py" "\$@"
EOF
chmod +x "$HOME/.local/bin/lldash" "$HOME/.local/bin/lldash-cli" "$HOME/.local/bin/llm-launch"

say "Done."
cat <<EOF
  lldash                      start the dashboard, then open http://127.0.0.1:3000 (launch models from its left panel)
  lldash-cli start bonsai2.5-27b-88k   start a profile (name or unique start of it), wait until ready; lldash-cli stop
  lldash-cli profile list     the launch profiles
  llm-launch --attach         open opencode on the model the dashboard is serving (same opencode setup as llm-launch)
  llm-launch                  the launcher on its own, no dashboard needed
After downloading another model: python3 $KIT/make-profiles.py   (adds its profile)
(~/.local/bin must be on your PATH: open a new terminal if these commands are not found.)
EOF
