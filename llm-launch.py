#!/usr/bin/env python3
"""llm-launch: run a local model on an 8 GB NVIDIA laptop GPU and open opencode on it.

    python3 llm-launch.py            (run it from the project folder you want opencode to work in)
    python3 llm-launch.py --update         get the latest kit from GitHub (new plugins, a new tested setup); the launcher
                                           also checks at each start and asks (--no-update skips that)
    python3 llm-launch.py --fetch-plugins  get the plugins kit.json names (setup.sh does this; so does each launch)
    python3 llm-launch.py --add-dir DIR    also look for models in DIR
    python3 llm-launch.py --results        the measurements llm-tune saved on this machine (context that fits, speed)
    python3 llm-launch.py --inventory      find the models this kit tested anywhere in your home folder, say what is known
                                           about each, and add their folders to the search path (setup.sh runs this)
    python3 llm-launch.py --attach [--port 8080]
                                     a model server is already running (e.g. started from LL-Dash): just open opencode on
                                     it, with the same opencode setup

It asks, in order:
  1. which model (the .gguf files in your Hugging Face download folder and in ~/models)
  2. MTP on or off (only for models that have an MTP head)
  3. images on the CPU or the GPU (only when the model has an image projector file next to it)
  4. how much context (it suggests the largest that was tested to fit in 8 GB; for a model it has no number for, you
     enter one, and it remembers what you entered for that model next time)
then it adds the model to your opencode config, starts llama-server, and starts opencode pointed at it. When you leave
opencode, it stops the server.

Server settings that match the tested setup: q4_0 K/V cache, all layers on the GPU, flash attention, one request at a
time, the model's own recommended sampling. opencode gets the plugins and settings kit.json names, for these sessions
only. Today: the task driver (opencode-task-driver 0.3.1, focus mode: splits a request with several problems into tasks
and works them one at a time, each on a clean slate) and the "slim" part of the compaction-anchor plugin
(plugins/compaction-anchor.js: a short system prompt in place of opencode's long one, shorter tool descriptions; your
project's AGENTS.md is kept). A newer tested setup arrives as a kit update (--update).
Files it writes: ~/.config/llm-launch/ (your choices, downloaded plugins, llm-tune's measurements), ~/.cache/llm-launch/
(server logs), and the model entry in ~/.config/opencode/opencode.json (a backup is kept next to it the first time).
Python 3.8+, standard library only.
"""
import glob, json, os, re, shutil, signal, socket, struct, subprocess, sys, time, urllib.request

HOME = os.path.expanduser("~")
STATE_DIR = os.path.join(HOME, ".config", "llm-launch")
STATE_FILE = os.path.join(STATE_DIR, "state.json")
KIT = os.path.dirname(os.path.abspath(__file__))
KIT_JSON = os.path.join(KIT, "kit.json")           # the opencode setup for these sessions: plugins, options, settings
PLUGIN_DIR = os.path.join(STATE_DIR, "plugins")    # downloaded plugins (checked by sha256)
SETUP_VERSION_FILE = os.path.join(STATE_DIR, "setup_version")   # the kit.json setup_version setup.sh last ran for
LOG_DIR = os.path.join(HOME, ".cache", "llm-launch")
OC_CONFIG = os.path.join(HOME, ".config", "opencode", "opencode.json")
PROVIDER = "llama-local"
KV = "q4_0"                 # K/V cache type used in every tested number below
DEFAULT_PORT = 8080

# Tested on an RTX 3060 held to exactly what an 8 GB RTX 3070 Laptop offers (8192 MiB minus the driver's reservation
# minus a 200 MiB safety margin, with nothing else on the card), PrismML llama.cpp prism-b12178, q4_0 K/V: the largest
# context that loaded AND survived a prompt filling it to 97%. Keyed by file name. "mtp_on" is None when MTP does not fit
# usefully; "mib_per_1k" (VRAM per 1,000 tokens of context) is used to shrink the suggestion when images go on the GPU.
RECOMMENDED = {
    # suggested = the largest context that left >= 64 MiB spare under the cap (llama.cpp allocates a little more during a
    # session -- CUDA graphs for new shapes -- and a model filled to the last MiB crashed that way in testing). The absolute
    # maxima measured are in the comments.
    "Ternary-Bonsai-2.5-27B-PTQ1_0.gguf":      {"mtp_off": 88064, "mib_per_1k": 18.55},                  # max 90112
    "Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf":  {"mtp_off": 88064, "mtp_on": 41984, "mib_per_1k": 18.55}, # max 90112 / 44032
    "Ornith-1.5-9B-Q4_K_M.gguf":               {"mtp_off": 249856, "mtp_on": 180224, "mib_per_1k": 9.77}, # max 254976 / 185344
    "Ternary-Bonsai-2-9B-PTQ1_0.gguf":         {"mtp_off": 262144, "mib_per_1k": 9.77},                  # the model's own max; 4.6 GB used
    "Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf": {"mtp_off": 262144, "mtp_on": 262144, "mib_per_1k": 11.8}, # 5.0 / 5.9 GB used
    "Ternary-Bonsai-2-27B-PTQ1_0.gguf":        {"mtp_off": 88064, "mib_per_1k": 18.55},                  # 90112 held; 93184 crashed later
    # not measured, inferred: the same Bonsai 2 9B weights, packed between the tested PTQ1_0 (4.6 GB used at 262k) and the
    # PQ2_0-MTP file (5.0 GB used at 262k with MTP off), so 262k leaves ~3 GB spare on an 8 GB card
    "Ternary-Bonsai-2-9B-PQ2_0.gguf":          {"mtp_off": 262144, "mib_per_1k": 9.77, "inferred": True},
    "Ternary-Bonsai-2-9B-Q2_0.gguf":           {"mtp_off": 262144, "mib_per_1k": 9.77, "inferred": True},
}
# the files the numbers above were measured with, per model (shown when a different file of that model is found)
TESTED_FILES = {"Ternary-Bonsai-2.5-27B": "PTQ1_0 (or PTQ1_0-mtp)", "Ternary-Bonsai-2-27B": "PTQ1_0",
                "Ternary-Bonsai-2-9B": "PTQ1_0 or PQ2_0-MTP-Q8_0", "Ornith-1.5-9B": "Q4_K_M"}
# Each model's own recommended sampling (thinking mode; Ornith: its "precise coding" set), from the model cards, as
# llama-server flags. The tests use exactly these. A model not listed gets llama.cpp's defaults plus whatever its GGUF
# file sets.
SAMPLING = {
    "Ternary-Bonsai-2.5-27B": "--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0",
    "Ternary-Bonsai-2-27B": "--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05 --presence-penalty 0",
    "Ternary-Bonsai-2-9B": "--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 1.0 --repeat-penalty 1.0 --repeat-last-n 64",
    "Ornith-1.5": "--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0",
}
DRAFT_N = {"Ternary-Bonsai-2.5-27B": 3}     # MTP draft length (PrismML's preset for 2.5 27B); 2 for the rest
# models worth having, for the download menu (repo, file, image projector)
DOWNLOADS = [
    ("prism-ml/Ternary-Bonsai-2.5-27B-GGUF", "Ternary-Bonsai-2.5-27B-PTQ1_0.gguf", "Ternary-Bonsai-2.5-27B-mmproj-Q8_0.gguf"),
    ("prism-ml/Ternary-Bonsai-2.5-27B-GGUF", "Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf", "Ternary-Bonsai-2.5-27B-mmproj-Q8_0.gguf"),
    ("ornith-ai/Ornith-1.5-9B-GGUF", "Ornith-1.5-9B-Q4_K_M.gguf", None),
    ("prism-ml/Ternary-Bonsai-2-9B-gguf", "Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf", "Ternary-Bonsai-2-9B-mmproj-Q8_0.gguf"),
]
# (the short system prompt lives in plugins/compaction-anchor.js, its SLIM_PROMPT; kit.json says which plugins a session gets)

B = "\033[1m" if sys.stdout.isatty() else ""
D = "\033[2m" if sys.stdout.isatty() else ""
R = "\033[31m" if sys.stdout.isatty() else ""
N = "\033[0m" if sys.stdout.isatty() else ""


# ------------------------------------------------------------------ small helpers
def load_state():
    try:
        with open(STATE_FILE) as f: return json.load(f)
    except (OSError, ValueError): return {}


def save_state(st):
    os.makedirs(STATE_DIR, exist_ok=True)
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f: json.dump({k: v for k, v in st.items() if not k.startswith("_")}, f, indent=1)
    os.replace(tmp, STATE_FILE)


def ask(prompt, default=None):
    try:
        v = input(f"{prompt}" + (f" [{default}]" if default not in (None, "") else "") + ": ").strip()
    except EOFError:
        print(); sys.exit(1)
    return v if v else (default if default is not None else "")


def yes(prompt, default=True):
    v = ask(prompt + (" (Y/n)" if default else " (y/N)")).lower()
    return default if not v else v.startswith("y")


def parse_ctx(s):
    s = s.strip().lower().replace(",", "").replace("_", "")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(k?)", s)
    if not m: return None
    n = float(m.group(1)) * (1024 if m.group(2) else 1)
    return int(n)


def gb(n): return f"{n / 1e9:.1f} GB"


# ------------------------------------------------------------------ GGUF header (MTP head? native context?)
_T = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def gguf_meta(path, want=("nextn_predict_layers", "context_length", "general.architecture")):
    """The few header keys this script needs (no tensor data is read)."""
    out = {}
    def rd(f, fmt): return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    def rs(f): return f.read(rd(f, "<Q")).decode("utf-8", "replace")
    def val(f, t):
        if t in _T: return rd(f, _T[t])
        if t == 8: return rs(f)
        if t == 9:
            et, n = rd(f, "<I"), rd(f, "<Q")
            for _ in range(n): val(f, et)
            return None
        raise ValueError(t)
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF": return out
            rd(f, "<I"); rd(f, "<Q"); nkv = rd(f, "<Q")
            for _ in range(nkv):
                k = rs(f); v = val(f, rd(f, "<I"))
                if any(k.endswith(w) for w in want): out[k.split(".")[-1] if k != "general.architecture" else "arch"] = v
    except Exception:
        pass
    return out


# ------------------------------------------------------------------ finding models
# where downloaded models usually are, besides the Hugging Face cache (LM Studio's folders included)
DEFAULT_DIRS = [os.path.join(HOME, "models"), os.path.join(HOME, "Downloads"), os.path.join(HOME, ".lmstudio", "models"),
                os.path.join(HOME, ".cache", "lm-studio", "models")]
# packings of the tested models that hold the same weights in more space: on 8 GB they leave little or no room for context
BIGGER_PACKS = {"Ternary-Bonsai-2.5-27B-PQ2_0": "7.3 GB", "Ternary-Bonsai-2.5-27B-Q2_0": "7.7 GB",
                "Ternary-Bonsai-2-27B-PQ2_0": "7.2 GB", "Ternary-Bonsai-2-27B-Q2_0": "7.6 GB"}


def hf_cache():
    if os.environ.get("HF_HUB_CACHE"): return os.environ["HF_HUB_CACHE"]
    return os.path.join(os.environ.get("HF_HOME") or os.path.join(HOME, ".cache", "huggingface"), "hub")


def scan_models(st):
    found = {}
    def add(path, repo):
        name = os.path.basename(path)
        if "mmproj" in name.lower(): return
        m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", name)
        if m and m.group(1) != "00001": return                     # only the first part of a split model
        real = os.path.realpath(path)
        if real in found or not os.path.isfile(real): return
        found[real] = {"path": path, "name": name, "repo": repo, "size": os.path.getsize(real), "dir": os.path.dirname(path)}
    hub = hf_cache()
    for repo_dir in glob.glob(os.path.join(hub, "models--*")):
        repo = os.path.basename(repo_dir)[len("models--"):].replace("--", "/")
        snaps = sorted(glob.glob(os.path.join(repo_dir, "snapshots", "*")), key=os.path.getmtime, reverse=True)
        for s in snaps:
            for p in glob.glob(os.path.join(s, "**", "*.gguf"), recursive=True): add(p, repo)
    for d in list(DEFAULT_DIRS) + [x for x in st.get("model_dirs", []) if x not in DEFAULT_DIRS]:
        d = os.path.expanduser(d)
        if not os.path.isdir(d): continue
        for p in glob.glob(os.path.join(d, "**", "*.gguf"), recursive=True):
            rel = os.path.relpath(os.path.dirname(p), d)
            add(p, (d.replace(HOME, "~", 1) if rel == "." else os.path.join(d.replace(HOME, "~", 1), rel)))
    return sorted(found.values(), key=lambda m: m["name"].lower())


def projector_for(model):
    """An image projector file next to the model (same download folder), Q8_0 preferred."""
    cands = [p for p in glob.glob(os.path.join(model["dir"], "*.gguf")) if "mmproj" in os.path.basename(p).lower()]
    if not cands: return None
    cands.sort(key=lambda p: (0 if "q8_0" in p.lower() else 1 if "f16" in p.lower() else 2, p))
    return cands[0]


# ------------------------------------------------------------------ llama-server
def find_server(st):
    cands = [st.get("llama_server"), shutil.which("llama-server"),
             os.path.join(HOME, "llama-prism", "llama-server"), os.path.join(HOME, "llama-prism", "bin", "llama-server")]
    for c in cands:
        if c and os.path.isfile(c) and os.access(c, os.X_OK): return c
    print(f"\n{B}Where is llama-server?{N} (the PrismML llama.cpp build: see README.md)")
    while True:
        p = os.path.expanduser(ask("Path to llama-server"))
        if os.path.isdir(p): p = os.path.join(p, "llama-server")
        if os.path.isfile(p) and os.access(p, os.X_OK):
            st["llama_server"] = p; save_state(st); return p
        print(f"{R}No runnable llama-server at {p}{N}")


def opencode_bin():
    """opencode on PATH, else where its installer puts it (a terminal opened before the install has no PATH entry yet)."""
    p = shutil.which("opencode") or os.path.join(HOME, ".opencode", "bin", "opencode")
    return p if os.path.isfile(p) and os.access(p, os.X_OK) else None


def port_free(port):
    with socket.socket() as s:
        try: s.bind(("127.0.0.1", port)); return True
        except OSError: return False


def health(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            return json.loads(r.read()).get("status") == "ok"
    except Exception:
        return False


def start_server(server, model, ctx, mtp, draft_n, proj, proj_gpu, port, log):
    argv = [server, "-m", model["path"], "-c", str(ctx), "-ngl", "999", "-fa", "on", "-np", "1", "--jinja", "-fit", "off",
            "--cache-type-k", KV, "--cache-type-v", KV, "--host", "127.0.0.1", "--port", str(port)]
    argv += next((v.split() for k, v in SAMPLING.items() if model["name"].startswith(k)), [])
    if mtp:
        argv += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(draft_n), "--spec-draft-p-min", "0", "-ctkd", KV, "-ctvd", KV]
    if proj:
        argv += ["--mmproj", proj] + ([] if proj_gpu else ["--no-mmproj-offload"])
    env = dict(os.environ, LLAMA_ARG_CACHE_RAM="2048", LLAMA_ARG_CTX_CHECKPOINTS="4",
               LD_LIBRARY_PATH=os.path.dirname(server) + (":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else ""))
    os.makedirs(LOG_DIR, exist_ok=True)
    fh = open(log, "w")
    fh.write("$ " + " ".join(argv) + "\n\n"); fh.flush()
    p = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    return p, argv


def wait_ready(p, port, timeout=600):
    t0 = time.time(); dots = 0
    while time.time() - t0 < timeout:
        if p.poll() is not None: return False
        if health(port):
            print(); return True
        time.sleep(1); dots += 1
        if dots % 5 == 0: print(".", end="", flush=True)
    print(); return False


def warm_up(port):
    """One tiny request: some GPU buffers are only allocated at the first request, so a context that loads can still run
    out of memory here."""
    body = json.dumps({"prompt": "Hello", "n_predict": 2, "cache_prompt": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r: r.read(); return True
    except Exception:
        return False


def stop_server(p):
    if p is None or p.poll() is not None: return
    try:
        os.killpg(p.pid, signal.SIGTERM); p.wait(20)
    except Exception:
        try: os.killpg(p.pid, signal.SIGKILL)
        except Exception: pass


ERR = re.compile(r"error|failed|out of memory|cudaMalloc|unknown (model )?(type|architecture)|invalid|exception|abort", re.I)


def explain_failure(log, mtp, proj, proj_gpu):
    try: lines = open(log, errors="replace").read().splitlines()
    except OSError: lines = []
    errs = [l for l in lines if ERR.search(l)]
    print(f"\n{R}{B}llama-server did not start.{N} Its error output:\n")
    for l in (errs[-15:] or lines[-20:]): print("   " + l[-220:])
    print(f"\n   (full log: {log})\n")
    text = "\n".join(lines).lower()
    oom = "out of memory" in text or "failed to allocate" in text or "cudamalloc" in text
    if "unknown" in text and ("type" in text or "architecture" in text):
        print(f"{B}This model's file type is not known to this llama-server.{N} Ternary Bonsai PTQ1_0 / PQ2_0 files need the PrismML\n"
              f"llama.cpp build (README.md), and a model newer than the build may need a newer one.\n")
    print(f"{B}If it did not fit in the GPU's memory{N}" + (" (it says out of memory above)" if oom else "") + ", you can:")
    print("   a) use a smaller context")
    if mtp: print("   b) turn MTP off (it needs its own memory on the GPU)")
    if proj and proj_gpu: print("   c) process images on the CPU instead of the GPU")
    print("   - and check nothing else is using the GPU:")
    for line in gpu_busy_advice("     "): print(line)


# ------------------------------------------------------------------ opencode
def model_id(name):
    return re.sub(r"[^a-z0-9.]+", "-", re.sub(r"\.gguf$", "", name.lower())).strip("-")


def jsonc_loads(text):
    """JSON with // and /* */ comments and trailing commas (what opencode accepts) -> object; ValueError if broken."""
    out, i, n, in_str = [], 0, len(text), False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n: out.append(text[i + 1]); i += 2; continue
            if c == '"': in_str = False
            i += 1; continue
        if c == '"': in_str = True; out.append(c); i += 1; continue
        if text.startswith("//", i):
            j = text.find("\n", i); i = n if j < 0 else j; continue
        if text.startswith("/*", i):
            j = text.find("*/", i + 2); i = n if j < 0 else j + 2; continue
        out.append(c); i += 1
    return json.loads(re.sub(r",(\s*[}\]])", r"\1", "".join(out)))


def check_opencode_config():
    """Before anything is loaded: opencode will not start at all with a broken ~/.config/opencode/opencode.json. Offer to
    set it aside (kept as a backup) so a fresh one is written. Returns False to stop."""
    if not os.path.exists(OC_CONFIG): return True
    text = open(OC_CONFIG, errors="replace").read()
    try:
        json.loads(text); return True
    except ValueError:
        pass
    try:
        jsonc_loads(text); return True        # comments / trailing commas: fine for opencode; this script won't edit it
    except ValueError as e:
        err = e
    line = getattr(err, "lineno", None)
    print(f"\n{R}{B}Your opencode config has an error, and opencode will not start with it:{N} {OC_CONFIG}")
    if line:
        print(f"   line {line}: {text.splitlines()[line - 1].strip() if line <= len(text.splitlines()) else ''}")
    print(f"   ({err.msg if hasattr(err, 'msg') else err})")
    if yes("Set it aside (kept as a backup) and let this script write a fresh one?", default=True):
        bak = OC_CONFIG + time.strftime(".broken-%Y%m%d-%H%M%S")
        os.replace(OC_CONFIG, bak)
        print(f"   moved to {bak}; anything you want from it can be copied back by hand")
        return True
    print("   Fix that line by hand (or remove the file), then run this again.")
    return False


def update_opencode_config(mid, label, ctx, port):
    """Add / update this model's entry under provider llama-local. A config with comments (JSONC) is not edited: the
    entry is printed for you to paste, and this session gets it anyway (OPENCODE_CONFIG_CONTENT)."""
    entry = {"name": label, "limit": {"context": ctx, "input": ctx, "output": ctx}}
    prov = {"npm": "@ai-sdk/openai-compatible", "name": "Local llama.cpp",
            "options": {"baseURL": f"http://127.0.0.1:{port}/v1", "apiKey": "sk-local"}, "models": {mid: entry}}
    try:
        cfg = json.load(open(OC_CONFIG)) if os.path.exists(OC_CONFIG) else {"$schema": "https://opencode.ai/config.json"}
    except ValueError:
        # JSONC (comments): left as it is; this session gets the entry through OPENCODE_CONFIG_CONTENT anyway
        print(f"   {D}{OC_CONFIG} has comments, so it is left unchanged; this session gets the model entry anyway{N}")
        return prov
    p = cfg.setdefault("provider", {}).setdefault(PROVIDER, {})
    before = json.dumps(cfg, sort_keys=True)
    p.setdefault("npm", prov["npm"]); p.setdefault("name", prov["name"])
    opts = p.setdefault("options", {}); opts["baseURL"] = prov["options"]["baseURL"]; opts.setdefault("apiKey", "sk-local")
    p.setdefault("models", {})[mid] = entry
    if json.dumps(cfg, sort_keys=True) != before:
        os.makedirs(os.path.dirname(OC_CONFIG), exist_ok=True)
        bak = OC_CONFIG + ".bak-llm-launch"
        if os.path.exists(OC_CONFIG) and not os.path.exists(bak): shutil.copy2(OC_CONFIG, bak)
        tmp = OC_CONFIG + ".tmp"
        with open(tmp, "w") as f: json.dump(cfg, f, indent=2)
        os.replace(tmp, OC_CONFIG)
        print(f"   opencode config: {PROVIDER}/{mid} set to {ctx:,} tokens of context ({OC_CONFIG})")
    return prov


def load_kit():
    """kit.json: the opencode setup this kit's sessions use (its plugins with their options, and opencode settings)."""
    with open(KIT_JSON) as f: return json.load(f)


def _sha256(path):
    import hashlib
    try:
        with open(path, "rb") as f: return hashlib.sha256(f.read()).hexdigest()
    except OSError: return None


def plugin_path(p, quiet=False):
    """The local file of one kit.json plugin: a file in this kit, or a download (once, into ~/.config/llm-launch/plugins/)
    kept only if it matches its sha256. None if it cannot be had (no network): the session then runs without it."""
    if p.get("file"):
        path = os.path.join(KIT, p["file"])
        if os.path.isfile(path) and (not p.get("sha256") or _sha256(path) == p["sha256"]): return path
        print(f"{R}   the plugin file {path} is missing or changed{N}"); return None
    dest = os.path.join(PLUGIN_DIR, p.get("save_as") or os.path.basename(p["url"]))
    if _sha256(dest) == p["sha256"]: return dest
    os.makedirs(PLUGIN_DIR, exist_ok=True)
    old = os.path.join(STATE_DIR, os.path.basename(dest))          # where versions before kit.json kept it
    if _sha256(old) == p["sha256"]:
        os.replace(old, dest); return dest
    tmp = dest + ".part"
    try:
        if not quiet: print(f"   downloading the plugin: {p['name']} ...")
        with urllib.request.urlopen(p["url"], timeout=60) as r, open(tmp, "wb") as fh: fh.write(r.read())
        if _sha256(tmp) != p["sha256"]:
            os.remove(tmp); print(f"{R}   the downloaded plugin ({p['name']}) does not match its checksum; not used{N}")
            return None
        os.replace(tmp, dest); return dest
    except Exception as e:
        print(f"{R}   could not download the plugin {p['name']} ({e}){N}")
        return None


def _subst(v):
    """{kit} and {state} in kit.json option values: this kit's folder, ~/.config/llm-launch."""
    if isinstance(v, str): return v.replace("{kit}", KIT).replace("{state}", STATE_DIR)
    if isinstance(v, list): return [_subst(x) for x in v]
    if isinstance(v, dict): return {k: _subst(x) for k, x in v.items()}
    return v


def fetch_plugins(kit=None, quiet=False):
    """[(plugin, local path or None)] for every plugin kit.json names."""
    kit = kit or load_kit()
    return [(p, plugin_path(p, quiet)) for p in kit.get("plugins", [])]


def session_config(prov):
    """What this launcher adds for its own opencode sessions only (your opencode.json keeps none of it): the plugins and
    settings kit.json names (today: the task driver in focus mode, and the compaction-anchor plugin's "slim" part --
    opencode's long built-in instructions swapped for a short prompt, your project's AGENTS.md kept, the longest tool
    descriptions shortened -- the skill tool off, and the compaction reserve used in the tests), plus the model entry
    (in case opencode.json could not be edited)."""
    kit = load_kit()
    plugins = []
    for p, path in fetch_plugins(kit):
        if path: plugins.append(["file://" + path, _subst(p.get("options") or {})])
        else: print(f"{R}   this session runs without: {p['name']}{N}")
    cfg = {"provider": {PROVIDER: prov}, "plugin": plugins}
    cfg.update(_subst(kit.get("opencode_config") or {}))
    return json.dumps(cfg)


# ------------------------------------------------------------------ kit updates (git pull from GitHub)
def _git(*args, timeout=30):
    return subprocess.run(["git", "-C", KIT, *args], capture_output=True, text=True, timeout=timeout,
                          env=dict(os.environ, GIT_TERMINAL_PROMPT="0"))


def kit_is_clone():
    return os.path.isdir(os.path.join(KIT, ".git")) and shutil.which("git") is not None


def updates_waiting(timeout=8):
    """The changes on GitHub this copy of the kit does not have yet (their one-line descriptions, newest first): [] when
    it is up to date, None when it could not check (offline, not a git clone)."""
    if not kit_is_clone(): return None
    try:
        if _git("fetch", "--quiet", timeout=timeout).returncode: return None
        r = _git("log", "--format=%s", "HEAD..@{u}")
    except (subprocess.TimeoutExpired, OSError):
        return None
    return [s for s in r.stdout.split("\n") if s] if r.returncode == 0 else None


def update(ask_first=False, changes=None):
    """Bring the kit up to date (git pull --ff-only), then restart this script, so the rest runs on the new code:
    post_update() says what changed and gets the plugins the new kit.json names. False if nothing was updated."""
    if not kit_is_clone():
        print(f"{R}This copy of the kit is not a git clone, so it cannot update itself.{N} Clone it again (README.md)."); return False
    if changes is None:
        changes = updates_waiting(timeout=30)
        if changes is None: print(f"{R}Could not reach GitHub to check for updates.{N}"); return False
    if not changes:
        print("The kit is up to date."); return False
    print(f"{B}A kit update is available{N} ({len(changes)} change{'s' if len(changes) > 1 else ''}):")
    for s in changes[:12]: print(f"   - {s}")
    if len(changes) > 12: print(f"   ... and {len(changes) - 12} more")
    if ask_first and not yes("Update now?", default=True): return False
    head = _git("rev-parse", "--short", "HEAD").stdout.strip()
    setup = (load_kit().get("setup") or "")
    r = _git("pull", "--ff-only", "--quiet", timeout=180)
    if r.returncode:
        print(f"{R}The update did not apply:{N}")
        for line in (r.stderr or r.stdout).strip().splitlines()[-6:]: print(f"   {line}")
        print(f"   Usually that means files in the kit folder were changed on this machine. See which:  git -C {KIT} status\n"
              f"   To set those changes aside and update:  git -C {KIT} stash && llm-launch --update")
        return False
    args = sys.argv[1:]                            # with --update: post_update() runs, then it stops there
    os.environ["LLM_LAUNCH_UPDATED_FROM"] = json.dumps([head, setup])
    sys.stdout.flush()
    os.execv(sys.executable, [sys.executable, os.path.join(KIT, "llm-launch.py")] + args)


def post_update():
    """Right after update() restarted this script: what changed, the new plugins, and whatever else the new kit needs."""
    try: head, setup = json.loads(os.environ.pop("LLM_LAUNCH_UPDATED_FROM"))
    except (KeyError, ValueError): return False
    kit = load_kit()
    print(f"   updated: {head} -> {_git('rev-parse', '--short', 'HEAD').stdout.strip()}")
    if kit.get("setup") != setup: print(f"   opencode setup: {setup} -> {B}{kit.get('setup')}{N}")
    for p, path in fetch_plugins(kit):
        if not path: print(f"{R}   missing: {p['name']} (the launcher tries again at each start){N}")
    check_setup(kit)
    print()
    return True


def check_setup(kit=None):
    """The kit's one-time setup or its opencode version moved on since this machine last ran setup.sh: offer to run it."""
    kit = kit or load_kit()
    try: done = int(open(SETUP_VERSION_FILE).read().strip())
    except (OSError, ValueError): done = 1                # setup.sh from before it recorded this: version 1
    want = kit.get("setup_version", 1)
    oc = opencode_bin()
    have_oc = ""
    if oc:
        try: have_oc = subprocess.run([oc, "--version"], capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception: pass
    why = []
    if want > done: why.append("the kit's one-time setup has new steps")
    if kit.get("opencode") and have_oc != kit["opencode"]:
        why.append(f"this setup was tested with opencode {kit['opencode']} (you have {have_oc or 'none'})")
    if not why: return True
    print(f"{B}Setup:{N} " + "; ".join(why) + ".")
    if not yes("Run setup.sh now (it skips what is already done)?", default=True):
        print(f"   Later:  bash {os.path.join(KIT, 'setup.sh')}"); return False
    return subprocess.call(["bash", os.path.join(KIT, "setup.sh"), "--no-launch"]) == 0


# ------------------------------------------------------------------ the graphics mode, and what else uses the GPU
def graphics_mode():
    """Pop!_OS's graphics mode (system76-power): compute / hybrid / nvidia / integrated; None elsewhere."""
    if not shutil.which("system76-power"): return None
    try: return subprocess.run(["system76-power", "graphics"], capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception: return None


def gpu_users():
    """[(pid, name, MiB)]: the programs computing on the NVIDIA GPU now (nvidia-smi's compute apps)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return []
    rows = []
    for line in out.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) == 3 and f[0].isdigit(): rows.append((int(f[0]), os.path.basename(f[1]), int(f[2]) if f[2].isdigit() else 0))
    return rows


def gpu_busy_advice(indent="   "):
    """Lines that say what else holds GPU memory and what to do about it (for the graphics mode this machine is in)."""
    lines = []
    users = gpu_users()
    if users:
        lines.append(indent + "Using the GPU now: " + ", ".join(f"{n} (pid {p}, {m:,} MiB)" for p, n, m in users)
                     + ". Close those, or stop them (kill <pid>).")
    mode = graphics_mode()
    if mode == "nvidia":
        lines.append(indent + "The desktop runs on the NVIDIA card (graphics mode: nvidia), which takes some of its memory. Switch")
        lines.append(indent + "to Compute: choose Graphics: Compute in the top-bar menu (or: system76-power graphics compute), then reboot.")
    elif not users:
        lines.append(indent + "Run  nvidia-smi  to see what is using it.")
    return lines


# ------------------------------------------------------------------ downloads
def download(st):
    os.makedirs(os.path.join(HOME, "models"), exist_ok=True)
    print(f"\n{B}Download a model{N} (into ~/models/<repo>/):")
    for i, (repo, f, proj) in enumerate(DOWNLOADS, 1):
        print(f"  {i}) {f}   {D}{repo}{' + image projector' if proj else ''}{N}")
    v = ask("Number (Enter to go back)")
    if not v.isdigit() or not 1 <= int(v) <= len(DOWNLOADS): return
    repo, f, proj = DOWNLOADS[int(v) - 1]
    dest = os.path.join(HOME, "models", repo.replace("/", "__"))
    os.makedirs(dest, exist_ok=True)
    for name in [f] + ([proj] if proj else []):
        out = os.path.join(dest, name)
        if os.path.exists(out): print(f"   have {name}"); continue
        url = f"https://huggingface.co/{repo}/resolve/main/{name}"
        print(f"   downloading {name} ...")
        hf = shutil.which("hf")
        if hf:
            if subprocess.call([hf, "download", repo, name, "--local-dir", dest]) != 0: print(f"{R}   download failed{N}")
            continue
        tmp = out + ".part"
        try:
            with urllib.request.urlopen(url) as r, open(tmp, "wb") as fh:
                total = int(r.headers.get("Content-Length") or 0); got = 0; last = 0
                while True:
                    b = r.read(1 << 22)
                    if not b: break
                    fh.write(b); got += len(b)
                    if time.time() - last > 2:
                        last = time.time(); print(f"\r   {gb(got)} / {gb(total)}", end="", flush=True)
            os.replace(tmp, out); print(f"\r   done: {out}            ")
        except Exception as e:
            print(f"\n{R}   download failed: {e}{N}")


# ------------------------------------------------------------------ the session
def add_dir(st, d):
    d = os.path.abspath(os.path.expanduser(d))
    if not os.path.isdir(d): print(f"{R}   no folder {d}{N}"); return False
    dirs = st.setdefault("model_dirs", [])
    if d not in dirs: dirs.append(d); save_state(st)
    return True


def inventory(st):
    """Find the models this kit knows about anywhere in the home folder (and other mounted drives), say what is known about each, and add their
    folders to the launcher's search path. Exit status 0 when a tested Bonsai 2.5 27B file is among them, else 3."""
    skip = {os.path.join(HOME, x) for x in (".local/share/Trash", "snap", ".npm", ".cargo", ".rustup", ".nvm")}
    found = []
    user = os.path.basename(HOME)
    tops = [HOME] + [d for d in (f"/media/{user}", f"/run/media/{user}", "/mnt") if os.path.isdir(d)]   # other drives too
    walk = (x for top in tops for x in os.walk(top, followlinks=False))
    for root, dirs, files in walk:
        top = next(t for t in tops if root == t or root.startswith(t + os.sep))
        if root in skip or root[len(top):].count(os.sep) >= 7:
            dirs[:] = []; continue
        dirs[:] = [d for d in dirs if d not in ("node_modules", ".git", "__pycache__")]
        for f in files:
            if f.lower().endswith(".gguf") and "mmproj" not in f.lower() and any(f.startswith(k) for k in TESTED_FILES):
                if not re.search(r"-(\d{5})-of-\d{5}\.gguf$", f) or "-00001-of-" in f: found.append(os.path.join(root, f))
    have_b25 = False
    if not found: print("   none of the models this kit tested were found")
    for path in sorted(found, key=lambda p: os.path.basename(p).lower()):
        name = os.path.basename(path); rec = RECOMMENDED.get(name)
        try: size = gb(os.path.getsize(path))
        except OSError: size = "?"
        proj = projector_for({"dir": os.path.dirname(path)})
        big = next((v for k, v in BIGGER_PACKS.items() if name.startswith(k)), None)
        if rec:
            ctx = max(x for x in (rec.get("mtp_off"), rec.get("mtp_on")) if x)
            what = f"{'expected' if rec.get('inferred') else 'tested'} on 8 GB: up to {ctx:,} tokens"
            if name.startswith("Ternary-Bonsai-2.5-27B"): have_b25 = True
        elif big: what = f"a bigger packing ({big}): on 8 GB use the PTQ1_0 file instead"
        else:
            fam = next(k for k in TESTED_FILES if name.startswith(k))
            what = f"not tested here (the tested file is {TESTED_FILES[fam]}): you choose its context"
        print(f"   {name}  {D}{size}{N}\n      {what}{'; image file next to it' if proj else ''}\n      {D}{path.replace(HOME, '~', 1)}{N}")
        d = os.path.dirname(path)
        if not d.startswith(hf_cache()): add_dir(st, d)
    return 0 if have_b25 else 3


def pick_model(st):
    while True:
        models = scan_models(st)
        tuned = load_tuned()
        print(f"\n{B}Models{N}  {D}(Hugging Face downloads, ~/models, ~/Downloads, LM Studio's folders, and folders you added){N}")
        if not models: print("   none found yet")
        for i, m in enumerate(models, 1):
            rec = RECOMMENDED.get(m["name"]) or {}
            best = max([x for x in (rec.get("mtp_off"), rec.get("mtp_on")) if x] or [0])
            tag = f"  {D}{'expected' if rec.get('inferred') else 'tested'} on 8 GB: up to {best:,} tokens{N}" if best else ""
            here = {k.split("|", 1)[1]: r for k, r in tuned.items() if k.split("|", 1)[0] == m["name"]}
            if here:   # this machine's own numbers win (the MTP-off / images-CPU setting first)
                k0 = "mtp=off,images=cpu" if "mtp=off,images=cpu" in here else sorted(here)[0]
                r0 = here[k0]; sp = r0.get("speed") or {}; ds = sorted(sp, key=lambda x: int(x))
                wr = f", writing {sp[ds[0]].get('writing')}-{sp[ds[-1]].get('writing')} t/s" if ds else ""
                tag = (f"  {D}measured here: " + (f"up to {r0.get('suggested', 0):,} tokens{wr}" if r0.get("suggested") else "does not fit")
                       + (f" (+{len(here) - 1} more setting{'s' if len(here) > 2 else ''})" if len(here) > 1 else "") + f"{N}")
            print(f"  {i:2d}) {m['name']}  {D}{gb(m['size'])}  {m['repo']}{N}{tag}")
        print(f"   d) download a model    f) add a folder to search    r) measurements on this machine    q) quit")
        v = ask("Model").lower()
        if v == "q": sys.exit(0)
        if v == "r":
            print(f"\n{B}Measured on this machine{N} (llm-tune):"); print("\n".join(results_table(tuned))); continue
        if v == "d": download(st); continue
        if v == "f":
            add_dir(st, ask("Folder with .gguf files"))
            continue
        if v.isdigit() and 1 <= int(v) <= len(models): return models[int(v) - 1]


def load_tuned():
    """This machine's own measurements (llm-tune.py): {"<model file>|mtp=..,images=..": record}."""
    try:
        with open(os.path.join(STATE_DIR, "tuned.json")) as f: return json.load(f)
    except (OSError, ValueError): return {}


def tuned_for(key):
    return load_tuned().get(key)


def setting_label(key):
    """'mtp=off,images=cpu' -> 'MTP off, images CPU'"""
    kv = dict(x.split("=") for x in key.split(","))
    return f"MTP {kv.get('mtp', 'off')}, images {kv.get('images', 'cpu').upper()}"


def kshort(n):
    return f"{n // 1000}k" if n >= 1000 else str(n)


def speed_lines(rec, indent="   "):
    """The depth / reading / writing table of one measurement."""
    sp = rec.get("speed") or {}
    out = [f"{indent}{'depth':>9}   {'reading':>8}   {'writing':>8}   (tokens per second)"]
    for d in sorted(sp, key=lambda x: int(x)):
        out.append(f"{indent}{int(d):>9,}   {sp[d].get('reading', 0):>8,}   {sp[d].get('writing', 0):>8}")
    return out


def results_table(tuned=None):
    """Every measurement saved on this machine, one line each."""
    tuned = load_tuned() if tuned is None else tuned
    if not tuned: return ["   none yet: run llm-tune to measure a model on this machine"]
    rows = [("model", "setting", "held", "suggested", "reading / writing at each depth (t/s)", "measured")]
    for k in sorted(tuned):
        name, setting = k.split("|", 1); r = tuned[k]; sp = r.get("speed") or {}
        w = "  ".join(f"{kshort(int(d))}: {sp[d].get('reading')}/{sp[d].get('writing')}" for d in sorted(sp, key=lambda x: int(x)))
        if not r.get("max_ctx"): w = r.get("note") or "does not fit"
        rows.append((name.replace(".gguf", ""), setting_label(setting), f"{r.get('max_ctx', 0):,}", f"{r.get('suggested', 0):,}", w or "-", r.get("measured", "")))
    wid = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    line = lambda r: "   " + "  ".join(str(c).ljust(wid[i]) for i, c in enumerate(r))
    return [line(rows[0]), "   " + "  ".join("-" * w for w in wid)] + [line(r) for r in rows[1:]]


def choose(st, model, ask_ctx=True):
    """MTP, images, context (in that order). Returns a dict, or None to pick another model. ask_ctx=False (llm-tune):
    stop after MTP / images."""
    big = next((v for k, v in BIGGER_PACKS.items() if model["name"].startswith(k)), None)
    if big:
        print(f"\n{R}Note:{N} this packing is {big}. The PTQ1_0 file holds the same weights in 6.0 GB, which on an 8 GB card leaves")
        print("   room for about 88,000 tokens of context. With this one there is far less, or none. (Download PTQ1_0: choose d.)")
    meta = gguf_meta(model["path"])
    rec = RECOMMENDED.get(model["name"], {})
    has_mtp = bool(meta.get("nextn_predict_layers"))
    mem = st.setdefault("models", {}).setdefault(model["name"], {})
    mtp = False
    if not has_mtp:
        other = model["name"].replace(".gguf", "-mtp.gguf") if model["name"].startswith("Ternary-Bonsai-2.5-27B-PTQ1_0.gguf") else None
        print(f"\n{B}MTP{N} (a draft head that speeds up writing): this file has none, so it runs without it."
              + (f"\n   {D}{other} has one: choose d at the model menu to get it.{N}" if other else ""))
    if has_mtp:
        print(f"\n{B}MTP{N} (a built-in draft head that makes generation faster).")
        print("   Turning MTP off leaves more room for context, at the cost of running slower.")
        on, off = rec.get("mtp_on"), rec.get("mtp_off")
        if on and off:
            print(f"   {D}Tested on 8 GB: up to {on:,} tokens with MTP on, {off:,} with it off.{N}")
        elif off and not on:
            print(f"   {D}On an 8 GB card this model fits much more context with MTP off.{N}")
        # default: on when it costs little context (or nothing is known), off when it costs a lot
        cheap = bool(on) and (not off or on >= 0.9 * off)
        mtp = yes("Use MTP?", default=mem.get("mtp", cheap or not rec))
    proj = projector_for(model)
    proj_gpu = False
    if proj:
        print(f"\n{B}Images{N} ({os.path.basename(proj)}): processing them on the CPU keeps the GPU's memory for context;")
        print("   on the GPU, images are read faster but there is less room for context.")
        proj_gpu = not yes("Process images on the CPU?", default=not mem.get("images_gpu", False))
    key = f"mtp={'on' if mtp else 'off'},images={'gpu' if (proj and proj_gpu) else 'cpu'}"
    draft_n = next((n for k, n in DRAFT_N.items() if model["name"].startswith(k)), 2)
    if not ask_ctx:
        return {"mtp": mtp, "proj": proj, "proj_gpu": proj_gpu, "ctx": None, "key": key, "draft_n": draft_n}
    mine = tuned_for(f"{model['name']}|{key}")
    suggested = rec.get("mtp_on" if mtp else "mtp_off")
    if suggested and proj and proj_gpu and rec.get("mib_per_1k"):
        mib = os.path.getsize(proj) / 2**20 + 300          # the projector + its working memory
        suggested = max(4096, int(suggested - mib / rec["mib_per_1k"] * 1000) // 1024 * 1024)
    last = st.get("_tried", {}).get(f"{model['name']}|{key}") or mem.get("ctx", {}).get(key)
    native = meta.get("context_length")
    print(f"\n{B}Context{N} (how many tokens the model can keep in view at once).")
    if mine and not mine.get("suggested"):
        print(f"   {R}Measured on this machine (llm-tune, {mine.get('measured', '')}): {mine.get('note') or 'it does not fit'}{N}")
        mine = None
    if mine:
        suggested = mine["suggested"]
        print(f"   Measured on this machine (llm-tune, {mine.get('measured', '')}): held {mine.get('max_ctx', 0):,}, suggested {suggested:,}")
        if mine.get("speed"): print("\n".join(speed_lines(mine, indent="   ")))
    elif suggested and rec.get("inferred"): print(f"   Expected to fit on an 8 GB card (same weights as a tested file, about the same size): {suggested:,}")
    elif suggested: print(f"   Tested to fit on an 8 GB RTX 3070 Laptop with these settings: {suggested:,}")
    else: print("   No tested number for this model: enter one (if it does not fit, the server will say so), or measure it: llm-tune")
    if last: print(f"   You used {last:,} last time.")
    if native: print(f"   {D}The model's own maximum: {native:,}{N}")
    while True:
        v = ask("Context", default=str(last or suggested or ""))
        ctx = parse_ctx(v) if v else None
        if ctx and ctx >= 2048:
            if native and ctx > native: print(f"   (more than the model's maximum {native:,}; using {native:,})"); ctx = native
            break
        print("   Enter a number of tokens, e.g. 65536 or 64k.")
    return {"mtp": mtp, "proj": proj, "proj_gpu": proj_gpu, "ctx": ctx, "key": key, "draft_n": draft_n}


RUNNING = [None]          # the server this run started: stopped if the terminal closes or the script is killed


def _on_signal(signum, _frame):
    stop_server(RUNNING[0])
    sys.exit(128 + signum)


def attach(port):
    """Open opencode on a llama-server that is already running (LL-Dash started it, or another llm-launch)."""
    if not check_opencode_config(): return 1
    if not health(port):
        print(f"{R}No model server answering on port {port}.{N} Start one (LL-Dash, or llm-launch without --attach)."); return 1
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=5) as r: props = json.loads(r.read())
    except Exception as e:
        print(f"{R}Could not read the server's settings: {e}{N}"); return 1
    path = props.get("model_path") or ""
    ctx = (props.get("default_generation_settings") or {}).get("n_ctx") or props.get("n_ctx")
    if not path or not ctx:
        print(f"{R}The server did not say which model it runs or its context.{N}"); return 1
    name = os.path.basename(path); mid = model_id(name)
    print(f"Model server on port {port}: {name}, {ctx:,} tokens of context")
    prov = update_opencode_config(mid, f"{name} ({ctx // 1000}k)", ctx, port)
    env = dict(os.environ, OPENCODE_CONFIG_CONTENT=session_config(prov))
    print(f"Starting opencode with {PROVIDER}/{mid} in {os.getcwd()}")
    print(f"   opencode setup: {load_kit().get('setup')}  {D}(kit.json){N}\n")
    return subprocess.call([opencode_bin(), "-m", f"{PROVIDER}/{mid}"], env=env)


def main():
    try: sys.stdout.reconfigure(line_buffering=True)
    except Exception: pass
    for sg in (signal.SIGHUP, signal.SIGTERM): signal.signal(sg, _on_signal)
    if "--results" in sys.argv:
        print(f"{B}Measured on this machine{N} (llm-tune):"); print("\n".join(results_table())); return 0
    if "--inventory" in sys.argv:
        return inventory(load_state())
    if "--add-dir" in sys.argv:
        st = load_state()
        return 0 if add_dir(st, sys.argv[sys.argv.index("--add-dir") + 1]) else 1
    if "--fetch-plugins" in sys.argv:
        kit = load_kit(); got = fetch_plugins(kit)
        for p, path in got: print(f"   {p['name']}: " + (path if path else f"{R}missing{N}"))
        return 0 if all(path for _, path in got) else 1
    updated = post_update()                        # only right after an update restarted this script
    if "--update" in sys.argv:
        if not updated: update()                   # (restarts this script when it updated)
        return 0
    if sys.stdin.isatty() and "--no-update" not in sys.argv:
        changes = updates_waiting()
        if changes: update(ask_first=True, changes=changes); print()
        elif changes is None and kit_is_clone(): print(f"{D}(could not check GitHub for kit updates){N}")
    if "--attach" in sys.argv:
        if not opencode_bin(): print(f"{R}opencode is not installed (bash setup.sh installs it).{N}"); return 1
        port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else DEFAULT_PORT
        return attach(port)
    st = load_state()
    st.pop("_tried", None)
    server = find_server(st)
    if not opencode_bin():
        print(f"{R}opencode is not installed (bash setup.sh installs it).{N}"); sys.exit(1)
    if not check_opencode_config(): return 1
    model = pick_model(st)
    while True:
        c = choose(st, model)
        port = DEFAULT_PORT
        while not port_free(port): port += 1
        log = os.path.join(LOG_DIR, f"llama-server-{time.strftime('%Y%m%d-%H%M%S')}.log")
        mid = model_id(model["name"])
        print(f"\nStarting llama-server: {model['name']}, {c['ctx']:,} tokens, MTP {'on' if c['mtp'] else 'off'}"
              + (f", images on the {'GPU' if c['proj_gpu'] else 'CPU'}" if c["proj"] else "") + f" (port {port})")
        p, argv = start_server(server, model, c["ctx"], c["mtp"], c["draft_n"], c["proj"], c["proj_gpu"], port, log)
        RUNNING[0] = p
        st.setdefault("_tried", {})[f"{model['name']}|{c['key']}"] = c["ctx"]      # proposed again on a retry in this run
        ok = wait_ready(p, port) and warm_up(port) and p.poll() is None
        if not ok:
            stop_server(p)
            explain_failure(log, c["mtp"], c["proj"], c["proj_gpu"])
            if yes("\nTry again with different settings?", default=True): continue
            sys.exit(1)
        print("   ready.")
        st.pop("_tried", None)                     # it worked: remember these settings for this model
        mem = st.setdefault("models", {}).setdefault(model["name"], {})
        mem["mtp"] = c["mtp"]; mem["images_gpu"] = c["proj_gpu"]; mem.setdefault("ctx", {})[c["key"]] = c["ctx"]
        save_state(st)
        prov = update_opencode_config(mid, f"{model['name']} ({c['ctx'] // 1000}k)", c["ctx"], port)
        env = dict(os.environ, OPENCODE_CONFIG_CONTENT=session_config(prov))
        print(f"Starting opencode with {PROVIDER}/{mid} in {os.getcwd()}  {D}(server log: {log}){N}")
        print(f"   opencode setup: {load_kit().get('setup')}  {D}(kit.json){N}\n")
        time.sleep(1)
        try:
            rc = subprocess.call([opencode_bin(), "-m", f"{PROVIDER}/{mid}"], env=env)
        except KeyboardInterrupt:
            rc = 130
        died = p.poll() is not None
        if died:
            print(f"\n{R}The model server stopped while opencode was running.{N}")
            explain_failure(log, c["mtp"], c["proj"], c["proj_gpu"])
        else:
            if yes("\nStop the model server?", default=True): stop_server(p)
            else: print(f"   It keeps running on port {port} (pid {p.pid}); stop it with: kill {p.pid}")
        return rc



if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(); sys.exit(130)
