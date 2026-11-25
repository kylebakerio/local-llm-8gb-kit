#!/usr/bin/env python3
"""llm-launch: run a local model on an 8 GB NVIDIA laptop GPU and open opencode on it.

    python3 llm-launch.py            (run it from the project folder you want opencode to work in)
    python3 llm-launch.py --fetch-plugins  download the task-driver plugin (setup.sh does this; so does the first launch)
    python3 llm-launch.py --add-dir DIR    also look for models in DIR
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
time, the model's own recommended sampling. opencode gets two plugins for these sessions: the task driver
(opencode-task-driver 0.3.1, focus mode: splits a request with several problems into tasks and works them one at a time,
each on a clean slate) and the "slim" part of the compaction-anchor plugin (a short system prompt in place of opencode's
long one, shorter tool descriptions; your project's AGENTS.md is kept).
Files it writes: ~/.config/llm-launch/ (your choices, the plugin), ~/.cache/llm-launch/ (server logs), and the model
entry in ~/.config/opencode/opencode.json (a backup is kept next to it the first time).
Python 3.8+, standard library only.
"""
import glob, json, os, re, shutil, signal, socket, struct, subprocess, sys, time, urllib.request

HOME = os.path.expanduser("~")
STATE_DIR = os.path.join(HOME, ".config", "llm-launch")
STATE_FILE = os.path.join(STATE_DIR, "state.json")
PLUGIN_FILE = os.path.join(STATE_DIR, "compaction-anchor.js")
# the task driver, published at github.com/kylebakerio/opencode-task-driver (release v0.3.1), checked by its sha256
TD_VERSION = "0.3.1"
TD_URL = f"https://github.com/kylebakerio/opencode-task-driver/releases/download/v{TD_VERSION}/opencode-task-driver.js"
TD_SHA256 = "07e7c273a4bc389a04b348f1f03ca1873859da98e2b9246a4fd4c880272f64eb"
TD_FILE = os.path.join(STATE_DIR, f"opencode-task-driver-{TD_VERSION}.js")
LOG_DIR = os.path.join(HOME, ".cache", "llm-launch")
OC_CONFIG = os.path.join(HOME, ".config", "opencode", "opencode.json")
PROVIDER = "llama-local"
KV = "q4_0"                 # K/V cache type used in every tested number below
RESERVE = 22528             # opencode's compaction reserve in the tests (tokens kept free for the model's next step)
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
# (the short system prompt lives in the plugin below: PLUGIN_SRC, its SLIM_PROMPT)

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
    print("   - and check nothing else is using the GPU: run  nvidia-smi  (the desktop should be on the Intel graphics,\n"
          "     see README.md)")


# ------------------------------------------------------------------ opencode
def model_id(name):
    return re.sub(r"[^a-z0-9.]+", "-", re.sub(r"\.gguf$", "", name.lower())).strip("-")


def update_opencode_config(mid, label, ctx, port):
    """Add / update this model's entry under provider llama-local. A config with comments (JSONC) is not edited: the
    entry is printed for you to paste, and this session gets it anyway (OPENCODE_CONFIG_CONTENT)."""
    entry = {"name": label, "limit": {"context": ctx, "input": ctx, "output": ctx}}
    prov = {"npm": "@ai-sdk/openai-compatible", "name": "Local llama.cpp",
            "options": {"baseURL": f"http://127.0.0.1:{port}/v1", "apiKey": "sk-local"}, "models": {mid: entry}}
    try:
        cfg = json.load(open(OC_CONFIG)) if os.path.exists(OC_CONFIG) else {"$schema": "https://opencode.ai/config.json"}
    except ValueError:
        print(f"\n{R}{OC_CONFIG} could not be read as plain JSON (comments?), so it was not changed.{N} Add this under \"provider\":")
        print(json.dumps({PROVIDER: prov}, indent=2))
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


def fetch_task_driver(quiet=False):
    """The task-driver plugin file, downloaded once from its GitHub release and checked against TD_SHA256. None if it
    cannot be had (no network): the session then runs without it, and says so."""
    import hashlib
    def ok(path):
        try:
            with open(path, "rb") as f: return hashlib.sha256(f.read()).hexdigest() == TD_SHA256
        except OSError: return False
    if ok(TD_FILE): return TD_FILE
    os.makedirs(STATE_DIR, exist_ok=True)
    tmp = TD_FILE + ".part"
    try:
        if not quiet: print(f"   downloading the task-driver plugin {TD_VERSION} from GitHub ...")
        with urllib.request.urlopen(TD_URL, timeout=60) as r, open(tmp, "wb") as fh: fh.write(r.read())
        if not ok(tmp):
            os.remove(tmp); print(f"{R}   the downloaded task-driver plugin does not match its checksum; not used{N}"); return None
        os.replace(tmp, TD_FILE); return TD_FILE
    except Exception as e:
        print(f"{R}   could not download the task-driver plugin ({e}); this session runs without it{N}")
        return None


def session_config(prov):
    """What this launcher adds for its own opencode sessions only (your opencode.json keeps none of it): the task driver
    (focus mode) and the compaction-anchor plugin's "slim" part (opencode's long built-in instructions swapped for a
    short prompt -- your project's AGENTS.md is kept -- and the longest tool descriptions shortened); the skill tool off;
    the compaction reserve used in the tests; and the model entry (in case opencode.json could not be edited)."""
    write_plugin()
    plugins = [["file://" + PLUGIN_FILE, {"parts": ["slim"]}]]
    td = fetch_task_driver()
    if td: plugins.append(["file://" + td, {"mode": "focus"}])
    return json.dumps({
        "provider": {PROVIDER: prov},
        "plugin": plugins,
        "tools": {"skill": False},
        "compaction": {"reserved": RESERVE},
    })


def write_plugin():
    os.makedirs(STATE_DIR, exist_ok=True)
    if not os.path.exists(PLUGIN_FILE) or open(PLUGIN_FILE).read() != PLUGIN_SRC:
        with open(PLUGIN_FILE, "w") as f: f.write(PLUGIN_SRC)


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
        print(f"\n{B}Models{N}  {D}(Hugging Face downloads, ~/models, ~/Downloads, LM Studio's folders, and folders you added){N}")
        if not models: print("   none found yet")
        for i, m in enumerate(models, 1):
            rec = RECOMMENDED.get(m["name"]) or {}
            best = max([x for x in (rec.get("mtp_off"), rec.get("mtp_on")) if x] or [0])
            tag = f"  {D}{'expected' if rec.get('inferred') else 'tested'} on 8 GB: up to {best:,} tokens{N}" if best else ""
            print(f"  {i:2d}) {m['name']}  {D}{gb(m['size'])}  {m['repo']}{N}{tag}")
        print(f"   d) download a model    f) add a folder to search    q) quit")
        v = ask("Model").lower()
        if v == "q": sys.exit(0)
        if v == "d": download(st); continue
        if v == "f":
            add_dir(st, ask("Folder with .gguf files"))
            continue
        if v.isdigit() and 1 <= int(v) <= len(models): return models[int(v) - 1]


def choose(st, model):
    """MTP, images, context (in that order). Returns a dict, or None to pick another model."""
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
    suggested = rec.get("mtp_on" if mtp else "mtp_off")
    if suggested and proj and proj_gpu and rec.get("mib_per_1k"):
        mib = os.path.getsize(proj) / 2**20 + 300          # the projector + its working memory
        suggested = max(4096, int(suggested - mib / rec["mib_per_1k"] * 1000) // 1024 * 1024)
    last = st.get("_tried", {}).get(f"{model['name']}|{key}") or mem.get("ctx", {}).get(key)
    native = meta.get("context_length")
    print(f"\n{B}Context{N} (how many tokens the model can keep in view at once).")
    if suggested and rec.get("inferred"): print(f"   Expected to fit on an 8 GB card (same weights as a tested file, about the same size): {suggested:,}")
    elif suggested: print(f"   Tested to fit on an 8 GB RTX 3070 Laptop with these settings: {suggested:,}")
    else: print("   No tested number for this model: enter one (if it does not fit, the server will say so).")
    if last: print(f"   You used {last:,} last time.")
    if native: print(f"   {D}The model's own maximum: {native:,}{N}")
    while True:
        v = ask("Context", default=str(last or suggested or ""))
        ctx = parse_ctx(v) if v else None
        if ctx and ctx >= 2048:
            if native and ctx > native: print(f"   (more than the model's maximum {native:,}; using {native:,})"); ctx = native
            break
        print("   Enter a number of tokens, e.g. 65536 or 64k.")
    return {"mtp": mtp, "proj": proj, "proj_gpu": proj_gpu, "ctx": ctx, "key": key,
            "draft_n": next((n for k, n in DRAFT_N.items() if model["name"].startswith(k)), 2)}


RUNNING = [None]          # the server this run started: stopped if the terminal closes or the script is killed


def _on_signal(signum, _frame):
    stop_server(RUNNING[0])
    sys.exit(128 + signum)


def attach(port):
    """Open opencode on a llama-server that is already running (LL-Dash started it, or another llm-launch)."""
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
    print(f"Starting opencode with {PROVIDER}/{mid} in {os.getcwd()}\n")
    return subprocess.call([opencode_bin(), "-m", f"{PROVIDER}/{mid}"], env=env)


def main():
    try: sys.stdout.reconfigure(line_buffering=True)
    except Exception: pass
    for sg in (signal.SIGHUP, signal.SIGTERM): signal.signal(sg, _on_signal)
    if "--inventory" in sys.argv:
        return inventory(load_state())
    if "--add-dir" in sys.argv:
        st = load_state()
        return 0 if add_dir(st, sys.argv[sys.argv.index("--add-dir") + 1]) else 1
    if "--fetch-plugins" in sys.argv:
        write_plugin(); td = fetch_task_driver()
        print(f"   plugins ready in {STATE_DIR}" if td else f"{R}   task-driver plugin missing{N}")
        return 0 if td else 1
    if "--attach" in sys.argv:
        if not opencode_bin(): print(f"{R}opencode is not installed (bash setup.sh installs it).{N}"); return 1
        port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else DEFAULT_PORT
        return attach(port)
    st = load_state()
    st.pop("_tried", None)
    server = find_server(st)
    if not opencode_bin():
        print(f"{R}opencode is not installed (bash setup.sh installs it).{N}"); sys.exit(1)
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
        print(f"Starting opencode with {PROVIDER}/{mid} in {os.getcwd()}  {D}(server log: {log}){N}\n")
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


# The compaction-anchor plugin with the "slim" part (shorter tool descriptions), exactly as tested.
PLUGIN_SRC = r'''// Session history for opencode.
//
// When opencode summarizes (compacts) a long session, the summary replaces the conversation and details get lost:
// exact error text, file contents that were read, the wording of the original request, the items not started yet.
// Each part below can be switched on alone (options.parts); with no list given, "reinsert" and "pointer" are on.
//   - always: each time a session is summarized, its complete conversation so far (every message, tool call with its
//     arguments, and tool result in full, oldest first) is written to a readable text file, <dir>/<session id>.md,
//     rewritten whole from the session store each time.
//   - "checklist": the summarizer is asked to start the summary with a task checklist -- every item of the original
//     request (given to it word for word, with the session's todo list) marked done / in progress / not started. The
//     todo list as it stood at that moment is shown under the summary, then a line asking the model to make its todo
//     list (todowrite) match the checklist, work from it and check it before it finishes.
//   - "reinsert": after a summary, the first message starts with the original request, word for word.
//   - "pointer": the summary ends with one line saying where the full conversation file is.
//   - "no-stop-and-ask": the automatic turn after a summary reads "Continue with the task."
//   - "slim": fewer fixed tokens in every request, for a small context window. Unlike the other parts it applies from the
//     first request. (a) opencode's own general instructions at the top of the system prompt ("You are opencode, an
//     interactive CLI tool ...", up to the model / environment lines) are replaced by a short prompt; the environment
//     block and the project's AGENTS.md instructions that follow are kept as they are. (Setting the config's
//     agent.build.prompt instead also drops AGENTS.md in opencode 1.18.34.) (b) The longest built-in tools (bash,
//     todowrite, edit, read, webfetch) get shorter descriptions. The skill tool is switched off in the config.
// Everything except the summarizer's instructions is a view of the context sent to the model: nothing stored changes.
// Before the first summary nothing else is changed at all.
//
// Options (plugin entry [path, {dir, parts}]): dir = where the files go (default ~/.local/share/opencode/history).
// <dir>/.events.jsonl records what the plugin did (loaded, wrote, asked, shown, errors), one JSON object per line.
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const TASK_MAX = 32000;      // characters of the original request repeated after a summary
const PARTS = ["checklist", "reinsert", "pointer", "no-stop-and-ask", "slim"];
const STOP_ASK = "Continue if you have next steps, or stop and ask for clarification if you are unsure how to proceed.";
const CONTINUE = "Continue with the task.";
// "slim": the short system prompt that replaces opencode's own instructions (the project's AGENTS.md is kept).
const SLIM_PROMPT = [
  "You are a coding agent working in the user's project through tools.",
  "- Find and read the relevant code before changing it; follow the project's conventions, style and libraries.",
  "- Make the change the request asks for, then check it (run the relevant tests, or a quick command that exercises it) before you finish.",
  "- When the request has several items or steps, keep a todo list (todowrite) and work through it.",
  "- Use the file tools (read, grep, glob, edit, write) for files and bash for commands.",
  "- Do not commit or push unless asked.",
  "- Keep replies short: when done, say what you changed and how you checked it.",
].join("\n");
const OC_PROMPT_START = "You are opencode, an interactive CLI tool";
// where opencode's own instructions end: the first of these that follows them (model line, environment block, instructions)
const OC_PROMPT_ENDS = ["You are powered by the model named", "Here is some useful information about the environment", "<env>", "Instructions from:"];
function slimSystem(text) {
  const i = text.indexOf(OC_PROMPT_START);
  if (i < 0) return null;
  const ends = OC_PROMPT_ENDS.map((m) => text.indexOf(m, i + OC_PROMPT_START.length)).filter((j) => j > i);
  if (!ends.length) return i === 0 ? SLIM_PROMPT : null;     // an element that is opencode's prompt alone: all of it
  const j = Math.min(...ends);
  return text.slice(0, i) + SLIM_PROMPT + "\n" + text.slice(j);
}
// "slim": the tool descriptions sent to the model (parameters unchanged). Each keeps the rules the original states that
// change what a call does or returns; usage advice and examples are dropped.
const SLIM_TOOLS = {
  bash: "Run a bash command (Linux) for builds, tests, git and other terminal work. Use the workdir parameter instead of " +
        "`cd dir && command`. Quote paths that contain spaces. Default timeout 120000 ms (set timeout to change it). Output " +
        "over 2000 lines or 51200 bytes is cut and the full output is saved to a file you can read or grep. Use /tmp/opencode " +
        "for temporary files outside the workspace. For files use the file tools instead: glob to find, grep to search, read " +
        "to read, edit or write to change. Only commit or push when asked.",
  todowrite: "Create or update the todo list for this session (send the whole list each time; each item has content, status " +
             "pending / in_progress / completed / cancelled, and priority high / medium / low). Use it when the work has three " +
             "or more steps or the request has several items. Keep exactly one item in_progress; mark an item completed only " +
             "when its work is done and checked.",
  edit: "Replace exact text in a file. oldString must match the file exactly, including indentation, without the line-number " +
        "prefix the read tool adds, and must occur once (or set replaceAll to replace every occurrence). Read the file before " +
        "editing it.",
  read: "Read a file (up to 2000 lines from offset, 1-indexed; each line is prefixed `<line>: `; lines over 2000 characters " +
        "are cut) or list a directory. Use an absolute path. Read a larger window rather than many small slices.",
  webfetch: "Fetch a web page (a full URL; http is upgraded to https) and return it as markdown (default), text or html. " +
            "Read-only; very large pages may be summarized.",
};

export const SessionHistory = async ({ client, directory }, options = {}) => {
  const dir = String((options && options.dir) || path.join(os.homedir(), ".local", "share", "opencode", "history"));
  const parts = new Set(Array.isArray(options && options.parts) ? options.parts.filter((x) => PARTS.includes(x)) : ["reinsert", "pointer"]);
  const todoSnap = new Map();   // session id -> the todo list as it stood at the latest summary (also <dir>/<id>.todos.json)
  const known = new Map();   // session id -> { task, file } (cached per process)
  const shown = new Set();   // "<session>:<summary message id>": the first view after each summary is logged once
  const event = (o) => {
    try {
      fs.mkdirSync(dir, { recursive: true });
      fs.appendFileSync(path.join(dir, ".events.jsonl"), JSON.stringify({ t: Date.now(), ...o }) + "\n");
    } catch {}
  };
  const fileOf = (sid) => path.join(dir, `${sid}.md`);
  event({ event: "loaded", directory, parts: [...parts] });

  async function messages(sid) {
    const r = await client.session.messages({ path: { id: sid } });
    const data = Array.isArray(r) ? r : r && r.data;
    if (!Array.isArray(data)) throw new Error("could not read the session: " + JSON.stringify((r && r.error) || r).slice(0, 300));
    return data;
  }

  // the first request of the session: the text the user typed (parts opencode added itself, such as an attached file's
  // contents, are marked synthetic and left out)
  function taskOf(msgs) {
    for (const m of msgs) {
      if (m.info.role !== "user" || m.parts.some((p) => p.type === "compaction")) continue;
      const text = m.parts.filter((p) => p.type === "text" && !p.synthetic && !p.ignored && p.text).map((p) => p.text).join("\n").trim();
      if (text) return text;
    }
    return null;
  }

  function render(msgs, sid) {
    const out = [];
    out.push("# Conversation history", "");
    out.push(`Session ${sid}. Everything said and done in this conversation up to its most recent summary, oldest first:`);
    out.push("every message, every tool call with its arguments, and every tool result in full. Rewritten at each summary.", "");
    let n = 0, calls = 0, summaries = 0;
    for (const m of msgs) {
      const role = m.info.role;
      if (role === "user" && m.parts.some((p) => p.type === "compaction")) continue;   // the summary request itself
      if (role === "assistant" && m.info.summary === true) {
        summaries += 1;
        out.push(`======== conversation summarized here (summary ${summaries}) ========`, "");
        for (const p of m.parts) if (p.type === "text" && p.text) out.push(p.text.trim(), "");
        continue;
      }
      n += 1;
      out.push(`======== [${n}] ${role} ========`, "");
      for (const p of m.parts) {
        if (p.type === "text" && p.text) {
          out.push(p.synthetic ? `(added automatically)\n${p.text}` : p.text, "");
        } else if (p.type === "reasoning" && p.text) {
          out.push("(thinking)", p.text, "");
        } else if (p.type === "file") {
          out.push(`(attached file: ${p.filename || p.url || "file"}${p.mime ? ", " + p.mime : ""})`, "");
        } else if (p.type === "tool") {
          calls += 1;
          const st = p.state || {};
          out.push(`--- tool call ${calls}: ${p.tool} ---`);
          for (const [k, v] of Object.entries(st.input || {})) {
            if (typeof v === "string") out.push(v.includes("\n") ? `${k}:\n${v}` : `${k}: ${v}`);
            else out.push(`${k}: ${JSON.stringify(v)}`);
          }
          if (st.status === "completed") out.push(`--- result of tool call ${calls} ---`, String(st.output ?? ""));
          else if (st.status === "error") out.push(`--- error from tool call ${calls} ---`, String(st.error ?? ""));
          else out.push(`--- tool call ${calls} has no result (${st.status || "unknown"}) ---`);
          out.push("");
        } else if (p.type === "subtask") {
          out.push(`(sub-task: ${p.description || p.agent || ""})`, p.prompt || "", "");
        }
      }
    }
    return { text: out.join("\n") + "\n", messages: n, calls, summaries };
  }

  async function write(sid, why) {
    const msgs = await messages(sid);
    const r = render(msgs, sid);
    fs.mkdirSync(dir, { recursive: true });
    const file = fileOf(sid), tmp = `${file}.tmp`;
    fs.writeFileSync(tmp, r.text);
    fs.renameSync(tmp, file);
    const task = taskOf(msgs);
    known.set(sid, { task, file });
    event({ event: "wrote", why, session: sid, file, bytes: Buffer.byteLength(r.text), messages: r.messages, tool_calls: r.calls,
            summaries: r.summaries, task_chars: task ? task.length : 0 });
    return known.get(sid);
  }

  // what a session needs after a summary; a new process (a later turn of the same session) reads it from the store
  async function stateOf(sid) {
    if (known.has(sid)) return known.get(sid);
    if (fs.existsSync(fileOf(sid))) {
      const task = taskOf(await messages(sid));
      known.set(sid, { task, file: fileOf(sid) });
      return known.get(sid);
    }
    return await write(sid, "late");
  }

  // the session's todo list (opencode's todowrite store). The plugin API can READ it (GET /session/{id}/todo); nothing
  // in it writes todos -- only the model's own todowrite tool does -- so the list is shown, never changed.
  async function todosOf(sid) {
    const r = await client.session.todo({ path: { id: sid } });
    const data = Array.isArray(r) ? r : r && r.data;
    return Array.isArray(data) ? data.map((t) => ({ content: String(t.content || ""), status: String(t.status || ""), priority: String(t.priority || "") })) : [];
  }
  const todoLines = (todos) => todos.map((t) => `- [${t.status}] ${t.content}`).join("\n");
  function snapOf(sid) {
    if (todoSnap.has(sid)) return todoSnap.get(sid);
    let todos = [];
    try { todos = JSON.parse(fs.readFileSync(path.join(dir, `${sid}.todos.json`), "utf8")).todos || []; } catch {}
    todoSnap.set(sid, todos);
    return todos;
  }
  function checklistAsk(task, todos) {
    const out = [
      "Additional requirement for this summary, which takes precedence over the template above: start the summary with a section \"## Task checklist\", placed before \"## Objective\"; every other section follows as the template says.",
      "- One line for every distinct task, issue or requirement in the original request below, in the request's order, using the request's own numbering when it has one. A request that asks for a single thing gets a one-line checklist.",
      "- Begin each line with its status: [done], [in progress] or [not started]. Never leave an item out: an item with no recorded work is [not started], whether or not the conversation or an earlier summary mentions it.",
      "- For a [done] item, add one short piece of evidence on the same line (the change made, or the check that passed).",
    ];
    if (todos.length) out.push("- The session's todo list is below. Use it as evidence of progress; the checklist still follows the original request.");
    out.push("", "<original-request>", task.length > TASK_MAX ? task.slice(0, TASK_MAX) + "\n[...]" : task, "</original-request>");
    if (todos.length) out.push("", "<todo-list>", todoLines(todos), "</todo-list>");
    return out.join("\n");
  }

  const norm = (s) => String(s || "").replace(/\s+/g, " ").trim();
  const textOf = (m) => m.parts.filter((p) => p.type === "text" && !p.synthetic && p.text).map((p) => p.text).join("\n");

  const slimmed = new Set();
  let promptSlims = 0, promptMisses = 0;
  return {
    "experimental.chat.system.transform": async (input, output) => {
      if (!parts.has("slim") || !output || !Array.isArray(output.system)) return;
      let done = false;
      for (let k = 0; k < output.system.length; k++) {
        const t = output.system[k];
        if (typeof t !== "string" || !t.includes(OC_PROMPT_START)) continue;
        const n = slimSystem(t);
        if (n === null) continue;
        output.system[k] = n; done = true;
        if (promptSlims++ === 0) event({ event: "slimmed", what: "system", chars_before: t.length, chars_after: n.length, session: input && input.sessionID });
        break;
      }
      // a request that carries opencode's own prompt but could not be slimmed (its text changed): say so once
      if (!done && output.system.some((t) => typeof t === "string" && t.includes(OC_PROMPT_START)) && promptMisses++ === 0)
        event({ event: "error", where: "slim-system", error: "opencode's prompt is there but its end was not found; left as it is" });
    },
    "tool.definition": async (input, output) => {
      if (!parts.has("slim")) return;
      const d = SLIM_TOOLS[input && input.toolID];
      if (!d) return;
      const before = String((output && output.description) || "").length;
      output.description = d;
      if (!slimmed.has(input.toolID)) { slimmed.add(input.toolID); event({ event: "slimmed", tool: input.toolID, chars_before: before, chars_after: d.length }); }
    },
    "experimental.session.compacting": async (input, output) => {
      const sid = input.sessionID;
      let st = null;
      try { st = await write(sid, "summary"); }
      catch (e) { event({ event: "error", where: "write", session: sid, error: String((e && e.stack) || e).slice(0, 600) }); }
      if (!parts.has("checklist")) return;
      try {
        let todos = [];
        try { todos = await todosOf(sid); }
        catch (e) { event({ event: "error", where: "todos", session: sid, error: String((e && e.stack) || e).slice(0, 300) }); }
        todoSnap.set(sid, todos);
        fs.mkdirSync(dir, { recursive: true });
        fs.writeFileSync(path.join(dir, `${sid}.todos.json`), JSON.stringify({ t: Date.now(), todos }));
        const task = st ? st.task : (await stateOf(sid)).task;
        if (task) output.context.push(checklistAsk(task, todos));
        event({ event: "asked", session: sid, task_chars: task ? task.length : 0, todos: todos.length });
      } catch (e) { event({ event: "error", where: "checklist", session: sid, error: String((e && e.stack) || e).slice(0, 600) }); }
    },
    "experimental.chat.messages.transform": async (_input, output) => {
      const msgs = output.messages;
      for (let i = 0; i + 1 < msgs.length; i++) {
        const u = msgs[i], a = msgs[i + 1];
        if (u.info.role !== "user" || !u.parts.some((p) => p.type === "compaction")) continue;
        if (a.info.role !== "assistant" || a.info.summary !== true) continue;
        const sid = u.info.sessionID;
        let st;
        try { st = await stateOf(sid); }
        catch (e) { event({ event: "error", where: "state", session: sid, error: String((e && e.stack) || e).slice(0, 600) }); continue; }
        const added = {};
        // the original request at the top, unless opencode already put it back after the summary (it re-sends the last
        // request when a provider rejected it as too large)
        const replayed = st.task && msgs.slice(i + 2).some((m) => m.info.role === "user" && norm(textOf(m)) === norm(st.task));
        if (parts.has("reinsert") && st.task && !replayed) {
          const task = st.task.length > TASK_MAX
            ? st.task.slice(0, TASK_MAX) + `\n\n[... the rest of this request is the first message in ${st.file}]`
            : st.task;
          const part = { id: `${u.info.id}-request`, sessionID: sid, messageID: u.info.id, type: "text", synthetic: true,
                         text: `My original request, repeated word for word from the start of this conversation:\n\n${task}\n\n` };
          msgs[i] = { ...u, parts: [part, ...u.parts] };
          added.task_chars = task.length;
        }
        const extra = [];
        let todoAsk = null;   // last, after the pointer: the line right before the turn that continues the work
        if (parts.has("checklist")) {
          const summary = a.parts.filter((p) => p.type === "text").map((p) => p.text || "").join("\n");
          added.checklist = /^\s*#+\s*Task checklist/im.test(summary);
          added.not_started = (summary.match(/\[not started\]/gi) || []).length;
          const todos = snapOf(sid);
          if (todos.length) {
            extra.push({ id: `${a.info.id}-todos`, sessionID: sid, messageID: a.info.id, type: "text", synthetic: true,
                         text: `\n\nTodo list when this summary was written:\n${todoLines(todos)}` });
            added.todos = todos.length;
          }
          // Kyle 10-30: the model keeps the checklist in its own todo list (only its todowrite tool can write it) and
          // checks it before it finishes
          const what = added.checklist ? "match the task checklist above, one item per line" : "cover every task in the original request";
          todoAsk = { id: `${a.info.id}-todoask`, sessionID: sid, messageID: a.info.id, type: "text", synthetic: true,
                       text: `\n\nNext: use the todowrite tool to make the todo list ${what}, each with its current status ` +
                             `(completed, in_progress or pending). Then continue with the items that are not completed, keep the ` +
                             `todo list up to date as you go, and check it before you finish.` };
          added.todo_ask = true;
        }
        if (parts.has("pointer") && st.file) {
          const line = `The full conversation before this summary (every message, tool call and complete tool output) is saved in ${st.file}. ` +
                       `Search it (for example with grep -n) or read the part you need whenever a detail you need is missing from this summary.`;
          extra.push({ id: `${a.info.id}-history`, sessionID: sid, messageID: a.info.id, type: "text", synthetic: true, text: `\n\n${line}` });
          added.pointer = st.file;
        }
        if (todoAsk) extra.push(todoAsk);
        if (extra.length) {
          const ps = [...a.parts];
          let at = ps.length;
          for (let k = ps.length - 1; k >= 0; k--) if (ps[k].type === "text") { at = k + 1; break; }
          ps.splice(at, 0, ...extra);
          msgs[i + 1] = { ...a, parts: ps };
        }
        const key = `${sid}:${a.info.id}`;
        if (!shown.has(key)) { shown.add(key); event({ event: "shown", session: sid, summary: a.info.id, replayed: !!replayed, ...added }); }
        i += 1;
      }
      if (parts.has("no-stop-and-ask")) {
        for (let i = 0; i < msgs.length; i++) {
          const m = msgs[i];
          if (m.info.role !== "user") continue;
          const k = m.parts.findIndex((p) => p.type === "text" && p.synthetic && typeof p.text === "string" && p.text.endsWith(STOP_ASK));
          if (k < 0) continue;
          const ps = [...m.parts];
          ps[k] = { ...ps[k], text: ps[k].text.slice(0, -STOP_ASK.length) + CONTINUE };
          msgs[i] = { ...m, parts: ps };
          const key = `${m.info.sessionID}:${m.info.id}:continue`;
          if (!shown.has(key)) { shown.add(key); event({ event: "continue-replaced", session: m.info.sessionID, message: m.info.id }); }
        }
      }
    },
  };
};
'''

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(); sys.exit(130)
