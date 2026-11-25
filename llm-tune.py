#!/usr/bin/env python3
"""llm-tune: measure, on THIS machine, how much context a model fits in the GPU and how fast it runs.

    python3 llm-tune.py            (or: llm-tune)    every model found, every MTP setting, the ones not measured yet
    llm-tune --redo                                  measure everything again
    llm-tune --one                                   pick one model and its settings (MTP, images on the CPU or the GPU)

For each model and setting:
  1. FIT: finds the largest context that loads (by halving the range), fills it to 97% with real text, then sends
     requests at a few depths -- llama.cpp takes a little more memory the first time it meets each kind of request, and a
     context filled to the last megabyte can crash on that later -- stepping down until all of it holds. The suggestion
     leaves at least 64 MiB of the GPU free.
  2. SPEED: reading and writing speed (tokens per second) at 4k, 32k and near the maximum, three answers per depth,
     with the model's own recommended sampling.
  3. Saves the result in ~/.config/llm-launch/tuned.json; llm-launch then suggests this machine's number.

Close other programs that use the GPU first (nvidia-smi shows them); keep the desktop on the Intel graphics. A run takes
roughly 10-25 minutes per model and setting: most of it is reading long prompts. Ctrl-C stops it and the server.
Python 3.8+, standard library only; uses llm-launch.py from the same folder.
"""
import importlib.util, json, os, signal, subprocess, sys, threading, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("llm_launch", os.path.join(HERE, "llm-launch.py"))
L = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(L)
B, D, R, N = L.B, L.D, L.R, L.N
TUNED = os.path.join(L.STATE_DIR, "tuned.json")
PORT = 8095
HEADROOM_MIB = 64
CURRENT = [None]


# ------------------------------------------------------------------ GPU memory (nvidia-smi)
def gpu_index():
    """The NVIDIA GPU to measure: the only one, or the one with the most memory (a laptop dGPU)."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    rows = [[x.strip() for x in l.split(",")] for l in out if l.strip()]
    if not rows: sys.exit(f"{R}nvidia-smi lists no GPU{N}")
    best = max(rows, key=lambda r: int(r[2]))
    return best[0], best[1], int(best[2])


def mem(idx):
    out = subprocess.run(["nvidia-smi", "-i", idx, "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    used, free = (int(x.strip()) for x in out.split(","))
    return used, free


class Sampler:
    """Lowest free GPU memory while a request runs (sampled every 0.25 s)."""
    def __init__(self, idx):
        self.idx, self.low, self._stop = idx, None, threading.Event()
    def __enter__(self):
        def run():
            while not self._stop.is_set():
                try:
                    f = mem(self.idx)[1]; self.low = f if self.low is None else min(self.low, f)
                except Exception: pass
                time.sleep(0.25)
        self._t = threading.Thread(target=run, daemon=True); self._t.start(); return self
    def __exit__(self, *a):
        self._stop.set(); self._t.join()


# ------------------------------------------------------------------ the server
def start(server, model, ctx, s, log):
    p, argv = L.start_server(server, model, ctx, s["mtp"], s["draft_n"], s["proj"], s["proj_gpu"], PORT, log)
    CURRENT[0] = p
    ok = L.wait_ready(p, PORT, timeout=900)
    return p, ok and p.poll() is None


def stop():
    L.stop_server(CURRENT[0]); CURRENT[0] = None
    time.sleep(1)


def post(path, body, timeout=3600):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read())


_TEXT = None
def text_tokens(n):
    """n tokens of real source code (Python's own standard library), tokenized by the running server."""
    global _TEXT
    if _TEXT is None:
        lib = os.path.dirname(os.__file__); parts, size = [], 0
        for name in sorted(os.listdir(lib)):
            if name.endswith(".py"):
                try: t = open(os.path.join(lib, name), errors="replace").read()
                except OSError: continue
                parts.append(t); size += len(t)
                if size > 4_000_000: break
        _TEXT = post("/tokenize", {"content": "".join(parts)})["tokens"]
    t = _TEXT
    while len(t) < n: t = t + _TEXT
    return t[:n]


def ask_at_depth(depth, n_predict):
    """A chat request with about `depth` tokens of code before the question; returns llama-server's timings."""
    code = post("/detokenize", {"tokens": text_tokens(depth)})["content"]
    body = {"messages": [{"role": "user", "content": code + "\n\nExplain what the code above does, part by part."}],
            "max_tokens": n_predict}
    r = post("/v1/chat/completions", body)
    return r.get("timings") or {}


# ------------------------------------------------------------------ fit
def fit(server, model, s, idx, native, logdir):
    """(largest context that held, suggested context, notes)."""
    def loads(ctx):
        p, ok = start(server, model, ctx, s, os.path.join(logdir, f"fit-{ctx}.log"))
        used = mem(idx)[0] if ok else None
        stop()
        print(f"   {ctx:>7,} tokens: {'loads' if ok else 'does not fit'}" + (f" ({used:,} MiB in use)" if ok else ""), flush=True)
        return ok, used
    def holds(ctx):
        """Fill to 97%, then a short and a medium request on a fresh server (new request shapes take a little more)."""
        p, ok = start(server, model, ctx, s, os.path.join(logdir, f"hold-{ctx}.log"))
        low = None
        try:
            if not ok: return False, None
            with Sampler(idx) as sm:
                post("/completion", {"prompt": text_tokens(int(ctx * 0.97) - 64), "n_predict": 32, "cache_prompt": False})
                for d in (2048, min(32768, ctx // 3)):
                    ask_at_depth(d, 64)
            low = sm.low
            return p.poll() is None, low
        except Exception:
            return False, low
        finally:
            stop()
    lo, hi = 4096, native
    pts = {}
    ok, u = loads(lo)
    if not ok: return None, None, "it does not fit even at 4,096 tokens"
    pts[lo] = u
    ok, u = loads(hi)
    if ok: pts[hi] = u; lo = hi
    while hi - lo > 1024:
        mid = (lo + hi) // 2 // 1024 * 1024
        if mid in (lo, hi): break
        ok, u = loads(mid)
        if ok: pts[mid] = u; lo = mid
        else: hi = mid
    ctx = lo
    while ctx >= 4096:
        print(f"   checking {ctx:,} with a full prompt and requests at a few depths ...", flush=True)
        ok, low = holds(ctx)
        if ok: break
        print(f"   {ctx:,} did not hold; trying {ctx - 2048:,}", flush=True)
        ctx -= 2048
    if ctx < 4096: return None, None, "no context held under a full prompt"
    # suggestion: leave HEADROOM_MIB free, using the memory each 1,000 tokens took during the search
    ks = sorted(pts)
    slope = (pts[ks[-1]] - pts[ks[0]]) / max(1, ks[-1] - ks[0]) if len(ks) > 1 else 0.02
    need = max(0, HEADROOM_MIB - (low if low is not None else HEADROOM_MIB))
    sug = ctx if need == 0 or slope <= 0 else max(4096, int((ctx - need / slope) // 1024 * 1024))
    return ctx, sug, f"{low} MiB of the GPU stayed free at the fullest" if low is not None else ""


# ------------------------------------------------------------------ speed
def speed(server, model, s, ctx, logdir):
    depths = sorted({d for d in (4096, 32768, int(ctx * 0.9)) if d < ctx * 0.95})
    p, ok = start(server, model, ctx, s, os.path.join(logdir, f"speed-{ctx}.log"))
    out = {}
    try:
        if not ok: return None
        for d in depths:
            first = ask_at_depth(d, 256)                     # reads the whole prompt, then writes
            reps = [ask_at_depth(d, 256) for _ in range(2)]  # the same prompt again: only the writing is new
            w = sorted(x.get("predicted_per_second", 0) for x in [first] + reps)
            out[d] = {"reading": round(first.get("prompt_per_second", 0)), "writing": round(w[len(w) // 2], 1)}
            print(f"   at {d:>7,} tokens: reading {out[d]['reading']:>5,} t/s, writing {out[d]['writing']:>5} t/s", flush=True)
        return out
    finally:
        stop()


# ------------------------------------------------------------------ main
def load_tuned():
    try:
        with open(TUNED) as f: return json.load(f)
    except (OSError, ValueError): return {}


def save(key, rec):
    t = load_tuned(); t[key] = rec
    os.makedirs(L.STATE_DIR, exist_ok=True)
    with open(TUNED + ".tmp", "w") as f: json.dump(t, f, indent=1)
    os.replace(TUNED + ".tmp", TUNED)
    return t


def tune_one(server, model, s, setting_key, idx, gname):
    """Fit, then speed, for one model + setting; saved either way (a model that does not fit is recorded as such)."""
    meta = L.gguf_meta(model["path"])
    native = min(int(meta.get("context_length") or 262144), 262144)
    logdir = os.path.join(L.LOG_DIR, "tune-" + time.strftime("%Y%m%d-%H%M%S")); os.makedirs(logdir, exist_ok=True)
    label = L.setting_label(setting_key)
    print(f"\n{B}{model['name']}{N}  ({label})")
    print(f"{B}Fit{N}: the largest context, searching 4,096 - {native:,}")
    t0 = time.time()
    held, sug, note = fit(server, model, s, idx, native, logdir)
    rec = {"max_ctx": held or 0, "suggested": sug or 0, "speed": None, "gpu": gname, "measured": time.strftime("%Y-%m-%d %H:%M")}
    if not held:
        print(f"{R}   {note}{N}  (logs: {logdir})")
        rec.update(note=note, minutes=round((time.time() - t0) / 60, 1))
        save(f"{model['name']}|{setting_key}", rec); return rec
    print(f"   {B}held: {held:,} tokens; suggested: {sug:,}{N}  {D}{note}{N}")
    print(f"{B}Speed{N} at {sug:,} tokens of context:")
    rec["speed"] = speed(server, model, s, sug, logdir)
    rec["minutes"] = round((time.time() - t0) / 60, 1)
    save(f"{model['name']}|{setting_key}", rec)
    print(f"{B}Result{N} ({rec['minutes']} min): held {held:,}, suggested {sug:,} (llm-launch suggests this from now on)")
    if rec["speed"]: print("\n".join(L.speed_lines(rec, indent="   ")))
    return rec


def settings_for(model):
    """Every setting worth measuring, without questions: MTP off, plus MTP on when the file has the head; images on the
    CPU (the projector loaded there when it is next to the model)."""
    meta = L.gguf_meta(model["path"])
    proj = L.projector_for(model)
    draft_n = next((n for k, n in L.DRAFT_N.items() if model["name"].startswith(k)), 2)
    out = []
    for mtp in ([False, True] if meta.get("nextn_predict_layers") else [False]):
        out.append(({"mtp": mtp, "draft_n": draft_n, "proj": proj, "proj_gpu": False},
                    f"mtp={'on' if mtp else 'off'},images=cpu"))
    return out


def main():
    try: sys.stdout.reconfigure(line_buffering=True)
    except Exception: pass
    def bye(*_):
        stop(); print(); sys.exit(130)
    signal.signal(signal.SIGINT, bye); signal.signal(signal.SIGTERM, bye)
    redo = "--redo" in sys.argv
    st = L.load_state()
    server = L.find_server(st)
    idx, gname, total = gpu_index()
    used, free = mem(idx)
    print(f"{B}GPU{N}: {gname}, {total:,} MiB, {used:,} MiB in use now")
    if used > 300:
        print(f"{R}   {used:,} MiB is already in use.{N} The result is only right for a GPU in the same state: close what nvidia-smi")
        print("   lists (a model server, a browser on the NVIDIA GPU), or switch the desktop to the Intel graphics, then run this again.")
        if not L.yes("Measure anyway?", default=False): return 1
    if not L.port_free(PORT):
        sys.exit(f"{R}port {PORT} is in use (another llm-tune?){N}")
    one = "--one" in sys.argv
    if not one:
        models = L.scan_models(st)
        tuned = load_tuned()
        todo, have = [], []
        for m in models:
            for s, k in settings_for(m):
                (have if (f"{m['name']}|{k}" in tuned and not redo) else todo).append((m, s, k))
        print(f"\n{B}Every model found{N}: {len(todo)} to measure" + (f", {len(have)} already measured (llm-tune --redo measures them again)" if have else ""))
        for m, s, k in todo: print(f"   {m['name']}  ({L.setting_label(k)})")
        if not todo:
            print("   nothing left to measure"); print("\n".join(L.results_table(tuned))); return 0
        print(f"   about {len(todo) * 10}-{len(todo) * 25} minutes in all. Leave the laptop plugged in; Ctrl-C stops it (finished ones stay saved).")
        a = L.ask("Measure all of these (a), or pick one model (p)?", "a").lower()
        one = a.startswith("p")
    if one:
        model = L.pick_model(st)
        c = L.choose(st, model, ask_ctx=False)          # the MTP / images questions
        todo = [(model, {"mtp": c["mtp"], "draft_n": c["draft_n"], "proj": c["proj"], "proj_gpu": c["proj_gpu"]}, c["key"])]
    t0 = time.time()
    for n, (m, s, k) in enumerate(todo, 1):
        if len(todo) > 1: print(f"\n{D}[{n} of {len(todo)}]{N}", end="")
        try: tune_one(server, m, s, k, idx, gname)
        except Exception as e:
            stop(); print(f"{R}   {m['name']} ({L.setting_label(k)}) stopped with an error: {e}{N}")
    print(f"\n{B}All measurements on this machine{N}  ({round((time.time() - t0) / 60)} min this run; also: llm-launch --results, or r at its model menu)")
    print("\n".join(L.results_table()))
    print(f"{D}Saved in {TUNED}{N}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
