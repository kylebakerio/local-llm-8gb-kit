#!/usr/bin/env python3
"""llm-tune: measure, on THIS machine, how much context a model fits in the GPU and how fast it runs.

    python3 llm-tune.py            (or: llm-tune)    every model found, every MTP setting, the ones not measured yet;
                                                     it asks whether to run the fit test, the speed test, or both
    llm-tune --fit                                   the fit test only (does it load, the largest context that holds)
    llm-tune --speed                                 the speed test only (at the context the fit test found)
    llm-tune --redo                                  measure again what was measured before
    llm-tune --one                                   pick one model and its settings (MTP, images on the CPU or the GPU)
    (they combine: llm-tune --fit --one, llm-tune --speed --redo, ...)

Settings measured for a model: MTP off, plus MTP on when it has an MTP head, in its file or as a separate head file
next to it (llama-server loads that file as the draft model); when MTP on does not fit, MTP off is measured right after.

For each model and setting:
  1. FIT: first checks system RAM: a model that keeps part of itself in RAM (mixture of experts) needs that RAM free,
     and one that does not get it would load, pass the GPU check, then swap. Then finds the largest context that loads
     (by halving the range), fills it to 97% with real text, then sends requests at a few depths -- llama.cpp takes a
     little more memory the first time it meets each kind of request, and a context filled to the last megabyte can
     crash on that later -- stepping down until all of it holds. The suggestion leaves at least 64 MiB of the GPU free.
     It also records the lowest free RAM, any swapping, and how often the server had to read its weights back from
     disk (major page faults: what running short of RAM looks like for weights mapped from the file) while the context
     was full.
  2. SPEED: reading and writing speed (tokens per second) at 4k, 32k and near the maximum, three answers per depth,
     with the model's own recommended sampling, at the context the fit test suggested. The GPU first cools (up to 4
     minutes) for a 4k number on a cool card; 4k is measured again at the end, after the long run. Laptop GPUs slow
     themselves down when hot: the run records the temperature, the clock and how often the GPU slowed itself, and says
     how much speed heat cost (and what helps).
  3. Saves the result in ~/.config/llm-launch/tuned.json; llm-launch then suggests this machine's number and shows
     its speed and heat.

Close other programs that use the GPU first (nvidia-smi shows them), and the ones that use a lot of RAM (a web browser)
before measuring a model that runs partly from RAM; keep the desktop on the Intel graphics (Pop's Compute or Hybrid
graphics mode). Roughly, per model and setting: fit 5-15 minutes, speed 10-15 (a model with experts in RAM: fit 15-45,
speed 15-30); most of it is reading long prompts. Ctrl-C stops it and the server.
Python 3.8+, standard library only; uses llm-launch.py from the same folder.
"""
import glob, importlib.util, json, os, re, signal, subprocess, sys, threading, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("llm_launch", os.path.join(HERE, "llm-launch.py"))
L = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(L)
B, D, R, N = L.B, L.D, L.R, L.N
TUNED = os.path.join(L.STATE_DIR, "tuned.json")
PORT = 8095
HEADROOM_MIB = 64
MOE_FIT_MAX = 131072   # the fit search's ceiling for a model with experts in RAM: it reads prompts at ~150 tokens/s, so a
                       # full-prompt check at 262k would take most of an hour (the tests ran at 88k)
CURRENT = [None]
IDLE_C = [None]        # the GPU's temperature when this run started (the cool-down aims near it)


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


# ------------------------------------------------------------------ heat: temperature, clocks, slowdowns (nvidia-smi)
HEAT_FIELDS = ["temperature.gpu", "clocks.sm", "clocks.max.sm", "power.draw", "utilization.gpu",
               "clocks_throttle_reasons.sw_thermal_slowdown", "clocks_throttle_reasons.hw_thermal_slowdown",
               "clocks_throttle_reasons.hw_power_brake_slowdown"]
BUSY_PCT = 50          # samples with the GPU at least this busy count for clocks (an idle GPU clocks down by design)
COOL_WAIT_S = 240      # longest wait for the GPU to cool before the cool-card measurement


def _num(x):
    try: return float(x)
    except (TypeError, ValueError): return None


def heat_now(idx, fields=None):
    """One reading: {temp, sm, max_sm, power, util, thermal, brake} (thermal / brake: the GPU is slowing itself for heat /
    because the charger cannot keep up; None when the driver does not report them)."""
    fields = fields or HEAT_FIELDS
    r = subprocess.run(["nvidia-smi", "-i", idx, "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=15)
    if r.returncode or not r.stdout.strip():
        if fields is HEAT_FIELDS: return heat_now(idx, HEAT_FIELDS[:5])      # an older driver: no slowdown reasons
        return None
    v = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
    act = lambda i: (v[i].lower() == "active") if len(v) > i and v[i] not in ("[N/A]", "N/A") else None
    sw, hw, brake = act(5), act(6), act(7)
    return {"t": time.time(), "temp": _num(v[0]), "sm": _num(v[1]), "max_sm": _num(v[2]), "power": _num(v[3]),
            "util": _num(v[4]), "thermal": (bool(sw or hw) if (sw is not None or hw is not None) else None), "brake": brake}


class HeatLog:
    """GPU temperature, clock and slowdown reasons every 2 s, for one model + setting (fit and speed)."""
    def __init__(self, idx):
        self.idx, self.samples, self._stop = idx, [], threading.Event()
    def __enter__(self):
        def run():
            while not self._stop.is_set():
                try:
                    h = heat_now(self.idx)
                    if h: self.samples.append(h)
                except Exception: pass
                self._stop.wait(2)
        self._t = threading.Thread(target=run, daemon=True); self._t.start(); return self
    def __exit__(self, *a):
        self._stop.set(); self._t.join()

    def stats(self, t0=None, t1=None):
        """Over [t0, t1]: peak and average temperature, the share of busy time the GPU slowed itself for heat (and for
        the charger), and its average clock while busy."""
        w = [s for s in self.samples if (t0 is None or s["t"] >= t0) and (t1 is None or s["t"] <= t1)]
        busy = [s for s in w if (s["util"] or 0) >= BUSY_PCT and s["sm"]]
        temps = [s["temp"] for s in w if s["temp"] is not None]
        share = lambda key: (sum(1 for s in busy if s[key]) / len(busy)) if busy and any(s[key] is not None for s in busy) else None
        pw = [s["power"] for s in busy if s["power"] is not None]
        return {"peak_c": max(temps) if temps else None, "avg_c": round(sum(temps) / len(temps)) if temps else None,
                "slowed": share("thermal"), "brake": share("brake"),
                "clock": round(sum(s["sm"] for s in busy) / len(busy)) if busy else None,
                "power_w": round(sum(pw) / len(pw), 1) if pw else None, "busy_n": len(busy)}

    def cool_clock(self):
        """The GPU's clock under load while it was NOT slowing itself for heat (90th percentile), None if it always was."""
        c = sorted(s["sm"] for s in self.samples if (s["util"] or 0) >= BUSY_PCT and s["sm"] and s["thermal"] is False)
        return round(c[int(len(c) * 0.9) - 1 if len(c) > 1 else 0]) if c else None

    def max_clock(self):
        return next((round(s["max_sm"]) for s in reversed(self.samples) if s["max_sm"]), None)


def cool_down(idx, target):
    """Wait (up to COOL_WAIT_S) for the GPU to cool to `target` °C; returns the temperature it reached."""
    h = heat_now(idx); t = h and h["temp"]
    if t is None or t <= target: return t
    print(f"   letting the GPU cool from {t:.0f} °C toward {target:.0f} °C first (for a cool-card number) ", end="", flush=True)
    end = time.time() + COOL_WAIT_S
    while time.time() < end:
        time.sleep(10); h = heat_now(idx); t = h and h["temp"]
        print(".", end="", flush=True)
        if t is None or t <= target: break
    print(f" {t:.0f} °C" if t is not None else "")
    return t


def ac_power():
    """True on a charger (the barrel plug or USB-C), False on battery, None if this machine does not say."""
    seen = None
    for d in glob.glob("/sys/class/power_supply/*"):
        try:
            if open(os.path.join(d, "type")).read().strip() in ("Mains", "USB"):
                seen = bool(seen) or open(os.path.join(d, "online")).read().strip() == "1"
        except OSError: pass
    return seen


def power_profile():
    """The power mode in effect (Pop's system76-power profile, else the ACPI platform profile), for the record."""
    try:
        out = subprocess.run(["system76-power", "profile"], capture_output=True, text=True, timeout=10).stdout
        m = re.search(r"Power Profile:\s*(\S+)", out)
        if m: return m.group(1)
    except Exception: pass
    try: return open("/sys/firmware/acpi/platform_profile").read().strip()
    except OSError: return None


def severity(writing_loss, slowed):
    """How much heat cost: from the writing speed lost against a cool card when there is one, else from how much of
    the time the GPU slowed itself."""
    if writing_loss is not None:
        if writing_loss < 0.05 and (slowed or 0) < 0.05: return "none"
        return "little" if writing_loss < 0.10 else "moderate" if writing_loss < 0.30 else "heavy"
    if slowed is None: return None
    return "none" if slowed < 0.05 else "little" if slowed < 0.25 else "moderate" if slowed < 0.60 else "heavy"


def heat_lines(heat, indent="   "):
    """What heat did to this measurement, in words."""
    if not heat: return []
    out = []
    sev = heat.get("severity")
    cool, hot = heat.get("cool") or {}, heat.get("hot") or {}
    head = f"{indent}{B}Heat: {sev or 'unknown'}{N}"
    if heat.get("writing_loss") is not None:
        head += (f". Writing at {L.kshort(hot.get('depth', 4096))}: {cool.get('writing')} t/s on a cool card ({cool.get('temp_c', 0):.0f} °C), "
                 f"{hot.get('writing')} t/s after a long run ({hot.get('temp_c', 0):.0f} °C): {heat['writing_loss']:.0%} slower")
        if heat.get("reading_loss") is not None: head += f"; reading {heat['reading_loss']:.0%} slower"
    out.append(head + ".")
    bits = []
    unreported = (heat.get("writing_loss") or 0) >= 0.10 and (heat.get("slowed") or 0) < 0.05
    if heat.get("peak_c") is not None: bits.append(f"the GPU peaked at {heat['peak_c']:.0f} °C")
    if heat.get("slowed") is not None and not unreported: bits.append(f"slowed itself for heat {heat['slowed']:.0%} of the speed test")
    if heat.get("clock"):
        c = f"its clock averaged {heat['clock']:,} MHz"
        if heat.get("clock_cool"): c += f" ({heat['clock_cool']:,} MHz when it was not slowed)"
        elif heat.get("clock_max"): c += f" (its top clock: {heat['clock_max']:,} MHz)"
        bits.append(c)
    if bits:
        line = "; ".join(bits)
        out.append(f"{indent}   " + line[0].upper() + line[1:] + ".")
    if unreported and cool.get("clock") and hot.get("clock"):
        pw = lambda x: f"{x['power_w']:.0f} W, " if x.get("power_w") is not None else ""
        out.append(f"{indent}   The driver reported no heat slowdown, but the GPU ran at {pw(cool)}{cool['clock']:,} MHz on the cool card and "
                   f"{pw(hot)}{hot['clock']:,} MHz after the long run.")
    if heat.get("brake"): out.append(f"{indent}   {R}The charger could not keep up {heat['brake']:.0%} of the time{N} (the GPU's \"power brake\").")
    return out


HEAT_TIPS = """{B}To lose less speed to heat{N} (then measure again: llm-tune --redo --one):
   - Raise the back of the laptop 2-3 cm (a stand, or a book under the back edge), on a hard, flat surface: not a bed,
     a couch or a lap. The fans pull air in from underneath.
   - Keep the vents clear. On an older laptop, blow the dust out of the vents and fans.
   - A cooling pad (a stand with fans) under the laptop, and a cooler room, both help.
   - Use the full-size charger the laptop came with: a weaker or USB-C charger makes the GPU hold back.
   - Power mode (the top-bar menu): measured in "{profile}". If that is Battery Life or Balanced, try Performance while
     plugged in, and compare. Whether it helps depends on the laptop."""


MAJFLT_TIGHT = 2000   # major page faults during the requests after the full prompt that mean RAM ran short: the weights
                      # kept in RAM are mapped from the model file, so when RAM runs out the kernel drops them and reads
                      # them back from disk (major faults) -- MemAvailable and swap stay quiet while that happens


def majflt(pid, stat=None):
    """The process's major page faults so far (/proc/<pid>/stat field 12), None if unknown."""
    try:
        with open(stat or f"/proc/{pid}/stat") as f: return int(f.read().rsplit(")", 1)[1].split()[9])
    except (OSError, ValueError, IndexError): return None


def swap_used():
    mi = L.meminfo()
    return mi.get("SwapTotal", 0) - mi.get("SwapFree", 0) if mi else None


class Sampler:
    """Lowest free GPU memory while a request runs, and the lowest free system RAM (MemAvailable) and the most swap in
    use (sampled every 0.25 s)."""
    def __init__(self, idx):
        self.idx, self.low, self.ram_low, self.swap_high, self._stop = idx, None, None, None, threading.Event()
    def __enter__(self):
        def run():
            while not self._stop.is_set():
                try:
                    f = mem(self.idx)[1]; self.low = f if self.low is None else min(self.low, f)
                except Exception: pass
                try:
                    mi = L.meminfo()
                    a = mi.get("MemAvailable")
                    if a is not None: self.ram_low = a if self.ram_low is None else min(self.ram_low, a)
                    if "SwapTotal" in mi:
                        sw = mi["SwapTotal"] - mi.get("SwapFree", 0)
                        self.swap_high = sw if self.swap_high is None else max(self.swap_high, sw)
                except Exception: pass
                time.sleep(0.25)
        self._t = threading.Thread(target=run, daemon=True); self._t.start(); return self
    def __exit__(self, *a):
        self._stop.set(); self._t.join()


# ------------------------------------------------------------------ the server
def start(server, model, ctx, s, log):
    p, argv = L.start_server(server, model, ctx, s["mtp"], s["draft_n"], s["proj"], s["proj_gpu"], PORT, log, s.get("cpu_layers"))
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
def text_tokens(n, offset=0):
    """n tokens of real source code (Python's own standard library), tokenized by the running server; `offset` starts
    further into it (a prompt that shares nothing with earlier ones, so none of it comes from the server's cache)."""
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
    t = _TEXT[offset % len(_TEXT):] + _TEXT
    while len(t) < n: t = t + _TEXT
    return t[:n]


def ask_at_depth(depth, n_predict, offset=0):
    """A chat request with about `depth` tokens of code before the question; returns llama-server's timings."""
    code = post("/detokenize", {"tokens": text_tokens(depth, offset)})["content"]
    body = {"messages": [{"role": "user", "content": code + "\n\nExplain what the code above does, part by part."}],
            "max_tokens": n_predict}
    r = post("/v1/chat/completions", body)
    return r.get("timings") or {}


# ------------------------------------------------------------------ fit
def fit(server, model, s, idx, native, logdir, info=None):
    """(largest context that held, suggested context, notes). info (a dict) gets the system RAM record: need_mib and
    avail_mib before the search (short_mib when it was not enough: then nothing is started), and while the context was
    full, min_avail_mib, swap_growth_mib and tight."""
    info = {} if info is None else info
    r = L.ram_check(model, s["mtp"], s.get("proj"), s.get("proj_gpu"))
    info.update(need_mib=r["need"], avail_mib=r["avail"])
    if r["short"]:
        info["short_mib"] = r["short"]
        return None, None, (f"not enough free system RAM: the model server needs about {L.gib(r['need'])}, plus about "
                            f"{L.gib(L.RAM_HEADROOM_MIB)} for opencode and the desktop; {L.gib(r['avail'])} was free. Close other "
                            f"programs (a web browser often holds 1-3 GB) and measure again")
    swap0 = swap_used()
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
                f0 = majflt(p.pid)                # the full prompt has touched every weight once: the rest should be in RAM
                for d in (2048, min(32768, ctx // 3)):
                    ask_at_depth(d, 64)
                f1 = majflt(p.pid)
            if f0 is not None and f1 is not None: info["major_faults"] = f1 - f0
            low = sm.low
            if sm.ram_low is not None: info["min_avail_mib"] = sm.ram_low
            if sm.swap_high is not None and swap0 is not None: info["swap_growth_mib"] = max(0, sm.swap_high - swap0)
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
    info["tight"] = bool((info.get("min_avail_mib") is not None and info["min_avail_mib"] < L.RAM_HEADROOM_MIB)
                         or (info.get("swap_growth_mib") or 0) > 256 or (info.get("major_faults") or 0) > MAJFLT_TIGHT)
    note = f"{low} MiB of the GPU stayed free at the fullest" if low is not None else ""
    if info.get("min_avail_mib") is not None: note += f"; {L.gib(info['min_avail_mib'])} of RAM stayed free"
    return ctx, sug, note


# ------------------------------------------------------------------ speed
def measure(d, offset=0):
    """Reading and writing speed with about d tokens before the question: one answer that reads the whole prompt, two
    more on the same prompt (only the writing is new); writing is the median of the three."""
    first = ask_at_depth(d, 256, offset)
    reps = [ask_at_depth(d, 256, offset) for _ in range(2)]
    w = sorted(x.get("predicted_per_second", 0) for x in [first] + reps)
    return {"reading": round(first.get("prompt_per_second", 0)), "writing": round(w[len(w) // 2], 1)}


def heat_note(hl, t0, t1):
    st = hl.stats(t0, t1)
    bits = [f"{st['peak_c']:.0f} °C" if st["peak_c"] is not None else None, f"{st['clock']:,} MHz" if st["clock"] else None,
            f"slowed for heat {st['slowed']:.0%}" if st["slowed"] else None]
    bits = [b for b in bits if b]
    return f"  {D}({', '.join(bits)}){N}" if bits else ""


def window(hl, idx, t0):
    """Since t0: the highest temperature (the reading now if no sample fell in the window), and the GPU's average
    clock and power while busy."""
    st = hl.stats(t0, time.time())
    p = st["peak_c"]
    if p is None:
        h = heat_now(idx); p = h and h["temp"]
    return {"temp_c": p, "clock": st["clock"], "power_w": st["power_w"]}


def speed(server, model, s, ctx, logdir, idx, hl):
    """(speed by depth, heat record). The card first cools, then: 4k (the cool-card number), 32k, near the maximum, and
    4k again at the end (on a fresh part of the text) -- the number after a long run, which is what the table keeps."""
    short = min(4096, ctx // 2)                       # the cool / hot depth (a small context gets a smaller one)
    depths = sorted({d for d in (32768, int(ctx * 0.9)) if d < ctx * 0.95 and d > short})
    p, ok = start(server, model, ctx, s, os.path.join(logdir, f"speed-{ctx}.log"))
    out = {}
    try:
        if not ok: return None, None
        text_tokens(1)
        ask_at_depth(min(1024, short), 8, offset=len(_TEXT) // 3)   # a first request pays one-time setup costs: not the cool one
        target = max(50, (IDLE_C[0] or 40) + 8)
        cool_down(idx, target)
        h = heat_now(idx); t0 = time.time()
        cool = measure(short); cool.update(window(hl, idx, t0))
        if h and h["temp"] is not None: cool["temp_c"] = h["temp"]      # the cool number: the temperature it started at
        print(f"   at {short:>7,} tokens: reading {cool['reading']:>5,} t/s, writing {cool['writing']:>5} t/s   cool card"
              + (f"  {D}(started at {cool['temp_c']:.0f} °C){N}" if cool.get("temp_c") is not None else ""), flush=True)
        t_speed = t0
        for d in depths:
            t0 = time.time(); out[d] = measure(d)
            print(f"   at {d:>7,} tokens: reading {out[d]['reading']:>5,} t/s, writing {out[d]['writing']:>5} t/s"
                  + heat_note(hl, t0, time.time()), flush=True)
        t0 = time.time(); out[short] = measure(short, offset=len(_TEXT) // 2)
        hot = dict(out[short], **window(hl, idx, t0), depth=short)
        print(f"   at {short:>7,} tokens: reading {hot['reading']:>5,} t/s, writing {hot['writing']:>5} t/s   after the long run"
              + heat_note(hl, t0, time.time()), flush=True)
        st = hl.stats(t_speed, time.time())
        wl = (1 - hot["writing"] / cool["writing"]) if cool["writing"] else None
        rl = (1 - hot["reading"] / cool["reading"]) if cool["reading"] else None
        heat = {"severity": severity(None if wl is None else max(0.0, wl), st["slowed"]),
                "writing_loss": None if wl is None else round(max(0.0, wl), 3),
                "reading_loss": None if rl is None else round(max(0.0, rl), 3),
                "cool": cool, "hot": hot, "peak_c": st["peak_c"], "avg_c": st["avg_c"],
                "slowed": None if st["slowed"] is None else round(st["slowed"], 3),
                "brake": None if st["brake"] is None else round(st["brake"], 3),
                "clock": st["clock"], "clock_cool": hl.cool_clock(), "clock_max": hl.max_clock(),
                "profile": power_profile(), "ac": ac_power()}
        return {k: out[k] for k in sorted(out)}, heat
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


def needs(rec, what, redo=False):
    """What a setting still needs for this run: (fit, speed). A fit recorded as short of system RAM counts as not
    measured (close programs, run again); a fit-only record still needs its speed."""
    has_fit = bool(rec) and not (rec.get("ram") or {}).get("short_mib")
    has_speed = has_fit and bool(rec.get("speed"))
    held = has_fit and bool(rec.get("suggested"))
    f = what in ("fit", "both") and (redo or not has_fit)
    sp = what in ("speed", "both") and (redo or not has_speed) and (held or not has_fit or f)
    return f, sp


def minutes(model, s, what):
    """Rough minutes (low, high) for one setting: most of it is reading long prompts."""
    moe = bool(L.cpu_layers_for(model["name"], s["mtp"]))
    f, sp = what
    return ((15 if moe else 5) * f + (15 if moe else 10) * sp, (45 if moe else 15) * f + (30 if moe else 15) * sp)


def tune_one(server, model, s, setting_key, idx, gname, what="both", redo=False):
    """The fit test, the speed test, or both, for one model + setting; saved after each (a model that does not fit is
    recorded as such). The speed test runs at the context the fit test suggested; with no fit yet, the fit runs first."""
    meta = L.gguf_meta(model["path"])
    native = min(int(meta.get("context_length") or 262144), 262144)
    if L.cpu_layers_for(model["name"], s["mtp"]): native = min(native, MOE_FIT_MAX)
    logdir = os.path.join(L.LOG_DIR, "tune-" + time.strftime("%Y%m%d-%H%M%S")); os.makedirs(logdir, exist_ok=True)
    label = L.setting_label(setting_key)
    print(f"\n{B}{model['name']}{N}  ({label})")
    t0 = time.time()
    with HeatLog(idx) as hl:
        return _tune_one(server, model, s, setting_key, idx, gname, native, logdir, t0, hl, what, redo)


def _tune_one(server, model, s, setting_key, idx, gname, native, logdir, t0, hl, what="both", redo=False):
    key = f"{model['name']}|{setting_key}"
    old = load_tuned().get(key)
    do_fit, do_speed = needs(old, what, redo)
    if not do_fit and not do_speed:
        print(f"   nothing to measure: {(old or {}).get('note') or 'it did not fit'} (the fit test measures it again: llm-tune --fit --one)")
        return old
    if do_speed and not do_fit and not (old and old.get("suggested") and not (old.get("ram") or {}).get("short_mib")):
        print("   no fit measured yet for this setting; the speed test needs one, so the fit test runs first")
        do_fit = True
    if do_fit:
        print(f"{B}Fit{N}: the largest context, searching 4,096 - {native:,}")
        info = {}
        held, sug, note = fit(server, model, s, idx, native, logdir, info)
        rec = {"max_ctx": held or 0, "suggested": sug or 0, "speed": None, "gpu": gname, "measured": time.strftime("%Y-%m-%d %H:%M"),
               "ram": info}
        if old and old.get("speed") and held and old.get("suggested") == sug:      # same context: its speed still holds
            rec.update(speed=old["speed"], **({"heat": old["heat"]} if old.get("heat") else {}))
        if not held:
            print(f"{R}   {note}{N}  (logs: {logdir})")
            rec.update(note=note, minutes=round((time.time() - t0) / 60, 1))
            save(key, rec); return rec
        print(f"   {B}held: {held:,} tokens; suggested: {sug:,}{N}  {D}{note}{N}")
        if info.get("tight"):
            print(f"   {R}System RAM ran short while the context was full{N}"
                  + (f" ({L.gib(info['min_avail_mib'])} free at the lowest" if info.get("min_avail_mib") is not None else " (")
                  + (f", {info['swap_growth_mib']:,} MiB went to swap" if info.get("swap_growth_mib") else "")
                  + (f", {info['major_faults']:,} reads of the model back from disk)" if (info.get("major_faults") or 0) > MAJFLT_TIGHT else ")")
                  + ": with opencode running too it may slow down a lot. Close other programs while using this model.")
        rec["fit_minutes"] = round((time.time() - t0) / 60, 1)
        save(key, rec)                                     # the fit is kept even if the speed test fails
        if not do_speed:
            rec["minutes"] = rec["fit_minutes"]; save(key, rec)
            print(f"{B}Result{N} ({rec['minutes']} min): held {held:,}, suggested {sug:,} (llm-launch suggests this from now on)"
                  + ("" if rec.get("speed") else f"; speed not measured: llm-tune --speed"))
            return rec
    else:
        rec = dict(old)
    sug = rec["suggested"]
    print(f"{B}Speed{N} at {sug:,} tokens of context" + ("" if do_fit else f" (the fit test's suggestion, {rec.get('measured', '')})") + ":")
    t1 = time.time()
    try:
        rec["speed"], heat = speed(server, model, s, sug, logdir, idx, hl)
    except Exception as e:
        stop(); heat = None
        print(f"{R}   the speed test stopped with an error: {e}{N}  (logs: {logdir})")
    if heat: rec["heat"] = heat
    rec["speed_measured"] = time.strftime("%Y-%m-%d %H:%M")
    rec["speed_minutes"] = round((time.time() - t1) / 60, 1)
    rec["minutes"] = round((time.time() - t0) / 60, 1)
    save(key, rec)
    print(f"{B}Result{N} ({rec['minutes']} min): held {rec['max_ctx']:,}, suggested {sug:,} (llm-launch suggests this from now on)")
    if rec["speed"]: print("\n".join(L.speed_lines(rec, indent="   ", heat_summary=False)))
    for line in heat_lines(heat): print(line)
    return rec


def settings_for(model):
    """Every setting worth measuring, without questions: MTP off, plus MTP on when the model has a head (in its file, or
    a separate head file next to it); images on the CPU (the projector loaded there when it is next to the model); a
    mixture-of-experts file in its tested layout."""
    meta = L.gguf_meta(model["path"])
    proj = L.projector_for(model)
    draft_n = next((n for k, n in L.DRAFT_N.items() if model["name"].startswith(k)), 2)
    out = []
    for mtp in ([False, True] if meta.get("nextn_predict_layers") or L.separate_mtp_head(model) else [False]):
        out.append(({"mtp": mtp, "draft_n": draft_n, "proj": proj, "proj_gpu": False},
                    L.setting_key(model["name"], mtp, False, proj)))
    return out


def run_todo(server, todo, idx, gname, what, redo):
    """Measure each (model, setting, key) in turn. A setting with MTP on that does not fit (in the GPU, or in system
    RAM) falls back to MTP off: that setting is measured right after it, unless it is already measured or queued.
    Returns the heat records that came out moderate or heavy."""
    queue, done, hot = list(todo), set(), []
    i = 0
    while i < len(queue):
        m, s, k = queue[i]; i += 1
        if len(queue) > 1: print(f"\n{D}[{i} of {len(queue)}]{N}", end="")
        try:
            rec = tune_one(server, m, s, k, idx, gname, what, redo)
        except Exception as e:
            stop(); print(f"{R}   {m['name']} ({L.setting_label(k)}) stopped with an error: {e}{N}"); continue
        done.add((m["name"], k))
        if ((rec or {}).get("heat") or {}).get("severity") in ("moderate", "heavy"): hot.append(rec["heat"])
        if s.get("mtp") and rec is not None and not rec.get("max_ctx"):
            k_off = L.setting_key(m["name"], False, s.get("proj_gpu"), s.get("proj"))
            off = load_tuned().get(f"{m['name']}|{k_off}")
            later = any(x[0]["name"] == m["name"] and x[2] == k_off for x in queue[i:])
            if (m["name"], k_off) in done or later: continue
            if needs(off, "fit")[0]:
                print(f"   {B}MTP on does not fit here, so the fit test tries MTP off next{N} (the launcher then suggests MTP off)")
                queue.insert(i, (m, dict(s, mtp=False), k_off))
            elif off and off.get("suggested"):
                print(f"   MTP on does not fit here; MTP off was measured before ({off['suggested']:,} tokens): the launcher suggests it")
    return hot


WHAT = {"f": "fit", "s": "speed", "b": "both"}


def ask_what():
    """--fit / --speed on the command line, else asked."""
    if "--fit" in sys.argv and "--speed" not in sys.argv: return "fit"
    if "--speed" in sys.argv and "--fit" not in sys.argv: return "speed"
    if "--fit" in sys.argv: return "both"
    print(f"\n{B}What to measure{N}:")
    print("   f) the fit test: does it load, and the largest context that fits (in the GPU, with enough system RAM)")
    print("   s) the speed test: reading and writing speed at that context, and what heat costs (needs a fit; runs it first if there is none)")
    print("   b) both")
    while True:
        v = L.ask("Which", "b").lower()[:1]
        if v in WHAT: return WHAT[v]


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
    h0 = heat_now(idx); IDLE_C[0] = h0 and h0["temp"]
    print(f"{B}GPU{N}: {gname}, {total:,} MiB, {used:,} MiB in use now" + (f", {IDLE_C[0]:.0f} °C" if IDLE_C[0] is not None else ""))
    mi = L.meminfo()
    if mi.get("MemAvailable") is not None:
        print(f"{B}System RAM{N}: {L.gib(mi['MemTotal'])}, {L.gib(mi['MemAvailable'])} free now")
    if ac_power() is False:
        print(f"{R}   The laptop is on battery.{N} On battery the GPU is held back hard, so the speeds would not be its real ones.")
        if not L.yes("Measure anyway?", default=False): return 1
    if used > 300:
        print(f"{R}   {used:,} MiB is already in use.{N} The result is only right for a GPU in the same state:")
        for line in L.gpu_busy_advice(): print(line)
        if not L.yes("Measure anyway?", default=False): return 1
    if not L.port_free(PORT):
        sys.exit(f"{R}port {PORT} is in use (another llm-tune?){N}")
    one = "--one" in sys.argv
    what = ask_what()
    if not one:
        models = L.scan_models(st)
        exl = [m for m in models if m.get("engine") == "exl3"]
        if exl: print(f"   {D}(not measured here: {', '.join(m['name'] for m in exl)} -- EXL3 models; llm-tune measures llama.cpp models only){N}")
        models = [m for m in models if m.get("engine") != "exl3"]
        tuned = load_tuned()
        todo, have = [], []
        for m in models:
            for s, k in settings_for(m):
                n = needs(tuned.get(f"{m['name']}|{k}"), what, redo)
                (todo if any(n) else have).append((m, s, k, n))
        label = {"fit": "the fit test", "speed": "the speed test", "both": "the fit and speed tests"}[what]
        print(f"\n{B}Every model found, {label}{N}: {len(todo)} to measure"
              + (f", {len(have)} already measured (llm-tune --redo measures them again)" if have else ""))
        short = []
        for m, s, k, n in todo:
            r = L.ram_check(m, s["mtp"], s.get("proj"), s.get("proj_gpu")) if L.cpu_layers_for(m["name"], s["mtp"]) else None
            steps = " + ".join(x for x, on in (("fit", n[0]), ("speed", n[1])) if on)
            print(f"   {m['name']}  ({L.setting_label(k)}): {steps}"
                  + (f"  {R}needs about {L.gib(r['need'] + L.RAM_HEADROOM_MIB)} of free RAM, {L.gib(r['avail'])} free now{N}" if r and r["short"] else ""))
            if r and r["short"]: short.append(m["name"])
        if not todo:
            print("   nothing left to measure"); print("\n".join(L.results_table(tuned))); return 0
        lo = sum(minutes(m, s, n)[0] for m, s, k, n in todo); hi = sum(minutes(m, s, n)[1] for m, s, k, n in todo)
        print(f"   about {lo}-{hi} minutes in all. Leave the laptop plugged in; Ctrl-C stops it (finished ones stay saved).")
        if short:
            print(f"   {R}Not enough free RAM now for {len(short)} of them:{N} those are recorded as such and skipped. Close other programs "
                  f"(a web browser often holds 1-3 GB) and run llm-tune again to measure them.")
        a = L.ask("Measure all of these (a), or pick one model (p)?", "a").lower()
        one = a.startswith("p")
        todo = [(m, s, k) for m, s, k, n in todo]
    if one:
        model = L.pick_model(st)
        if model.get("engine") == "exl3":
            print(f"{R}llm-tune measures llama.cpp models only{N}; an EXL3 model's context is set in the launcher."); return 1
        c = L.choose(st, model, ask_ctx=False)          # the MTP / images questions
        todo = [(model, {"mtp": c["mtp"], "draft_n": c["draft_n"], "proj": c["proj"], "proj_gpu": c["proj_gpu"]}, c["key"])]
        redo = True                                     # one model picked by hand: measure it (again)
    t0 = time.time()
    hot = run_todo(server, todo, idx, gname, what, redo)
    print(f"\n{B}All measurements on this machine{N}  ({round((time.time() - t0) / 60)} min this run; also: llm-launch --results, or r at its model menu)")
    print("\n".join(L.results_table()))
    print(f"{D}Saved in {TUNED}{N}")
    if hot:
        print(); print(HEAT_TIPS.format(B=B, N=N, profile=hot[-1].get("profile") or "unknown"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
