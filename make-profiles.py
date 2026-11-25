#!/usr/bin/env python3
"""Write LL-Dash's config and launch profiles for the models on this machine (run by install-lldash.sh; safe to re-run
after downloading another model).

  make-profiles.py [--lab ~/llm-lab]

  - <lab>/dashboard/dashboard.config.json, when there is none: the PrismML llama-server at ~/llama-prism, ~/models,
    the dashboard on 127.0.0.1:3000 only (its API can start programs; nothing else on the network should reach it).
  - <lab>/logs/profiles.json: one profile per tested model file found (Hugging Face downloads and ~/models), with MTP
    on and off where the file has an MTP head, each at the context tested to fit in 8 GB, q4_0 K/V, the model's own
    recommended sampling, the image projector (when it is next to the model) on the CPU. Profiles you made yourself
    are kept; ones this script made before are updated.
The numbers and settings come from llm-launch.py (the same folder), so the launcher and the dashboard agree."""
import importlib.util, json, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("llm_launch", os.path.join(HERE, "llm-launch.py"))
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
TAG = "llm-launch preset"          # marks the profiles this script owns (config.note)
FLAG_FIELDS = {"--temp": "temp", "--top-k": "topK", "--top-p": "topP", "--min-p": "minP",
               "--presence-penalty": "presencePenalty", "--repeat-penalty": "repeatPenalty"}


def sampling_fields(name):
    flags = next((v.split() for k, v in L.SAMPLING.items() if name.startswith(k)), [])
    fields, rest = {}, []
    i = 0
    while i < len(flags):
        f = flags[i]; v = flags[i + 1] if i + 1 < len(flags) else None
        if f in FLAG_FIELDS: fields[FLAG_FIELDS[f]] = float(v)
        else: rest += [f, v]
        i += 2
    return fields, rest


# short, typeable profile names: <model>[-mtp]-<context in thousands>k, e.g. bonsai2.5-27b-88k (lldash-cli start takes a prefix)
SLUG = {"Ternary-Bonsai-2.5-27B-PTQ1_0.gguf": "bonsai2.5-27b", "Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf": "bonsai2.5-27b",
        "Ornith-1.5-9B-Q4_K_M.gguf": "ornith1.5-9b", "Ternary-Bonsai-2-9B-PTQ1_0.gguf": "bonsai2-9b",
        "Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf": "bonsai2-9b", "Ternary-Bonsai-2-27B-PTQ1_0.gguf": "bonsai2-27b",
        "Ternary-Bonsai-2-9B-PQ2_0.gguf": "bonsai2-9b", "Ternary-Bonsai-2-9B-Q2_0.gguf": "bonsai2-9b"}


def nice(name):
    return SLUG.get(name) or re.sub(r"[^a-z0-9.]+", "-", re.sub(r"\.gguf$", "", name.lower())).strip("-")


def profiles_for(model):
    rec = L.RECOMMENDED.get(model["name"])
    if not rec: return {}
    meta = L.gguf_meta(model["path"])
    has_mtp = bool(meta.get("nextn_predict_layers"))
    proj = L.projector_for(model)
    fields, rest = sampling_fields(model["name"])
    out = {}
    for mtp in ([True, False] if has_mtp else [False]):
        ctx = rec.get("mtp_on" if mtp else "mtp_off")
        if not ctx: continue
        cfg = {"modelPath": model["path"], "build": "prism", "ctx": ctx, "ngl": 999, "port": 8080, "devices": [],
               "fa": True, "cacheK": L.KV, "cacheV": L.KV, "jinja": True, "verbosity": 3,
               "argString": " ".join(["-np", "1", "-fit", "off"] + rest), "note": TAG, **fields}
        if mtp:
            n = next((v for k, v in L.DRAFT_N.items() if model["name"].startswith(k)), 2)
            cfg.update(specType="draft-mtp", specDraftNMax=n, specDraftNMin=0)
            cfg["argString"] += f" --spec-draft-p-min 0 -ctkd {L.KV} -ctvd {L.KV}"
        if proj:
            cfg.update(mmprojPath=proj, mmprojOffload=False)      # images on the CPU (turn Images on in the launcher)
        label = f"{nice(model['name'])}{'-mtp' if mtp else ''}-{ctx // 1000}k"
        out[label] = cfg
    return out


def main():
    lab = os.path.expanduser(sys.argv[sys.argv.index("--lab") + 1] if "--lab" in sys.argv else "~/llm-lab")
    dash, logs = os.path.join(lab, "dashboard"), os.path.join(lab, "logs")
    os.makedirs(logs, exist_ok=True)
    cfg_file = os.path.join(dash, "dashboard.config.json")
    if not os.path.exists(cfg_file):
        server = L.find_server(L.load_state())
        cfg = {"listenHost": "127.0.0.1", "port": 3000, "modelsRoot": "~/models",
               "llamaServerBuilds": [{"id": "prism", "label": "PrismML llama.cpp", "path": server}]}
        with open(cfg_file, "w") as f: json.dump(cfg, f, indent=2)
        print(f"wrote {cfg_file} (llama-server: {server})")
    pf = os.path.join(logs, "profiles.json")
    try: doc = json.load(open(pf))
    except (OSError, ValueError): doc = {"version": 3, "profiles": {}}
    doc.setdefault("version", 3); profs = doc.setdefault("profiles", {})
    for k in [k for k, v in profs.items() if (v.get("config") or {}).get("note") == TAG]: del profs[k]
    now = int(time.time() * 1000); made = []
    chosen = {}
    for m in L.scan_models(L.load_state()):
        for label, c in profiles_for(m).items():
            # one profile per model and MTP choice: a file with an MTP head run with MTP off is the same as the plain file
            # (and a tested file wins over an inferred one: PTQ1_0 over PQ2_0 / Q2_0 for Bonsai 2 9B)
            if label in chosen:
                prev = os.path.basename(chosen[label]["modelPath"]); cur = os.path.basename(c["modelPath"])
                rank = lambda n: (bool(L.RECOMMENDED.get(n, {}).get("inferred")), "mtp" in n.lower())
                if rank(prev) <= rank(cur): continue
            chosen[label] = c
    for label, c in sorted(chosen.items()):
        profs[label] = {"config": c, "savedAt": now, "lastLoadedAt": None}; made.append((label, c["modelPath"]))
    tmp = pf + ".tmp"
    with open(tmp, "w") as f: json.dump(doc, f, indent=2)
    os.replace(tmp, pf)
    if made:
        print(f"{len(made)} launch profile(s) in {pf}:")
        for label, p in made: print(f"   {label}")
    else:
        print("No tested model files found yet. Download one (python3 llm-launch.py, then d), then run this again.")


if __name__ == "__main__":
    main()
