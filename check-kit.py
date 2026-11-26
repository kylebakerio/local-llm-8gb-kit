#!/usr/bin/env python3
"""Check kit.json before pushing a change to it (for whoever maintains this kit; the launcher does not need it).

    python3 check-kit.py            the plugins and files it names exist, prompt texts pass the rules below
    python3 check-kit.py --fetch    also download each url plugin and compare its sha256

Rules: every plugin has a 'file' in this repo or a 'url' with a 'sha256'; every {kit}/... path in plugin options exists;
a prompt text (any .md / .txt file an option names) never mentions how it was tested ("codebench", "benchmark"),
since the model reads it. A changed prompt text needs a new file name: the setup name then tells the sessions apart.
"""
import hashlib, json, os, re, sys, urllib.request

KIT = os.path.dirname(os.path.abspath(__file__))
BANNED = re.compile(r"codebench|benchmark", re.I)


def main():
    kit = json.load(open(os.path.join(KIT, "kit.json")))
    bad = []
    for key in ("setup", "setup_version", "opencode", "plugins"):
        if key not in kit: bad.append(f"kit.json has no '{key}'")
    for p in kit.get("plugins", []):
        name = p.get("name", "?")
        if p.get("file"):
            path = os.path.join(KIT, p["file"])
            if not os.path.isfile(path): bad.append(f"{name}: no file {p['file']}")
            elif p.get("sha256") and hashlib.sha256(open(path, "rb").read()).hexdigest() != p["sha256"]:
                bad.append(f"{name}: {p['file']} does not match its sha256")
        elif p.get("url") and p.get("sha256"):
            if "--fetch" in sys.argv:
                data = urllib.request.urlopen(p["url"], timeout=60).read()
                if hashlib.sha256(data).hexdigest() != p["sha256"]: bad.append(f"{name}: the download does not match its sha256")
        else:
            bad.append(f"{name}: needs 'file', or 'url' and 'sha256'")
        for v in re.findall(r'"(\{kit\}/[^"]+)"', json.dumps(p.get("options") or {})):
            path = v.replace("{kit}", KIT)
            if not os.path.exists(path): bad.append(f"{name}: option path {v} does not exist")
            elif path.endswith((".md", ".txt")) and BANNED.search(open(path, encoding="utf-8").read()):
                bad.append(f"{name}: {v} mentions the benchmark (the model reads this text)")
    for v in re.findall(r'"([^"]*)"', json.dumps([p.get("options") for p in kit.get("plugins", [])])):
        if BANNED.search(v): bad.append(f"a plugin option mentions the benchmark: {v[:60]}")
    # opencode's own prompt can be replaced once: the slim part and a system-prompt plugin together were never tested
    # (the test harness refuses that pair)
    slim = [p for p in kit.get("plugins", []) if "slim" in ((p.get("options") or {}).get("parts") or [])]
    prompt = [p for p in kit.get("plugins", []) if "system-prompt" in (p.get("file") or p.get("url") or "")]
    if slim and prompt: bad.append("the slim part and a system-prompt plugin both replace opencode's prompt: keep one")
    for line in bad: print("FAIL", line)
    if not bad: print(f"ok: {kit.get('setup')} ({len(kit.get('plugins', []))} plugins)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
