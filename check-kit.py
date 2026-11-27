#!/usr/bin/env python3
"""Check kit.json before pushing a change to it (for whoever maintains this kit; the launcher does not need it).

    python3 check-kit.py            the setups, the plugins and files they name, the prompt texts (rules below)
    python3 check-kit.py --fetch    also download each url plugin and compare its sha256

Rules:
  - kit.json has setup, setup_version, opencode and setups; exactly one setup is the default; ids are unique; every
    non-default setup has an 'ask' text (the launcher's yes / no question).
  - Every plugin has a 'file' in this repo (with a 'sha256' it must match) or a 'url' with a 'sha256'; every {kit}/...
    path in plugin options exists.
  - A prompt text (any .md / .txt file an option names) never mentions how it was tested ("codebench", "benchmark"),
    since the model reads it; nor does any option value. system-prompt.js uses its file exactly as it is, so the text has
    no leading or trailing whitespace (the tested text is stripped), and matches its 'prompt_sha256'.
  - Each setup is a combination the test harness runs (it refuses the others):
      * at most one plugin replaces opencode's own prompt (compaction-anchor's "slim" part, system-prompt.js);
      * at most one replaces opencode's turn after a summary (system-prompt.js with continueText, compaction-anchor's
        "no-stop-and-ask", a task driver: it handles that turn itself);
      * a task driver runs alone or with system-prompt.js only: no compaction-anchor parts, focus mode;
      * each plugin appears once;
      * every setup has compaction-anchor with its reinsert part, or a task driver: without either, a long session
        can lose the request to a summary (Kyle 11-26).
  - One plugin the harness never ran is allowed beside a tested set: interaction-mode.js, whose autonomous mode changes
    nothing the model sees (plan-first and interactive, untested, replace the system prompt's "# Questions" section).
    Its kit.json entry says so with "untested": true; its options.default is "autonomous" (the launcher passes the
    person's choice at each start); it appears at most once per setup, after system-prompt.js, whose prompt text must
    have a "# Questions" section; and it counts as neither the reinsert part nor a task driver. Its file holds text the
    model reads, so the whole file is checked for the words above. Any other plugin is refused, "untested" or not.
A changed prompt text needs a new file name: the setup name then tells the sessions apart.
"""
import hashlib, json, os, re, sys, urllib.request

KIT = os.path.dirname(os.path.abspath(__file__))
BANNED = re.compile(r"codebench|benchmark", re.I)
ANCHOR_PARTS = ("checklist", "reinsert", "pointer", "no-stop-and-ask", "slim")


def read(path, binary=False):
    with open(path, "rb") if binary else open(path, encoding="utf-8") as f: return f.read()


def kind(p):
    """compaction-anchor / system-prompt / task-driver / interaction-mode / other, from the plugin's file or url."""
    src = os.path.basename(p.get("file") or p.get("url") or "")
    if "compaction-anchor" in src or "session-history" in src: return "compaction-anchor"
    if "system-prompt" in src: return "system-prompt"
    if "task-driver" in src: return "task-driver"
    if src.startswith("interaction-mode"): return "interaction-mode"
    return "other"


def check_plugin(p, fetch, bad, sha_cache):
    name = p.get("name", "?")
    if p.get("file"):
        path = os.path.join(KIT, p["file"])
        if not os.path.isfile(path): bad.append(f"{name}: no file {p['file']}")
        elif not p.get("sha256"): bad.append(f"{name}: a file plugin needs its sha256 (the tested build)")
        elif hashlib.sha256(read(path, True)).hexdigest() != p["sha256"]:
            bad.append(f"{name}: {p['file']} does not match its sha256")
        if os.path.isfile(path) and kind(p) == "interaction-mode" and BANNED.search(read(path)):
            bad.append(f"{name}: {p['file']} mentions the benchmark (the model reads the text it adds)")
    elif p.get("url") and p.get("sha256"):
        if fetch and p["url"] not in sha_cache:
            body = urllib.request.urlopen(p["url"], timeout=60).read()
            sha_cache[p["url"]] = (hashlib.sha256(body).hexdigest(), bool(BANNED.search(body.decode("utf-8", "replace"))))
        if fetch and sha_cache[p["url"]][0] != p["sha256"]: bad.append(f"{name}: the download does not match its sha256")
        if fetch and kind(p) == "interaction-mode" and sha_cache[p["url"]][1]:
            bad.append(f"{name}: the download mentions the benchmark (the model reads the text it adds)")
    else:
        bad.append(f"{name}: needs 'file' (+ 'sha256'), or 'url' and 'sha256'")
    for v in re.findall(r'"(\{kit\}/[^"]+)"', json.dumps(p.get("options") or {})):
        path = v.replace("{kit}", KIT)
        if not os.path.exists(path): bad.append(f"{name}: option path {v} does not exist")
        elif path.endswith((".md", ".txt")):
            text = read(path)
            if BANNED.search(text): bad.append(f"{name}: {v} mentions the benchmark (the model reads this text)")
            if kind(p) == "system-prompt" and v == (p.get("options") or {}).get("file"):
                if text != text.lstrip("﻿").strip():
                    bad.append(f"{name}: {v} has leading or trailing whitespace (system-prompt.js sends it as it is; the tested text is stripped)")
                if p.get("prompt_sha256") and hashlib.sha256(text.encode("utf-8")).hexdigest() != p["prompt_sha256"]:
                    bad.append(f"{name}: {v} does not match its prompt_sha256 (a changed text needs a new file name)")
    for v in re.findall(r'"([^"]*)"', json.dumps(p.get("options") or {})):
        if BANNED.search(v): bad.append(f"{name}: an option mentions the benchmark: {v[:60]}")


def _prompt_text(p):
    """The text of a system-prompt.js plugin's prompt file, or None when it cannot be read."""
    f = str((p.get("options") or {}).get("file") or "").replace("{kit}", KIT)
    try: return read(f) if f else None
    except OSError: return None


def check_setup(s, bad):
    """The combination rules (see the module doc) for one setup."""
    sid = s.get("id", "?")
    ps = s.get("plugins") or []
    kinds = [kind(p) for p in ps]
    for k in set(kinds):
        if kinds.count(k) > 1: bad.append(f"setup {sid}: {k} appears {kinds.count(k)} times")
    anchor = [p for p in ps if kind(p) == "compaction-anchor"]
    parts = set(((anchor[0].get("options") or {}).get("parts") or []) if anchor else [])
    if anchor and not (anchor[0].get("options") or {}).get("parts"):
        bad.append(f"setup {sid}: compaction-anchor without a parts list (its defaults include parts no test used)")
    for x in sorted(parts - set(ANCHOR_PARTS)): bad.append(f"setup {sid}: compaction-anchor has no part {x!r}")
    sp = [p for p in ps if kind(p) == "system-prompt"]
    td = [p for p in ps if kind(p) == "task-driver"]
    cont = bool(sp and (sp[0].get("options") or {}).get("continueText"))
    if sp and not (sp[0].get("options") or {}).get("file"): bad.append(f"setup {sid}: system-prompt.js needs its file option")
    if sp and cont and not str(sp[0]["options"]["continueText"]).strip(): bad.append(f"setup {sid}: an empty continueText")
    if "slim" in parts and sp: bad.append(f"setup {sid}: the slim part and system-prompt.js both replace opencode's prompt: keep one")
    turn = [x for x, on in (("system-prompt.js continueText", cont), ("compaction-anchor no-stop-and-ask", "no-stop-and-ask" in parts),
                            ("a task driver", bool(td))) if on]
    if len(turn) > 1: bad.append(f"setup {sid}: {' and '.join(turn)} would both replace the turn after a summary: keep one")
    if td and anchor: bad.append(f"setup {sid}: a task driver runs without the compaction-anchor parts (it handles compaction itself)")
    if td and (td[0].get("options") or {}).get("mode", "focus") != "focus": bad.append(f"setup {sid}: the task driver runs in focus mode only")
    if not td and "reinsert" not in parts:
        bad.append(f"setup {sid}: needs compaction-anchor with reinsert, or a task driver (without either, a long session can lose the request to a summary)")
    other = [p.get("name", "?") for p in ps if kind(p) == "other"]
    if other: bad.append(f"setup {sid}: plugins no tested setup used: {', '.join(other)}")
    for p in ps:
        if kind(p) != "interaction-mode": continue
        if p.get("untested") is not True:
            bad.append(f"setup {sid}: {p.get('name', '?')} was never tested: its entry needs \"untested\": true")
        if (p.get("options") or {}).get("default") != "autonomous":
            bad.append(f"setup {sid}: {p.get('name', '?')} needs options.default \"autonomous\" (the tested mode; the launcher passes the person's choice)")
        if sp and ps.index(p) < ps.index(sp[0]):
            bad.append(f"setup {sid}: {p.get('name', '?')} must come after system-prompt.js (it edits that prompt's Questions section)")
        text = _prompt_text(sp[0]) if sp else None
        if text is not None and not re.search(r"(?m)^# Questions$", text):
            bad.append(f"setup {sid}: the system prompt has no '# Questions' section for {p.get('name', '?')} to replace")


def check(kit, fetch=False):
    """The list of problems with a kit.json object ([] = ok)."""
    bad = []
    for key in ("setup", "setup_version", "opencode", "setups"):
        if key not in kit: bad.append(f"kit.json has no '{key}'")
    if "plugins" in kit: bad.append("kit.json has a top-level 'plugins' list: plugins go inside a setup")
    sets = kit.get("setups") or []
    defaults = [s for s in sets if s.get("default")]
    if len(defaults) != 1: bad.append(f"kit.json needs exactly one default setup (has {len(defaults)})")
    elif kit.get("setup") != defaults[0].get("name"): bad.append("kit.json 'setup' is not the default setup's name")
    ids = [s.get("id") for s in sets]
    if len(set(ids)) != len(ids) or not all(ids): bad.append("every setup needs its own id")
    sha_cache = {}
    for s in sets:
        if not s.get("default") and not s.get("ask"): bad.append(f"setup {s.get('id')}: needs an 'ask' text (the launcher's question)")
        if not s.get("name"): bad.append(f"setup {s.get('id')}: needs a name")
        for p in s.get("plugins") or []: check_plugin(p, fetch, bad, sha_cache)
        check_setup(s, bad)
    return bad


def main():
    kit = json.loads(read(os.path.join(KIT, "kit.json")))
    bad = check(kit, "--fetch" in sys.argv)
    for line in bad: print("FAIL", line)
    if not bad: print("ok: " + "; ".join(f"{s['id']}: {s['name']} ({len(s.get('plugins', []))} plugins)" for s in kit["setups"]))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
