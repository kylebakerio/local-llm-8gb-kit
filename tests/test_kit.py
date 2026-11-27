"""Unit tests for the launcher and llm-tune (no GPU, no network, no model server): python3 -m unittest discover -s tests

They use tiny synthetic GGUF files (a real header, sparse data) and stand-ins for nvidia-smi, /proc/meminfo and
llama-server, so they run anywhere. The interaction-mode plugin's tests run it under node (skipped when node is missing)."""
import importlib.util, json, os, shutil, struct, subprocess, sys, tempfile, unittest
from unittest import mock

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(KIT, file))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


T = load("llm_tune", "llm-tune.py")
L = T.L            # llm-tune's own copy of the launcher: patch this one so both see it
V12_SHA = "3583abaede2d9155461515d5f826c4bb66912433fc1ff173edd72c81abb35885"   # the v1.2 prompt text the tests ran
MIB = 2**20


def write_gguf(path, tensors, kv=None, data_size=None):
    """A GGUF v3 file: the given header keys and tensors (name -> bytes), tensor data as a sparse hole."""
    kv = dict(kv or {})
    def s(x): b = x.encode(); return struct.pack("<Q", len(b)) + b
    out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv)))
    for k, v in kv.items():
        out += s(k)
        if isinstance(v, str): out += struct.pack("<I", 8) + s(v)
        else: out += struct.pack("<I", 4) + struct.pack("<I", v)
    off = 0
    for name, size in tensors.items():
        out += s(name) + struct.pack("<I", 1) + struct.pack("<Q", 1) + struct.pack("<I", 0) + struct.pack("<Q", off)
        off += size
    pad = (32 - len(out) % 32) % 32
    out += b"\0" * pad
    with open(path, "wb") as f:
        f.write(out); f.truncate(len(out) + (data_size if data_size is not None else off))
    return path


class MoEModel:
    """A synthetic 4-layer mixture-of-experts file registered in the launcher's RECOMMENDED table."""
    NAME = "test-moe-4L.gguf"

    def __init__(self, d):
        t = {"token_embd.weight": 10 * MIB}
        for i in range(4):
            t[f"blk.{i}.attn_q.weight"] = 1 * MIB
            t[f"blk.{i}.ffn_up_exps.weight"] = 100 * MIB
            t[f"blk.{i}.ffn_down_exps.weight"] = 100 * MIB
        self.path = write_gguf(os.path.join(d, self.NAME), t, {"general.architecture": "qwen35moe", "qwen35moe.context_length": 65536,
                                                              "qwen35moe.nextn_predict_layers": 1})
        self.model = {"path": self.path, "name": self.NAME, "size": os.path.getsize(self.path), "dir": d, "repo": "test"}
        self.rec = {"mtp_off": 32768, "mtp_on": 32768, "cpu_layers": {"mtp_off": 2, "mtp_on": 3}, "estimated": ["mtp_on"]}


class TestRamCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.m = MoEModel(self.tmp.name)
        self.p = mock.patch.dict(L.RECOMMENDED, {MoEModel.NAME: self.m.rec}); self.p.start()

    def tearDown(self):
        self.p.stop(); self.tmp.cleanup()

    def test_fit_refuses_when_ram_is_short_without_starting_a_server(self):
        """llm-tune's fit must budget system RAM, not only the GPU: with too little MemAvailable for the experts kept in
        RAM it records 'not enough RAM' and never starts llama-server (which would pass the VRAM probe, then swap)."""
        started = []
        def fake_start(*a, **k): started.append(a); return None, False
        s = {"mtp": False, "draft_n": 2, "proj": None, "proj_gpu": False}
        with mock.patch.object(T, "start", fake_start), mock.patch.object(T, "mem", lambda idx: (0, 8000)), \
             mock.patch.object(L, "meminfo", lambda: {"MemAvailable": 1500, "SwapTotal": 0, "SwapFree": 0}, create=True):
            out = T.fit("llama-server", self.m.model, s, "0", 65536, self.tmp.name)
        self.assertEqual(started, [], "a server was started although RAM was short")
        held = out[0] if isinstance(out, tuple) else out.get("held")
        self.assertFalse(held)
        self.assertIn("RAM", out[2] if isinstance(out, tuple) else out.get("note", ""))

    def test_ram_need_counts_the_experts_in_ram_and_the_embeddings(self):
        need_off = L.ram_resident_mib(self.m.model, False)       # layers 0-1: 4 x 100 MiB + 10 MiB embeddings
        need_on = L.ram_resident_mib(self.m.model, True)         # layers 0-2: 6 x 100 MiB + 10 MiB
        self.assertAlmostEqual(need_off, 410, delta=1)
        self.assertAlmostEqual(need_on, 610, delta=1)
        r = L.ram_check(self.m.model, False, avail=410 + L.RAM_ANON_MIB + L.CACHE_RAM_MIB + L.RAM_HEADROOM_MIB + 1)
        self.assertEqual(r["short"], 0)
        r = L.ram_check(self.m.model, False, avail=1000)
        self.assertGreater(r["short"], 0)


class TestTuneSteps(unittest.TestCase):
    """llm-tune runs the fit test and the speed test separately, per model and setting."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tuned = os.path.join(self.tmp.name, "tuned.json")
        self.p = [mock.patch.object(T, "TUNED", self.tuned), mock.patch.object(T, "HeatLog", FakeHeat)]
        for x in self.p: x.start()
        self.model = {"path": write_gguf(os.path.join(self.tmp.name, "dense.gguf"), {"token_embd.weight": MIB}), "name": "dense.gguf",
                      "size": 1, "dir": self.tmp.name, "repo": "test"}
        self.s = {"mtp": False, "draft_n": 2, "proj": None, "proj_gpu": False}
        self.calls = []

    def tearDown(self):
        for x in self.p: x.stop()
        self.tmp.cleanup()

    def fake_fit(self, *a, **k):
        self.calls.append("fit"); return 40960, 38912, "ok"

    def fake_speed(self, server, model, s, ctx, *a, **k):
        self.calls.append(("speed", ctx)); return {"4096": {"reading": 100, "writing": 10.0}}, None

    def run_one(self, what, redo=False):
        with mock.patch.object(T, "fit", self.fake_fit), mock.patch.object(T, "speed", self.fake_speed):
            return T._tune_one("srv", self.model, self.s, "mtp=off,images=cpu", "0", "gpu", 65536, self.tmp.name, 0, None, what, redo)

    def test_fit_only_saves_the_fit_and_no_speed(self):
        rec = self.run_one("fit")
        self.assertEqual(self.calls, ["fit"])
        self.assertEqual((rec["max_ctx"], rec["suggested"], rec["speed"]), (40960, 38912, None))
        with open(self.tuned) as f: self.assertEqual(json.load(f)["dense.gguf|mtp=off,images=cpu"]["suggested"], 38912)

    def test_speed_only_uses_the_saved_fit(self):
        self.run_one("fit"); self.calls.clear()
        rec = self.run_one("speed")
        self.assertEqual(self.calls, [("speed", 38912)])          # no new fit, the fit's suggestion
        self.assertTrue(rec["speed"])

    def test_speed_without_a_fit_runs_the_fit_first(self):
        self.run_one("speed")
        self.assertEqual(self.calls, ["fit", ("speed", 38912)])

    def test_needs(self):
        fit_only = {"max_ctx": 1, "suggested": 1, "speed": None}
        both = dict(fit_only, speed={"4096": {}})
        nofit = {"max_ctx": 0, "suggested": 0, "speed": None, "note": "it does not fit"}
        ramshort = {"max_ctx": 0, "suggested": 0, "ram": {"short_mib": 500}}
        self.assertEqual(T.needs(None, "both"), (True, True))
        self.assertEqual(T.needs(fit_only, "both"), (False, True))
        self.assertEqual(T.needs(fit_only, "fit"), (False, False))
        self.assertEqual(T.needs(both, "speed"), (False, False))
        self.assertEqual(T.needs(both, "speed", redo=True), (False, True))
        self.assertEqual(T.needs(nofit, "speed"), (False, False))      # nothing to measure at a context that does not fit
        self.assertEqual(T.needs(ramshort, "fit"), (True, False))      # short of RAM last time: tried again


class FakeHeat:
    def __init__(self, idx): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass


class TestMoELaunch(unittest.TestCase):
    def test_tested_layouts_match_the_lab_launch_lines(self):
        """The 35B-A3B / Gemma layouts: the experts of layers 0..n-1 in RAM with the tested -ot pattern, MTP n-max 1."""
        m = {"path": "/x/Qwen_Qwen3.6-35B-A3B-IQ3_XXS.gguf", "name": "Qwen_Qwen3.6-35B-A3B-IQ3_XXS.gguf"}
        a = L.server_argv("llama-server", m, 88064, False, 1, None, False, 8080)
        self.assertEqual(a[a.index("-ot") + 1], "blk\\.(" + "|".join(map(str, range(25))) + ")\\.ffn_.*_exps.*=CPU")
        self.assertEqual(" ".join(a[a.index("--temp"):a.index("-ot")]), "--temp 1.0 --top-p 0.95 --top-k 20")
        a = L.server_argv("llama-server", m, 88064, True, 1, None, False, 8080)
        self.assertIn("--spec-draft-n-max 1 --spec-draft-p-min 0 -ctkd q4_0 -ctvd q4_0", " ".join(a))
        self.assertTrue(a[a.index("-ot") + 1].startswith("blk\\.(0|1|") and "|26)" in a[a.index("-ot") + 1])
        g = {"path": "/x/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf", "name": "gemma-4-26B-A4B-it-UD-IQ4_XS.gguf"}
        a = L.server_argv("llama-server", g, 88064, False, 2, None, False, 8080)
        self.assertIn("|19)\\.ffn_", a[a.index("-ot") + 1]); self.assertIn("--top-k 64", " ".join(a))
        o = {"path": "/x/Ornith-1.5-35B-A3B-IQ3_XXS.gguf", "name": "Ornith-1.5-35B-A3B-IQ3_XXS.gguf"}
        a = L.server_argv("llama-server", o, 88064, True, 1, None, False, 8080)
        self.assertIn("--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0", " ".join(a))
        self.assertEqual(next(n for k, n in L.DRAFT_N.items() if o["name"].startswith(k)), 1)
        self.assertEqual(next((n for k, n in L.DRAFT_N.items() if "Ornith-1.5-9B-Q4_K_M.gguf".startswith(k)), 2), 2)
        b = {"path": "/x/Ternary-Bonsai-2.5-27B-PTQ1_0.gguf", "name": "Ternary-Bonsai-2.5-27B-PTQ1_0.gguf"}
        self.assertNotIn("-ot", L.server_argv("llama-server", b, 88064, False, 3, None, False, 8080))

    def test_setting_keys(self):
        self.assertEqual(L.setting_key("Ternary-Bonsai-2.5-27B-PTQ1_0.gguf", False, False), "mtp=off,images=cpu")
        self.assertEqual(L.setting_key("Qwen_Qwen3.6-35B-A3B-IQ3_XXS.gguf", True, False), "mtp=on,images=cpu,ram_layers=27")
        self.assertIn("experts of 27 layers in RAM", L.setting_label("mtp=on,images=cpu,ram_layers=27"))


class TestDownloads(unittest.TestCase):
    def test_every_download_is_pinned_and_checked(self):
        for d in L.DOWNLOADS:
            self.assertRegex(d["rev"], r"^[0-9a-f]{40}$", d["label"])
            for name, size, sha in d["files"]:
                self.assertRegex(sha, r"^[0-9a-f]{64}$", name)
                if d.get("kind") != "dir": self.assertGreater(size, 10**8, name)
        names = [d["files"][0][0] for d in L.DOWNLOADS]
        for f in ("Ternary-Bonsai-2.5-27B-Uncensored-PTQ1_0-mtp.gguf", "gemma-4-26B-A4B-it-UD-IQ4_XS.gguf",
                  "Qwen_Qwen3.6-35B-A3B-IQ3_XXS.gguf", "Ornith-1.5-35B-A3B-IQ3_XXS.gguf"):
            self.assertIn(f, names); self.assertIn(f, L.RECOMMENDED)

    def test_a_file_that_does_not_match_its_checksum_is_not_kept(self):
        import hashlib, io
        body = b"x" * 1000
        class R(io.BytesIO):
            status = 200
            def __enter__(self): return self
            def __exit__(self, *a): pass
        d = {"repo": "a/b", "rev": "0" * 40}
        with tempfile.TemporaryDirectory() as t, mock.patch.object(L.shutil, "which", lambda x: None), \
             mock.patch.object(L.urllib.request, "urlopen", lambda *a, **k: R(body)):
            self.assertFalse(L.fetch_file(d, "m.gguf", 1000, "0" * 64, t))
            self.assertEqual(os.listdir(t), [])
            self.assertTrue(L.fetch_file(d, "m.gguf", 1000, hashlib.sha256(body).hexdigest(), t))
            self.assertEqual(os.listdir(t), ["m.gguf"])

    def test_no_room_no_download(self):
        d = {"label": "big", "repo": "a/b", "rev": "0" * 40, "files": [("big.gguf", 10**12, "0" * 64)]}
        with tempfile.TemporaryDirectory() as t, mock.patch.object(L, "MODELS_DIR", t), mock.patch.object(L, "scan_models", lambda st: []), \
             mock.patch.object(L, "fetch_file", lambda *a: self.fail("downloaded with no room")), mock.patch.object(L, "yes", lambda *a, **k: True):
            self.assertFalse(L.get_download(d, {}))


C = load("check_kit", "check-kit.py")


class TestCheckKit(unittest.TestCase):
    """check-kit.py accepts the two tested plugin sets and refuses every other combination of these plugins."""
    A = {"file": "plugins/compaction-anchor.js", "options": {"parts": ["checklist", "reinsert"]}}
    SP_CONT = {"file": "plugins/system-prompt.js", "options": {"file": "{kit}/prompts/v1-1.md", "continueText": "Continue with the task."}}
    SP = {"file": "plugins/system-prompt.js", "options": {"file": "{kit}/prompts/v1-1.md"}}
    TD = {"file": "plugins/opencode-task-driver-0.4.2.js", "options": {"mode": "focus"}}

    def bad(self, *plugins):
        out = []; C.check_setup({"id": "x", "plugins": list(plugins)}, out); return out

    def test_the_two_tested_sets_pass(self):
        self.assertEqual(self.bad(self.A, self.SP_CONT), [])          # standard: v1.1 + continue text + checklist + reinsert
        self.assertEqual(self.bad(self.SP, self.TD), [])              # task driver: v1.1 without the continue text + the driver

    def test_other_combinations_are_refused(self):
        slim = {"file": "plugins/compaction-anchor.js", "options": {"parts": ["slim"]}}
        nsa = {"file": "plugins/compaction-anchor.js", "options": {"parts": ["checklist", "no-stop-and-ask"]}}
        for combo in [(slim, self.SP), (self.SP_CONT, self.TD), (self.A, self.SP, self.TD), (self.A, self.TD),
                      (nsa, self.SP_CONT), (self.SP, self.SP_CONT), (dict(self.TD, options={"mode": "tasks"}), self.SP),
                      (self.SP, {"file": "plugins/something-else.js"}), ({"file": "plugins/compaction-anchor.js"}, self.SP_CONT)]:
            self.assertTrue(self.bad(*combo), [p.get("options") for p in combo])

    def test_this_kit_passes(self):
        with open(os.path.join(KIT, "kit.json")) as f: self.assertEqual(C.check(json.load(f)), [])

    def test_a_setup_without_reinsert_or_a_task_driver_is_refused(self):
        """Kyle 11-26: every session has the compaction plugin (with reinsert) or the task driver. Without either, a
        long session can lose the request to a summary (a model asked for the bug texts back after 14 summaries)."""
        checklist_only = {"file": "plugins/compaction-anchor.js", "options": {"parts": ["checklist"]}}
        for combo in [(self.SP_CONT,), (self.SP,), (), (checklist_only, self.SP_CONT)]:
            out = self.bad(*combo)
            self.assertTrue(any("reinsert" in x and "task driver" in x for x in out), (combo, out))
        self.assertEqual(self.bad(self.A, self.SP_CONT), [])
        self.assertEqual(self.bad(self.SP, self.TD), [])


class TestKeeperPlugins(unittest.TestCase):
    """The launcher never starts a session without the compaction plugin or the task driver: if the chosen setup's one
    cannot be loaded, it uses the other setup (saying so), and if neither can be, it stops before starting anything."""
    KIT_JSON = {"setups": [
        {"id": "standard", "name": "std", "default": True, "plugins": [
            {"name": "anchor", "file": "plugins/compaction-anchor.js", "options": {"parts": ["checklist", "reinsert"]}},
            {"name": "sp", "file": "plugins/system-prompt.js"}]},
        {"id": "task-driver", "name": "td", "ask": "?", "plugins": [
            {"name": "sp", "file": "plugins/system-prompt.js"},
            {"name": "driver", "file": "plugins/opencode-task-driver-0.4.2.js"}]}]}

    def run_with(self, missing, sid):
        have = lambda p, quiet=False: None if p["file"] in missing else "/x/" + p["file"]
        with mock.patch.object(L, "plugin_path", have), mock.patch("builtins.print"):
            return L.usable_setup(sid, self.KIT_JSON)

    def test_the_chosen_setup_is_kept_when_its_plugins_load(self):
        self.assertEqual(self.run_with(set(), "task-driver"), "task-driver")
        self.assertEqual(self.run_with({"plugins/system-prompt.js"}, "standard"), "standard")   # the prompt alone may be missing

    def test_a_missing_keeper_switches_to_the_other_setup(self):
        self.assertEqual(self.run_with({"plugins/opencode-task-driver-0.4.2.js"}, "task-driver"), "standard")
        self.assertEqual(self.run_with({"plugins/compaction-anchor.js"}, "standard"), "task-driver")

    def test_neither_keeper_stops_before_a_session(self):
        with self.assertRaises(SystemExit):
            self.run_with({"plugins/compaction-anchor.js", "plugins/opencode-task-driver-0.4.2.js"}, "standard")

    def test_a_prompt_with_a_trailing_newline_is_refused(self):
        with tempfile.TemporaryDirectory() as t, mock.patch.object(C, "KIT", t):
            os.makedirs(os.path.join(t, "prompts")); os.makedirs(os.path.join(t, "plugins"))
            for f, text in (("prompts/p.md", "A prompt.\n"), ("plugins/system-prompt.js", "x")):
                with open(os.path.join(t, f), "w") as fh: fh.write(text)
            import hashlib
            p = {"name": "sp", "file": "plugins/system-prompt.js", "sha256": hashlib.sha256(b"x").hexdigest(), "options": {"file": "{kit}/prompts/p.md"}}
            out = []; C.check_plugin(p, False, out, {})
            self.assertTrue(any("whitespace" in x for x in out), out)


class TestCheckKitModes(unittest.TestCase):
    """The interaction-mode plugin rides along in both tested sets, after the system prompt it edits, only with its
    "untested" marker and the tested mode (autonomous) as its default; it never stands in for the reinsert part or the task
    driver, and other plugins stay refused."""
    A, TD = TestCheckKit.A, TestCheckKit.TD
    SP_CONT = {"file": "plugins/system-prompt.js", "options": {"file": "{kit}/prompts/v1-2.md", "continueText": "Continue with the task."}}
    SP = {"file": "plugins/system-prompt.js", "options": {"file": "{kit}/prompts/v1-2.md"}}
    IM = {"name": "im", "file": "plugins/interaction-mode.js", "untested": True, "options": {"default": "autonomous"}}

    def bad(self, *plugins):
        out = []; C.check_setup({"id": "x", "plugins": list(plugins)}, out); return out

    def test_both_tested_sets_pass_with_it(self):
        self.assertEqual(self.bad(self.A, self.SP_CONT, self.IM), [])
        self.assertEqual(self.bad(self.SP, self.TD, self.IM), [])

    def test_it_needs_the_untested_marker_and_autonomous_as_default(self):
        for im in (dict(self.IM, untested=None), {k: v for k, v in self.IM.items() if k != "untested"}, dict(self.IM, untested="yes"),
                   dict(self.IM, options={"default": "interactive"}), dict(self.IM, options={"default": "plan-first"}), dict(self.IM, options={})):
            self.assertTrue(self.bad(self.A, self.SP_CONT, im), im)

    def test_it_comes_after_a_prompt_with_a_questions_section(self):
        out = self.bad(self.A, self.IM, self.SP_CONT)
        self.assertTrue(any("after system-prompt.js" in x for x in out), out)
        v11 = dict(self.SP_CONT, options={"file": "{kit}/prompts/v1-1.md", "continueText": "Continue with the task."})   # no Questions section
        out = self.bad(self.A, v11, self.IM)
        self.assertTrue(any("Questions" in x for x in out), out)

    def test_the_old_bad_combinations_stay_refused_with_it(self):
        slim = {"file": "plugins/compaction-anchor.js", "options": {"parts": ["slim"]}}
        nsa = {"file": "plugins/compaction-anchor.js", "options": {"parts": ["checklist", "no-stop-and-ask"]}}
        for combo in [(slim, self.SP), (self.SP_CONT, self.TD), (self.A, self.SP, self.TD), (self.A, self.TD), (nsa, self.SP_CONT),
                      (self.SP, self.SP_CONT), (dict(self.TD, options={"mode": "tasks"}), self.SP), (self.SP, {"file": "plugins/something-else.js"}),
                      ({"file": "plugins/compaction-anchor.js"}, self.SP_CONT), (self.A, self.SP_CONT, self.IM, self.IM)]:
            self.assertTrue(self.bad(*combo, self.IM), [p.get("options") for p in combo])

    def test_it_is_not_the_reinsert_part_or_a_task_driver(self):
        for combo in [(self.IM,), (self.SP_CONT, self.IM), (self.SP, self.IM)]:
            out = self.bad(*combo)
            self.assertTrue(any("reinsert" in x and "task driver" in x for x in out), (combo, out))

    def test_the_untested_marker_does_not_let_other_plugins_in(self):
        out = self.bad(self.A, self.SP_CONT, {"name": "x", "file": "plugins/something-else.js", "untested": True, "options": {"default": "autonomous"}})
        self.assertTrue(any("no tested setup used" in x for x in out), out)

    def test_the_kit_runs_the_tested_v12_text_with_the_plugin_last(self):
        with open(os.path.join(KIT, "kit.json")) as f: kit = json.load(f)
        for s in kit["setups"]:
            sp = [p for p in s["plugins"] if p.get("file") == "plugins/system-prompt.js"]
            self.assertEqual((sp[0]["options"]["file"], sp[0]["prompt_sha256"]), ("{kit}/prompts/v1-2.md", V12_SHA), s["id"])
            im = [p for p in s["plugins"] if p.get("file") == "plugins/interaction-mode.js"]
            self.assertEqual(len(im), 1, s["id"])
            self.assertEqual((im[0]["untested"], im[0]["options"]), (True, {"default": "autonomous"}))
            self.assertIs(s["plugins"][-1], im[0])      # last: after the prompt it edits and the task driver's planning block
        import hashlib
        with open(os.path.join(KIT, "prompts", "v1-2.md"), "rb") as f: self.assertEqual(hashlib.sha256(f.read()).hexdigest(), V12_SHA)

    def test_its_file_is_checked_for_the_banned_words(self):
        """The plugin's text goes to the model and the repo is public: the whole file, comments included, is checked."""
        import hashlib
        with tempfile.TemporaryDirectory() as t, mock.patch.object(C, "KIT", t):
            os.makedirs(os.path.join(t, "plugins"))
            body = b"// built for the benchmark\nexport const X = async () => ({});\n"
            with open(os.path.join(t, "plugins", "interaction-mode.js"), "wb") as f: f.write(body)
            out = []; C.check_plugin(dict(self.IM, sha256=hashlib.sha256(body).hexdigest()), False, out, {})
            self.assertTrue(any("mentions the benchmark" in x for x in out), out)
        with open(os.path.join(KIT, "kit.json")) as f:
            real = next(p for p in json.load(f)["setups"][0]["plugins"] if p["file"] == "plugins/interaction-mode.js")
        out = []; C.check_plugin(real, False, out, {})
        self.assertEqual(out, [])


class TestEXL3(unittest.TestCase):
    """The EXL3 path (experimental): detection, TabbyAPI's settings, its health answer. TabbyAPI itself is not run here."""
    def make(self, d, size=8570690830):
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump({"quantization_config": {"quant_method": "exl3", "bits": 2.0},
                       "text_config": {"mtp_num_hidden_layers": 1, "max_position_embeddings": 262144}}, f)
        for name, n in (("model-00001-of-00002.safetensors", size), ("model-00002-of-00002.safetensors", 1000)):
            with open(os.path.join(d, name), "wb") as f: f.truncate(n)          # sparse: no disk used
        return d

    def test_found_as_an_exl3_model_and_matched_to_its_tested_settings(self):
        with tempfile.TemporaryDirectory() as t, mock.patch.object(L, "DEFAULT_DIRS", [t]), mock.patch.object(L, "hf_cache", lambda: os.path.join(t, "none")):
            self.make(os.path.join(t, "turboderp__Qwen3.8-27B-exl3__SCP_1.75bpw"), size=8587658013)
            with open(os.path.join(t, "config.json"), "w") as f: json.dump({"model_type": "llama"}, f)      # not EXL3: ignored
            ms = [m for m in L.scan_models({}) if m.get("engine") == "exl3"]
            self.assertEqual(len(ms), 1)
            m = ms[0]
            self.assertTrue(m["exl3"]["mtp"])
            # the lab's measured 8 GB fit: 57,344 with MTP off, minus the head's 390 MiB at 18.06 KiB/token with it on
            self.assertEqual(L.EXL3_RECOMMENDED[m["exl3"]["sig"]]["ctx"], {"mtp_off": 57344, "mtp_on": 34816})
            self.assertNotIn("model-00001-of-00002.safetensors:8570690830", L.EXL3_RECOMMENDED)      # 2.00 bpw: dropped

    def test_tabby_config_is_the_lab_layout(self):
        m = {"path": "/m/turboderp__Qwen3.8-27B-exl3__SCP_2.00bpw/", "name": "x"}
        c = L.exl3_config(m, 40960, True, 2, "4,4", 8081)
        self.assertEqual((c["network"]["host"], c["network"]["port"], c["network"]["allowed_origins"]), ("127.0.0.1", 8081, []))
        self.assertEqual((c["model"]["model_dir"], c["model"]["model_name"]), ("/m", "turboderp__Qwen3.8-27B-exl3__SCP_2.00bpw"))
        self.assertEqual((c["model"]["max_seq_len"], c["model"]["cache_size"], c["model"]["cache_mode"]), (40960, 40960, "4,4"))
        self.assertEqual(c["draft_model"], {"draft_mode": "mtp", "draft_cache_mode": "4,4", "draft_num_tokens": 2})
        self.assertTrue(c["model"]["use_generation_config"]); self.assertEqual(c["model"]["tool_format"], "auto")
        self.assertEqual(L.exl3_config(m, 40960, False, 2, "4,4", 8081)["draft_model"], {"draft_mode": "disabled"})

    def test_context_is_a_multiple_of_256(self):
        st = {}
        m = {"name": "x", "exl3": {"sig": "none", "bits": 2.0, "mtp": False, "native": 262144}}
        with mock.patch.object(L, "ask", lambda *a, **k: "40000"):
            self.assertEqual(L.choose_exl3(st, m)["ctx"], 39936)
        # 1.75 bpw: MTP on by default, at the context that fits with the head; after an MTP-on start failure, off
        m = {"name": "y", "exl3": {"sig": "model-00001-of-00002.safetensors:8587658013", "bits": 1.75, "mtp": True, "native": 262144}}
        asked = []
        with mock.patch.object(L, "yes", lambda q, default=True: asked.append(default) or default), mock.patch.object(L, "ask", lambda q, default=None: default):
            c = L.choose_exl3({}, m)
            self.assertEqual((asked, c["mtp"], c["ctx"]), ([True], True, 34816))
            c = L.choose_exl3({"_mtp_failed": {"y": True}}, m)
            self.assertEqual((asked[-1], c["mtp"], c["ctx"]), (False, False, 57344))

    def test_health_accepts_tabbys_answer(self):
        import http.server, threading
        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                b = b'{"status": "healthy", "issues": []}'
                self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
            def log_message(self, *a): pass
        srv = http.server.HTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
        try: self.assertTrue(L.health(srv.server_address[1]))
        finally: srv.shutdown(); srv.server_close()

    def test_exl3_downloads_are_whole_pinned_folders(self):
        ds = [d for d in L.DOWNLOADS if d.get("kind") == "dir"]
        self.assertEqual([d["subdir"] for d in ds], ["turboderp__Qwen3.8-27B-exl3__SCP_1.75bpw"])     # 2.00 bpw dropped
        for d in ds:
            names = {f[0] for f in d["files"]}
            self.assertTrue({"config.json", "model-00001-of-00002.safetensors", "tokenizer.json"} <= names)
            self.assertTrue(all(len(f[2]) == 64 for f in d["files"]))
            w = next(f for f in d["files"] if f[0] == "model-00001-of-00002.safetensors")
            self.assertIn(f"{w[0]}:{w[1]}", L.EXL3_RECOMMENDED)
            with tempfile.TemporaryDirectory() as t, mock.patch.object(L, "MODELS_DIR", t):
                self.assertEqual(len(L.missing_files(d)), len(d["files"]))


class TestSeparateMtpHead(unittest.TestCase):
    """Qwen3.8-27B's 1-bit files keep their MTP head in a separate file (MTP/mtp-Qwen3.8-27B-Q4_0.gguf). The download
    menu gets it with the model, the launcher offers MTP only when it is there and loads it as the draft model, and
    llm-tune measures such a model with MTP off and on."""
    REV = "4af51b3542f775b0a8e4b310bfb3f17e01c306c9"
    HEAD = ("MTP/mtp-Qwen3.8-27B-Q4_0.gguf", 1369590656, "50d9ce5a6da381bbcfb31061cf73df94a90e6faf8efeddee379a9cb8f1501c6e")
    FILES = {"Qwen3.8-27B-UD-IQ1_S.gguf": (6192222208, "3895b6eaa91e705c06ad1938d16c22e86f073c6a67df86260a1da79be3d1f887"),
             "Qwen3.8-27B-UD-IQ1_M.gguf": (6729166848, "1b5165a7149ea51e683c8eaf23372188ad9fc9d1a795386f7a1b558acf847dc6")}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.t = self.tmp.name

    def tearDown(self): self.tmp.cleanup()

    def model(self, d, name="Qwen3.8-27B-UD-IQ1_S.gguf", nextn=0):
        os.makedirs(d, exist_ok=True)
        p = write_gguf(os.path.join(d, name), {"token_embd.weight": MIB, "blk.0.attn_q.weight": MIB},
                       {"general.architecture": "qwen35", "qwen35.nextn_predict_layers": nextn, "qwen35.context_length": 262144})
        return {"path": p, "name": name, "size": os.path.getsize(p), "dir": d, "repo": "test"}

    def head(self, d, size=None):
        p = os.path.join(d, self.HEAD[0]); os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f: f.truncate(self.HEAD[1] if size is None else size)       # sparse: no disk used
        return p

    def test_the_two_downloads_are_pinned_with_their_head(self):
        ds = [d for d in L.DOWNLOADS if d["repo"] == "unsloth/Qwen3.8-27B-GGUF"]
        self.assertEqual(len(ds), 2)
        for d in ds:
            self.assertEqual((d["rev"], d["new"]), (self.REV, "2026-11-27"))
            (name, size, sha), head = d["files"]
            self.assertEqual((size, sha), self.FILES[name])
            self.assertEqual(head, self.HEAD)
            self.assertIn("being tested", d["label"])
            self.assertFalse(any(c in f[0] for f in d["files"] for c in "*?["), "exact names only, no patterns")
        self.assertEqual(sorted(d["files"][0][0] for d in ds), sorted(self.FILES))

    def test_a_file_in_the_repos_mtp_folder_is_saved_there(self):
        import hashlib, io
        body = b"h" * 1000
        class R(io.BytesIO):
            status = 200
            def __enter__(self): return self
            def __exit__(self, *a): pass
        d = {"repo": "a/b", "rev": "0" * 40}
        with mock.patch.object(L.shutil, "which", lambda x: None), mock.patch.object(L.urllib.request, "urlopen", lambda *a, **k: R(body)):
            self.assertTrue(L.fetch_file(d, "MTP/mtp-x.gguf", 1000, hashlib.sha256(body).hexdigest(), self.t))
        self.assertTrue(os.path.isfile(os.path.join(self.t, "MTP", "mtp-x.gguf")))

    def test_the_head_is_not_listed_as_a_model(self):
        m = self.model(self.t); self.head(self.t)
        write_gguf(os.path.join(self.t, "MTP", "mtp-other.gguf"), {"blk.64.nextn.eh_proj.weight": MIB})
        with mock.patch.object(L, "DEFAULT_DIRS", [self.t]), mock.patch.object(L, "hf_cache", lambda: os.path.join(self.t, "none")):
            self.assertEqual([x["name"] for x in L.scan_models({})], [m["name"]])

    def test_the_head_is_found_where_the_menu_puts_it(self):
        m = self.model(os.path.join(self.t, "models", "x"))
        self.assertIsNone(L.mtp_head_for(m))
        self.assertEqual(L.mtp_head_for(m), None)
        p = self.head(m["dir"])
        self.assertEqual(L.mtp_head_for(m), p)                                         # in the MTP folder next to it
        os.remove(p); self.head(m["dir"], size=1000)
        self.assertIsNone(L.mtp_head_for(m))                                           # a cut-off file is not used
        # a model in the Hugging Face cache: the menu puts the head in ~/models/<repo>/MTP/, and finds it there
        hub = os.path.join(self.t, "hub"); snap = os.path.join(hub, "models--unsloth--Qwen3.8-27B-GGUF", "snapshots", self.REV)
        cached = self.model(snap); cached["repo"] = "unsloth/Qwen3.8-27B-GGUF"
        models = os.path.join(self.t, "dl")
        with mock.patch.object(L, "MODELS_DIR", models), mock.patch.object(L, "hf_cache", lambda: hub), \
             mock.patch.object(L, "DEFAULT_DIRS", []), mock.patch.object(L, "scan_models", lambda st: [cached]):
            entry = next(d for d in L.DOWNLOADS if d["files"][0][0] == "Qwen3.8-27B-UD-IQ1_S.gguf")
            with mock.patch.object(L, "have_file", lambda n, s, st=None: cached["path"] if n == cached["name"] else None):
                self.assertEqual(L.target_dir(entry, {}), L.dl_dest(entry))
                self.assertEqual([f[0] for f in L.missing_files(entry, {})], [self.HEAD[0]])
                p = self.head(L.dl_dest(entry))
                self.assertEqual(L.missing_files(entry, {}), [])                       # the menu counts it as there ...
                self.assertEqual(L.mtp_head_for(cached), p)                            # ... and the launcher finds it
        bonsai = self.model(self.t, "Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf", nextn=1)
        self.assertIsNone(L.mtp_head_for(bonsai))                                      # a built-in head: no separate file

    def run_choose(self, m, tuned=None, mem=None):
        asked = []
        st = {"models": {m["name"]: mem}} if mem else {}
        with mock.patch.object(L, "load_tuned", lambda: tuned or {}), mock.patch.object(L, "yes", lambda q, default=True: asked.append((q, default)) or default), \
             mock.patch("builtins.print"):
            c = L.choose(st, m, ask_ctx=False)
        return c, asked

    def test_mtp_is_offered_only_with_the_head(self):
        m = self.model(self.t)
        c, asked = self.run_choose(m)
        self.assertEqual((c["mtp"], [q for q, _ in asked if "MTP" in q]), (False, []))   # no head file: not asked
        self.head(self.t)
        c, asked = self.run_choose(m)
        self.assertEqual([d for q, d in asked if "MTP" in q], [False])                   # asked; off until measured
        k_on = L.setting_key(m["name"], True, False)
        c, asked = self.run_choose(m, {f"{m['name']}|{k_on}": {"max_ctx": 20480, "suggested": 18432}})
        self.assertEqual([d for q, d in asked if "MTP" in q], [True])                    # llm-tune found MTP on fits
        self.assertTrue(c["mtp"])

    def test_the_server_loads_the_head_as_the_draft_model(self):
        m = self.model(self.t); head = self.head(self.t)
        a = " ".join(L.server_argv("llama-server", m, 16384, True, 2, None, False, 8080))
        self.assertIn(f"--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0 -ctkd q4_0 -ctvd q4_0 -md {head} -ngld 999", a)
        self.assertIn("--temp 1.0 --top-p 0.95 --top-k 20", a)                           # Qwen3.8-27B's own sampling
        a = L.server_argv("llama-server", m, 16384, False, 2, None, False, 8080)
        self.assertNotIn("-md", a); self.assertNotIn("--spec-type", a)
        b = self.model(self.t, "Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf", nextn=1)
        self.assertNotIn("-md", L.server_argv("llama-server", b, 41984, True, 3, None, False, 8080))
        q = self.model(os.path.join(self.t, "own"), nextn=1); self.head(q["dir"])          # a file with its own head: that one is used
        self.assertNotIn("-md", L.server_argv("llama-server", q, 16384, True, 2, None, False, 8080))

    def test_llm_tune_measures_mtp_off_and_on_when_the_head_is_there(self):
        m = self.model(self.t)
        self.assertEqual([s["mtp"] for s, k in T.settings_for(m)], [False])
        self.head(self.t)
        got = T.settings_for(m)
        self.assertEqual([(s["mtp"], k) for s, k in got], [(False, "mtp=off,images=cpu"), (True, "mtp=on,images=cpu")])

    def test_the_inventory_lists_them_as_being_tested(self):
        home = os.path.join(self.t, "home"); d = os.path.join(home, "models", "q")
        self.model(d); self.head(d); self.model(d, "Qwen3.8-27B-UD-IQ1_M.gguf")
        os.remove(os.path.join(d, self.HEAD[0])); self.head(d)
        out = []
        real_isdir = os.path.isdir
        with mock.patch.object(L, "HOME", home), mock.patch.object(L, "save_state", lambda st: None), \
             mock.patch.object(L.os.path, "isdir", lambda p: False if p in ("/mnt",) or p.startswith(("/media/", "/run/media/")) else real_isdir(p)), \
             mock.patch("builtins.print", lambda *a, **k: out.append(" ".join(map(str, a)))):
            L.inventory({})
        text = "\n".join(out)
        self.assertIn("Qwen3.8-27B-UD-IQ1_S.gguf", text); self.assertIn("Qwen3.8-27B-UD-IQ1_M.gguf", text)
        self.assertIn("being tested for 8 GB", text); self.assertIn("its MTP head file is there", text)
        self.assertNotIn("mtp-Qwen3.8-27B-Q4_0.gguf  ", text)


class TestBuildPin(unittest.TestCase):
    """The kit installs PrismML's prism-b12232-f702476: the CUDA 12.8 build for driver 570 or newer, else 12.4, each
    checked by its published sha256; setup_version moved on, so the launcher offers to run setup.sh after the update."""
    TAG = "prism-b12232-f702476"
    SHA = {"12.8": "5aeef1a6c543765a488920a771c4cde0931b95f4cb52dfa9aa3c7e0e106d329f",
           "12.4": "b5750a2e4eb7369093d5ec7a1e82bf3e3772fefd5e398a46e8f245d4fb3490ed"}

    def read(self, f):
        with open(os.path.join(KIT, f)) as fh: return fh.read()

    def test_setup_installs_the_new_build(self):
        s = self.read("setup.sh")
        self.assertIn(f"PRISM={self.TAG}", s)
        self.assertIn('grep -q "build 12232"', s)                   # its --version prints "version: ... (build 12232, commit f702476)"
        self.assertIn(f'if [ "${{major:-0}}" -ge 570 ]; then cu=12.8; sha={self.SHA["12.8"]}', s)
        self.assertIn(f'else cu=12.4; sha={self.SHA["12.4"]}; fi', s)
        self.assertIn('releases/download/$PRISM/$t', s)
        self.assertNotIn("12178", s)
        with open(os.path.join(KIT, "kit.json")) as f: self.assertEqual(json.load(f)["setup_version"], 3)

    def test_the_readme_install_steps_match(self):
        r = self.read("README.md")
        t = f"llama-{self.TAG}-bin-linux-cuda-12.8-x64.tar.gz"
        self.assertIn(f"releases/download/{self.TAG}/{t}", r)
        self.assertIn(f"sha256sum {t}   # {self.SHA['12.8']}", r)
        self.assertIn("PrismML build prism-b12232", r)
        self.assertIn("## What was measured (8 GB, q4_0 K/V cache, PrismML prism-b12178)", r)   # the measurements keep their build


class TestRamSignals(unittest.TestCase):
    def test_majflt_reads_field_12(self):
        with tempfile.NamedTemporaryFile("w", suffix=".stat", delete=False) as f:
            f.write("4242 (llama-server x) S 1 4242 4242 0 -1 4194560 120 0 77 0 5 6 0 0 20 0 9 0 100 200 300\n"); path = f.name
        try: self.assertEqual(T.majflt(4242, path), 77)
        finally: os.remove(path)
        self.assertIsNone(T.majflt(-1, "/nonexistent/stat"))


class TestMtpFallback(unittest.TestCase):
    """MTP stays on by default for the 35B-A3B models; when the fit test finds MTP on does not fit on this machine (the
    GPU or system RAM), it measures MTP off right after, and the launcher then suggests MTP off."""
    QWEN = "Qwen_Qwen3.6-35B-A3B-IQ3_XXS.gguf"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tuned = os.path.join(self.tmp.name, "tuned.json")
        self.m = {"path": "/x/" + self.QWEN, "name": self.QWEN, "size": 1, "dir": "/x", "repo": "t"}
        self.s_on = {"mtp": True, "draft_n": 1, "proj": None, "proj_gpu": False}
        self.k_on, self.k_off = L.setting_key(self.QWEN, True, False), L.setting_key(self.QWEN, False, False)

    def tearDown(self): self.tmp.cleanup()

    def fake_tune(self, calls, fit_on):
        def f(server, m, s, k, idx, gname, what, redo):
            calls.append(k)
            rec = ({"max_ctx": 90112, "suggested": 88064} if (fit_on or not s["mtp"]) else
                   {"max_ctx": 0, "suggested": 0, "note": "not enough free system RAM", "ram": {"short_mib": 700}})
            T.save(f"{m['name']}|{k}", rec); return rec
        return f

    def test_mtp_on_that_does_not_fit_is_followed_by_mtp_off(self):
        calls = []
        with mock.patch.object(T, "TUNED", self.tuned), mock.patch.object(T, "tune_one", self.fake_tune(calls, False)):
            T.run_todo("srv", [(self.m, self.s_on, self.k_on)], "0", "gpu", "fit", True)
        self.assertEqual(calls, [self.k_on, self.k_off])

    def test_no_fallback_when_mtp_on_fits_or_off_is_queued(self):
        calls = []
        with mock.patch.object(T, "TUNED", self.tuned), mock.patch.object(T, "tune_one", self.fake_tune(calls, True)):
            T.run_todo("srv", [(self.m, self.s_on, self.k_on)], "0", "gpu", "fit", True)
        self.assertEqual(calls, [self.k_on])
        calls = []
        with mock.patch.object(T, "TUNED", self.tuned), mock.patch.object(T, "tune_one", self.fake_tune(calls, False)):
            T.run_todo("srv", [(self.m, self.s_on, self.k_on), (self.m, dict(self.s_on, mtp=False), self.k_off)], "0", "gpu", "fit", True)
        self.assertEqual(calls, [self.k_on, self.k_off])                  # measured once, in its own place

    def test_launcher_defaults_mtp_on_and_falls_back_to_off_from_the_fit(self):
        tuned = {}
        asked = []
        def run():
            with mock.patch.object(L, "gguf_meta", lambda p, *a: {"nextn_predict_layers": 1, "context_length": 262144}), \
                 mock.patch.object(L, "projector_for", lambda m: None), mock.patch.object(L, "load_tuned", lambda: tuned), \
                 mock.patch.object(L, "ram_check", lambda *a, **k: {"need": 10000, "avail": 14000, "short": 0}), \
                 mock.patch.object(L, "yes", lambda q, default=True: asked.append(default) or default):
                return L.choose({}, self.m, ask_ctx=False)
        self.assertTrue(run()["mtp"])                                     # nothing measured here: MTP on (the tested default)
        tuned[f"{self.QWEN}|{self.k_on}"] = {"max_ctx": 0, "suggested": 0, "note": "not enough free system RAM"}
        tuned[f"{self.QWEN}|{self.k_off}"] = {"max_ctx": 90112, "suggested": 88064}
        self.assertFalse(run()["mtp"])                                    # llm-tune: MTP on does not fit, MTP off does
        self.assertEqual(asked, [True, False])


class TestOutputCap(unittest.TestCase):
    def test_no_per_response_output_cap(self):
        """opencode caps a response at 32,000 tokens unless OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX lifts it (the lab's
        no-output-cap rule): the launcher lifts it, and the model entry's output limit is the whole context."""
        env = L.opencode_env('{"x": 1}')
        self.assertEqual(env["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"], "1048576")
        self.assertEqual(env["OPENCODE_CONFIG_CONTENT"], '{"x": 1}')
        with tempfile.TemporaryDirectory() as t, mock.patch.object(L, "OC_CONFIG", os.path.join(t, "opencode.json")):
            prov = L.update_opencode_config("m", "m", 88064, 8080)
        self.assertEqual(prov["models"]["m"]["limit"], {"context": 88064, "input": 88064, "output": 88064})
        src = open(os.path.join(KIT, "llm-launch.py")).read()
        self.assertEqual(src.count("subprocess.call([opencode_bin()"), src.count("env=env)"))   # every opencode start gets that env


NODE = shutil.which("node")
# Loads plugins as opencode does (each export called with (input, options); input.client a stand-in for opencode's client
# that keeps each session's permission rules) and replays opencode's hook calls.
PLUGIN_RUNNER = r"""
import fs from "node:fs"; import path from "node:path"; import { pathToFileURL } from "node:url";
const spec = JSON.parse(fs.readFileSync(process.argv[1], "utf8"));
const perms = spec.perms || {}, calls = [];
const client = {
  session: {
    get: async ({ path: p }) => ({ data: { id: p.id, permission: perms[p.id] || [] } }),
    update: async ({ path: p, body }) => { for (const r of body.permission || []) calls.push([p.id, r.permission, r.action]); perms[p.id] = [...(perms[p.id] || []), ...(body.permission || [])]; return { data: {} }; },
  },
  tui: { showToast: async () => ({ data: true }) },
};
const hooks = [];
for (const [file, opts] of spec.plugins) {
  const dst = path.join(spec.tmp, hooks.length + "-" + path.basename(file).replace(/\.js$/, ".mjs"));   // .mjs: an ES module on any node
  fs.copyFileSync(file, dst);
  for (const f of Object.values(await import(pathToFileURL(dst).href))) hooks.push(await f({ directory: spec.tmp, client }, opts));
}
const each = async (name, input, output) => { for (const h of hooks) if (h[name]) await h[name](input, output); return output; };
const out = [];
for (const [op, ...a] of spec.ops) {
  if (op === "keys") out.push(hooks.map((h) => Object.keys(h).sort()));
  else if (op === "config") { for (const h of hooks) if (h.config) await h.config(a[0]); out.push(a[0]); }
  else if (op === "event") { await each("event", { event: a[0] }, undefined); out.push(null); }
  else if (op === "idle") { await each("event", { event: { type: "session.status", properties: { sessionID: a[0], status: { type: "idle" } } } }, undefined); out.push(null); }
  else if (op === "system") out.push((await each("experimental.chat.system.transform", { sessionID: a[0], model: {} }, { system: a[1] })).system);
  else if (op === "chat") { await each("chat.message", { sessionID: a[0] }, { message: {}, parts: a[1] }); out.push(null); }
  else if (op === "command") { const parts = a[2], o = { parts }; await each("command.execute.before", { command: a[0], sessionID: a[1], arguments: "" }, o); out.push({ same: o.parts === parts, parts }); }
  else if (op === "before") { try { await each("tool.execute.before", { tool: a[1], sessionID: a[0], callID: "c" }, { args: a[2] || {} }); out.push(null); } catch (e) { out.push(String(e.message)); } }
  else if (op === "after") { await each("tool.execute.after", { tool: a[1], sessionID: a[0], callID: "c", args: {} }, { title: "", output: "", metadata: {} }); out.push(null); }
  else if (op === "key") { const m = globalThis[Symbol.for("opencode.interaction-mode")]; out.push(m ? (m.get(a[0]) ?? null) : null); }
  else if (op === "calls") out.push(calls.slice());
  else if (op === "read") { try { out.push(JSON.parse(fs.readFileSync(path.join(spec.tmp, a[0]), "utf8"))); } catch { out.push(null); } }
}
console.log(JSON.stringify(out));
"""
# opencode 1.18.34's system message for the main agent (its own prompt, the environment lines, the project's AGENTS.md,
# the skills), the title writer's and the summarizer's (their own prompt only)
ENV_LINES = ("You are powered by the model named m. The exact model ID is llama-local/m\n"
             "Here is some useful information about the environment you are running in:\n<env>\n  Working directory: /p\n</env>\n")
AGENTS = "Instructions from: /p/AGENTS.md\n# Questions\n- The project's own section, never touched.\n"
MAIN = ("You are opencode, an interactive CLI tool that helps users with software engineering tasks.\n\nBe concise.\n" + ENV_LINES + AGENTS +
        "Skills provide specialized instructions and workflows for specific tasks.")
TITLE = "You are a title generator. You output ONLY a thread title. Nothing else."
SUMMARY = "You are a helpful AI assistant tasked with summarizing conversations."
REFUSED = "Questions wait until the end: make the most sensible assumption, note it, and keep going; list it in your final message."
TYPED = [{"type": "text", "text": "Fix the two bugs"}]


@unittest.skipUnless(NODE, "node is not installed")
class TestModePlugin(unittest.TestCase):
    """plugins/interaction-mode.js: autonomous sends the tested v1.2 text exactly; plan-first and interactive (untested)
    replace only its Questions section; plan-first's window ends at the first sign of work; the question tool follows the
    mode; /interactive, /plan-first and /autonomous switch the session."""
    IM = os.path.join(KIT, "plugins", "interaction-mode.js")
    SP = os.path.join(KIT, "plugins", "system-prompt.js")

    def run_ops(self, plugins, ops, perms=None):
        with tempfile.TemporaryDirectory() as t:
            plugins = [[f, dict(o, dir=t)] for f, o in plugins]          # their event logs go here, not under ~
            spec = os.path.join(t, "spec.json")
            with open(spec, "w") as f: json.dump({"tmp": t, "plugins": plugins, "ops": ops, "perms": perms or {}}, f)
            r = subprocess.run([NODE, "--no-warnings", "--input-type=module", "-e", PLUGIN_RUNNER, spec], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(r.stdout)

    def im(self, default): return [self.IM, {"default": default}]
    def sp(self): return [self.SP, {"file": os.path.join(KIT, "prompts", "v1-2.md"), "continueText": "Continue with the task."}]

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(KIT, "prompts", "v1-2.md")) as f: cls.V12 = f.read()
        cls.TESTED = cls.V12 + "\n" + ENV_LINES + AGENTS + "Skills provide specialized instructions and workflows for specific tasks."
        cls.AUTO = cls.V12[cls.V12.index("# Questions\n"):cls.V12.index("\n\n# Plan and track")]

    def section(self, text):
        """The Questions section of the prompt part (before the environment lines)."""
        head = text[:text.index("You are powered by")]
        return head[head.index("# Questions\n"):head.index("\n\n# Plan and track")]

    def test_autonomous_sends_the_tested_v12_text(self):
        ops = [["config", {}], ["chat", "s1", TYPED], ["system", "s1", [MAIN]], ["system", "s1", [TITLE]], ["system", "s1", [SUMMARY]],
               ["before", "s1", "write"], ["idle", "s1"], ["chat", "s1", TYPED], ["system", "s1", [MAIN]], ["key", "s1"]]
        out = self.run_ops([self.sp(), self.im("autonomous")], ops)
        self.assertEqual(out[2], [self.TESTED])                                  # v1.2 exactly, as the tested sessions get it
        self.assertEqual(out[3:5], [[TITLE], [SUMMARY]])
        self.assertEqual(out[8], [self.TESTED])
        self.assertEqual(out[9], "autonomous")
        self.assertEqual(self.run_ops([self.sp()], [["system", "s1", [MAIN]]]), [[self.TESTED]])    # the prompt plugin alone: the same
        self.assertEqual(self.run_ops([self.sp(), self.im("bogus")], [["system", "s1", [MAIN]]]), [[self.TESTED]])   # unknown default: autonomous
        keys = self.run_ops([self.im("autonomous")], [["keys"]])[0][0]            # nothing that changes messages, tools or parameters
        self.assertEqual(keys, sorted(["config", "event", "chat.message", "command.execute.before", "tool.execute.before",
                                       "tool.execute.after", "experimental.chat.system.transform"]))

    def test_the_other_modes_replace_only_the_questions_section(self):
        for mode, first in (("interactive", "- The user is at the keyboard and can answer questions."),
                            ("plan-first", "- The user is here now but will be away while you work.")):
            got = self.run_ops([self.sp(), self.im(mode)], [["system", "s1", [MAIN]], ["system", "s1", [TITLE]]])
            text = got[0][0]
            sec = self.section(text)
            self.assertTrue(sec.startswith("# Questions\n" + first), (mode, sec))
            self.assertNotIn("Work without asking the user anything", text, mode)
            start = self.TESTED.index(self.AUTO)
            self.assertEqual(text[:start] + self.AUTO + text[start + len(sec):], self.TESTED, mode)    # only the section differs
            self.assertIn("List the assumptions you made, and any questions that blocked you", text)   # the final-message line stays
            self.assertIn(AGENTS, text)                                                                 # AGENTS.md's own section untouched
            self.assertEqual(got[1], [TITLE])
            self.assertIsNone(C.BANNED.search(sec))
        # opencode's own prompt (no Questions section of its own): the section goes at the end
        text = self.run_ops([self.im("interactive")], [["system", "s1", [MAIN]]])[0][0]
        self.assertTrue(text.startswith(MAIN + "\n\n# Questions\n- The user is at the keyboard"), text[-300:])

    def test_plan_first_window_ends_at_the_first_sign_of_work(self):
        def window(*ops):
            out = self.run_ops([self.sp(), self.im("plan-first")], [["chat", "s1", TYPED], ["key", "s1"], *ops, ["key", "s1"], ["system", "s1", [MAIN]]])
            self.assertEqual(out[1], "plan-first")
            return out[-2], out[-1][0]
        bash = lambda c: ["before", "s1", "bash", {"command": c}]
        for ends in ([["before", "s1", t]] for t in ("edit", "write", "patch", "apply_patch", "file_tasks", "add_task")):
            self.assertEqual(window(*ends), ("autonomous", self.TESTED), ends)    # then the tested text again
        for c in ("echo x > a.txt", "cat > a.py <<'EOF'\nx = 1\nEOF", "sed -i 's/a/b/' f", "rm -f a", "mkdir -p d", "git commit -m x",
                  "git checkout -- f", "cat a | tee b", "npm install", "pip install x", "printf hi >> log"):
            self.assertEqual(window(bash(c))[0], "autonomous", c)
        for c in ("ls -la", "cat a.txt", "grep -n x f 2>&1", "git status", "git diff", "npm test", "ls > /dev/null", 'python3 -c "print(1 > 2)"',
                  "echo 'a > b'", "pytest -q"):
            self.assertEqual(window(bash(c))[0], "plan-first", c)
        for t in ("read", "grep", "glob", "todowrite", "task"):
            self.assertEqual(window(["before", "s1", t])[0], "plan-first", t)
        self.assertEqual(window(["before", "s1", "question"], ["after", "s1", "question"])[0], "autonomous")   # a question-tool answer
        self.assertEqual(window(["idle", "s1"], ["chat", "s1", [{"type": "text", "text": "Use option A"}]]), ("autonomous", self.TESTED))
        self.assertEqual(window(["chat", "s1", [{"type": "text", "text": "Also this"}]])[0], "plan-first")   # before any reply: still planning
        self.assertEqual(window(["idle", "s1"], ["chat", "s1", [{"type": "text", "text": "Continue with the task.", "synthetic": True}]])[0], "plan-first")
        planning = self.run_ops([self.sp(), self.im("plan-first")], [["system", "s1", [MAIN]]])[0][0]
        self.assertIn("ask all your questions in one round", self.section(planning))

    def test_the_question_tool_follows_the_mode(self):
        q = lambda sid: ["before", sid, "question"]
        out = self.run_ops([self.im("autonomous")], [q("s1")])
        self.assertEqual(out, [REFUSED])
        self.assertEqual(self.run_ops([self.im("interactive")], [["chat", "s1", TYPED], q("s1"), ["before", "s1", "write"], q("s1")]), [None, None, None, None])
        out = self.run_ops([self.im("plan-first")], [["chat", "s1", TYPED], q("s1"), ["before", "s1", "edit"], q("s1"), ["calls"]])
        self.assertEqual(out, [None, None, None, REFUSED, []])                    # after the window: refused, the tool list left alone
        # taken out of the tool list for an autonomous session (a session rule), once; put back only if this plugin took it out
        out = self.run_ops([self.im("autonomous")], [["config", {}], ["chat", "s1", TYPED], ["chat", "s1", TYPED], ["calls"],
                                                    ["command", "interactive", "s1", [{"type": "text", "text": "t"}]], ["calls"],
                                                    ["command", "autonomous", "s1", [{"type": "text", "text": "t"}]], ["calls"]])
        self.assertEqual(out[3], [["s1", "question", "deny"]])
        self.assertEqual(out[5], [["s1", "question", "deny"], ["s1", "question", "allow"]])
        self.assertEqual(out[7], [["s1", "question", "deny"], ["s1", "question", "allow"], ["s1", "question", "deny"]])
        run_session = {"r1": [{"permission": "question", "pattern": "*", "action": "deny"}]}     # as `opencode run` makes it
        out = self.run_ops([self.im("autonomous")], [["config", {}], ["chat", "r1", TYPED], ["command", "interactive", "r1", [{"type": "text", "text": "t"}]], ["calls"]], run_session)
        self.assertEqual(out[-1], [])                                             # already out; its own rule is never undone
        self.assertEqual(self.run_ops([self.im("interactive")], [["chat", "s1", TYPED], ["calls"]])[-1], [])
        self.assertEqual(self.run_ops([self.im("autonomous")], [["event", {"type": "session.created", "properties": {"info": {"id": "k", "parentID": "s1"}}}],
                                                               ["chat", "k", TYPED], q("k"), ["calls"]])[2:], [None, []])   # sub-agent sessions left alone

    def test_the_three_commands(self):
        cfg = self.run_ops([self.im("autonomous")], [["config", {"command": {"mine": {"template": "m"}}}]])[0]["command"]
        self.assertEqual(sorted(cfg), ["autonomous", "interactive", "mine", "plan-first"])
        part = lambda: [{"type": "text", "text": "template text"}]
        ops = [["config", {}], ["chat", "s1", TYPED]]
        for name in ("interactive", "plan-first", "autonomous", "plan-first"):
            ops += [["command", name, "s1", part()], ["key", "s1"], ["system", "s1", [MAIN]]]
        ops += [["before", "s1", "write"], ["key", "s1"], ["command", "init", "s1", part()], ["key", "s2"]]
        out = self.run_ops([self.sp(), self.im("autonomous")], ops)
        for i, (name, mode) in enumerate((("interactive", "interactive"), ("plan-first", "plan-first"), ("autonomous", "autonomous"), ("plan-first", "plan-first"))):
            cmd, key, system = out[2 + 3 * i: 5 + 3 * i]
            self.assertTrue(cmd["same"], name)                                    # opencode keeps using its own array
            self.assertEqual(len(cmd["parts"]), 1)
            self.assertEqual((cmd["parts"][0]["type"], cmd["parts"][0]["synthetic"]), ("text", True))   # a task driver skips it
            self.assertNotIn("\n", cmd["parts"][0]["text"])
            self.assertEqual(key, mode)
            self.assertEqual(system[0] == self.TESTED, mode == "autonomous", name)
        self.assertEqual(out[-4:], [None, "autonomous", {"same": True, "parts": part()}, None])   # the reopened window closes; other commands untouched; another session unknown until seen
        # a person's own command of the same name is theirs: left alone
        out = self.run_ops([self.im("autonomous")], [["config", {"command": {"interactive": {"template": "mine"}}}],
                                                    ["command", "interactive", "s1", part()], ["key", "s1"]])
        self.assertEqual((out[0]["command"]["interactive"], out[1]["parts"], out[2]), ({"template": "mine"}, part(), None))


@unittest.skipUnless(NODE, "node is not installed")
class TestModeAndDriver(unittest.TestCase):
    """The task driver 0.4.2 reads the mode the interaction-mode plugin publishes (kit.json's order: the driver, then the
    mode plugin). Interactive: her typed answer joins the request; autonomous: 0.4.1's handling (a new request)."""
    def tasks(self, mode):
        drv = [os.path.join(KIT, "plugins", "opencode-task-driver-0.4.2.js"), {"mode": "focus"}]
        im = [os.path.join(KIT, "plugins", "interaction-mode.js"), {"default": mode}]
        ops = [["chat", "s1", [{"type": "text", "text": "Fix the two checkout bugs: totals and the coupon field"}]],
               ["chat", "s1", [{"type": "text", "text": "Use the store currency for totals"}]], ["read", "s1.json"], ["key", "s1"]]
        out = TestModePlugin.run_ops(self, [drv, im], ops)
        return out[2]["tasks"], out[3]

    def test_an_answer_joins_the_request_when_she_is_there(self):
        tasks, key = self.tasks("interactive")
        self.assertEqual(key, "interactive")
        self.assertEqual(len(tasks), 1)
        self.assertIn("Use the store currency for totals", tasks[0]["details"])
        self.assertEqual(tasks[0]["status"], "open")
        tasks, key = self.tasks("autonomous")
        self.assertEqual(key, "autonomous")
        self.assertEqual([t["status"] for t in tasks], ["done", "open"])                # 0.4.1: the message is a new request


class TestDefaultSetup(unittest.TestCase):
    """v1.2 + the task driver 0.4.2 is the default setup; v1.2 + the compaction plugin is the lighter one, offered at each
    start. An answer saved while the other setup was the default no longer counts."""
    DRIVER_SHA = "34faa56d695c973c9e7d55b53d6b20d42718efa24724a6d41a4edb038efe40d0"

    def kit(self):
        with open(os.path.join(KIT, "kit.json")) as f: return json.load(f)

    def test_the_driver_setup_is_the_default(self):
        k = self.kit()
        d = next(s for s in k["setups"] if s.get("default"))
        self.assertEqual((d["id"], k["setup"]), ("task-driver", d["name"]))
        files = [p["file"] for p in d["plugins"]]
        self.assertEqual(files, ["plugins/system-prompt.js", "plugins/opencode-task-driver-0.4.2.js", "plugins/interaction-mode.js"])
        sp, drv = d["plugins"][0], d["plugins"][1]
        self.assertNotIn("continueText", sp["options"])                              # the driver handles the turn after a summary
        self.assertEqual((drv["sha256"], drv["options"]), (self.DRIVER_SHA, {"mode": "focus"}))
        light = next(s for s in k["setups"] if not s.get("default"))
        self.assertIn("much more reliable on complex tasks, but can take much longer", light["ask"])
        self.assertIn("plugins/compaction-anchor.js", [p["file"] for p in light["plugins"]])
        self.assertFalse(os.path.exists(os.path.join(KIT, "plugins", "opencode-task-driver-0.4.1.js")))
        cfg = json.loads(L.session_config({"models": {}}, None, "autonomous"))
        self.assertTrue(any(f.endswith("/plugins/opencode-task-driver-0.4.2.js") for f, o in cfg["plugin"]))

    def test_an_answer_from_before_the_switch_no_longer_counts(self):
        class TTY:
            def isatty(self): return True
        asked, answers, saved = [], [], []
        def fake_yes(q, default=True):
            asked.append(default); v = answers.pop(0); return default if v is None else v
        st = {"setup": "standard"}                                 # saved under the old default (the driver was off)
        with mock.patch.object(L.sys, "stdin", TTY()), mock.patch.object(L, "yes", fake_yes), \
             mock.patch.object(L, "save_state", lambda x: saved.append(dict(x))), mock.patch("builtins.print"):
            answers[:] = [None]; self.assertEqual(L.choose_setup(st), "task-driver")      # Enter: the new default
            answers[:] = [True]; self.assertEqual(L.choose_setup(st), "standard")         # she picks the lighter one
            answers[:] = [None]; self.assertEqual(L.choose_setup(st), "standard")         # remembered now
        self.assertEqual(asked, [False, False, True])
        self.assertEqual(saved[-1], {"setup": "standard", "setup_default": "task-driver"})
        self.assertEqual(L.choose_setup({"setup": "standard"}), "task-driver")               # no terminal: the same rule
        self.assertEqual(L.choose_setup({"setup": "standard", "setup_default": "task-driver"}), "standard")


class TestLauncherMode(unittest.TestCase):
    """The launcher asks how the agent should work at each start (remembered), and starts the mode plugin in that mode in
    every setup; a fresh install starts autonomous, the tested mode."""
    PROV = {"npm": "@ai-sdk/openai-compatible", "models": {}}

    def plugin_opts(self, sid, mode):
        cfg = json.loads(L.session_config(self.PROV, sid, mode))
        return [o for f, o in cfg["plugin"] if f.endswith("/plugins/interaction-mode.js")]

    def test_every_setup_gets_the_plugin_with_the_chosen_mode(self):
        for sid in ("standard", "task-driver"):
            for mode in ("autonomous", "plan-first", "interactive"):
                self.assertEqual(self.plugin_opts(sid, mode), [{"default": mode}], (sid, mode))
            self.assertEqual(self.plugin_opts(sid, None), [{"default": "autonomous"}])    # no answer: kit.json's default
        self.assertEqual(L.kit_mode_default(), "autonomous")

    def test_the_question_and_its_remembered_answer(self):
        class TTY:
            def isatty(self): return True
        asked, answers, saved, printed = [], [], [], []
        def fake_ask(q, default=None):
            asked.append((q, default)); v = answers.pop(0); return v if v else default
        st = {}
        with mock.patch.object(L.sys, "stdin", TTY()), mock.patch.object(L, "ask", fake_ask), mock.patch.object(L, "save_state", lambda s: saved.append(dict(s))), \
             mock.patch("builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a)))):
            for given, want in (([""], "autonomous"), (["p"], "plan-first"), ([""], "plan-first"), (["i"], "interactive"),
                                (["plan first"], "plan-first"), (["maybe", "A"], "autonomous")):
                answers[:] = given; self.assertEqual(L.choose_mode(st), want, given)
        text = "\n".join(printed)
        for line in ("How should the agent work?", "(a) as autonomously as possible: good for long or overnight sessions",
                     "lists its assumptions and any blocking questions at the end", "(p) plan first: it asks its questions up front, then works on its own (untested)",
                     "(i) interactive: it asks its questions up front and as they come up (untested)"):
            self.assertIn(line, text)
        self.assertEqual([d for _, d in asked], ["a", "a", "p", "p", "i", "p", "p"])        # the last answer is the next default
        self.assertEqual(saved[-1]["mode"], "autonomous")
        with mock.patch.object(L, "ask", lambda *a, **k: self.fail("asked without a terminal")):
            self.assertEqual(L.choose_mode({"mode": "plan-first"}), "plan-first")        # no terminal: the remembered mode
            self.assertEqual(L.choose_mode({"mode": "bogus"}), "autonomous")
        self.assertIsNone(L.choose_mode({}, {"setups": [{"id": "s", "default": True, "plugins": [{"file": "plugins/system-prompt.js"}]}]}))

    def test_the_mode_reaches_opencode_and_the_printed_setup_line(self):
        with open(os.path.join(KIT, "llm-launch.py")) as f: src = f.read()
        self.assertEqual(src.count("session_config(prov, sid, mode)"), 2)               # the launch and --attach
        self.assertEqual(src.count("mode = choose_mode(st)"), 2)
        self.assertEqual([L.mode_label(m) for m in ("autonomous", "plan-first", "interactive", None)],
                         [", autonomous", ", plan first (untested)", ", interactive (untested)", ""])


if __name__ == "__main__":
    unittest.main()
