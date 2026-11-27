"""Unit tests for the launcher and llm-tune (no GPU, no network, no model server): python3 -m unittest discover -s tests

They use tiny synthetic GGUF files (a real header, sparse data) and stand-ins for nvidia-smi, /proc/meminfo and
llama-server, so they run anywhere. The interactive plugin's tests run it under node (skipped when node is missing)."""
import importlib.util, json, os, shutil, struct, subprocess, sys, tempfile, unittest
from unittest import mock

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(KIT, file))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


T = load("llm_tune", "llm-tune.py")
L = T.L            # llm-tune's own copy of the launcher: patch this one so both see it
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
    TD = {"file": "plugins/opencode-task-driver-0.4.1.js", "options": {"mode": "focus"}}

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
            {"name": "driver", "file": "plugins/opencode-task-driver-0.4.1.js"}]}]}

    def run_with(self, missing, sid):
        have = lambda p, quiet=False: None if p["file"] in missing else "/x/" + p["file"]
        with mock.patch.object(L, "plugin_path", have), mock.patch("builtins.print"):
            return L.usable_setup(sid, self.KIT_JSON)

    def test_the_chosen_setup_is_kept_when_its_plugins_load(self):
        self.assertEqual(self.run_with(set(), "task-driver"), "task-driver")
        self.assertEqual(self.run_with({"plugins/system-prompt.js"}, "standard"), "standard")   # the prompt alone may be missing

    def test_a_missing_keeper_switches_to_the_other_setup(self):
        self.assertEqual(self.run_with({"plugins/opencode-task-driver-0.4.1.js"}, "task-driver"), "standard")
        self.assertEqual(self.run_with({"plugins/compaction-anchor.js"}, "standard"), "task-driver")

    def test_neither_keeper_stops_before_a_session(self):
        with self.assertRaises(SystemExit):
            self.run_with({"plugins/compaction-anchor.js", "plugins/opencode-task-driver-0.4.1.js"}, "standard")

    def test_a_prompt_with_a_trailing_newline_is_refused(self):
        with tempfile.TemporaryDirectory() as t, mock.patch.object(C, "KIT", t):
            os.makedirs(os.path.join(t, "prompts")); os.makedirs(os.path.join(t, "plugins"))
            for f, text in (("prompts/p.md", "A prompt.\n"), ("plugins/system-prompt.js", "x")):
                with open(os.path.join(t, f), "w") as fh: fh.write(text)
            import hashlib
            p = {"name": "sp", "file": "plugins/system-prompt.js", "sha256": hashlib.sha256(b"x").hexdigest(), "options": {"file": "{kit}/prompts/p.md"}}
            out = []; C.check_plugin(p, False, out, {})
            self.assertTrue(any("whitespace" in x for x in out), out)


class TestCheckKitInteractive(unittest.TestCase):
    """The untested interactive plugin rides along in both tested sets, only with its "untested" marker and off by default;
    it never stands in for the reinsert part or the task driver, and other plugins stay refused."""
    A, SP_CONT, SP, TD = TestCheckKit.A, TestCheckKit.SP_CONT, TestCheckKit.SP, TestCheckKit.TD
    IA = {"name": "ia", "file": "plugins/interactive.js", "untested": True, "options": {"default": "autonomous"}}

    def bad(self, *plugins):
        out = []; C.check_setup({"id": "x", "plugins": list(plugins)}, out); return out

    def test_both_tested_sets_pass_with_it(self):
        self.assertEqual(self.bad(self.A, self.SP_CONT, self.IA), [])
        self.assertEqual(self.bad(self.SP, self.TD, self.IA), [])

    def test_it_needs_the_untested_marker_and_autonomous_as_default(self):
        for ia in (dict(self.IA, untested=None), {k: v for k, v in self.IA.items() if k != "untested"}, dict(self.IA, untested="yes"),
                   dict(self.IA, options={"default": "interactive"}), dict(self.IA, options={})):
            self.assertTrue(self.bad(self.A, self.SP_CONT, ia), ia)

    def test_the_old_bad_combinations_stay_refused_with_it(self):
        slim = {"file": "plugins/compaction-anchor.js", "options": {"parts": ["slim"]}}
        nsa = {"file": "plugins/compaction-anchor.js", "options": {"parts": ["checklist", "no-stop-and-ask"]}}
        for combo in [(slim, self.SP), (self.SP_CONT, self.TD), (self.A, self.SP, self.TD), (self.A, self.TD), (nsa, self.SP_CONT),
                      (self.SP, self.SP_CONT), (dict(self.TD, options={"mode": "tasks"}), self.SP), (self.SP, {"file": "plugins/something-else.js"}),
                      ({"file": "plugins/compaction-anchor.js"}, self.SP_CONT), (self.A, self.SP_CONT, self.IA, self.IA)]:
            self.assertTrue(self.bad(*combo, self.IA), [p.get("options") for p in combo])

    def test_it_is_not_the_reinsert_part_or_a_task_driver(self):
        for combo in [(self.IA,), (self.SP_CONT, self.IA), (self.SP, self.IA)]:
            out = self.bad(*combo)
            self.assertTrue(any("reinsert" in x and "task driver" in x for x in out), (combo, out))

    def test_the_untested_marker_does_not_let_other_plugins_in(self):
        out = self.bad(self.A, self.SP_CONT, {"name": "x", "file": "plugins/something-else.js", "untested": True, "options": {"default": "autonomous"}})
        self.assertTrue(any("no tested setup used" in x for x in out), out)

    def test_the_kit_has_it_in_every_setup_off_by_default(self):
        with open(os.path.join(KIT, "kit.json")) as f: kit = json.load(f)
        for s in kit["setups"]:
            ia = [p for p in s["plugins"] if p.get("file") == "plugins/interactive.js"]
            self.assertEqual(len(ia), 1, s["id"])
            self.assertEqual((ia[0]["untested"], ia[0]["options"]), (True, {"default": "autonomous"}))
            self.assertIs(s["plugins"][-1], ia[0])      # last: after the task driver's planning block, whose filing it asks about

    def test_its_file_is_checked_for_the_banned_words(self):
        """The plugin's text goes to the model and the repo is public: the whole file, comments included, is checked."""
        import hashlib
        with tempfile.TemporaryDirectory() as t, mock.patch.object(C, "KIT", t):
            os.makedirs(os.path.join(t, "plugins"))
            body = b"// built for the benchmark\nexport const X = async () => ({});\n"
            with open(os.path.join(t, "plugins", "interactive.js"), "wb") as f: f.write(body)
            out = []; C.check_plugin(dict(self.IA, sha256=hashlib.sha256(body).hexdigest()), False, out, {})
            self.assertTrue(any("mentions the benchmark" in x for x in out), out)
        with open(os.path.join(KIT, "kit.json")) as f:
            real = next(p for p in json.load(f)["setups"][0]["plugins"] if p["file"] == "plugins/interactive.js")
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
# Loads plugins as opencode does (each export called with (input, options)) and replays opencode's hook calls.
PLUGIN_RUNNER = r"""
import fs from "node:fs"; import path from "node:path"; import { pathToFileURL } from "node:url";
const spec = JSON.parse(fs.readFileSync(process.argv[1], "utf8"));
const hooks = [];
for (const [file, opts] of spec.plugins) {
  const dst = path.join(spec.tmp, hooks.length + "-" + path.basename(file).replace(/\\.js$/, ".mjs"));   // .mjs: an ES module on any node
  fs.copyFileSync(file, dst);
  for (const f of Object.values(await import(pathToFileURL(dst).href))) hooks.push(await f({ directory: spec.tmp }, opts));
}
const each = async (name, input, output) => { for (const h of hooks) if (h[name]) await h[name](input, output); return output; };
const out = [];
for (const [op, ...a] of spec.ops) {
  if (op === "keys") out.push(hooks.map((h) => Object.keys(h).sort()));
  else if (op === "config") { for (const h of hooks) if (h.config) await h.config(a[0]); out.push(a[0]); }
  else if (op === "event") { await each("event", { event: a[0] }, undefined); out.push(null); }
  else if (op === "system") out.push((await each("experimental.chat.system.transform", { sessionID: a[0], model: {} }, { system: a[1] })).system);
  else if (op === "command") { const parts = a[2], o = { parts }; await each("command.execute.before", { command: a[0], sessionID: a[1], arguments: "" }, o); out.push({ same: o.parts === parts, parts }); }
}
console.log(JSON.stringify(out));
"""
# opencode 1.18.34's system message for the main agent (its own prompt, the environment lines, the skills), the title
# writer's and the summarizer's (their own prompt only)
MAIN = ("You are opencode, an interactive CLI tool that helps users with software engineering tasks.\n\nBe concise.\n"
        "You are powered by the model named m. The exact model ID is llama-local/m\n"
        "Here is some useful information about the environment you are running in:\n<env>\n  Working directory: /p\n</env>\n"
        "Skills provide specialized instructions and workflows for specific tasks.")
TITLE = "You are a title generator. You output ONLY a thread title. Nothing else."
SUMMARY = "You are a helpful AI assistant tasked with summarizing conversations."


@unittest.skipUnless(NODE, "node is not installed")
class TestInteractivePlugin(unittest.TestCase):
    """plugins/interactive.js (untested mode): off, it changes nothing the model sees; on, it appends its block to the
    main agent's one system message; /interactive and /autonomous switch the session."""
    IA = os.path.join(KIT, "plugins", "interactive.js")
    SP = os.path.join(KIT, "plugins", "system-prompt.js")

    def run_ops(self, plugins, ops):
        with tempfile.TemporaryDirectory() as t:
            plugins = [[f, dict(o, dir=t)] for f, o in plugins]          # their event logs go here, not under ~
            spec = os.path.join(t, "spec.json")
            with open(spec, "w") as f: json.dump({"tmp": t, "plugins": plugins, "ops": ops}, f)
            r = subprocess.run([NODE, "--no-warnings", "--input-type=module", "-e", PLUGIN_RUNNER, spec], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(r.stdout)

    def ia(self, default): return [self.IA, {"default": default}]
    def sp(self): return [self.SP, {"file": os.path.join(KIT, "prompts", "v1-1.md"), "continueText": "Continue with the task."}]

    def block(self):
        main = self.run_ops([self.ia("interactive")], [["system", "s1", [MAIN]]])[0][0]
        self.assertTrue(main.startswith(MAIN + "\n\n"))
        return main[len(MAIN) + 2:]

    def test_off_adds_nothing(self):
        ops = [["config", {}], ["system", "s1", [MAIN]], ["system", "s1", [TITLE]], ["system", "s1", [SUMMARY]],
               ["command", "autonomous", "s1", [{"type": "text", "text": "x"}]], ["system", "s1", [MAIN]], ["system", None, [MAIN]]]
        out = self.run_ops([self.ia("autonomous")], ops)
        self.assertEqual(out[1:4], [[MAIN], [TITLE], [SUMMARY]])
        self.assertEqual(out[5:], [[MAIN], [MAIN]])
        self.assertEqual(self.run_ops([self.ia("bogus")], [["system", "s1", [MAIN]]]), [[MAIN]])      # an unknown default: off
        # with the v1.1 prompt plugin, in kit.json's order: exactly what the prompt plugin alone sends
        alone = self.run_ops([self.sp()], [["system", "s1", [MAIN]], ["system", "s1", [TITLE]]])
        self.assertEqual(self.run_ops([self.sp(), self.ia("autonomous")], [["system", "s1", [MAIN]], ["system", "s1", [TITLE]]]), alone)
        self.assertNotEqual(alone[0], [MAIN])                                                       # (the prompt plugin did act)
        # no hook that could change messages, tools or parameters
        self.assertEqual(self.run_ops([self.ia("autonomous")], [["keys"]])[0][0],
                         sorted(["config", "event", "command.execute.before", "experimental.chat.system.transform"]))

    def test_on_appends_the_block(self):
        block = self.block()
        self.assertTrue(block.startswith("<interaction>\n") and block.endswith("\n</interaction>"), block)
        self.assertLess(len(block), 1200)
        self.assertIsNone(C.BANNED.search(block))
        ops = [["system", "s1", [MAIN]], ["system", "s1", [MAIN]], ["system", "s1", [TITLE]], ["system", "s1", [SUMMARY]], ["system", None, [MAIN]],
               ["event", {"type": "session.created", "properties": {"info": {"id": "child", "parentID": "s1"}}}], ["system", "child", [MAIN]]]
        out = self.run_ops([self.ia("interactive")], ops)
        self.assertEqual(out[0], [MAIN + "\n\n" + block])                       # one system message, the block at its end
        self.assertEqual(out[1], out[0])                                          # each request once
        self.assertEqual(out[2:5], [[TITLE], [SUMMARY], [MAIN]])                  # not the title writer, the summarizer, no session
        self.assertEqual(out[6], [MAIN])                                          # not a sub-agent's session
        # after the v1.1 prompt plugin: its text, then the block
        alone = self.run_ops([self.sp()], [["system", "s1", [MAIN]]])[0][0]
        self.assertEqual(self.run_ops([self.sp(), self.ia("interactive")], [["system", "s1", [MAIN]]])[0], [alone + "\n\n" + block])

    def test_the_commands_switch_the_session(self):
        cfg = self.run_ops([self.ia("autonomous")], [["config", {"command": {"mine": {"template": "m"}}}]])[0]["command"]
        self.assertEqual(sorted(cfg), ["autonomous", "interactive", "mine"])
        self.assertTrue(all(cfg[k]["template"] and cfg[k]["description"] for k in ("autonomous", "interactive")))
        block = self.block()
        part = lambda: [{"type": "text", "text": "template text"}]
        ops = [["config", {}], ["system", "s1", [MAIN]],
               ["command", "interactive", "s1", part()], ["system", "s1", [MAIN]], ["system", "s2", [MAIN]],
               ["command", "autonomous", "s1", part()], ["system", "s1", [MAIN]],
               ["command", "init", "s1", part()]]
        out = self.run_ops([self.ia("autonomous")], ops)
        self.assertEqual(out[1], [MAIN])
        on = out[2]
        self.assertTrue(on["same"])                                               # opencode keeps using its own array
        self.assertEqual(len(on["parts"]), 1)
        self.assertEqual((on["parts"][0]["type"], on["parts"][0]["synthetic"]), ("text", True))   # a task driver skips it
        self.assertIn("interactive", on["parts"][0]["text"]); self.assertNotIn("\n", on["parts"][0]["text"])
        self.assertEqual(out[3], [MAIN + "\n\n" + block])                       # switched on for this session
        self.assertEqual(out[4], [MAIN])                                          # not for another one
        self.assertIn("autonomous", out[5]["parts"][0]["text"])
        self.assertEqual(out[6], [MAIN])                                          # off again
        self.assertEqual(out[7]["parts"], part())                                 # other commands untouched
        # on by default, /autonomous turns it off for the session
        out = self.run_ops([self.ia("interactive")], [["config", {}], ["command", "autonomous", "s1", part()], ["system", "s1", [MAIN]], ["system", "s2", [MAIN]]])
        self.assertEqual(out[2:], [[MAIN], [MAIN + "\n\n" + block]])
        # a person's own command of the same name is theirs: left alone
        out = self.run_ops([self.ia("autonomous")], [["config", {"command": {"interactive": {"template": "mine"}}}],
                                                     ["command", "interactive", "s1", part()], ["system", "s1", [MAIN]]])
        self.assertEqual((out[0]["command"]["interactive"], out[1]["parts"], out[2]), ({"template": "mine"}, part(), [MAIN]))


class TestLauncherMode(unittest.TestCase):
    """The launcher asks interactive or autonomous at each start (remembered), and starts the plugin in that mode in every
    setup; a fresh install starts autonomous, the tested behaviour."""
    PROV = {"npm": "@ai-sdk/openai-compatible", "models": {}}

    def plugin_opts(self, sid, mode):
        cfg = json.loads(L.session_config(self.PROV, sid, mode))
        return [o for f, o in cfg["plugin"] if f.endswith("/plugins/interactive.js")]

    def test_every_setup_gets_the_plugin_with_the_chosen_mode(self):
        for sid in ("standard", "task-driver"):
            for mode in ("interactive", "autonomous"):
                self.assertEqual(self.plugin_opts(sid, mode), [{"default": mode}], (sid, mode))
            self.assertEqual(self.plugin_opts(sid, None), [{"default": "autonomous"}])    # no answer: kit.json's default
        self.assertEqual(L.kit_mode_default(), "autonomous")

    def test_the_question_and_its_remembered_answer(self):
        class TTY:
            def isatty(self): return True
        asked, answers, saved = [], [], []
        def fake_ask(q, default=None):
            asked.append((q, default)); v = answers.pop(0); return v if v else default
        st = {}
        with mock.patch.object(L.sys, "stdin", TTY()), mock.patch.object(L, "ask", fake_ask), mock.patch.object(L, "save_state", lambda s: saved.append(dict(s))), \
             mock.patch("builtins.print"):
            answers[:] = [""]; self.assertEqual(L.choose_mode(st), "autonomous")            # a fresh install: autonomous
            answers[:] = ["i"]; self.assertEqual(L.choose_mode(st), "interactive")
            answers[:] = [""]; self.assertEqual(L.choose_mode(st), "interactive")           # remembered as the next default
            answers[:] = ["maybe", "Autonomous"]; self.assertEqual(L.choose_mode(st), "autonomous")
        self.assertIn("Will you be at the keyboard (interactive: it asks when it needs a decision) or leaving it to work "
                      "(autonomous: it won't stop to ask)?", asked[0][0])
        self.assertEqual([d for _, d in asked], ["autonomous", "autonomous", "interactive", "interactive", "interactive"])
        self.assertEqual(saved[-1]["mode"], "autonomous")
        with mock.patch.object(L, "ask", lambda *a, **k: self.fail("asked without a terminal")):
            self.assertEqual(L.choose_mode({"mode": "interactive"}), "interactive")        # no terminal: the remembered mode
        self.assertIsNone(L.choose_mode({}, {"setups": [{"id": "s", "default": True, "plugins": [{"file": "plugins/system-prompt.js"}]}]}))

    def test_the_mode_reaches_opencode_and_the_printed_setup_line(self):
        with open(os.path.join(KIT, "llm-launch.py")) as f: src = f.read()
        self.assertEqual(src.count("session_config(prov, sid, mode)"), 2)               # the launch and --attach
        self.assertEqual(src.count("mode = choose_mode(st)"), 2)
        self.assertEqual((L.mode_label("interactive"), L.mode_label("autonomous"), L.mode_label(None)), (", interactive (untested)", ", autonomous", ""))


if __name__ == "__main__":
    unittest.main()
