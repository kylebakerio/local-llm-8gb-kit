"""Unit tests for the launcher and llm-tune (no GPU, no network, no model server): python3 -m unittest discover -s tests

They use tiny synthetic GGUF files (a real header, sparse data) and stand-ins for nvidia-smi, /proc/meminfo and
llama-server, so they run anywhere."""
import importlib.util, json, os, struct, sys, tempfile, unittest
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

    def test_a_prompt_with_a_trailing_newline_is_refused(self):
        with tempfile.TemporaryDirectory() as t, mock.patch.object(C, "KIT", t):
            os.makedirs(os.path.join(t, "prompts")); os.makedirs(os.path.join(t, "plugins"))
            for f, text in (("prompts/p.md", "A prompt.\n"), ("plugins/system-prompt.js", "x")):
                with open(os.path.join(t, f), "w") as fh: fh.write(text)
            import hashlib
            p = {"name": "sp", "file": "plugins/system-prompt.js", "sha256": hashlib.sha256(b"x").hexdigest(), "options": {"file": "{kit}/prompts/p.md"}}
            out = []; C.check_plugin(p, False, out, {})
            self.assertTrue(any("whitespace" in x for x in out), out)


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
            self.make(os.path.join(t, "turboderp__Qwen3.8-27B-exl3__SCP_2.00bpw"))
            with open(os.path.join(t, "config.json"), "w") as f: json.dump({"model_type": "llama"}, f)      # not EXL3: ignored
            ms = [m for m in L.scan_models({}) if m.get("engine") == "exl3"]
            self.assertEqual(len(ms), 1)
            m = ms[0]
            self.assertTrue(m["exl3"]["mtp"]); self.assertEqual(m["exl3"]["bits"], 2.0)
            self.assertEqual(L.EXL3_RECOMMENDED[m["exl3"]["sig"]]["ctx"], 40960)

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
        self.assertEqual(len(ds), 2)
        for d in ds:
            names = {f[0] for f in d["files"]}
            self.assertTrue({"config.json", "model-00001-of-00002.safetensors", "tokenizer.json"} <= names)
            self.assertTrue(all(len(f[2]) == 64 for f in d["files"]))
            w = next(f for f in d["files"] if f[0] == "model-00001-of-00002.safetensors")
            self.assertIn(f"{w[0]}:{w[1]}", L.EXL3_RECOMMENDED)
            with tempfile.TemporaryDirectory() as t, mock.patch.object(L, "MODELS_DIR", t):
                self.assertEqual(len(L.missing_files(d)), len(d["files"]))


if __name__ == "__main__":
    unittest.main()
