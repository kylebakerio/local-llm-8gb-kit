#!/usr/bin/env python3
"""llm-launch: run a local model on an 8 GB NVIDIA laptop GPU and open opencode on it.

    python3 llm-launch.py            (run it from the project folder you want opencode to work in)
    python3 llm-launch.py --update         get the latest kit from GitHub (new plugins, a new tested setup); the launcher
                                           also checks at each start and asks (--no-update skips that)
    python3 llm-launch.py --fetch-plugins  get the plugins kit.json names (setup.sh does this; so does each launch)
    python3 llm-launch.py --download       the download menu (the models this kit tested, pinned and checked by sha256)
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
time, the model's own recommended sampling. A mixture-of-experts model bigger than the card (Gemma 4 26B-A4B, Qwen 3.6 /
Ornith 1.5 35B-A3B) keeps the experts of its first layers in system RAM (-ot ...=CPU), the layout that was tested.
opencode gets the plugins and settings kit.json names, for these sessions only. Today (kit.json "setups"):
  - standard (the default): the v1.1 system prompt in place of opencode's own instructions (plugins/system-prompt.js +
    prompts/v1-1.md; your project's AGENTS.md is kept), "Continue with the task." after a summary, and two parts of the
    compaction-anchor plugin: after a summary, the original request word for word and a task checklist.
  - task driver (asked at each start, off unless you turn it on): opencode-task-driver 0.4.1 in focus mode (a request with
    several problems is split into tasks, worked one at a time) with the v1.1 prompt.
A newer tested setup arrives as a kit update (--update).
EXL3 models (experimental): a folder whose config.json says quant_method "exl3" runs on TabbyAPI + exllamav3 instead
(install-exl3.sh installs them into ~/exl3); the launcher writes TabbyAPI's config.yml with the lab's settings.
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
    # not measured, inferred: the same Bonsai 2 9B weights, packed between the tested PTQ1_0 (4.6 GB used at 262k) and the
    # PQ2_0-MTP file (5.0 GB used at 262k with MTP off), so 262k leaves ~3 GB spare on an 8 GB card
    "Ternary-Bonsai-2-9B-PQ2_0.gguf":          {"mtp_off": 262144, "mib_per_1k": 9.77, "inferred": True},
    "Ternary-Bonsai-2-9B-Q2_0.gguf":           {"mtp_off": 262144, "mib_per_1k": 9.77, "inferred": True},
    # (Bonsai 2 27B was measured too, and dropped 11-25: Bonsai 2.5 27B does as well and adds images and MTP)
    # not measured, inferred: the uncensored edit of Bonsai 2.5 27B (kelsey1337's MTP repack), the same size and tensor
    # layout as the stock -mtp file, so the same numbers
    "Ternary-Bonsai-2.5-27B-Uncensored-PTQ1_0-mtp.gguf": {"mtp_off": 88064, "mtp_on": 41984, "mib_per_1k": 18.55, "inferred": True},
    # Mixture-of-experts models bigger than the card: the experts (blk.N.ffn_*_exps) of layers 0 .. cpu_layers-1 stay in
    # system RAM (-ot ...=CPU, read through mmap); the rest of the model and the K/V cache are on the GPU. Measured on the
    # test GPU at 88,064 tokens with MTP off: Gemma 4 26B-A4B 7,418 MiB peak (experts of layers 0-19 in RAM), Qwen 3.6 /
    # Ornith 1.5 35B-A3B 7,218 MiB (layers 0-24), all three under the 8 GB cap. Their K/V cache is small (about 5.5 MiB
    # per 1,000 tokens), so a smaller context frees little: the MTP heads' memory (measured: Qwen 1,126 MiB, Ornith
    # 720 MiB) is made room for by moving more layers' experts to RAM instead (Qwen layers 25-26: 634 MiB, Ornith layer
    # 25: 340 MiB). That MTP-on layout is "estimated" (worked out from the measured numbers, not run under the 8 GB cap):
    # llm-tune --fit measures it on this machine.
    "gemma-4-26B-A4B-it-UD-IQ4_XS.gguf":  {"mtp_off": 88064, "cpu_layers": {"mtp_off": 20}},
    "Qwen_Qwen3.6-35B-A3B-IQ3_XXS.gguf":  {"mtp_off": 88064, "mtp_on": 88064, "cpu_layers": {"mtp_off": 25, "mtp_on": 27},
                                           "estimated": ["mtp_on"]},
    "Ornith-1.5-35B-A3B-IQ3_XXS.gguf":    {"mtp_off": 88064, "mtp_on": 88064, "cpu_layers": {"mtp_off": 25, "mtp_on": 26},
                                           "estimated": ["mtp_on"]},
}
# the files the numbers above were measured with, per model (shown when a different file of that model is found)
TESTED_FILES = {"Ternary-Bonsai-2.5-27B": "PTQ1_0 (or PTQ1_0-mtp)", "Ternary-Bonsai-2-9B": "PTQ1_0 or PQ2_0-MTP-Q8_0",
                "Ornith-1.5-9B": "Q4_K_M", "Ornith-1.5-35B-A3B": "IQ3_XXS (bartowski)", "Qwen_Qwen3.6-35B-A3B": "IQ3_XXS (bartowski)",
                "gemma-4-26B-A4B-it": "UD-IQ4_XS (unsloth)"}
# Each model's own recommended sampling (thinking mode; Ornith: its "precise coding" set), from the model cards, as
# llama-server flags. The tests use exactly these. A model not listed gets llama.cpp's defaults plus whatever its GGUF
# file sets. Gemma 4 (Google's card) and Qwen 3.6 (its generation_config) leave min-p to llama.cpp's default, 0.05, as
# in the tests.
SAMPLING = {
    "Ternary-Bonsai-2.5-27B": "--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0",
    "Ternary-Bonsai-2-27B": "--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05 --presence-penalty 0",
    "Ternary-Bonsai-2-9B": "--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 1.0 --repeat-penalty 1.0 --repeat-last-n 64",
    "Ornith-1.5": "--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0",
    "Qwen_Qwen3.6-35B-A3B": "--temp 1.0 --top-p 0.95 --top-k 20",
    "gemma-4-26B-A4B-it": "--temp 1.0 --top-p 0.95 --top-k 64",
}
# MTP draft length: PrismML's preset for Bonsai 2.5 27B; 1 for the 35B-A3B models (fastest with their experts in RAM:
# Qwen wrote 39 t/s at 1 vs 32 at 2, Ornith 40 vs 29); 2 for the rest
DRAFT_N = {"Ternary-Bonsai-2.5-27B": 3, "Qwen_Qwen3.6-35B-A3B": 1, "Ornith-1.5-35B-A3B": 1}
CACHE_RAM_MIB = 2048      # LLAMA_ARG_CACHE_RAM: llama.cpp's prompt cache in system RAM (every tested run used 2048)
RAM_ANON_MIB = 1024       # the server's own working memory in system RAM, besides the weights it maps (0.6-1.0 GiB measured)
RAM_HEADROOM_MIB = 1024   # system RAM to leave free besides the server: opencode and the desktop
DISK_MARGIN = 5 * 10**9   # free disk to keep after a download
# models worth having, for the download menu: each pinned to a repository revision, every file checked by its sha256
# (files: name, bytes, sha256; the first is the model, then its image projector if any). new: offered once after a kit
# update brings it.
HF = "https://huggingface.co"
DOWNLOADS = [
    {"label": "Bonsai 2.5 27B PTQ1_0 (the 8 GB file) + image file", "repo": "prism-ml/Ternary-Bonsai-2.5-27B-GGUF",
     "rev": "28a7093cc25bc151cc2b284c5570f12ce343fdcf",
     "files": [("Ternary-Bonsai-2.5-27B-PTQ1_0.gguf", 5993834848, "929560ec9d37cf3cadd4b00cc87b51c570082a628dd830b4692dc0b2ed18dbbe"),
               ("Ternary-Bonsai-2.5-27B-mmproj-Q8_0.gguf", 629246976, "6807ede61d570bb86ba34b756a0fa109edc33668604de867c6ea6d8f1d631903")]},
    {"label": "Bonsai 2.5 27B PTQ1_0 with the MTP head + image file", "repo": "prism-ml/Ternary-Bonsai-2.5-27B-GGUF",
     "rev": "28a7093cc25bc151cc2b284c5570f12ce343fdcf",
     "files": [("Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf", 6232818944, "e0fc5f3d47b9c926d52677d303b19df356a2376614b3e1cb9973865fc04ca431"),
               ("Ternary-Bonsai-2.5-27B-mmproj-Q8_0.gguf", 629246976, "6807ede61d570bb86ba34b756a0fa109edc33668604de867c6ea6d8f1d631903")]},
    {"label": "Bonsai 2.5 27B Uncensored with the MTP head (kelsey1337's repack; being tested)",
     "repo": "kelsey1337/Ternary-Bonsai-2.5-27B-Uncensored-MTP-GGUF", "rev": "ad67f3e0770aefcdac514848ee3595d7edcdf7ad", "new": "2026-11-26",
     "files": [("Ternary-Bonsai-2.5-27B-Uncensored-PTQ1_0-mtp.gguf", 6232818944, "29331c5477242fad9f667cc428bcf18ab22f304ba6b02684bf9a37ae935e8e9c")]},
    {"label": "Gemma 4 26B-A4B UD-IQ4_XS (experts in RAM; being tested)", "repo": "unsloth/gemma-4-26B-A4B-it-GGUF",
     "rev": "c099eb48e663fd284577b04978a94ffccb261841", "new": "2026-11-26",
     "files": [("gemma-4-26B-A4B-it-UD-IQ4_XS.gguf", 13597177568, "babd1e389d386352f71600765d37390f7dc993fbfad6725caccf996ffe34aecf")]},
    {"label": "Qwen 3.6 35B-A3B IQ3_XXS (experts in RAM, MTP; being tested)", "repo": "bartowski/Qwen_Qwen3.6-35B-A3B-GGUF",
     "rev": "5c2410d71524f4f72b023ce8daf7a80528226d5f", "new": "2026-11-26",
     "files": [("Qwen_Qwen3.6-35B-A3B-IQ3_XXS.gguf", 15761446528, "4eb30af2df0dde76fe8d0a60f0e88f285ee668dcb1b8da892ee5c13dd6b00399")]},
    {"label": "Ornith 1.5 35B-A3B IQ3_XXS (experts in RAM, MTP; being tested)", "repo": "bartowski/Ornith-1.5-35B-A3B-GGUF",
     "rev": "64b0493d34a5ca4c1b4ad67bb99b41d74b4f07d6", "new": "2026-11-26",
     "files": [("Ornith-1.5-35B-A3B-IQ3_XXS.gguf", 15340447392, "8918ccb9ee29abe3875efec0c3e86f0e35ef0b8a03e3f0e1c422869859a518d0")]},
    {"label": "Qwen3.8-27B EXL3 2.00 bits per weight (turboderp; needs the EXL3 engine: experimental)", "kind": "dir",
     "repo": "turboderp/Qwen3.8-27B-exl3", "rev": "98f2f8dfab4cb73ed1ae3428a1671483ab05c490", "subdir": "turboderp__Qwen3.8-27B-exl3__SCP_2.00bpw", "needs": "exl3",
     "new": "2026-11-26",
     "files": [
               (".gitattributes", 1570, "34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930"),
               ("LICENSE", 11544, "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"),
               ("README.md", 65012, "57e4bdb258ee1a7d2635c5174ebd4e56abe392505cdb5f8bbb356b0dc4293641"),
               ("cal_trace.json", 3410836, "7a3436f016f3f1822bc5de1cdbb23797965c145411b3c5c600fb27812d87dd2d"),
               ("cal_trace.md", 2274406, "1356cd7eaa5eaf5eb4182529d925d5fa579eb28ac749032248a8e08632ccdcbc"),
               ("cal_trace.safetensors", 4096088, "a034bee4c7920b4ac2e62b9d881a9b118488c414a81cf89aa64bd9274c47d685"),
               ("chat_template.jinja", 9004, "af16f654968d47d09a6c604e00370b12dbc21f76456f59c9e98b82e84d9f6375"),
               ("config.json", 4645, "4dfda0f243b79870356ae053cc8c5961abb25b6bf028bf9fa7fb86dcbc4a231a"),
               ("crc32.txt", 238, "b42dd291f5f98b05807e458f5b47849969a9b3326acbf0c97d28b05826740b83"),
               ("generation_config.json", 202, "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e"),
               ("kld_table.json", 190470, "aae4c43ba9b95749bd8cd1c0eb92012830d23d95706350538a80319829325720"),
               ("measurement.json", 60916, "0ee08dec3537a438830198570b7dd492ee546b941901247b2d9654ed9deaaf6b"),
               ("merges.txt", 3353259, "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d"),
               ("model-00001-of-00002.safetensors", 8570690830, "b6dc6ac9660936a3b3431546085c64680d11f04c29aedcf5181d01cd3486df49"),
               ("model-00002-of-00002.safetensors", 1274185392, "a9c41b299c294b69d926efae202b4ee383e75fbfe8bf59fc336010c579c2e03f"),
               ("model.safetensors.index.json", 295619, "ca964393247aa41add1369e1c1fa657b254098f0b8bb72b602ac918b155ced90"),
               ("noise_attrib.json", 257094, "82c41c57a5970de325cdb5bc137a62b67fd0ba3bbe5be94f0d009099dd6182cd"),
               ("preprocessor_config.json", 390, "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"),
               ("qbench_prompts_div.json", 1920186, "4800f95d62e910fd098703b107e3bb26c376005bb2c53dfc7d3fe2323d1914f2"),
               ("qbench_prompts_div.md", 1314356, "0158b7fe7e790e187e334fbed7abe88740e7c6692c1757583a8c736c639de4b8"),
               ("qbench_prompts_gen.json", 344109, "6070481d810633355e264a618f8a3727af0fc135b3c3671c917928dc6dc215ed"),
               ("qbench_prompts_gen.md", 228736, "a5a75cf17d725e15440afe644f41a090c2b860fe02f5ea1e69bff3bcd750b3a4"),
               ("quantization_config.json", 629282, "1c29cdc8932a561ece5d47a19ebee78d7761eda3db315ff421d9a12de6e3c064"),
               ("recipe_1.40bpw.yaml", 21487, "395099be69595d56444d53b8d3f3ae05141cbd02814e07256ca8e487acc28dc7"),
               ("recipe_1.60bpw.yaml", 21487, "3ed1ef00984736488668efe456180ec3d1835a2009712aff2efe357c597f61e2"),
               ("recipe_1.80bpw.yaml", 21487, "5fd281e12bb5fa0107d93ae470ba1c4f5d1f8d495e37f4734656b0c80854aa03"),
               ("recipe_2.00bpw.yaml", 21490, "ef5ea53d7566b52638e6481ed7efedf0be26c868a4d50b5abb8e7ee08222f879"),
               ("recipe_2.20bpw.yaml", 21484, "fc08aed44aa0cc993b53aaff36cc98dba2a1aa81fc5055812568e1c086d0872d"),
               ("recipe_3.00bpw.yaml", 21487, "d977b2198b21721ae0e42667d89e2eb8107c15b8a8a3c3b16fe618d8f7e14353"),
               ("recipe_4.00bpw.yaml", 21487, "0b4a5b64580cbc3abf3d08d0bea1f526c50c6d5738c638f41288595069cec3b4"),
               ("recipe_5.00bpw.yaml", 21486, "1736b92c76728bb865fc38ed8a929dc6c21b077066fcb7815decfe20e5ccaad2"),
               ("recipe_6.00bpw.yaml", 21486, "a95a52fb59b3976ee615c02df6f68efa758af95def6a519c8a4848b4964cd9c4"),
               ("rfn_2.00bpw.json", 85427, "e66a8173c003f2b7e7fe89b7c10d6b0064a6ab1798078eedcf7d9b8555fa31c3"),
               ("tokenizer.json", 12809320, "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"),
               ("tokenizer_config.json", 17928, "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27"),
               ("video_preprocessor_config.json", 385, "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"),
               ("vocab.json", 6722759, "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003")]},
    {"label": "Qwen3.8-27B EXL3 1.75 bits per weight (turboderp; needs the EXL3 engine: experimental)", "kind": "dir",
     "repo": "turboderp/Qwen3.8-27B-exl3", "rev": "2c1a38c2de43eab08cc8fba99656dc488e698cfb", "subdir": "turboderp__Qwen3.8-27B-exl3__SCP_1.75bpw", "needs": "exl3",
     "new": "2026-11-26",
     "files": [
               (".gitattributes", 1570, "34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930"),
               ("LICENSE", 11544, "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"),
               ("README.md", 65012, "57e4bdb258ee1a7d2635c5174ebd4e56abe392505cdb5f8bbb356b0dc4293641"),
               ("cal_trace.json", 3410836, "7a3436f016f3f1822bc5de1cdbb23797965c145411b3c5c600fb27812d87dd2d"),
               ("cal_trace.md", 2274406, "1356cd7eaa5eaf5eb4182529d925d5fa579eb28ac749032248a8e08632ccdcbc"),
               ("cal_trace.safetensors", 4096088, "a034bee4c7920b4ac2e62b9d881a9b118488c414a81cf89aa64bd9274c47d685"),
               ("chat_template.jinja", 9004, "af16f654968d47d09a6c604e00370b12dbc21f76456f59c9e98b82e84d9f6375"),
               ("config.json", 4648, "2563697d3f21f1cea50f72f9c12f1bbeccc3f786d3a9bde4efe3ea0dde8ecd42"),
               ("crc32.txt", 238, "b42dd291f5f98b05807e458f5b47849969a9b3326acbf0c97d28b05826740b83"),
               ("generation_config.json", 202, "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e"),
               ("kld_table.json", 190470, "aae4c43ba9b95749bd8cd1c0eb92012830d23d95706350538a80319829325720"),
               ("measurement.json", 60916, "0ee08dec3537a438830198570b7dd492ee546b941901247b2d9654ed9deaaf6b"),
               ("merges.txt", 3353259, "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d"),
               ("model-00001-of-00002.safetensors", 8587658013, "e4c048aed121cc7fe17679498226b04b66ee52aeea0e2ec9baca1787922aebbc"),
               ("model-00002-of-00002.safetensors", 416227710, "78b726c919466ff7c630f582165e6a0bffb9c02105caeccfaaaef309def879c3"),
               ("model.safetensors.index.json", 295619, "a8e8a7658d741139f3bdbd489ad9da5a1acc260c7856e1fc0988059cc816e1ee"),
               ("noise_attrib.json", 257094, "82c41c57a5970de325cdb5bc137a62b67fd0ba3bbe5be94f0d009099dd6182cd"),
               ("preprocessor_config.json", 390, "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"),
               ("qbench_prompts_div.json", 1920186, "4800f95d62e910fd098703b107e3bb26c376005bb2c53dfc7d3fe2323d1914f2"),
               ("qbench_prompts_div.md", 1314356, "0158b7fe7e790e187e334fbed7abe88740e7c6692c1757583a8c736c639de4b8"),
               ("qbench_prompts_gen.json", 344109, "6070481d810633355e264a618f8a3727af0fc135b3c3671c917928dc6dc215ed"),
               ("qbench_prompts_gen.md", 228736, "a5a75cf17d725e15440afe644f41a090c2b860fe02f5ea1e69bff3bcd750b3a4"),
               ("quantization_config.json", 629299, "f03a5b0e64061f0f6b921726dc2d9818910be6bada50013197d8ad9dbddfb777"),
               ("recipe_1.40bpw.yaml", 21487, "395099be69595d56444d53b8d3f3ae05141cbd02814e07256ca8e487acc28dc7"),
               ("recipe_1.60bpw.yaml", 21487, "3ed1ef00984736488668efe456180ec3d1835a2009712aff2efe357c597f61e2"),
               ("recipe_1.80bpw.yaml", 21487, "5fd281e12bb5fa0107d93ae470ba1c4f5d1f8d495e37f4734656b0c80854aa03"),
               ("recipe_2.00bpw.yaml", 21490, "ef5ea53d7566b52638e6481ed7efedf0be26c868a4d50b5abb8e7ee08222f879"),
               ("recipe_2.20bpw.yaml", 21484, "fc08aed44aa0cc993b53aaff36cc98dba2a1aa81fc5055812568e1c086d0872d"),
               ("recipe_3.00bpw.yaml", 21487, "d977b2198b21721ae0e42667d89e2eb8107c15b8a8a3c3b16fe618d8f7e14353"),
               ("recipe_4.00bpw.yaml", 21487, "0b4a5b64580cbc3abf3d08d0bea1f526c50c6d5738c638f41288595069cec3b4"),
               ("recipe_5.00bpw.yaml", 21486, "1736b92c76728bb865fc38ed8a929dc6c21b077066fcb7815decfe20e5ccaad2"),
               ("recipe_6.00bpw.yaml", 21486, "a95a52fb59b3976ee615c02df6f68efa758af95def6a519c8a4848b4964cd9c4"),
               ("rfn_2.00bpw.json", 85427, "e66a8173c003f2b7e7fe89b7c10d6b0064a6ab1798078eedcf7d9b8555fa31c3"),
               ("tokenizer.json", 12809320, "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"),
               ("tokenizer_config.json", 17928, "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27"),
               ("video_preprocessor_config.json", 385, "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"),
               ("vocab.json", 6722759, "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003")]},
    {"label": "Ornith 1.5 9B Q4_K_M", "repo": "ornith-ai/Ornith-1.5-9B-GGUF", "rev": "abdd624b12ebf020b767fff532ff44fe552b28c3",
     "files": [("Ornith-1.5-9B-Q4_K_M.gguf", 5780090816, "70c112196e0b7023803c9762752e46d29e612a92c83f995bc3ba1ceb07e8fab6")]},
    {"label": "Bonsai 2 9B PQ2_0 with the MTP head + image file", "repo": "prism-ml/Ternary-Bonsai-2-9B-gguf",
     "rev": "82265ccc78ddaac4700286a35e9badcc32376dd9",
     "files": [("Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf", 2675410720, "c79fcdfaf6b7bf47089b50dadcfd11f9ce7d3e96adf3bbefde7912e5ef69e22b"),
               ("Ternary-Bonsai-2-9B-mmproj-Q8_0.gguf", 624229664, "e0cb6f6988cea323ef798cd77d019bbb819af75582f034eaab0e7232314953db")]},
]
# EXL3 (EXPERIMENTAL, still being tested): EXL3 models run on TabbyAPI + exllamav3, installed by install-exl3.sh into
# ~/exl3. A folder is an EXL3 model when its config.json says quant_method "exl3"; the tested ones are told apart by their
# first weights file (name:size). Their contexts are the lab's targets for an 8 GB card, NOT yet measured: 2.00 bits per
# weight at about 40k tokens, 1.75 at about 80k; K/V cache 4-bit (cache_mode 4,4); MTP draft 2 tokens.
EXL3_HOME = os.environ.get("EXL3_HOME") or os.path.join(HOME, "exl3")
EXL3_RECOMMENDED = {
    "model-00001-of-00002.safetensors:8570690830": {"label": "Qwen3.8-27B EXL3 2.00 bpw", "ctx": 40960, "draft_n": 2, "cache_mode": "4,4"},
    "model-00001-of-00002.safetensors:8587658013": {"label": "Qwen3.8-27B EXL3 1.75 bpw", "ctx": 81920, "draft_n": 2, "cache_mode": "4,4"},
}
# (the system prompt is prompts/v1-1.md, delivered by plugins/system-prompt.js; kit.json says which plugins a session gets)

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


def _gguf_read(path, want=(), tensors=False):
    """(the header keys ending in one of `want`, {tensor name: bytes in the file} when tensors=True). No tensor data is
    read: a tensor's size is the distance to the next tensor's offset (the last one's: to the end of the file)."""
    out, sizes = {}, {}
    def rd(f, fmt): return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    def rs(f): return f.read(rd(f, "<Q")).decode("utf-8", "replace")
    def val(f, t):
        if t in _T: return rd(f, _T[t])
        if t == 8: return rs(f)
        if t == 9:
            et, n = rd(f, "<I"), rd(f, "<Q")
            if et in _T: f.seek(n * struct.calcsize(_T[et]), 1); return None
            for _ in range(n): val(f, et)
            return None
        raise ValueError(t)
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF": return out, sizes
            rd(f, "<I"); nt = rd(f, "<Q"); nkv = rd(f, "<Q")
            align = 32
            for _ in range(nkv):
                k = rs(f); v = val(f, rd(f, "<I"))
                if k == "general.alignment" and isinstance(v, int) and v > 0: align = v
                if any(k.endswith(w) for w in want): out[k.split(".")[-1] if k != "general.architecture" else "arch"] = v
            if tensors:
                offs = []
                for _ in range(nt):
                    name = rs(f); nd = rd(f, "<I"); f.seek(8 * nd + 4, 1); offs.append((rd(f, "<Q"), name))
                data = (f.tell() + align - 1) // align * align
                end = os.path.getsize(path) - data
                offs.sort()
                for i, (o, name) in enumerate(offs):
                    sizes[name] = (offs[i + 1][0] if i + 1 < len(offs) else end) - o
    except Exception:
        pass
    return out, sizes


def gguf_meta(path, want=("nextn_predict_layers", "context_length", "general.architecture")):
    """The few header keys this script needs (no tensor data is read)."""
    return _gguf_read(path, want)[0]


_TENSORS = {}
def gguf_tensors(path):
    """{tensor name: bytes} of a GGUF file ({} if it cannot be read); cached per file."""
    try: key = (os.path.realpath(path), os.path.getsize(path), os.path.getmtime(path))
    except OSError: return {}
    if key not in _TENSORS: _TENSORS[key] = _gguf_read(path, tensors=True)[1]
    return _TENSORS[key]


# ------------------------------------------------------------------ mixture-of-experts layout and system RAM
EXPERTS = re.compile(r"blk\.(\d+)\.ffn_.*_exps")


def cpu_layers_for(name, mtp):
    """How many layers keep their experts in system RAM for this file and MTP setting (None: everything on the GPU)."""
    return ((RECOMMENDED.get(name) or {}).get("cpu_layers") or {}).get("mtp_on" if mtp else "mtp_off")


def experts_override(n):
    """The -ot value that keeps the experts of layers 0 .. n-1 in system RAM (the tested runs' exact pattern)."""
    return "blk\\.(" + "|".join(str(i) for i in range(n)) + ")\\.ffn_.*_exps.*=CPU"


def setting_key(name, mtp, images_gpu, proj=None):
    """The key llm-tune saves a measurement under: 'mtp=off,images=cpu', plus ',ram_layers=N' for a mixture-of-experts
    layout (a different layout is a different measurement)."""
    k = f"mtp={'on' if mtp else 'off'},images={'gpu' if (proj and images_gpu) else 'cpu'}"
    n = cpu_layers_for(name, mtp)
    return k + (f",ram_layers={n}" if n else "")


def ram_resident_mib(model, mtp, proj=None, proj_gpu=False):
    """System RAM the model's weights keep while it runs: the experts of the layers kept in RAM, the token embeddings
    (llama.cpp keeps the input layer on the CPU), and the image projector when it runs on the CPU."""
    t = gguf_tensors(model["path"])
    n = cpu_layers_for(model["name"], mtp) or 0
    b = t.get("token_embd.weight", 0)
    for k, s in t.items():
        m = EXPERTS.match(k)
        if m and int(m.group(1)) < n: b += s
    if proj and not proj_gpu:
        try: b += os.path.getsize(proj)
        except OSError: pass
    return b / 2**20


def meminfo():
    """/proc/meminfo in MiB ({} where there is none)."""
    out = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1); out[k] = int(v.split()[0]) // 1024
    except (OSError, ValueError):
        pass
    return out


def ram_check(model, mtp, proj=None, proj_gpu=False, avail=None):
    """{need, avail, short} in MiB: what the server needs in system RAM (its weights there, its working memory and the
    prompt cache), what is available now (MemAvailable), and how much is missing to also leave RAM_HEADROOM_MIB for
    opencode and the desktop (0: enough; avail None: unknown)."""
    need = round(ram_resident_mib(model, mtp, proj, proj_gpu) + RAM_ANON_MIB + CACHE_RAM_MIB)
    if avail is None: avail = meminfo().get("MemAvailable")
    return {"need": need, "avail": avail, "short": max(0, need + RAM_HEADROOM_MIB - avail) if avail is not None else 0}


def gib(mib): return f"{mib / 1024:.1f} GB"


# ------------------------------------------------------------------ finding models
# where downloaded models usually are, besides the Hugging Face cache (LM Studio's folders included)
DEFAULT_DIRS = [os.path.join(HOME, "models"), os.path.join(HOME, "Downloads"), os.path.join(HOME, ".lmstudio", "models"),
                os.path.join(HOME, ".cache", "lm-studio", "models")]
# packings of the tested models that hold the same weights in more space: on 8 GB they leave little or no room for context
BIGGER_PACKS = {"Ternary-Bonsai-2.5-27B-PQ2_0": "7.3 GB", "Ternary-Bonsai-2.5-27B-Q2_0": "7.7 GB",
                "Ternary-Bonsai-2.5-27B-Uncensored-PQ2_0": "7.3 GB"}


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
    def add_exl3(cfg, repo, name=None):
        d = os.path.dirname(cfg); real = os.path.realpath(d)
        if real in found: return
        info = exl3_info(d)
        if not info: return
        name = name or os.path.basename(d.rstrip("/"))
        found[real] = {"path": d, "name": name, "repo": repo, "size": info["size"], "dir": d, "engine": "exl3", "exl3": info}
    hub = hf_cache()
    for repo_dir in glob.glob(os.path.join(hub, "models--*")):
        repo = os.path.basename(repo_dir)[len("models--"):].replace("--", "/")
        snaps = sorted(glob.glob(os.path.join(repo_dir, "snapshots", "*")), key=os.path.getmtime, reverse=True)
        for s in snaps:
            for p in glob.glob(os.path.join(s, "**", "*.gguf"), recursive=True): add(p, repo)
            for p in glob.glob(os.path.join(s, "config.json")): add_exl3(p, repo, f"{repo.split('/')[-1]}@{os.path.basename(s)[:8]}")
    for d in list(DEFAULT_DIRS) + [x for x in st.get("model_dirs", []) if x not in DEFAULT_DIRS]:
        d = os.path.expanduser(d)
        if not os.path.isdir(d): continue
        for p in glob.glob(os.path.join(d, "**", "*.gguf"), recursive=True):
            rel = os.path.relpath(os.path.dirname(p), d)
            add(p, (d.replace(HOME, "~", 1) if rel == "." else os.path.join(d.replace(HOME, "~", 1), rel)))
        for p in glob.glob(os.path.join(d, "**", "config.json"), recursive=True): add_exl3(p, d.replace(HOME, "~", 1))
    return sorted(found.values(), key=lambda m: m["name"].lower())


def exl3_info(d):
    """For a folder that holds an EXL3 model: {bits, mtp, sig, size, native}; None for anything else."""
    try:
        with open(os.path.join(d, "config.json")) as f: cfg = json.load(f)
    except (OSError, ValueError): return None
    q = cfg.get("quantization_config") or {}
    if not isinstance(q, dict) or q.get("quant_method") != "exl3": return None
    weights = sorted(p for p in glob.glob(os.path.join(d, "*.safetensors")) if re.fullmatch(r"model(-\d{5}-of-\d{5})?\.safetensors", os.path.basename(p)))
    if not weights: return None
    tc = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    return {"bits": q.get("bits"), "mtp": bool(tc.get("mtp_num_hidden_layers") or tc.get("num_nextn_predict_layers")),
            "sig": f"{os.path.basename(weights[0])}:{os.path.getsize(weights[0])}", "size": sum(os.path.getsize(w) for w in weights),
            "native": tc.get("max_position_embeddings")}


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
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)    # a just-closed server's TIME_WAIT is not "in use"
        try: s.bind(("127.0.0.1", port)); return True
        except OSError: return False


def health(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            return json.loads(r.read()).get("status") in ("ok", "healthy")      # llama-server / TabbyAPI
    except Exception:
        return False


def server_argv(server, model, ctx, mtp, draft_n, proj, proj_gpu, port, cpu_layers=None):
    """llama-server's command line for these settings. cpu_layers None: the tested layout for this file and MTP setting
    (a mixture-of-experts model keeps the experts of its first layers in system RAM)."""
    argv = [server, "-m", model["path"], "-c", str(ctx), "-ngl", "999", "-fa", "on", "-np", "1", "--jinja", "-fit", "off",
            "--cache-type-k", KV, "--cache-type-v", KV, "--host", "127.0.0.1", "--port", str(port)]
    argv += next((v.split() for k, v in SAMPLING.items() if model["name"].startswith(k)), [])
    if mtp:
        argv += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(draft_n), "--spec-draft-p-min", "0", "-ctkd", KV, "-ctvd", KV]
    if proj:
        argv += ["--mmproj", proj] + ([] if proj_gpu else ["--no-mmproj-offload"])
    n = cpu_layers_for(model["name"], mtp) if cpu_layers is None else cpu_layers
    if n: argv += ["-ot", experts_override(n)]
    return argv


def start_server(server, model, ctx, mtp, draft_n, proj, proj_gpu, port, log, cpu_layers=None):
    argv = server_argv(server, model, ctx, mtp, draft_n, proj, proj_gpu, port, cpu_layers)
    env = dict(os.environ, LLAMA_ARG_CACHE_RAM=str(CACHE_RAM_MIB), LLAMA_ARG_CTX_CHECKPOINTS="4",
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


def warm_up(port, engine=None):
    """One tiny request: some GPU buffers are only allocated at the first request, so a context that loads can still run
    out of memory here."""
    if engine == "exl3":
        body = json.dumps({"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 2, "stream": False}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r: r.read(); return True
        except Exception:
            return False
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


def explain_failure(log, mtp, proj, proj_gpu, model=None):
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
    moe = bool(model and cpu_layers_for(model["name"], mtp))
    if model and not moe and any(EXPERTS.match(k) for k in gguf_tensors(model["path"])) and model["size"] > 7.5e9:
        print(f"{B}This is a mixture-of-experts model bigger than the GPU,{N} and the kit has no tested layout for this file, so all\n"
              f"of it went to the GPU. The tested files of this kind keep part of the model in system RAM: get one with d at the\n"
              f"model menu (README.md, the model list).\n")
    if moe:
        r = ram_check(model, mtp, proj, proj_gpu)
        print(f"{B}This model keeps part of itself in system RAM{N} (about {gib(r['need'])} with the prompt cache; "
              + (f"{gib(r['avail'])} available now)." if r["avail"] is not None else ").")
              + (f"\n   {R}That is not enough:{N} close other programs (a web browser often holds 1-3 GB)." if r["short"] else "")
              + "\n   If the log above just stops without an error, the system most likely ran out of RAM.")
    print(f"{B}If it did not fit in the GPU's memory{N}" + (" (it says out of memory above)" if oom else "") + ", you can:")
    print("   a) use a smaller context")
    if mtp: print("   b) turn MTP off (it needs its own memory on the GPU" + ("; with it off, fewer experts stay in RAM too)" if moe else ")"))
    if proj and proj_gpu: print("   c) process images on the CPU instead of the GPU")
    print("   - and check nothing else is using the GPU:")
    for line in gpu_busy_advice("     "): print(line)
    if moe and (RECOMMENDED.get(model["name"]) or {}).get("estimated") and mtp:
        print(f"   - with MTP on, this model's layout is estimated, not measured: llm-tune --fit measures what fits on this machine")


# ------------------------------------------------------------------ the EXL3 engine (experimental): TabbyAPI + exllamav3
# TabbyAPI reads config.yml (JSON is valid YAML) and its templates / sampler_overrides from the folder it starts in; it
# is started through runpy so neither that folder nor the script's own is put on sys.path ahead of its tree.
EXL3_BOOT = ("import runpy, sys\nT = sys.argv[1]\nsys.path.insert(0, T)\nsys.argv = [T + '/main.py']\n"
             "runpy.run_path(T + '/main.py', run_name='__main__')\n")


def exl3_python(): return os.path.join(EXL3_HOME, "venv", "bin", "python")
def exl3_tabby(): return os.path.join(EXL3_HOME, "tabbyAPI")


def exl3_installed():
    """install-exl3.sh finished: its stamp, the venv's python and TabbyAPI's main.py are there."""
    return (os.path.isfile(os.path.join(EXL3_HOME, "INSTALLED")) and os.access(exl3_python(), os.X_OK)
            and os.path.isfile(os.path.join(exl3_tabby(), "main.py")))


def exl3_config(model, ctx, mtp, draft_n, cache_mode, port):
    """TabbyAPI's settings for one model: the lab's launch settings (loopback only, so no API key; one request at a time;
    the model's own generation_config.json sampling -- temperature 1.0, top-p 0.95, top-k 20 for Qwen3.8 -- for requests
    that set none, as opencode's; its chat template's reasoning and tool-call format)."""
    d = model["path"].rstrip("/")
    return {
        "network": {"host": "127.0.0.1", "port": port, "disable_auth": True, "allowed_origins": [], "web_ui": False,
                    "api_servers": ["OAI"], "send_tracebacks": False, "sse_ping_interval": 15, "access_log": False,
                    "disable_fetch_requests": True, "enable_metrics": False},
        "logging": {"log_prompt": False, "log_generation_params": False, "log_requests": False, "log_live_status": False,
                    "log_timestamps": True, "log_auth_keys": False, "log_chat_completion_requests": False},
        "model": {"model_dir": os.path.dirname(d), "model_name": os.path.basename(d), "inline_model_loading": False,
                  "use_dummy_models": False, "backend": "exllamav3", "max_seq_len": ctx, "cache_size": ctx,
                  "cache_mode": cache_mode, "max_batch_size": 1, "chunk_size": 2048, "output_chunking": True,
                  "tensor_parallel": False, "gpu_split_auto": True, "vision": False, "use_generation_config": True,
                  "warmup": True, "reasoning": True, "reasoning_start_token": "auto", "reasoning_end_token": "auto",
                  "start_in_reasoning": "auto", "tool_format": "auto"},
        "draft_model": ({"draft_mode": "mtp", "draft_cache_mode": cache_mode, "draft_num_tokens": draft_n} if mtp
                        else {"draft_mode": "disabled"}),
        "sampling": {"override_preset": "safe_defaults"},
        "memory": {"sysmem_recurrent_cache": 2048, "sysmem_kv_cache": 0},
        "developer": {"unsafe_launch": False, "disable_request_streaming": False, "realtime_process_priority": False},
    }


def start_exl3(model, c, port, log):
    """TabbyAPI with this model, in its own folder under ~/.cache/llm-launch/ (its config.yml is kept there)."""
    run = os.path.join(LOG_DIR, "exl3-" + time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run, exist_ok=True)
    cfg = exl3_config(model, c["ctx"], c["mtp"], c["draft_n"], c["cache_mode"], port)
    with open(os.path.join(run, "config.yml"), "w") as f: json.dump(cfg, f, indent=1)
    for sub in ("templates", "sampler_overrides"):
        src, dst = os.path.join(exl3_tabby(), sub), os.path.join(run, sub)
        if os.path.isdir(src) and not os.path.lexists(dst): os.symlink(src, dst)
    argv = [exl3_python(), "-I", "-c", EXL3_BOOT, exl3_tabby()]
    fh = open(log, "w")
    fh.write(f"$ (in {run}) {exl3_python()} -I -c <runpy main.py> {exl3_tabby()}\n{json.dumps(cfg)}\n\n"); fh.flush()
    p = subprocess.Popen(argv, cwd=run, stdout=fh, stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID"),
                         start_new_session=True)
    return p, argv


def choose_exl3(st, model):
    """MTP and context for an EXL3 model (no image or RAM questions: the lab runs it text-only, all of it on the GPU)."""
    info = model["exl3"]; rec = EXL3_RECOMMENDED.get(info["sig"]) or {}
    mem = st.setdefault("models", {}).setdefault(model["name"], {})
    print(f"\n{B}EXL3 engine{N} ({R}experimental{N}: our tests of it are still running). {info['bits']} bits per weight"
          + (f": {rec['label']}." if rec else "; not a file this kit tested."))
    mtp = False
    if info["mtp"]:
        print(f"\n{B}MTP{N} (a built-in draft head that makes generation faster).")
        mtp = yes("Use MTP?", default=mem.get("mtp", True))
    key = f"engine=exl3,mtp={'on' if mtp else 'off'}"
    suggested = rec.get("ctx")
    last = st.get("_tried", {}).get(f"{model['name']}|{key}") or mem.get("ctx", {}).get(key)
    native = info.get("native")
    print(f"\n{B}Context{N} (how many tokens the model can keep in view at once).")
    if suggested: print(f"   The lab's target for an 8 GB card: {suggested:,} (not measured yet: if it does not fit, try less)")
    else: print("   No number for this model: enter one (if it does not fit, the server will say so)")
    if last: print(f"   You used {last:,} last time.")
    while True:
        v = ask("Context", default=str(last or suggested or ""))
        ctx = parse_ctx(v) if v else None
        if ctx and ctx >= 2048:
            if native and ctx > native: ctx = native
            ctx = ctx // 256 * 256                     # exllamav3: a multiple of 256
            break
        print("   Enter a number of tokens, e.g. 40960 or 40k.")
    return {"mtp": mtp, "proj": None, "proj_gpu": False, "ctx": ctx, "key": key, "draft_n": rec.get("draft_n", 2),
            "cache_mode": rec.get("cache_mode", "4,4"), "engine": "exl3"}


def explain_exl3_failure(log, c):
    try: lines = open(log, errors="replace").read().splitlines()
    except OSError: lines = []
    errs = [l for l in lines if ERR.search(l)]
    print(f"\n{R}{B}The EXL3 engine (TabbyAPI) did not start.{N} Its error output:\n")
    for l in (errs[-15:] or lines[-20:]): print("   " + l[-220:])
    print(f"\n   (full log: {log})\n")
    text = "\n".join(lines).lower()
    print(f"{B}If it did not fit in the GPU's memory{N}" + (" (it says out of memory above)" if "out of memory" in text else "") + ", you can:")
    print("   a) use a smaller context")
    if c["mtp"]: print("   b) turn MTP off")
    print("   - and check nothing else is using the GPU:")
    for line in gpu_busy_advice("     "): print(line)
    print(f"   The EXL3 engine is experimental (README.md). To reinstall it:  bash {os.path.join(KIT, 'install-exl3.sh')}")


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


def kit_setups(kit=None):
    """kit.json's setups (each a tested set of plugins), the default first. A kit.json from before setups existed: its
    one plugin list is the only setup."""
    kit = kit or load_kit()
    sets = kit.get("setups") or [{"id": "standard", "name": kit.get("setup", ""), "default": True, "plugins": kit.get("plugins", [])}]
    return sorted(sets, key=lambda s: not s.get("default"))


def setup_of(setup_id=None, kit=None):
    sets = kit_setups(kit)
    return next((s for s in sets if s.get("id") == setup_id), sets[0])


def fetch_plugins(kit=None, quiet=False):
    """[(plugin, local path or None)] for every plugin kit.json names (in any of its setups; each once)."""
    kit = kit or load_kit()
    seen, out = set(), []
    for s in kit_setups(kit):
        for p in s.get("plugins", []):
            k = p.get("file") or p.get("url")
            if k in seen: continue
            seen.add(k); out.append((p, plugin_path(p, quiet)))
    return out


def choose_setup(st, kit=None):
    """Which of kit.json's setups this session uses: the default, or another one the person turns on (each non-default
    setup is a yes / no question with its own description; the answer is remembered as the next default)."""
    sets = kit_setups(kit)
    cur = setup_of(st.get("setup"), kit)
    if len(sets) < 2 or not sys.stdin.isatty(): return cur["id"]
    print()
    for s in sets[1:]:
        if yes(f"{B}{s.get('ask') or 'Use ' + s.get('name', s['id'])}{N}", default=cur["id"] == s["id"]):
            st["setup"] = s["id"]; save_state(st); return s["id"]
    st["setup"] = sets[0]["id"]; save_state(st)
    return sets[0]["id"]


def session_config(prov, setup_id=None):
    """What this launcher adds for its own opencode sessions only (your opencode.json keeps none of it): the plugins of
    the chosen kit.json setup (today's default: the v1.1 system prompt in place of opencode's own instructions, your
    project's AGENTS.md kept, "Continue with the task." after a summary, and the compaction-anchor plugin's checklist and
    reinsert parts), kit.json's opencode settings (the compaction reserve used in the tests), plus the model entry (in
    case opencode.json could not be edited)."""
    kit = load_kit()
    paths = {(p.get("file") or p.get("url")): path for p, path in fetch_plugins(kit)}
    plugins = []
    for p in setup_of(setup_id, kit).get("plugins", []):
        path = paths.get(p.get("file") or p.get("url"))
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
MODELS_DIR = os.path.join(HOME, "models")


def dl_dest(d):
    return os.path.join(MODELS_DIR, d["repo"].replace("/", "__"))


def have_file(name, size, st=None):
    """The path of a model file with this name and size anywhere the launcher looks (None if there is none)."""
    for m in scan_models(st if st is not None else load_state()) + [{"path": p} for p in glob.glob(os.path.join(MODELS_DIR, "*", name))]:
        p = m["path"]
        if os.path.basename(p) == name and os.path.isfile(p) and os.path.getsize(p) == size: return p
    return None


def target_dir(d, st=None):
    """Where a menu entry's files go: next to its model file when that is already somewhere writable (an image file
    must sit next to its model), else ~/models/<repo>/. A model that is a folder (EXL3): ~/models/<subdir>/."""
    if d.get("kind") == "dir": return os.path.join(MODELS_DIR, d["subdir"])
    p = have_file(d["files"][0][0], d["files"][0][1], st)
    if p and not os.path.realpath(p).startswith(os.path.realpath(hf_cache())) and os.access(os.path.dirname(p), os.W_OK):
        return os.path.dirname(p)
    return dl_dest(d)


def missing_files(d, st=None):
    """The files of a menu entry not there yet: the model anywhere the launcher looks, the others next to it (a model
    that is a folder: every file in its folder)."""
    if d.get("kind") == "dir":
        t = target_dir(d, st)
        return [f for f in d["files"] if not (os.path.isfile(os.path.join(t, f[0])) and os.path.getsize(os.path.join(t, f[0])) == f[1])]
    first, rest = d["files"][0], d["files"][1:]
    p = have_file(first[0], first[1], st)
    out = [] if p else [first]
    where = [os.path.dirname(p)] if p else []
    where.append(target_dir(d, st))
    for f in rest:
        if not any(os.path.isfile(os.path.join(w, f[0])) and os.path.getsize(os.path.join(w, f[0])) == f[1] for w in where): out.append(f)
    return out


def disk_free(path=MODELS_DIR):
    p = path
    while not os.path.isdir(p): p = os.path.dirname(p)
    return shutil.disk_usage(p).free


def sha256_file(path, label=None):
    import hashlib
    h, done, total, last = hashlib.sha256(), 0, os.path.getsize(path), 0
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 22)
            if not b: break
            h.update(b); done += len(b)
            if label and time.time() - last > 2:
                last = time.time(); print(f"\r   checking {label}: {done * 100 // max(1, total)}%", end="", flush=True)
    if label: print("\r" + " " * 60 + "\r", end="")
    return h.hexdigest()


def fetch_file(d, name, size, sha, dest):
    """Download one file of a pinned revision into dest; it is kept only if its sha256 matches. True when it is there."""
    out = os.path.join(dest, name)
    url = f"{HF}/{d['repo']}/resolve/{d['rev']}/{name}"
    hf = shutil.which("hf") if d.get("kind") != "dir" else None
    try:
        if hf:
            rc = subprocess.call([hf, "download", d["repo"], name, "--revision", d["rev"], "--local-dir", dest])
            if rc != 0 or not os.path.isfile(out): print(f"{R}   download failed{N}"); return False
            tmp = out
        else:
            tmp = out + ".part"                       # resumes a .part left by an earlier try
            got = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            if got > size: os.remove(tmp); got = 0
            if got < size:
                req = urllib.request.Request(url, headers={"Range": f"bytes={got}-"} if got else {})
                with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "ab" if got and r.status == 206 else "wb") as fh:
                    if r.status != 206: got = 0
                    last = 0
                    while True:
                        b = r.read(1 << 22)
                        if not b: break
                        fh.write(b); got += len(b)
                        if time.time() - last > 2:
                            last = time.time(); print(f"\r   {gb(got)} / {gb(size)}", end="", flush=True)
                print()
        if os.path.getsize(tmp) != size or sha256_file(tmp, name if size > 10**8 else None) != sha:
            os.remove(tmp)
            print(f"{R}   {name} does not match its published checksum: deleted (choose d to try again){N}")
            return False
        if tmp != out: os.replace(tmp, out)
        if size > 10**8 or d.get("kind") != "dir": print(f"   done, checksum ok: {out.replace(HOME, '~', 1)}")
        return True
    except KeyboardInterrupt:
        print(f"\n   stopped; choosing it again resumes the download"); return False
    except Exception as e:
        print(f"\n{R}   download failed: {e}{N} (choosing it again resumes it)")
        return False


def get_download(d, st=None, ask_first=True):
    """Every missing file of one menu entry, each asked first with its size and the free disk space."""
    miss = missing_files(d, st)
    if not miss: print(f"   have it: {d['label']}"); return True
    dest = target_dir(d, st)
    if d.get("needs") == "exl3" and not exl3_installed():
        print(f"   {D}(this model needs the EXL3 engine, which is experimental: bash {os.path.join(KIT, 'install-exl3.sh')}){N}")
    if d.get("kind") == "dir":         # many files: one question for the whole folder
        total = sum(f[1] for f in miss); free = disk_free(dest)
        if free - total < DISK_MARGIN:
            print(f"{R}   Not enough disk space for {d['label']} ({gb(total)}):{N} {gb(free)} free, and {gb(DISK_MARGIN)} should stay free.")
            return False
        if ask_first and not yes(f"   Download it ({len(miss)} files, {gb(total)}; {gb(free)} free, {gb(free - total)} after)?", default=True): return False
        os.makedirs(dest, exist_ok=True)
        print(f"   downloading into {dest.replace(HOME, '~', 1)} from {d['repo']} (revision {d['rev'][:8]}) ...")
        return all([fetch_file(d, name, size, sha, dest) for name, size, sha in miss])
    ok = True
    for name, size, sha in miss:
        free = disk_free(dest)
        part = os.path.join(dest, name + ".part")
        left = size - (os.path.getsize(part) if os.path.exists(part) else 0)
        if free - left < DISK_MARGIN:
            print(f"{R}   Not enough disk space for {name} ({gb(size)}):{N} {gb(free)} free in {MODELS_DIR.replace(HOME, '~', 1)}, "
                  f"and {gb(DISK_MARGIN)} should stay free. Delete models you no longer use, then choose d again.")
            return False
        if ask_first and not yes(f"   Download {name} ({gb(size)}; {gb(free)} free, {gb(free - left)} after)?", default=True):
            ok = False; continue
        os.makedirs(dest, exist_ok=True)
        print(f"   downloading {name} from {d['repo']} (revision {d['rev'][:8]}) ...")
        ok = fetch_file(d, name, size, sha, dest) and ok
    return ok


def download(st):
    """The download menu: every model this kit tested, pinned to a repository revision and checked by sha256."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"\n{B}Download a model{N} (into ~/models/<repo>/; {gb(disk_free())} free there):")
    for i, d in enumerate(DOWNLOADS, 1):
        miss = missing_files(d, st)
        size = sum(f[1] for f in miss) if miss else sum(f[1] for f in d["files"])
        print(f"  {i}) {d['label']}   {D}{gb(size)}{'  (have it)' if not miss else ''}{N}")
    v = ask("Number (Enter to go back)")
    if not v.isdigit() or not 1 <= int(v) <= len(DOWNLOADS): return
    get_download(DOWNLOADS[int(v) - 1], st)


def offer_new_downloads(st):
    """Once per model: the models a kit update added to the download menu, each asked separately (with its size and the
    free disk space). A no is remembered; they stay in the menu (d)."""
    offered = st.setdefault("offered_downloads", [])
    key = lambda d: d.get("subdir") or d["files"][0][0]
    new = [d for d in DOWNLOADS if d.get("new") and key(d) not in offered and not (d.get("needs") == "exl3" and not exl3_installed())]
    if not new: return
    todo = [d for d in new if missing_files(d, st)]
    for d in new: offered.append(key(d))
    save_state(st)
    if not todo: return
    print(f"\n{B}New in this kit: {len(todo)} model{'s' if len(todo) > 1 else ''} to try{N} (being tested for this laptop; README.md, the model list):")
    for d in todo: print(f"   - {d['label']}  {D}{gb(sum(f[1] for f in missing_files(d, st)))}{N}")
    print(f"   {gb(disk_free())} free in ~/models. Each is asked separately; a no keeps it in the model menu (d) for later.")
    for d in todo:
        print(f"\n   {B}{d['label']}{N}")
        get_download(d, st)


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
    found, exl3_found = [], []
    user = os.path.basename(HOME)
    tops = [HOME] + [d for d in (f"/media/{user}", f"/run/media/{user}", "/mnt") if os.path.isdir(d)]   # other drives too
    walk = (x for top in tops for x in os.walk(top, followlinks=False))
    for root, dirs, files in walk:
        top = next(t for t in tops if root == t or root.startswith(t + os.sep))
        if root in skip or root[len(top):].count(os.sep) >= 7:
            dirs[:] = []; continue
        dirs[:] = [d for d in dirs if d not in ("node_modules", ".git", "__pycache__")]
        if "config.json" in files and exl3_info(root): exl3_found.append(root)
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
            if rec.get("cpu_layers"): what += "; part of it runs from system RAM (mixture of experts)"
            if name.startswith("Ternary-Bonsai-2.5-27B"): have_b25 = True
        elif big: what = f"a bigger packing ({big}): on 8 GB use the PTQ1_0 file instead"
        else:
            fam = next(k for k in TESTED_FILES if name.startswith(k))
            what = f"not tested here (the tested file is {TESTED_FILES[fam]}): you choose its context"
        print(f"   {name}  {D}{size}{N}\n      {what}{'; image file next to it' if proj else ''}\n      {D}{path.replace(HOME, '~', 1)}{N}")
        d = os.path.dirname(path)
        if not d.startswith(hf_cache()): add_dir(st, d)
    for d in exl3_found:
        info = exl3_info(d); rec = EXL3_RECOMMENDED.get(info["sig"])
        print(f"   {os.path.basename(d)}  {D}{gb(info['size'])}{N}\n      EXL3 ({info['bits']} bits per weight; experimental)"
              + (f": the lab's target {rec['ctx']:,} tokens" if rec else ": not tested here")
              + ("" if exl3_installed() else "; needs the EXL3 engine (bash install-exl3.sh)") + f"\n      {D}{d.replace(HOME, '~', 1)}{N}")
        if not d.startswith(hf_cache()): add_dir(st, os.path.dirname(d))
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
            tag = (f"  {D}{'expected' if rec.get('inferred') else 'tested'} on 8 GB: up to {best:,} tokens"
                   + ("; experts in RAM" if rec.get("cpu_layers") else "") + f"{N}") if best else ""
            here = {k.split("|", 1)[1]: r for k, r in tuned.items() if k.split("|", 1)[0] == m["name"]}
            if here:   # this machine's own numbers win (the MTP-off / images-CPU setting first)
                k0 = next((k for k in sorted(here) if k.startswith("mtp=off,images=cpu")), sorted(here)[0])
                r0 = here[k0]; sp = r0.get("speed") or {}; ds = sorted(sp, key=lambda x: int(x))
                wr = f", writing {sp[ds[0]].get('writing')}-{sp[ds[-1]].get('writing')} t/s" if ds else ""
                if ((r0.get("heat") or {}).get("severity")) in ("moderate", "heavy"): wr += f" (slowed by heat: {r0['heat']['severity']})"
                tag = (f"  {D}measured here: " + (f"up to {r0.get('suggested', 0):,} tokens{wr}" if r0.get("suggested") else "does not fit")
                       + (f" (+{len(here) - 1} more setting{'s' if len(here) > 2 else ''})" if len(here) > 1 else "") + f"{N}")
            if m.get("engine") == "exl3":
                r = EXL3_RECOMMENDED.get(m["exl3"]["sig"]) or {}
                tag = f"  {D}EXL3, experimental" + (f": about {r['ctx']:,} tokens (not measured yet)" if r else "") + f"{N}"
                if not exl3_installed(): tag += f"  {R}needs the EXL3 engine: bash install-exl3.sh{N}"
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
        if v.isdigit() and 1 <= int(v) <= len(models):
            m = models[int(v) - 1]
            if m.get("engine") == "exl3" and not exl3_installed():
                print(f"{R}This EXL3 model needs the EXL3 engine (experimental).{N} Install it with:  bash {os.path.join(KIT, 'install-exl3.sh')}")
                continue
            return m


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
    return (f"MTP {kv.get('mtp', 'off')}, images {kv.get('images', 'cpu').upper()}"
            + (f", experts of {kv['ram_layers']} layers in RAM" if kv.get("ram_layers") else ""))


def kshort(n):
    return f"{n // 1000}k" if n >= 1000 else str(n)


def speed_lines(rec, indent="   ", heat_summary=True):
    """The depth / reading / writing table of one measurement (with the cool-card row and what heat cost, when
    llm-tune recorded them)."""
    sp = rec.get("speed") or {}
    heat = rec.get("heat") or {}
    cool = heat.get("cool") or {}
    out = [f"{indent}{'depth':>9}   {'reading':>8}   {'writing':>8}   (tokens per second)"]
    short = int((heat.get("hot") or {}).get("depth", 4096))
    if cool:
        out.append(f"{indent}{short:>9,}   {cool.get('reading', 0):>8,}   {cool.get('writing', 0):>8}   on a cool card"
                   + (f" ({cool['temp_c']:.0f} °C)" if cool.get("temp_c") is not None else ""))
    for d in sorted(sp, key=lambda x: int(x)):
        note = ""
        if cool and int(d) == short:
            hot = heat.get("hot") or {}
            note = "   after a long run" + (f" ({hot['temp_c']:.0f} °C)" if hot.get("temp_c") is not None else "")
        out.append(f"{indent}{int(d):>9,}   {sp[d].get('reading', 0):>8,}   {sp[d].get('writing', 0):>8}{note}")
    if heat_summary and heat_short(heat):
        out.append(f"{indent}heat: {heat_short(heat)}" + (" (what helps: README.md, section Heat)"
                                                         if heat.get("severity") in ("moderate", "heavy") else ""))
    return out


def heat_short(heat, compact=False):
    """'heavy: writing 54% slower when hot (88 °C)' (compact: 'heavy (-54%, 88 °C)') -- what heat cost in one
    measurement, or '' if not recorded."""
    if not heat or not heat.get("severity"): return ""
    s, loss, peak = heat["severity"], heat.get("writing_loss"), heat.get("peak_c")
    if compact:
        bits = ([f"-{loss:.0%}"] if loss is not None and s != "none" else []) + ([f"{peak:.0f} °C"] if peak is not None else [])
        return s + (f" ({', '.join(bits)})" if bits else "")
    if loss is not None and s != "none": s += f": writing {loss:.0%} slower when hot"
    if peak is not None: s += f" ({peak:.0f} °C)"
    return s


def results_table(tuned=None):
    """Every measurement saved on this machine, one line each."""
    tuned = load_tuned() if tuned is None else tuned
    if not tuned: return ["   none yet: run llm-tune to measure a model on this machine"]
    rows = [("model", "setting", "held", "suggested", "reading / writing at each depth (t/s)", "heat", "measured")]
    for k in sorted(tuned):
        name, setting = k.split("|", 1); r = tuned[k]; sp = r.get("speed") or {}
        w = "  ".join(f"{kshort(int(d))}: {sp[d].get('reading')}/{sp[d].get('writing')}" for d in sorted(sp, key=lambda x: int(x)))
        if not r.get("max_ctx"): w = r.get("note") or "does not fit"
        elif not sp: w = "speed not measured yet (llm-tune --speed)"
        if (r.get("ram") or {}).get("tight"): w += "  [RAM ran short]"
        rows.append((name.replace(".gguf", ""), setting_label(setting), f"{r.get('max_ctx', 0):,}", f"{r.get('suggested', 0):,}", w or "-",
                     heat_short(r.get("heat"), compact=True) or "-", r.get("measured", "")))
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
    proj = projector_for(model)
    moe = bool(rec.get("cpu_layers"))
    est = set(rec.get("estimated") or [])
    if moe:
        n_off = cpu_layers_for(model["name"], False)
        print(f"\n{B}System RAM{N}: this is a mixture-of-experts model bigger than the GPU. As tested, the experts of its first "
              f"{n_off} layers stay in system RAM" + (" (more with MTP on)." if has_mtp and cpu_layers_for(model["name"], True) else "."))
    ram = {m: ram_check(model, m, proj) for m in ([False, True] if has_mtp else [False])} if moe else {}
    if has_mtp:
        print(f"\n{B}MTP{N} (a built-in draft head that makes generation faster).")
        on, off = rec.get("mtp_on"), rec.get("mtp_off")
        if moe and on:
            extra = cpu_layers_for(model["name"], True) - cpu_layers_for(model["name"], False)
            print(f"   The head needs GPU memory, so with MTP on the experts of {extra} more layer{'s' if extra > 1 else ''} move to system RAM"
                  f" (about {gib(ram[True]['need'] - ram[False]['need'])} more).")
            print(f"   {D}Tested on 8 GB with MTP off: {off:,} tokens." + (f" With MTP on: {on:,}, estimated (not measured yet; llm-tune --fit measures it)."
                                                                         if "mtp_on" in est else f" With MTP on: {on:,}.") + f"{N}")
        else:
            print("   Turning MTP off leaves more room for context, at the cost of running slower.")
            if on and off:
                print(f"   {D}Tested on 8 GB: up to {on:,} tokens with MTP on, {off:,} with it off.{N}")
            elif off and not on:
                print(f"   {D}On an 8 GB card this model fits much more context with MTP off.{N}")
        # default: on when it costs little context (or nothing is known), off when it costs a lot -- or when it does not
        # leave enough system RAM free
        cheap = bool(on) and (not off or on >= 0.9 * off)
        default = mem.get("mtp", cheap or not rec)
        if moe and ram[True]["short"] and not ram[False]["short"]:
            default = False
            print(f"   {R}With MTP on there is not enough free RAM now{N} ({gib(ram[True]['avail'])} free, about "
                  f"{gib(ram[True]['need'] + RAM_HEADROOM_MIB)} needed): MTP off is suggested, or close other programs first.")
        mtp = yes("Use MTP?", default=default)
    proj_gpu = False
    if proj:
        print(f"\n{B}Images{N} ({os.path.basename(proj)}): processing them on the CPU keeps the GPU's memory for context;")
        print("   on the GPU, images are read faster but there is less room for context.")
        proj_gpu = not yes("Process images on the CPU?", default=not mem.get("images_gpu", False))
    if moe:
        r = ram_check(model, mtp, proj, proj_gpu)
        print(f"\n   {D}System RAM: the model server needs about {gib(r['need'])} (its weights in RAM, its working memory and up to "
              f"{gib(CACHE_RAM_MIB)} of prompt cache)" + (f"; {gib(r['avail'])} is free now." if r["avail"] is not None else ".") + f"{N}")
        if r["short"]:
            print(f"   {R}That leaves too little for opencode and the desktop{N} (keep about {gib(RAM_HEADROOM_MIB)} more free): close other "
                  f"programs first, a web browser often holds 1-3 GB. Running short makes it very slow, or the server stops.")
    key = setting_key(model["name"], mtp, proj_gpu, proj)
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
    elif suggested and ("mtp_on" if mtp else "mtp_off") in est:
        print(f"   Estimated to fit on an 8 GB card with these settings (worked out from measurements, not tested): {suggested:,}\n"
              f"   {D}To measure it on this laptop: llm-tune --fit --one{N}")
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
    name = os.path.basename(path.rstrip("/")); mid = model_id(name)
    print(f"Model server on port {port}: {name}, {ctx:,} tokens of context")
    st = load_state()
    sid = choose_setup(st)
    prov = update_opencode_config(mid, f"{name} ({ctx // 1000}k)", ctx, port)
    env = dict(os.environ, OPENCODE_CONFIG_CONTENT=session_config(prov, sid))
    print(f"Starting opencode with {PROVIDER}/{mid} in {os.getcwd()}")
    print(f"   opencode setup: {setup_of(sid).get('name')}  {D}(kit.json){N}\n")
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
    if "--download" in sys.argv:
        st = load_state()
        download(st); return 0
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
    if sys.stdin.isatty(): offer_new_downloads(st)
    model = pick_model(st)
    exl = model.get("engine") == "exl3"
    while True:
        c = choose_exl3(st, model) if exl else choose(st, model)
        sid = choose_setup(st)
        port = DEFAULT_PORT
        while not port_free(port): port += 1
        log = os.path.join(LOG_DIR, f"{'tabbyapi' if exl else 'llama-server'}-{time.strftime('%Y%m%d-%H%M%S')}.log")
        mid = model_id(model["name"])
        print(f"\nStarting {'the EXL3 engine (TabbyAPI)' if exl else 'llama-server'}: {model['name']}, {c['ctx']:,} tokens, MTP {'on' if c['mtp'] else 'off'}"
              + (f", images on the {'GPU' if c['proj_gpu'] else 'CPU'}" if c["proj"] else "") + f" (port {port})")
        if exl: p, argv = start_exl3(model, c, port, log)
        else: p, argv = start_server(server, model, c["ctx"], c["mtp"], c["draft_n"], c["proj"], c["proj_gpu"], port, log)
        RUNNING[0] = p
        st.setdefault("_tried", {})[f"{model['name']}|{c['key']}"] = c["ctx"]      # proposed again on a retry in this run
        ok = wait_ready(p, port, timeout=1800 if exl else 600) and warm_up(port, c.get("engine")) and p.poll() is None
        if not ok:
            stop_server(p)
            if exl: explain_exl3_failure(log, c)
            else: explain_failure(log, c["mtp"], c["proj"], c["proj_gpu"], model)
            if yes("\nTry again with different settings?", default=True): continue
            sys.exit(1)
        print("   ready.")
        st.pop("_tried", None)                     # it worked: remember these settings for this model
        mem = st.setdefault("models", {}).setdefault(model["name"], {})
        mem["mtp"] = c["mtp"]; mem["images_gpu"] = c["proj_gpu"]; mem.setdefault("ctx", {})[c["key"]] = c["ctx"]
        save_state(st)
        prov = update_opencode_config(mid, f"{model['name']} ({c['ctx'] // 1000}k)", c["ctx"], port)
        env = dict(os.environ, OPENCODE_CONFIG_CONTENT=session_config(prov, sid))
        print(f"Starting opencode with {PROVIDER}/{mid} in {os.getcwd()}  {D}(server log: {log}){N}")
        print(f"   opencode setup: {setup_of(sid).get('name')}  {D}(kit.json){N}\n")
        time.sleep(1)
        try:
            rc = subprocess.call([opencode_bin(), "-m", f"{PROVIDER}/{mid}"], env=env)
        except KeyboardInterrupt:
            rc = 130
        died = p.poll() is not None
        if died:
            print(f"\n{R}The model server stopped while opencode was running.{N}")
            if exl: explain_exl3_failure(log, c)
            else: explain_failure(log, c["mtp"], c["proj"], c["proj_gpu"], model)
        else:
            if yes("\nStop the model server?", default=True): stop_server(p)
            else: print(f"   It keeps running on port {port} (pid {p.pid}); stop it with: kill {p.pid}")
        return rc



if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(); sys.exit(130)
