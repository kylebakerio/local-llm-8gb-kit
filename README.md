# Local coding models on an 8 GB RTX 3070 Laptop (ThinkPad P1 Gen 4, Pop!_OS 22.04)

This sets up local coding models the way they were tested for this machine:
- **llama.cpp** (PrismML's build, which runs the Ternary Bonsai models) serves the model on the GPU.
- **opencode** is the coding agent.
- **`llm-launch.py`** ties them together:
  1. It asks which model to run.
  2. It asks whether to use MTP, where images go, and how much context.
  3. It starts the model server.
  4. It opens opencode on that model.

Every number below was measured on a GPU of the same generation (Ampere) as the 3070 Laptop. That GPU was held to exactly the memory an 8 GB card offers, with a 200 MB safety margin.

## One-time setup

### 1. Keep the NVIDIA GPU free for the model

Run the desktop on the Intel graphics, so all 8 GB on the NVIDIA card is free for the model.
- Use the top-bar menu: **Graphics: Hybrid**.
- Or run `system76-power graphics hybrid` in a terminal.

Then reboot and check:

```bash
nvidia-smi
```

**Memory-Usage** should be close to 0 MiB with no apps open. If the desktop runs on the NVIDIA card ("NVIDIA" graphics mode), it uses a few hundred MB of it, and you will need a smaller context than the table below.

### 2. NVIDIA driver 570 or newer

`nvidia-smi` shows the driver version in its top line. To update it:

```bash
sudo apt update && sudo apt install system76-driver-nvidia
```

### 3. Get this kit and run its setup

```bash
git clone https://github.com/kylebakerio/local-llm-8gb-kit ~/local-llm-8gb-kit
cd ~/local-llm-8gb-kit && bash setup.sh
```

No git? Use **Code → Download ZIP** on GitHub and unzip it to `~/local-llm-8gb-kit`.

`setup.sh` checks the GPU, the driver and the graphics mode, then installs the rest. Each step is skipped if it's already done, and every download is checked against its published checksum:

- **llama.cpp, PrismML build prism-b12178,** into `~/llama-prism`. This build runs the Ternary Bonsai models. It uses the CUDA 12.8 build, or 12.4 for an older driver.
- **opencode 1.18.34,** the version everything was tested with. **Don't upgrade opencode yet:** the plugins use opencode hooks that newer versions may change.
- **The opencode plugins:**
  - [opencode-task-driver](https://github.com/kylebakerio/opencode-task-driver) 0.3.1, from its GitHub release;
  - the "slim" part of the compaction-anchor plugin, which comes built into the launcher.
- **Your models.** Setup finds the models you already have anywhere in your home folder or on another mounted drive: the Hugging Face cache, `~/models`, `~/Downloads`, LM Studio's folders and so on.
  - It covers Bonsai 2.5 27B, Bonsai 2 27B, Bonsai 2 9B and Ornith 1.5 9B.
  - It says which files were tested and adds their folders to the launcher's search.
  - It downloads no models. To get one, use the launcher's model menu: choose `d`.
  - You can re-check any time with `python3 llm-launch.py --inventory`.
  - **For Bonsai 2.5 27B, use the PTQ1_0 file.** That is the 8 GB fit, and the same goes for Bonsai 2 27B. The PQ2_0 and Q2_0 files hold the same weights in 7.3 and 7.7 GB, and leave far less room for context, or none. If you don't have PTQ1_0, get it from the launcher: choose `d`. It is 6 GB, plus its 0.6 GB image file.
  - Get more models later from the launcher: choose `d` at its model menu.

To get updates later: `cd ~/local-llm-8gb-kit && git pull`.

**Use Bonsai 2.5 27B (the PTQ1_0 file) for real coding work.**
- In our coding tests on this 8 GB setup, it and Bonsai 2 27B each fixed 11 of 14 real bugs. The 9B models fixed 2 or 3. The details are under *What was measured*.
- Bonsai 2.5 gets the nod over Bonsai 2 because it can also read images.
- The 27B models are slow: in testing, a typical fix took about 40 minutes, and the longest took 2 hours. Her laptop measured slower than the test card (see *On her laptop* below), so expect longer.
- The `-mtp` file adds a speed-up head, but on this card that head halves the context (see the table). The coding tests ran with MTP off, and how well the model codes with half the context is untested.

<details><summary>Doing these steps by hand</summary>

```bash
# llama.cpp (PrismML build); for a driver older than 570 use ...-cuda-12.4-x64.tar.gz
mkdir -p ~/llama-prism && cd ~/llama-prism
curl -LO https://github.com/PrismML-Eng/llama.cpp/releases/download/prism-b12178-03ec600/llama-prism-b12178-03ec600-bin-linux-cuda-12.8-x64.tar.gz
sha256sum llama-prism-b12178-03ec600-bin-linux-cuda-12.8-x64.tar.gz   # a50997fb29bb0684c00cdf21bb1382923c6bc70085a74c9c3d913dcf76af1899
tar xzf llama-prism-b12178-03ec600-bin-linux-cuda-12.8-x64.tar.gz --strip-components=1

# opencode 1.18.34
curl -fsSL https://opencode.ai/install | bash -s -- --version 1.18.34

# the plugins
python3 ~/local-llm-8gb-kit/llm-launch.py --fetch-plugins

# a model: the Hugging Face CLI puts files in ~/.cache/huggingface/hub, which the launcher searches
pip install -U "huggingface_hub[cli]"
hf download prism-ml/Ternary-Bonsai-2.5-27B-GGUF Ternary-Bonsai-2.5-27B-PTQ1_0.gguf Ternary-Bonsai-2.5-27B-mmproj-Q8_0.gguf
```

</details>

## Every time

At the end, `setup.sh` offers to start the launcher. After that, open a terminal in your project folder and run:

```bash
llm-launch
```

Setup puts that command in `~/.local/bin`; in a terminal opened before setup, use `python3 ~/local-llm-8gb-kit/llm-launch.py`.

It asks, in order:
1. **Model:** pick one from the list.
2. **MTP:** a draft head that makes writing faster but leaves less room for context.
   - It is asked only for model files that contain one: Bonsai 2.5's `-mtp` file, Bonsai 2 9B's `PQ2_0-MTP` file, and Ornith 1.5 9B.
   - For any other file it says there is none and runs without it.
   - On 8 GB, MTP halves Bonsai 2.5's context but costs Bonsai 2 9B nothing. The launcher's default follows that.
3. **Images:** asked only if the model has an image file next to it. On the CPU (recommended) they cost no GPU memory. On the GPU, images are read faster, but there is less context.
4. **Context:**
   - It suggests the tested maximum for the model and settings.
   - For a model it has no number for, you type one.
   - It remembers the context that worked for each model.

If the model server fails to start:
- The launcher shows the error.
- It says what to change: a smaller context, MTP off, or images on the CPU.
- It offers to try again.

When you quit opencode, it asks whether to stop the server.

## Measure your own card: llm-tune

The contexts the launcher suggests were measured on a test GPU of the same generation as the 3070 Laptop. `llm-tune` measures them on this laptop instead: for any model file, including ones not in the table below, such as a different Ornith quant. With the setting you choose (MTP, images), it finds:

- **the largest context that loads and holds:** it fills the context to 97% with real text, then sends requests at a few depths;
- **reading and writing speed** at 4k, 32k and near that maximum.

At the end it prints a table of this run and of every measurement on this machine. The launcher then:
- suggests the measured context for that model and setting;
- shows the saved numbers beside the model in its list, and in full at the context question;
- shows the whole table on demand: `r` at its model menu, or `llm-launch --results`.

```bash
llm-tune            # every model found: MTP off, and on when the file has it; skips what is already measured
llm-tune --redo     # measure everything again
llm-tune --one      # pick one model and its settings (MTP, images on the CPU or the GPU)
```

It lists what it will measure and roughly how long that takes, asks once, then runs on its own. A model that does not fit at all is recorded as such.

**Before you run it:**
- Close other programs that use the GPU. `nvidia-smi` lists them.
- Keep the desktop on the Intel graphics.

It takes roughly 10 to 25 minutes per model and setting, mostly reading long prompts, so measuring everything can take an hour or two. Keep the laptop plugged in. `Ctrl-C` stops it, and the model server with it; finished measurements stay saved.

Results are saved in `~/.config/llm-launch/tuned.json`, and logs in `~/.cache/llm-launch/`.

## Optional: the LL-Dash dashboard

LL-Dash is a web dashboard for starting and stopping models and watching the GPU. It comes with a terminal version, `lldash-cli`.

```bash
cd ~/local-llm-8gb-kit && bash install-lldash.sh
```

It needs Node.js 18 or newer. Pop 22.04's built-in Node is too old; if so, the installer prints the two commands to get Node 20.

The installer:
- downloads the dashboard from this repo's release and checks it;
- installs it in `~/llm-lab`;
- sets it to listen on `127.0.0.1:3000` only, so nothing else on the network can reach it;
- writes one launch profile for each tested model it finds, with the same settings as the launcher;
- adds these commands to `~/.local/bin`:

| Command | What it does |
|---|---|
| `lldash` | Start the dashboard, then open http://127.0.0.1:3000 and launch a model from the left panel |
| `lldash-cli start bonsai2.5-27b-88k` | Start a profile by its name, or the start of it if that is unique, and wait until it is ready |
| `lldash-cli stop` | Stop it |
| `lldash-cli profile list` | List the launch profiles: `<model>[-mtp]-<context>k`, for example `bonsai2.5-27b-88k`, `bonsai2.5-27b-mtp-41k`, `ornith1.5-9b-249k` |
| `llm-launch --attach` | Open opencode on the model the dashboard is serving, with the same opencode setup as the launcher |

After downloading another tested model, run `python3 ~/local-llm-8gb-kit/make-profiles.py` to add its profile.

## What was measured (8 GB, q4_0 K/V cache, PrismML prism-b12178)

The test GPU (an RTX 3060, the same Ampere generation as the 3070 Laptop) was held to what an 8 GB card offers, minus a 200 MB safety margin.

**Context** is how much text the model keeps in view, in tokens. "Suggested" leaves a little room for memory that llama.cpp takes later in a session. A model filled to the last megabyte crashed that way in testing.

| Model file | MTP | Max context that held | Suggested by the launcher |
|---|---|---|---|
| Ternary-Bonsai-2.5-27B-PTQ1_0 (or the -mtp file with MTP off) | off | 90,112 | 88,064 |
| Ternary-Bonsai-2.5-27B-PTQ1_0-mtp | on | 44,032 | 41,984 |
| Ornith-1.5-9B-Q4_K_M | off | 254,976 | 249,856 |
| Ornith-1.5-9B-Q4_K_M | on | 185,344 | 180,224 |
| Ternary-Bonsai-2-9B (PTQ1_0, or PQ2_0-MTP) | off / on | 262,144 (its maximum; only 4.6 / 5.9 GB used) | 262,144 |
| Ternary-Bonsai-2-27B-PTQ1_0 | off | 90,112 (93,184 loaded but crashed later) | 88,064 |

**Speed on the test GPU**, in tokens per second, at three conversation lengths. Reading is how fast it takes in the conversation; writing is how fast it answers:

| Model | MTP | Reading 4k / 32k / deep | Writing 4k / 32k / deep |
|---|---|---|---|
| Bonsai 2.5 27B | off | 573 / 487 / 373 (86k) | 43 / 29 / 18 |
| Bonsai 2.5 27B | on | 533 / 465 / 441 (41k) | 86 / 61 / 65* |
| Ornith 1.5 9B | off | 1,726 / 1,511 / 989 (131k) | 56 / 47 / 29 |
| Bonsai 2 9B | off | 1,729 / 1,526 / 995 (131k) | 116 / 83 / 40 |
| Bonsai 2 9B | on | 1,403 / 1,366 / 865 (131k) | 136 / 108 / 78* |

\* **MTP's writing speed here is a best case.** It was measured continuing source code, where the draft head guesses right about 80% of the time. In normal agent work, expect roughly 1.5 to 1.9 times the MTP-off speed (PrismML's figure for Bonsai 2.5).

**On her laptop** (measured with `llm-tune` on 11-24, MTP off, images on the CPU):

| Model file | Max context that held | Suggested | Reading 4k / 32k / deep | Writing 4k / 32k / deep |
|---|---|---|---|---|
| Ternary-Bonsai-2.5-27B-PTQ1_0 | 100,352 | 97,280 | 257 / 198 / 144 (88k) | 13.6 / 7.5 / 4.8 |
| Ternary-Bonsai-2-27B-PTQ1_0 | 102,400 | 99,328 | 254 / 197 / 143 (89k) | 13.5 / 7.7 / 4.6 |
| Ornith-1.5-9B-AD-Q5_K-Q4_K | 240,640 | 235,520 | 734 / 447 / 317 (212k) | 13.1 / 10.2 / 4.7 |
| Ternary-Bonsai-2-9B-PQ2_0 | 262,144 | 262,144 | 833 / 574 / 332 (236k) | 28.4 / 20.2 / 6.4 |

- **Context:** her card holds about 10,000 more tokens than the test card did. The test card was held back with a safety margin, while on her laptop the desktop runs on the Intel graphics, so the model has nearly all of the NVIDIA card's memory. The launcher uses her own numbers from now on.
- **Speed:** her card ran at a third to half of the test card's speed. That test card was a desktop RTX 3060 in an external box with its own fans.
- **Why:** a laptop GPU under a long, full load heats up to its limit (87 °C) and then cuts its clocks. Our test laptop's RTX 4090, running the same kind of work, sat at 88 to 89 °C with its clocks at 500 to 630 MHz of a possible 3,105.
- **What that means:** her numbers were measured over two hours of nonstop load, so they are most likely her laptop's heat-limited speed. Expect shorter jobs to run faster, until the GPU heats up.

**To check whether her GPU is slowing itself,** run this in a second terminal while a model is writing (Ctrl-C stops it):

```bash
nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks.max.sm,power.draw,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.hw_power_brake_slowdown --format=csv -l 2
```

- **Thermal slowdown `Active`, with the temperature near 87:** that's the heat. Raise the back of the laptop for airflow, keep the vents clear, or use a cooling pad.
- **Power brake `Active`:** the charger can't supply enough. Use the full-size Lenovo charger the laptop came with, not a USB-C one.

**Images.** With the image file loaded on the CPU, Bonsai 2.5 still held 90,112 tokens with no extra GPU memory. Reading a 320×240 image and answering about it took about 3 seconds on the test machine's CPU. Her CPU is older, so expect a little longer, and longer again for bigger images.

### Coding tests

Each model got the same 14 real bugs from four web apps. Hidden tests decided pass or fail, and each bug was described the way a user would report it.

The test conditions:
- one try per bug, with a 2-hour limit;
- the GPU held to 8 GB, with MTP off;
- the context from the table above, and the launcher's slim system prompt.

| Model | Bugs fixed (of 14) | Typical time per bug | Longest |
|---|---|---|---|
| Bonsai 2.5 27B | 11 | 38 min | 2 h (the limit) |
| Bonsai 2 27B | 11 | 38 min | 2 h (the limit) |
| Ornith 1.5 9B | 3 | 14 min | 1 h 30 min |
| Bonsai 2 9B | 2 | 3 min | 23 min |

- **The limit:** two of Bonsai 2.5's fixes and one of Bonsai 2's were in place just as the 2-hour limit hit. They count as fixed, since there is no limit in real use.
- **27B vs 27B:** with one try per bug, a gap of one or two bugs is noise. The two 27B models are equally strong here.
- **The 9B models:** both are far behind.
  - Bonsai 2 9B usually finished within minutes with a fix that didn't work.
  - Ornith often repeated itself without changing anything until it was stopped.
- **Speed:** each model ran half its bugs on the RTX 3060 and half on an RTX 4090 Laptop, also held to 8 GB. Her card's speed falls between the two.
- **Plugins:** the tests used an earlier version of the compaction plugin in place of the task driver, and ran the 27B models at 90,112 tokens of context; the launcher suggests 88,064. A test of the launcher's exact plugin setup is running, and this section will be updated with what it shows.

Every model runs with its makers' recommended sampling settings, which the launcher sets itself:

| Model | temperature | top-p | top-k | min-p | presence penalty |
|---|---|---|---|---|---|
| Bonsai 2.5 27B | 1.0 | 0.95 | 20 | 0 | 0 |
| Bonsai 2 27B | 1.0 | 0.95 | 20 | 0.05 | 0 |
| Bonsai 2 9B | 1.0 | 0.95 | 20 | 0 | 1.0 |
| Ornith 1.5 9B (its "precise coding" setting) | 0.6 | 0.95 | 20 | 0 | 0 |

## What the launcher changes on your machine

- **`~/.config/opencode/opencode.json`:** it adds the model under a `llama-local` provider, with its context limit. The file is backed up once, to `opencode.json.bak-llm-launch`.
- **The opencode sessions it starts also get the following,** which is not saved in your opencode config. Opening opencode yourself gives you stock opencode, with these models available.
  - The **task driver** ([opencode-task-driver](https://github.com/kylebakerio/opencode-task-driver) 0.3.1, focus mode):
    - When a request holds several separate problems, it has the model file them as tasks.
    - It works them one at a time, each on a clean slate with a short handoff from the tasks before.
    - It runs the full test suite once, at the end.
    - A single-problem request runs as usual.
  - **A short system prompt** in place of opencode's long built-in one. Your project's `AGENTS.md` instructions are kept.
  - **Shorter tool descriptions.**
  - **The `skill` tool turned off.**
- **`~/.config/llm-launch/`:** your choices, and the two plugin files.
- **`~/.cache/llm-launch/`:** the model server's logs.
