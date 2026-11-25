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
- **A model, if you want it now:** Ternary Bonsai 2.5 27B, 6 GB plus its 0.6 GB image file, into `~/models`. Get more later from the launcher: choose `d` at its model menu.

To get updates later: `cd ~/local-llm-8gb-kit && git pull`.

**Bonsai 2.5 27B is expected to be the strongest model that fits.** That is PrismML's claim. Our own coding tests on this setup are running now, and this note will be updated with what they show. The `-mtp` file adds a speed-up head. On this card that head costs a lot of context (see the table).

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

```bash
cd ~/path/to/your/project
python3 ~/local-llm-8gb-kit/llm-launch.py
```

It asks, in order:
1. **Model:** pick one from the list.
2. **MTP**, only if the model has the head: it is faster, but leaves less room for context.
3. **Images**, only if the model has an image file next to it. On the CPU (recommended) they cost no GPU memory. On the GPU, images are read faster, but there is less context.
4. **Context:**
   - It suggests the tested maximum for the model and settings.
   - For a model it has no number for, you type one.
   - It remembers the context that worked for each model.

If the model server fails to start:
- The launcher shows the error.
- It says what to change: a smaller context, MTP off, or images on the CPU.
- It offers to try again.

When you quit opencode, it asks whether to stop the server.

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
| `lldash-cli start bonsai` | Start a profile by part of its name, and wait until it is ready |
| `lldash-cli stop` | Stop it |
| `lldash-cli profile list` | List the launch profiles |
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

**Her 3070 Laptop should match these or do a little better:**
- **Writing** depends mostly on memory bandwidth, and hers has about 24% more, so expect 10 to 20% faster.
- **Reading** depends mostly on compute, so expect about the same.

**Images.** With the image file loaded on the CPU, Bonsai 2.5 still held 90,112 tokens with no extra GPU memory. Reading a 320×240 image and answering about it took about 3 seconds on the test machine's CPU. Her CPU is older, so expect a little longer, and longer again for bigger images.

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
