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

## What changed in this update

- **New models to try.** Four models are being tested for this laptop right now. The launcher finds each one when its file is there, and runs it with the tested settings. Each download is optional and asked one at a time, with its size. Results come in a later update; until then, Bonsai 2.5 27B stays the one to use.

  | Model | Download | Notes |
  |---|---|---|
  | Bonsai 2.5 27B Uncensored (with the MTP head) | 6.2 GB | A community edit of Bonsai 2.5 27B with refusals removed. It has the same size and layout, so it gets the same contexts. |
  | Gemma 4 26B-A4B (UD-IQ4_XS) | 13.6 GB | A mixture-of-experts model: see *Models that run partly from system RAM*. It has no MTP head. |
  | Qwen 3.6 35B-A3B (IQ3_XXS) | 15.8 GB | Mixture of experts, with an MTP head. |
  | Ornith 1.5 35B-A3B (IQ3_XXS) | 15.3 GB | Mixture of experts, with an MTP head. It is built on the same 35B-A3B design as the Qwen model above, and packed the same way. |

  The three mixture-of-experts files need about 45 GB of disk between them. The launcher checks free space before each download.
- **A new system prompt, v1.1.** It replaces opencode's long built-in instructions. It also replaces the short "slim" prompt this kit used before. Your project's `AGENTS.md` is kept.
  - Why: its first version, v1, nearly doubled Bonsai 2.5 27B's score on our multi-bug tests, compared with opencode's own prompt: 11 of 15 bugs fixed against 6 of 15 (7/9 and 4/6, against 4/9 and 2/6). On single bugs the two were about level so far.
  - v1.1 is v1 with some extra work trimmed. The model keeps a todo list only when a request has more than one problem. It updates that list along with its next step. It doesn't re-read code it has just edited. In a first small test, on a different model, v1.1 scored about the same as v1: 9 of 11 against 10 of 11. It did not save time there. More tests are running.
  - When opencode summarizes a long session, the model is now told "Continue with the task." Before, it was told to continue or to stop and ask.
- **After a summary, the request comes back.** The original request returns word for word, along with a checklist of which of its items are done. That is the setup the new models are being tested with.
- **The task driver is removed for now.** Our tests are moving to the new prompt first.
- **`llm-tune` runs the fit test and the speed test separately.** For any model you can run either one. The fit test now also checks system RAM.
- **A second engine, EXL3 (experimental).** It runs Qwen3.8-27B squeezed to about 2 bits per weight. It is optional, and our tests of it are still running. See *EXL3 models (experimental)*.

## One-time setup

### 1. Keep the NVIDIA GPU free for the model

Run the desktop on the Intel graphics, so all 8 GB on the NVIDIA card is free for the model.
- **Compute** mode is best: the NVIDIA card then runs only the model. Choose **Graphics: Compute** in the top-bar menu, or run `system76-power graphics compute` in a terminal.
- **Hybrid** mode is fine too. Use it if an external monitor stops working in Compute mode.
- Only **NVIDIA** mode is a problem. In it, the desktop itself takes some of the card's memory. `setup.sh` offers to switch you to Compute.

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
- **The opencode plugins,** both in this kit: the system-prompt plugin with the v1.1 prompt, and two parts of the compaction-anchor plugin (see *What the launcher changes*).
- **Your models.** Setup finds the models you already have anywhere in your home folder or on another mounted drive: the Hugging Face cache, `~/models`, `~/Downloads`, LM Studio's folders and so on.
  - It covers Bonsai 2.5 27B (with the Uncensored edit), Bonsai 2 9B, Ornith 1.5 9B, and the new models above.
  - It says which files were tested and adds their folders to the launcher's search.
  - It downloads no models. To get one, use the launcher's model menu: choose `d`.
  - You can re-check any time with `python3 llm-launch.py --inventory`.
  - **For Bonsai 2.5 27B, use the PTQ1_0 file.** That is the 8 GB fit. The PQ2_0 and Q2_0 files hold the same weights in 7.3 and 7.7 GB, and leave far less room for context, or none. If you don't have PTQ1_0, get it from the launcher: choose `d`. It is 6 GB, plus its 0.6 GB image file.
  - Get more models later from the launcher: choose `d` at its model menu, or run `llm-launch --download`.
  - Every download in that menu is pinned to one published version of its files. Each file is checked against its published checksum and kept only if it matches.
  - After this update, the launcher offers each new model once. If you say no, it stays in the `d` menu.

- **Optional, experimental: the EXL3 engine** (TabbyAPI with exllamav3, into `~/exl3`). Setup asks; the answer defaults to no. See *EXL3 models (experimental)*.

**Updates:** at each start, `llm-launch` checks GitHub for a newer kit and asks before it updates. A newer kit can bring new plugin versions, or a system prompt that did better in our tests. To update by hand, run `llm-launch --update`. If an update also needs a setup step, such as a new opencode version, it offers to run `setup.sh` for you. Your settings and `llm-tune`'s measurements are kept.

**Use Bonsai 2.5 27B (the PTQ1_0 file) for real coding work.**
- In our coding tests on this 8 GB setup, it fixed 11 of 14 real bugs. The 9B models fixed 2 or 3. The details are under *What was measured*.
- Bonsai 2 27B also fixed 11, but Bonsai 2.5 can read images and has the MTP speed-up, so the kit sticks with 2.5 and no longer offers Bonsai 2 27B.
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
hf download prism-ml/Ternary-Bonsai-2.5-27B-GGUF Ternary-Bonsai-2.5-27B-PTQ1_0.gguf Ternary-Bonsai-2.5-27B-mmproj-Q8_0.gguf \
    --revision 28a7093cc25bc151cc2b284c5570f12ce343fdcf
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
   - It is asked only for model files that contain one: Bonsai 2.5's `-mtp` files (the Uncensored edit's too), Bonsai 2 9B's `PQ2_0-MTP` file, Ornith 1.5 9B, the two 35B-A3B models, and the EXL3 models.
   - For any other file it says there is none and runs without it.
   - On 8 GB, MTP halves Bonsai 2.5's context, costs Ornith 1.5 9B about a quarter of its context, and costs Bonsai 2 9B nothing. On the 35B-A3B models it keeps the context and moves more of the model into system RAM instead (see *Models that run partly from system RAM*). The launcher's default follows that.
3. **Images:** asked only if the model has an image file next to it. On the CPU (recommended) they cost no GPU memory. On the GPU, images are read faster, but there is less context.
4. **Context:**
   - It suggests the tested maximum for the model and settings.
   - For a model it has no number for, you type one.
   - It remembers the context that worked for each model.

For a model that runs partly from system RAM, it also says how much free RAM the model needs and how much there is. If there isn't enough, it warns you before starting.

If the model server fails to start:
- The launcher shows the error.
- It says what to change: a smaller context, MTP off, or images on the CPU.
- It offers to try again.

When you quit opencode, it asks whether to stop the server.

### Models that run partly from system RAM

Gemma 4 26B-A4B, Qwen 3.6 35B-A3B and Ornith 1.5 35B-A3B are bigger than the card: 13.6 to 15.8 GB. They are mixture-of-experts models. Each layer holds many small "experts", and for each word only a few of them run. So the launcher keeps the experts of the first layers in system RAM, and the CPU runs them. The rest of the model and the context stay on the GPU. That was the tested layout:

| Model | Experts in system RAM | Context | GPU memory (test card) | Reading / writing at 20k (test card) |
|---|---|---|---|---|
| Gemma 4 26B-A4B | the first 20 of 30 layers | 88,064 | 7,418 MiB | ~200 / 15.9 t/s |
| Qwen 3.6 35B-A3B, MTP off | the first 25 of 40 layers | 88,064 | 7,218 MiB | ~147 / 30 t/s |
| Ornith 1.5 35B-A3B, MTP off | the first 25 of 40 layers | 88,064 | 7,218 MiB | ~141 / 30 t/s |

**MTP on the two 35B-A3B models** made writing faster on the test card: 39 t/s for Qwen and 35 to 40 for Ornith, against 30 with it off. But the head needs GPU memory of its own: 1,126 MiB for Qwen and 720 MiB for Ornith. On these models a smaller context frees little room, so with MTP on, the launcher moves the experts of more layers into RAM instead: two more for Qwen, one more for Ornith. That layout is **estimated** from the measurements and has not been run on an 8 GB card yet. `llm-tune --fit` measures it on this laptop.

**System RAM is the tight part on a 16 GB laptop.** While running, these models need about 10.5 to 11.5 GB of RAM. That covers the experts, the server's own working memory, and up to 2 GB of prompt cache. opencode and the desktop need about 1 GB more. So **close the web browser** and other large programs first. If RAM runs out, the model slows down a lot, or the server stops.
- The launcher shows the need and what is free before it starts.
- When only MTP on would be short of RAM, it suggests MTP off.
- `llm-tune`'s fit test checks RAM first, and records it if RAM ran short.

### EXL3 models (experimental)

EXL3 is a different way of packing a model into fewer bits, run by a different engine: TabbyAPI with exllamav3, instead of llama.cpp. At about 2 bits per weight it puts the main weights of Qwen3.8-27B, a dense 27B model, on the card: about 6.2 GB at 2.00 bits and 5.4 GB at 1.75. The download is bigger, since it also holds parts that stay off the card, such as the image part. **It is experimental: our tests of it on an 8 GB card are still running, and none of its numbers below are measured yet.**

- **Install the engine** (about 5 GB to download, 9 GB on disk, into `~/exl3`): `bash install-exl3.sh`, or say yes when `setup.sh` asks.
  - It gets Python 3.12 through uv, so nothing on the system changes.
  - Every package is pinned to the version the lab installed and checked against its checksum. TabbyAPI is pinned to one git commit.
  - It needs git and NVIDIA driver 570 or newer.
- **Get a model** from the launcher's download menu (`d`). Each one is a folder of files, every file checked:

  | Model | Download | The lab's target context on 8 GB |
  |---|---|---|
  | Qwen3.8-27B EXL3, 2.00 bits per weight | 9.9 GB | about 40,000 tokens |
  | Qwen3.8-27B EXL3, 1.75 bits per weight | 9.0 GB | about 80,000 tokens |

- **The launcher** finds EXL3 model folders like the other models, and marks them *EXL3, experimental*.
  - It asks about MTP (a draft of 2 tokens) and the context, starts TabbyAPI, and opens opencode on it, with the same plugins as the other models.
  - The K/V cache is 4-bit, and sampling follows the model's own settings: temperature 1.0, top-p 0.95, top-k 20.
  - If the context does not fit, the launcher shows TabbyAPI's error. Try a smaller one.
- **Not covered yet:** `llm-tune` measures llama.cpp models only, LL-Dash has no profiles for EXL3, and `llm-launch --attach` does not work with it.

## Measure your own card: llm-tune

The contexts the launcher suggests were measured on a test GPU of the same generation as the 3070 Laptop. `llm-tune` measures them on this laptop instead: for any model file, including ones not in the table below, such as a different Ornith quant. With the setting you choose (MTP, images), it runs two tests. You can run them together or separately:

- **The fit test** checks whether the model loads and finds the largest context that loads and holds. It fills the context to 97% with real text, then sends requests at a few depths. It checks system RAM first: a model that runs partly from RAM is skipped when there isn't enough, and you can measure it again after closing programs. It also records the lowest free RAM while the context was full.
- **The speed test** measures reading and writing speed at 4k, 32k and near the context the fit test found, and *what heat costs* (see *Heat* below). It needs a fit result, and runs the fit test first if there is none.

At the end it prints a table of this run and of every measurement on this machine. The launcher then:
- suggests the measured context for that model and setting;
- shows the saved numbers beside the model in its list, and in full at the context question;
- shows the whole table on demand: `r` at its model menu, or `llm-launch --results`.

```bash
llm-tune                # every model found: MTP off, and on when the file has it; asks: fit test, speed test, or both
llm-tune --fit          # the fit test only (does it load, the largest context that fits)
llm-tune --speed        # the speed test only, at the context the fit test found
llm-tune --redo         # measure again what was measured before
llm-tune --one          # pick one model and its settings (MTP, images on the CPU or the GPU)
llm-tune --fit --one    # they combine
```

It lists what it will measure and roughly how long that takes, asks once, then runs on its own. A model that does not fit at all is recorded as such.

**For the new models, run the fit test first:** `llm-tune --fit`. It shows what fits on this laptop, for both MTP settings. The speed test can come later.

**Before you run it:**
- Close other programs that use the GPU. `nvidia-smi` lists them.
- Keep the desktop on the Intel graphics (Compute or Hybrid mode).
- Keep the laptop on its charger. On battery it warns you and asks first.

Most of the time goes into reading long prompts. Rough times per model and setting:
- Bonsai and the 9B models: fit 5 to 15 minutes, speed 10 to 15.
- The models that run partly from RAM read prompts much more slowly: fit 15 to 45 minutes, speed 15 to 30. Their fit test stops searching at 131,072 tokens of context.

Measuring everything can take several hours. Keep the laptop plugged in. `Ctrl-C` stops it, and the model server with it; finished measurements stay saved.

Results are saved in `~/.config/llm-launch/tuned.json`, and logs in `~/.cache/llm-launch/`.

### Heat

A laptop GPU slows itself down when it gets hot, and a long coding session keeps it busy for hours. So for each model and setting, `llm-tune` does three things:
- **A cool-card number:** before the speed test, it lets the GPU cool (up to 4 minutes) and measures 4k first.
- **A hot number:** it measures 4k again at the end, after the long run. The table keeps this one, because it is what long sessions get.
- **What happened in between:** every 2 seconds it records the GPU's temperature and clock, and whether the GPU said it was slowing itself for heat or because the charger could not keep up.

It then reports:
- how much slower writing and reading were when hot than when cool;
- the peak temperature;
- how much of the speed test the GPU spent slowed;
- its clock against its clock when not slowed.

It sums that up as **none**, **little** (writing under 10% slower), **moderate** (under 30%) or **heavy**. The launcher shows the same verdict beside the speeds.

If any model came out moderate or heavy, the run ends with what helps:
- **Raise the back of the laptop** 2 to 3 cm (a stand, or a book under the back edge), on a hard, flat surface: not a bed, a couch or a lap. The fans pull air in from underneath.
- **Keep the vents clear.** On an older laptop, blow the dust out of the vents and fans.
- **Add cooling:** a cooling pad (a stand with fans) and a cooler room both help.
- **Use the full-size charger** the laptop came with. A weaker or USB-C charger makes the GPU hold back.
- **Try the Performance power mode** (top-bar menu) while plugged in, then compare. Whether it helps depends on the laptop.

After a change, measure again with `llm-tune --redo --one` and compare.

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
- **Speed:** each model ran half its bugs on the RTX 3060 and half on an RTX 4090 Laptop, also held to 8 GB. Her laptop measured slower than both (see *On her laptop*), so her runs will take longer.
- **Plugins:** the tests used an earlier version of the compaction plugin in place of the task driver, and ran the 27B models at 90,112 tokens of context; the launcher suggests 88,064. A later test of the launcher's own setup was stopped partway, in favour of a new system prompt. On the 8 bugs it finished, every setup fixed 7 or 8, so the plugins made no clear difference there.
- **The new system prompt (v1):** on two tests that each hold several bugs, Bonsai 2.5 27B fixed 7 of 9 and 4 of 6 with it, against 4 of 9 and 2 of 6 with opencode's own prompt: 11 of 15 against 6 of 15. On single bugs the two were about level so far. This kit now uses v1.1 (see *What changed*). The new models are being tested with v1 and the checklist-and-request parts of the compaction plugin.

Every model runs with its makers' recommended sampling settings, which the launcher sets itself:

| Model | temperature | top-p | top-k | min-p | presence penalty |
|---|---|---|---|---|---|
| Bonsai 2.5 27B | 1.0 | 0.95 | 20 | 0 | 0 |
| Bonsai 2 9B | 1.0 | 0.95 | 20 | 0 | 1.0 |
| Ornith 1.5 9B and 35B-A3B (its "precise coding" setting) | 0.6 | 0.95 | 20 | 0 | 0 |
| Qwen 3.6 35B-A3B (its generation settings) | 1.0 | 0.95 | 20 | 0.05 (llama.cpp's default) | 0 |
| Gemma 4 26B-A4B (Google's card) | 1.0 | 0.95 | 64 | 0.05 (llama.cpp's default) | 0 |

The Uncensored Bonsai 2.5 edit runs with Bonsai 2.5's settings.

## What the launcher changes on your machine

- **`~/.config/opencode/opencode.json`:** it adds the model under a `llama-local` provider, with its context limit. The file is backed up once, to `opencode.json.bak-llm-launch`.
- **The opencode sessions it starts also get the following,** which is not saved in your opencode config. Opening opencode yourself gives you stock opencode, with these models available. The list comes from `kit.json`, and a kit update can change it. The launcher prints the current setup's name as opencode starts.
  - **The v1.1 system prompt** (`prompts/v1-1.md`, delivered by `plugins/system-prompt.js`). It replaces opencode's own general instructions. Opencode's lines about the model and the environment, your project's `AGENTS.md` instructions and the list of skills all stay as opencode writes them, in the one system message.
  - **"Continue with the task."** in place of opencode's automatic "continue, or stop and ask" turn after it summarizes a long session.
  - **Two parts of the compaction-anchor plugin** (`plugins/compaction-anchor.js`), which act only after a summary:
    - the original request again, word for word;
    - a task checklist that marks each item of the request as done, in progress, or not started. The model is asked to make its todo list match it.
  - Each time a session is summarized, the plugin also saves the whole conversation so far to a text file in `~/.local/share/opencode/history/`.
- **`~/.config/llm-launch/`:** your choices, downloaded plugins, and `llm-tune`'s measurements.
- **`~/models/<repo>/`:** the models you download from the launcher's menu.
- **`~/.cache/llm-launch/`:** the model server's logs.
