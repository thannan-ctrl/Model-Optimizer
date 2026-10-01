# LingBot-VA in FP8: calibration and accuracy check

**Goal:** run the LingBot-VA robot model's transformer in FP8 (8-bit numbers)
for speed on Thor, while keeping its outputs within **4%** of the original
bf16 model at every step.

**Result: passed.** On 100 held-out RoboTwin episodes (two separate sets of
50), with every model call checked, the worst call is off by 3.9% and the
average call by 0.9%. The robot's actions differ from bf16 by 0.05% of their
range on average.

## Results

Every model call checked (15,484 per set):

| Setup | Worst call | Average | Calls > 4% | Action drift (normalized) |
|---|---|---|---|---|
| **Final, eval set 1 (seed 1000)** | **3.92%** | 0.92% | 0 | 0.00053 |
| **Final, eval set 2 (seed 2000)** | **3.88%** | 0.93% | 0 | 0.00048 |
| FP8 everywhere | ≥ 13.6% | 3.9% | hundreds | 0.0014 |
| *bf16 TensorRT on Thor, for reference* | – | – | – | *0.00044* |

- **Action drift** is the mean absolute difference in the model's normalized
  [-1, 1] action space. It doesn't grow over an episode.
- **Margin is small.** Every worst call is video step 23, the step just
  before the bf16 ones.

**Estimated on Thor** (not measured; from bf16 TensorRT per-call times plus
the 1.27× FP8 speedup in `FP8_GUIDE.md`):
- **Speed:** about 1.25× faster than bf16 per chunk after the first, and
  about 1.21× per episode. Plain FP8 would be about 1.27×.
- **Memory:** both a bf16 and an FP8 engine are needed, about 13 GiB of
  weights vs 8.1 GiB for bf16 alone. A single FP8 weight set doesn't work:
  routing the same calls with FP8 weights and only bf16 activations fails,
  at 9.0% worst with 851 calls over 4% (`results/LOG.md`).

## How it works, in plain terms

- **Calibration.** FP8 needs one scale per layer so its limited range covers
  the values that layer sees. We find those scales by running the model on
  **real RoboTwin episodes** (camera frames, instruction, arm actions),
  replayed the same way the robot runs it: several action chunks in a row.
- **Checking accuracy.**
  - Run the original (bf16) and FP8 models side by side on the same inputs
    and compare every call.
  - Error = average absolute difference ÷ average size of the bf16 output.
  - Before each call, FP8 gets an exact copy of bf16's memory, so each number
    is that call's own error with no carry-over from earlier calls.
  - Separately, FP8 runs whole episodes on its own, and its final actions are
    compared with bf16's.
- **What makes it pass.** FP8 everywhere gives up to about 14% error. It
  concentrates in a few kinds of calls, and those run in bf16 instead:
  1. **Memory writes.** The model keeps a memory of past frames and actions
     (the KV cache). The 4 calls per chunk that write it run in bf16, so
     rounding errors aren't stored and carried forward.
  2. **The first chunk's video.** At the start of an episode the memory holds
     a single frame, and FP8 error is highest there.
  3. **The last denoising steps.** Each chunk refines its prediction over 25
     video and 50 action steps. The final, lowest-noise steps are the most
     sensitive, so the last video step and the last 2 action steps run in
     bf16.

  After the first chunk, that's about 7 of ~79 calls per chunk in bf16. The
  other ~91% run in FP8.

## Final setup

| | |
|---|---|
| **Model** | `robbyant/lingbot-va-posttrain-robotwin` (transformer only) |
| **FP8 layers** | 240 Linear layers: blocks 3–26 × 10 each. Blocks 0–2 and 27–29, embedders, output heads, norms and attention math stay bf16 |
| **Format** | FP8 E4M3, one scale per tensor, max calibration, fake-quant (no `--compress`). TensorRT-exportable |
| **Calibration data** | 50 RoboTwin episodes (one per task, both aug and clean scenes), 4 chunks each, including memory writes |
| **bf16 calls** | memory writes (`update_cache` 1 or 2), all video calls of chunk 0, video steps with t ≤ 172.4, action steps with t ≤ 40 |
| **Checkpoint** | `results/C/transformer.pt` (9.6 GB, not in git) |

## What we learned

- **Calibration data barely matters.** Dummy inputs, real first chunks and
  full multi-chunk replays all give the same error. Real data is still used
  because it's correct.
- **Error is spread over all quantized blocks,** not a single bad layer.
  Putting any one block, or any one layer type, back in bf16 barely helps.
- **Clipping outliers makes it much worse:** percentile calibration gives
  44–81%. The rare large activations carry real signal.
- **Smarter scales change nothing:** MSE for weights, activations or both
  lands on the same scales as max.
- **Check every call.** Sampling every 5th or 10th step missed the final
  steps, where the error peaks.
- **Big drift outliers can be quaternion sign flips.** q and −q are the same
  rotation. Compare rotations by angle.

## Reproduce

### 1. Setup (GB200)

```sh
# Code
git clone -b fix/exp-name-enametoolong git@github.com:thannan-ctrl/lingbot-va.git   # a23435d
git clone -b lingbot-va-fp8-quantization git@github.com:thannan-ctrl/Model-Optimizer.git

# Env: the lingbot-va conda env (see its INSTALL.md), plus:
pip install nvidia-modelopt==0.46.1 av        # av decodes the dataset's AV1 videos

# Model and data (skip the precomputed latents: not needed, and very large)
hf download robbyant/lingbot-va-posttrain-robotwin --local-dir $MODEL
hf download --repo-type dataset robbyant/robotwin-clean-and-aug-lerobot \
    --local-dir $DATA --exclude "*/latents/*"

# A GPU
salloc -p gb200nvl72_preprod -N1 --gres=gpu:1 -t 08:00:00
```

Set `LINGBOT=<lingbot-va checkout>`, `MODEL`, `DATA`, and run from
`Model-Optimizer/examples/diffusers/quantization`. Set
`LINGBOT_VA_DUMP_DIR=/tmp/dumps`: the server writes debug dumps on every
chunk, and they once filled the shared disk.

### 2. Calibrate (~1.7 h, 1 GPU)

```sh
python quantize.py --model lingbot-va --override-model-path $MODEL \
    --extra-param lingbot_va_repo=$LINGBOT \
    --extra-param lingbot_va_save_root=$LINGBOT_VA_DUMP_DIR \
    --extra-param robotwin_data_dir=$DATA \
    --extra-param robotwin_calib_mode=replay --extra-param robotwin_chunks=4 \
    --extra-param robotwin_calib_manifest=results/C/calib_manifest.json \
    --model-dtype BFloat16 --format fp8 --batch-size 1 --calib-size 50 \
    --quantized-torch-ckpt-save-path results/repro
```

`robotwin_calib_manifest` replays the exact 50 episodes used here. The
dataset path comes from `robotwin_data_dir`. Without the manifest, the same
list is drawn again from seed 0. The output goes to `results/repro`, so it
doesn't overwrite `results/C`.

### 3. Check accuracy (~3 h per eval set on 1 GPU; split across GPUs with `--episode-start` / `--num-episodes`)

```sh
python lingbot_va_parity.py --lingbot-va-repo $LINGBOT --model-path $MODEL \
    --test fp8 --fp8-ckpt results/repro/transformer.pt \
    --eval-manifest results/eval_manifest.json --robotwin-data-dir $DATA \
    --bf16-cache-writes --bf16-first-chunk-video \
    --bf16-last-video-steps 1 --bf16-last-action-steps 2 \
    --video-stride 1 --action-stride 1 --free-running \
    --out results/check.json
```

- **Expected result:** `WORST (judged): 3.9167e-02 … PASS (threshold 4%)`.
- **Second set:** use `results/eval_manifest_seed2000.json`, which should give
  3.88%.
- **Merging shards:** `--resummarize a.json b.json … --out merged.json`.
- **Sanity checks, both should give exactly 0:**
  - `--test bf16` (bf16 against itself)
  - `--test fp8-disabled` (FP8 checkpoint with quantizers off, which proves
    it holds the RoboTwin weights)

## Files

| File | What |
|---|---|
| `robotwin_episodes.py` | Picks and loads RoboTwin episodes; manifests |
| `robotwin_replay.py` | Replays an episode through the model like the robot client does |
| `calibration.py`, `quantize.py` | Calibration modes; `--act-calib` / `--quant-algo mse` options (tried, not used) |
| `lingbot_va_parity.py` | Accuracy check, drift, bf16 call routing |
| `results/` | Manifests, per-run commands, `LOG.md` (every run and the full history), `summaries/` |
| `LINGBOT_VA_FP8_THOR_HANDOFF.md` | What to build on Thor and how to route calls there |

## Gotchas

- **Keep the full bf16 cast.** `from_pretrained` leaves norms and a few
  tables in fp32, while the server runs everything in bf16. Skipping the cast
  causes a 4.9% false error.
- **`mto.restore` also loads the checkpoint's weights.** The first FP8
  attempt silently ran the *base* model's weights on Thor this way.
- **Don't use `--compress`.** TensorRT then can't build FP8 kernels.

## Next

1. Push the branch (signed-off commits, not pushed yet).
2. On Thor, per the handoff note:
   - build the FP8 engine next to the bf16 one
   - add the call routing
   - check parity
   - measure closed-loop success rate and timing
3. Optional: also route video step 23 to bf16 for more margin.
