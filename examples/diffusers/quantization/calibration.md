# LingBot-VA FP8 recalibration on RoboTwin

Branch `lingbot-va-fp8-quantization`. Plan: `~/.claude/plans/rustling-drifting-moon.md`.

## Current status (2026-09-30, 21:30 Munich): DONE, gate passed on two eval sets

**Candidate:** checkpoint C (`results/C/transformer.pt`, max calibration, 240
FP8 Linear layers in blocks 3–26). It runs these calls in bf16, and
everything else in FP8:
1. The 4 KV-cache-writing calls per chunk (bf16 KV cache retention).
2. All video calls of chunk 0, the first chunk of an episode.
3. The last video denoising step (t ≤ 172) and the last 2 action steps
   (t ≤ 40) of every chunk.

After chunk 0, that's about 7 of ~79 calls per chunk in bf16 (about 9%).
Chunk 0 has 26 more.

**Full eval set, every call checked** (50 episodes × K=4, 15,484 calls;
`results/drift/final2_merged.json`):

| | Worst | Mean | Calls > 4% |
|---|---|---|---|
| **This setup** | **3.92%** | 0.92% | **0** → **PASS** |
| Without the last-step bf16 (`final_merged.json`) | 7.5% | 1.06% | 123 |

- **Worst calls:** all are video denoising step 23 (t = 303), the step just
  before the bf16 ones, at 3.6–3.9%. The margin is small.
- **Action denoising:** 3.4% worst, 0.84% mean.
- **Cache writes and chunk-0 video:** 0% (bf16).

**End-to-end action drift** (FP8 runs on its own, compared with bf16):

| | Normalized mean abs | Worst single value |
|---|---|---|
| **This setup** | **0.00053** | 0.14 |
| Thor bf16 TensorRT (reference) | 0.00044 | 0.0039 |
| Plain FP8 | 0.0014 | 1.02 (quaternion sign flip) |

**How we got here:**
- **Calibration data:** A = B = C, so the data doesn't matter.
- **Localization:** the error is spread over all blocks.
- **Better scales:** percentile makes it worse; weight MSE and activation MSE
  change nothing.
- **bf16 KV cache retention:** 13.6% → 5.6% (strided check), 7.5% with every
  call checked.
- **bf16 for the last denoising steps:** 3.9%.

The strided checks had understated the worst error. This result checks every
call.

### Estimated speed on Thor

PyTorch fake-quant can't show real speed; it's about 2× slower than bf16. So
this is an estimate from two Thor measurements:
- **bf16 TensorRT per-call latency** (`RESULTS_SUMMARY.md`): video call 105
  ms, action call 72 ms, so 6.4 s per chunk for 26 video + 51 action calls.
- **FP8 vs bf16 TensorRT trunk** (`FP8_GUIDE.md`): 1.27× host latency
  (1.30× GPU compute), applied to every FP8 call.

| Setup | bf16 calls per chunk | Chunk 0 | Later chunks | vs bf16 | 4-chunk episode |
|---|---|---|---|---|---|
| bf16 TensorRT (measured) | all 77 | 6.40 s | 6.40 s | 1.00× | 25.6 s |
| Plain FP8 | 0 | 5.04 s | 5.04 s | 1.27× | 20.2 s |
| Without last-step bf16 | 2 | 5.67 s | 5.08 s | 1.26× | 20.9 s |
| **Final setup** | 5 (+ all video in chunk 0) | **5.67 s** | **5.13 s** | **1.25×** | **21.1 s (1.21×)** |

- **Speed:** the accuracy fix costs about 2% after the first chunk. The first
  chunk is slower because all its video runs in bf16.
- **Memory is the real cost:** both weight sets are needed, a bf16 engine
  (8.1 GiB) plus an FP8 engine (4.9 GiB). That's about **13 GiB of weights,
  vs 8.1 GiB for bf16 alone and 4.9 GiB for plain FP8.** So the final setup
  is faster than bf16 but uses more GPU memory.
- **Caveats:**
  - The 1.27× is from one input shape. Action calls (32 tokens) may gain
    more.
  - The `compute_kv_cache` calls (bf16 in both mixed setups) aren't in the
    77-call measurement.
  - End to end, the CPU text encoder (~9 s per episode) still dominates.

**Phase 4 (2026-09-30): complete.**
1. **Second eval seed: PASSED.** `results/eval_manifest_seed2000.json`:
   50 new episodes (30 aug / 20 clean), disjoint from calibration and from
   the first eval set. Same setup, every call checked
   (`results/drift/final2b_merged.json`).

   | Eval set | Calls | Worst | Mean | Calls > 4% | Drift (normalized mean abs) |
   |---|---|---|---|---|---|
   | seed 1000 | 15,484 | 3.92% | 0.92% | 0 | 5.3e-4 |
   | **seed 2000** | 15,484 | **3.88%** | 0.93% | **0** | **4.8e-4** |

   - **Where the worst calls are:** again all video denoising step 23
     (t = 303), at 3.7–3.9%. 11 calls are over 3.5%.
   - **The pass isn't specific to one eval set.** The margin is small but
     consistent.
2. **Committed: done.** Six commits on `lingbot-va-fp8-quantization`, all
   signed off (`-s`). **Not pushed:** the branch is 6 ahead of
   `fork/lingbot-va-fp8-quantization`.

   | Commit | Contents |
   |---|---|
   | `486199167` | RoboTwin replay calibration: episode planner, `robotwin_replay.py`, calibration modes |
   | `07a39b55b` | Calibration options: `--act-calib`, `--quant-algo mse`, `manifest_dir` |
   | `e90fd4341` | `lingbot_va_parity.py`: gate, drift, bf16 call routing, sharding |
   | `539c35f6b` | This file, plus `results/`: manifests, per-run commands, `LOG.md`, compact summaries in `results/summaries/` |
   | `cb390e103` | `LINGBOT_VA_FP8_THOR_HANDOFF.md`, and the speed estimate here |
   | `e576d23d6` | Second eval seed confirmation (3.88% worst, 0 calls > 4%) |

   The full per-call parity JSONs (8–13 MB each, about 150 MB in total) and
   the checkpoints stay on disk and are gitignored. Only compact summaries
   are committed (1.1 MB for 59 files).
3. **Thor handoff note: done.** `LINGBOT_VA_FP8_THOR_HANDOFF.md` covers:
   - the checkpoint, and that Thor needs both a bf16 and an FP8 engine
   - the call-routing rule, with a `RoutedTrunkForward` sketch for
     faster-wam's `trunk_wrapper.py`
   - pre-export checks: `mto.restore` weight equality, and the bf16 cast
   - Thor validation: fixture parity, TensorRT parity, closed-loop success
     rate, timing

**Checkpoints kept on disk:** `results/{A,B,C}/transformer.pt`, 9.6 GB each.
C is the candidate.

**State of the environment:**
- No GPU allocations held.
- 119 GB free on the shared scratch.
- VA_Server debug dumps now go to node-local `/tmp`.
- Working scripts are in `/home/scratch.thannan_wwfo/robotics/claude_scratch/`
  (`env.sh`, `calib.sh`, `parity.sh`, `drift.sh`, `final2*.sh`, ...), outside
  git.

**Remaining work (owner: you):**
1. Re-sign and push the 6 commits to `fork`.
2. On Thor, following `LINGBOT_VA_FP8_THOR_HANDOFF.md`:
   - build the FP8 trunk engine from `results/C/transformer.pt`, next to the
     existing bf16 engine
   - add `RoutedTrunkForward`
   - check fixture and TensorRT parity
   - measure closed-loop RoboTwin success rate and timing against bf16
3. Optional: margin is small (worst 3.9%, all at video step 23, t=303).
   Routing that step to bf16 too should add margin, at about one more bf16
   call per chunk.
4. Optional cleanup: the older base-model checkpoints in this directory
   (`lingbot_va_fp8*.pt`, `lingbot_va_fp8_hf/`, about 31 GB) predate the
   recalibration.

## Goal

An FP8 checkpoint of the LingBot-VA transformer, built from the RoboTwin
post-trained weights, whose **worst per-call error against bf16 is ≤ 4%**
on held-out RoboTwin episodes.

```
rel_mean = mean|fp8 - bf16| / mean|bf16|
```

This is the same metric as faster-wam `lingbot-va/parity_torch.py`.

### Where 4% comes from

You chose it on 2026-09-29, as Q1 of the previous planning session.

- **The reasoning:** on Thor, the bf16 TensorRT engine differs from PyTorch
  by about 1–4% rel_mean, while the old FP8 checkpoint differed by 40–180%.
  FP8 should be no worse than what bf16 TensorRT already costs.
- **faster-wam's own threshold is stricter.** `parity_torch.py` requires
  `max_rel < 2%` for `PARITY OK`, but that was meant for bf16 TensorRT vs bf16
  PyTorch. `RESULTS_SUMMARY.md` puts that engine's error at about 1% per call,
  with final actions agreeing to within 0.4% of the action range.

### Gate: worst call, outliers explained

Chosen 2026-09-29 and confirmed 2026-09-30.

1. **What gets checked.** For every checked call, each tensor is checked: the
   call's output, plus the K and V it writes in each of the 30 blocks (for
   commit and `compute_kv_cache` calls).
2. **Normal tensors** are judged on rel_mean.
3. **Near-zero tensors** are judged differently. A tensor counts as near zero
   if its mean|bf16| is below 10% of the median for its group. A group is the
   same mode and kind, and for K/V also the same block and tensor, across the
   whole run.
   - rel_mean blows up on these, so they're judged on absolute error instead:
     mean|fp8 − bf16| / group median.
   - They're flagged in the report, not failed.
4. **Discarded outputs aren't gated** (decided 2026-09-30). `VA_Server`
   throws away the outputs of cache-writing calls: the last video step
   (`video_exec_step=-1`), the last action step, and both
   `_compute_kv_cache` calls. Only the K/V those calls write is used. So their
   outputs are recorded but not gated.
5. **Pass/fail.** A call's error is the worst of its tensors. The run passes
   if the worst call is ≤ 4%.
6. **Reporting.** The raw worst rel_mean is always reported too.
   `lingbot_va_parity.py --resummarize parity.json [--threshold]
   [--near-zero-frac]` re-applies the gate to a finished run without rerunning
   the model.

It must stay TensorRT-exportable: E4M3 per-tensor QDQ on Linear layers only, with
no attention (softmax/BMM) quantization.

Everything runs on GB200 until the gate passes. After that, the checkpoint goes
to IGX Thor for the TensorRT rebuild, TensorRT parity and closed-loop eval.

## Why recalibrate

The first FP8 checkpoint (`FP8_GUIDE.md`) had two problems:

- It was calibrated on `robbyant/lingbot-va-base`, but Thor runs the RoboTwin
  post-trained weights. `mto.restore` loads the checkpoint's weights too, so
  on Thor it silently ran base weights.
- Calibration used one dummy prompt and all-black cameras for a single first
  chunk. It never saw real scenes, later chunks, or `compute_kv_cache` calls.

## Inputs

| What | Where |
|---|---|
| Weights | `robbyant/lingbot-va-posttrain-robotwin`, at `/home/scratch.thannan_wwfo/robotics/model/lingbot-va-posttrain-robotwin` (23 GB) |
| Dataset | `robbyant/robotwin-clean-and-aug-lerobot`, at `/home/scratch.thannan_wwfo/robotics/data/robotwin-clean-and-aug-lerobot` (latents excluded) |
| lingbot-va | `/home/scratch.thannan_wwfo/robotics/lingbot-va`, branch `fix/exp-name-enametoolong` |
| Config | `wan_va/configs/va_robotwin_cfg.py`, unchanged |
| Env | conda `lingbot-va` (aarch64), modelopt 0.46.1, plus `av` 17.1.0 |
| GPU | `gb200nvl72_preprod`, 1 GPU (`salloc -p gb200nvl72_preprod -N1 --gres=gpu:1 -t 08:00:00 --no-shell`) |

### Dataset facts that shaped the design

- **The same 50 tasks appear in two collections.** There are 100 LeRobot
  v2.1 datasets in total:
  - `lerobot_robotwin_eef_aug_500`: randomized scenes, 500 episodes per task
  - `lerobot_robotwin_eef_clean_50`: clean scenes, 50 episodes per task
- **Cameras.** `cam_high`, `cam_left_wrist` and `cam_right_wrist`, each
  480×640 AV1 at 50 fps. The env's OpenCV can't decode AV1, so PyAV was
  installed.
- **Actions.** 16-dim absolute end-effector actions: left xyz+quat, left
  gripper, right xyz+quat, right gripper. They're made relative to the
  episode's first frame with `get_relative_pose`, as in post-training.
- **Episode lengths.** 74 to 1084 frames (median 165).
  - 4 chunks need 113 frames, and about 10% of episodes are shorter.
  - `click_bell` (at most 85 frames) and `click_alarmclock` (at most 108) have
    no episode long enough for 4 chunks.

## Decisions

| # | Decision |
|---|---|
| Gate | Worst per-call rel_mean ≤ 4%. If it fails: localize the error first, then keep more layers in bf16, and add more data only last |
| Weights | RoboTwin post-trained weights everywhere; the base model is out |
| Tasks | All 50, evenly: one episode per task at calib-size 50 |
| Collections | Within each task, alternate between aug and clean, starting from a random one. That gives 26/24 at calib-size 50 |
| Short tasks | `click_bell` and `click_alarmclock` use their longest episodes and replay 3 chunks instead of 4 |
| Calibration data | Multi-chunk episode replays, including `compute_kv_cache` calls |
| `state` | Dataset actions converted to relative actions (teacher-forced) |
| Variants | A = dummy data, B = real data first chunk, C = real data 4-chunk replay |
| A → B → C | One change at a time. B reuses C's episode list and drops the `compute_kv_cache` calls, so B vs C differs only in the chunks |
| Gate per call | max(output rel_mean, worst block's rel_mean on the K/V the call writes) |
| Parity coverage | During the loop: every 5th video step, every 10th action step, and all cache writes. Phase 4 checks every call |
| Where | PyTorch on GB200 only. Thor comes after the gate passes |
| Git | Commit the code, manifests, parity JSONs and `results/LOG.md`. Keep `*.pt`, `server_dumps/` and logs out of git. Keep the A/B/C checkpoints and the best one; delete the rest. Commit with `-s`, and never push without an explicit ask |
| If C passes | Try quantizing more blocks (0–2 and 27–29, one at a time) while it still passes, and report the fastest passing config |
| bf16 fallback | Blocks can go back to bf16 without asking, down to 18 of 30 still quantized |
| Per-channel scales | Try them, but report separately: TensorRT support on Thor is unverified |
| Eval size | Use a smaller eval set for search iterations only; gating runs use the full set |

## How the replay works (`robotwin_replay.py`)

It mirrors the RoboTwin eval client (lingbot-va
`evaluation/robotwin/eval_polict_client_openpi.py:545-590`), with the recorded
episode standing in for the simulator. Robotwin config: `frame_chunk_size=2`,
`action_per_frame=16`.

1. `infer(reset=True, prompt=episode prompt)`.
2. Seed torch and numpy per episode (CRC32 of `task_dir:episode_index:seed`),
   so bf16 and FP8 runs start from the same noise.
3. For each chunk `k`:
   - `infer(obs=frame 0)`. This runs 26 video steps and 51 action steps, the
     last of each writing the cache. Only chunk 0 actually encodes the obs.
   - **Executed steps.** Chunk 0 executes 16 steps. Its first latent frame is
     the conditioning frame and isn't executed. Later chunks execute 32 steps.
   - **Key frames.** One every `action_per_frame // 4 = 4` executed steps: 4
     in chunk 0, 8 in later chunks. Dataset frame `t` is the observation after
     `t` steps.
   - **`state`.** The model's own action output, with the executed steps
     replaced by the dataset's relative actions. The conditioning slot keeps
     the model's value, as in the client.
   - `infer(obs=key_frames, compute_kv_cache=True, state=state)`. This makes
     2 calls with `update_cache=2`, one video and one action.

| Chunk | Dataset actions | Key frames |
|---|---|---|
| 0 | 0–15 | 4, 8, 12, 16 |
| 1 | 16–47 | 20 … 48 |
| 2 | 48–79 | 52 … 80 |
| 3 | 80–111 | 84 … 112 |

That indexing was verified on CPU with a fake server.

## Calibration (`calibration.py`)

- **`--extra-param robotwin_calib_mode`**:
  - `dummy`: the old recipe, used for variant A
  - `first_chunk`: chunk 0 only, with no `compute_kv_cache`
  - `replay`: the default
- **`--extra-param robotwin_chunks`**: default 4.
- **`--calib-size`**: the number of episodes. `--batch-size` has no effect for
  lingbot-va, and the OpenVid prompts aren't loaded.
- **Manifest.** Calibration writes `calib_manifest.json` (dataset dir, seed,
  chunks, episodes) into the checkpoint dir. To reuse it, pass
  `--extra-param robotwin_calib_manifest=<path>`.
- **Episode planning** (`robotwin_episodes.plan_episodes`):
  - Round-robin over task names, alternating collections within a task.
  - `min_length` is a preference, not a filter: episodes long enough for K
    chunks are drawn first.
  - `exclude` keeps eval episodes disjoint from calibration episodes.
- **Unchanged.** The skip-list in `filter_func_lingbot_va` keeps blocks 0–2
  and 27–29, the embedders and the output heads in bf16. That leaves 24 blocks
  × 10 Linears = **240 weight + 240 input FP8 quantizers**, all with max
  calibration. `--quantize-mha` stays off.

```sh
# Wrapper used for all runs: claude_scratch/calib.sh RUN MODE CALIB_SIZE CHUNKS
python quantize.py \
    --model lingbot-va \
    --override-model-path $MODEL \
    --extra-param lingbot_va_repo=$LINGBOT \
    --extra-param lingbot_va_save_root=$Q/results/$RUN/server_dumps \
    --extra-param robotwin_data_dir=$DATA \
    --extra-param robotwin_calib_mode=replay \
    --extra-param robotwin_chunks=4 \
    --model-dtype BFloat16 \
    --format fp8 --batch-size 1 --calib-size 50 --collect-method default \
    --quantized-torch-ckpt-save-path $Q/results/$RUN
```

The output is `results/$RUN/transformer.pt`, a fake-quant `mto.save`
checkpoint of about 9.6 GB, with `calib_manifest.json` next to it. Don't use
`--compress`; the Thor exporter needs fake-quant.

## Parity (`lingbot_va_parity.py`)

This is a port of faster-wam `parity_torch.py`. That script checks one
captured first chunk (calls 0, 1, 25, 26, 76 on the mug example) on Thor
through the LeRobot policy. The port checks RoboTwin episodes over several
chunks on GB200, through lingbot-va's own `VA_Server`, with no LeRobot needed.

- **Teacher forcing.**
  - Both transformers stay on the GPU: bf16 inside `VA_Server`, and the test
    copy.
  - For each selected call, the bf16 KV cache is copied into the test copy.
    The test copy runs on the same inputs, then bf16 runs.
  - So each number is that call's own quantization error, with no drift from
    earlier calls.
- **Cache-writing calls.** Commits (`update_cache=1`) and `compute_kv_cache`
  calls (`update_cache=2`) also compare the K/V they write, in all 30 blocks.
- **`--test` options:**
  - `fp8`: the gate
  - `bf16`: a second copy, which gives the noise floor
  - `fp8-disabled`: the checkpoint with every quantizer off. This must match
    bf16, which shows the restored weights are the RoboTwin weights.
- **Eval set.**
  - `--eval-manifest` is created if it doesn't exist:
    `plan_episodes(seed=1000)`, 50 episodes, the same length preference, and
    every `--exclude-manifest` calibration episode excluded.
  - After that the manifest is read and stays fixed for the whole loop.
- **Guards.** The script refuses to run if parameter dtypes differ from
  `VA_Server`'s, or if any attention quantizer is enabled. It also logs the
  number of enabled quantizers.
- **Output:**
  - a JSON with every checked call (episode, chunk, mode, kind, step, t, and
    for the output and each block's K/V: rel_mean, mean_abs, max_abs,
    ref_scale), the judged error, and any near-zero flags
  - a table of worst and mean by chunk × mode × kind
  - the top-20 worst calls
  - a PASS/FAIL line
- **`--free-running`** (optional, not part of the gate): runs FP8 end to end
  and reports the action drift for each chunk.

```sh
# Wrapper: claude_scratch/parity.sh RUN TEST [args]
python lingbot_va_parity.py \
    --lingbot-va-repo $LINGBOT \
    --model-path $MODEL \
    --test fp8 \
    --fp8-ckpt results/C/transformer.pt \
    --robotwin-data-dir $DATA \
    --eval-manifest results/eval_manifest.json \
    --exclude-manifest results/C/calib_manifest.json \
    --out results/C/parity_fp8.json
```

## Bugs found along the way

1. **`calibration.py` imported a deleted module.** It imported
   `robotwin_calib_sampler` (removed when `robotwin_episodes.py` replaced it) and
   read `quantized_torch_ckpt_path` from `CalibrationConfig`, which doesn't
   have it. The fix rewired calibration to `robotwin_episodes` and
   `robotwin_replay`, and added `CalibrationConfig.manifest_dir`, set from
   `--quantized-torch-ckpt-save-path`.
2. **The planner stratified over directories, not tasks.** With 100
   directories, 50 episodes would have come only from `aug_500`. It now
   stratifies by task name.
3. **The parity copy kept fp32 modules.** `from_pretrained` keeps
   `_keep_in_fp32_modules` (norms, `scale_shift_table`, `time_embedder`) in
   fp32, while `VA_Server._configure_model` casts everything to bf16. So the
   first bf16-vs-bf16 noise floor came out at **4.9%**, above the gate.
   - The fix casts the test copy to bf16 and adds a dtype guard.
   - The noise floor is now exactly 0.
   - This matters on Thor too: any FP8 path that loads the transformer
     without that cast runs different numerics from deployment.
4. **A harness time limit killed a GPU job.** The first Phase 2 launch ran as
   a harness background command, which has a 2-hour cap. When the cap hit,
   the srun step on the node died with it, 22 of 50 episodes into C's
   calibration. All GPU work now runs detached:
   `setsid nohup srun --jobid=... script > log &`.
5. **`--act-calib` checkpoints wouldn't load.** They store
   `config.FixedMethodHistogramCalibrator` in their modelopt state, and
   PyTorch's `weights_only` loader rejects unknown classes. The parity script
   now allowlists the class with `torch.serialization.add_safe_globals`.
   - **Thor:** whatever loads such a checkpoint must be able to import that
     class. Alternatively, reset the input quantizers' calibrator to `max`
     before `mto.save`.
6. **Activation MSE crashed on an input quantizer with no data.** Some Linear
   is never called during inference (the load log warns that
   `patch_embedding.*` is unused), so its histogram is `None`. The MSE
   search now returns `None` for it, as modelopt's calibrators do. This cost
   one AMSE/WAMSE calibration round.
8. **The shared disk filled up** (1000 GB, 100%) at 17:12 Munich on
   2026-09-30. Both AMSE and WAMSE calibrations died writing their outputs.
   - **Cause:** `VA_Server` writes latents, actions and observations for
     every chunk (`save_async`), and those dumps under `results/*/server_dumps`
     had reached 82 GB.
   - **Freed:** all `server_dumps/`, plus the rejected checkpoints (dry run,
     P9999, P99999, WMSE), as agreed (keep A/B/C and the best). That leaves
     119 GB free.
   - **Fix:** dumps now go to node-local `/tmp` (`LINGBOT_VA_DUMP_DIR` in
     `env.sh`, used by `calib.sh` and `lingbot_va_parity.py`).
   - AMSE and WAMSE were relaunched at 17:16 Munich.
7. **modelopt 0.46.1's histogram MSE is broken for FP8.**
   `HistogramCalibrator.compute_amax("mse")` calls `scaled_e4m3()` without its
   `M` argument. Our calibrator does the FP8 MSE search itself, with torch
   `float8_e4m3fn`.

## Results

### Dry run (harness check, not a baseline)

`replay` mode, calib-size 2, K=2. Eval set: 2 episodes × 2 chunks, 56 checked
calls.

| Check | Worst rel_mean | Mean | Result |
|---|---|---|---|
| Calibration | 5m42s wall (about 28 s per chunk plus about 3 min load/save), 11 GB host RSS | | 240 W + 240 I quantizers |
| bf16 vs bf16 (noise floor) | 0 | 0 | PASS |
| FP8, quantizers disabled | 0 | 0 | PASS: restored weights = RoboTwin bf16 weights |
| FP8 (dry checkpoint, calibrated on 2 episodes) | 11.7% | 4.1% | FAIL, as expected |

Parity costs about 24 s per episode at K=2 with strided checking.

Where the dry FP8 error sits:

| Call type | Worst | Mean |
|---|---|---|
| Video denoising | 4.9% | 2–4% |
| Action denoising | 1.7% | 0.5–1.4% |
| Video cache writes (commit, `compute_kv_cache`) | 9.7–11.7% | ~10% |
| Action cache writes | 7.8–8.4% | ~8% |

The failing number is the written K/V (9–12%) more than the call output
(4–7%). The worst blocks are the late quantized ones (23–27). Cache writes run
at t=0 on clean latents, so their activations probably differ in range from
the noisy denoising calls. The baselines will show whether that's a data
problem or a real hotspot.

### Budget estimate for Phase 2

| Run | Calibration | Parity (50 eval episodes, K=4) |
|---|---|---|
| A: `dummy`, 50 runs | ~5 min | ~40 min |
| C: `replay`, 50 episodes, K=4 | ~1.5 h | ~40 min |
| B: `first_chunk`, C's 50 episodes | ~25 min | ~40 min |

Parity for K=4 is extrapolated from the dry run.

### Baselines

**Eval set:** `results/eval_manifest.json`, fixed for the whole loop.
- 50 episodes, planned with seed 1000.
- Split 26 aug / 24 clean.
- 4 episodes are shorter than 113 frames and replay fewer chunks.
- It excludes C's calibration episodes. Those were planned up front into
  `results/C_planned_calib_manifest.json`; C's actual `calib_manifest.json`
  was checked to be identical, and B calibrated on the planned list.

**How it ran.** The serial pipeline (`phase2.sh`) was killed by the harness
time limit (bug 4). A, B and C then ran as parallel tracks on separate GPUs
(`claude_scratch/track{A,B,C}.sh`).

**Results** on the full eval set (50 episodes × K=4, 2744 checked calls),
re-scored with the current gate (discarded outputs excluded):

| Run | Calibration | Worst (gated) | Mean | Calls > 4% | Worst tensor |
|---|---|---|---|---|---|
| A | `dummy`, 50 runs | 13.8% | 3.9% | 835 | V written by block 24 (video commit) |
| B | `first_chunk`, C's 50 episodes | 13.6% | 3.9% | 833 | V, blocks 23/24 |
| C | `replay`, 50 episodes, K=4 (198 chunks) | 13.6% | 3.9% | 833 | V, blocks 23/24 |

With the earlier gate, which also counted the discarded outputs of
cache-writing calls, the worst calls were 16.1%, 15.0% and 15.9%.

Where the error sits, per call type (C; A and B are within ±1 point):

| Call type | Worst | Mean |
|---|---|---|
| Video cache writes: written K/V | 13.6% | 10.9% |
| Action cache writes: written K/V | ~12% | ~8.6% |
| Video denoising | 5.6% | 2.1% |
| Action denoising | 2.0% | 0.8% |

A's parity was re-run with the gate script. It gave identical numbers, so the
pipeline is deterministic end to end.

## Findings after Phase 2 (2026-09-30)

- **Calibration data is not the cause.** A, B and C are the same within about
  1 point. Re-scored with the discarded outputs excluded, all three are
  13.6–13.8%, and the worst call is always V written by blocks 23–24.
- **Localization sweep** (see `results/LOG.md`):
  - The error is spread over all 24 quantized blocks. Any single block back
    in bf16 moves the worst call by at most 0.8 points.
  - No single layer type fixes it. FFN up/down helps most (10–11%).
  - Activation quantization is the larger source: activation-only FP8 gives
    12.3%, weight-only FP8 gives 8.5%.
- **What the errors are.** The KV cache itself is bf16. The error in the
  written K/V is carried into `to_k`/`to_v` by a hidden state that has built
  up FP8 error through the earlier quantized blocks.
- **Consequence.** Moving blocks back to bf16 can't reach 4% while keeping at
  least 18 blocks quantized.

## Round 2 (launched 2026-09-30, 15:33 Munich, 12 GPUs)

**1. End-to-end action drift on C** (`--free-running`, 50 episodes × K=4,
split across 3 GPUs, merged with `--resummarize ... --out`). FP8 writes and
reads its own cache with no sync, and the actions are compared per chunk with
bf16, in physical units and in normalized [-1, 1] space. The normalized
numbers line up with Thor's bf16 TensorRT: max 3.9e-3, mean 4.4e-4.

| Setup | What it shows |
|---|---|
| FP8 | The real impact of FP8, including cache error propagating across chunks |
| FP8 + `--bf16-cache-writes` | Cache-writing calls (about 4 of 79 per chunk) run with quantizers off, so the cache is exactly bf16. On Thor this would mean a bf16 engine for cache writes and an FP8 engine for denoising, about 8–10 GB more weight memory |
| bf16 vs bf16 (5 episodes) | Noise floor for the drift numbers. Should be 0 |

**2. Better calibration.** Each variant recalibrates on C's episodes
(`replay`, K=2; A/B/C showed chunks don't matter). Then it runs a fast parity
check (10 episodes × K=2), with and without `--bf16-cache-writes`.

| Run | Options | What changes |
|---|---|---|
| P9999 | `--act-calib percentile --act-percentile 99.99` | Activation scale clips the top 0.01% |
| P99999 | `--act-calib percentile --act-percentile 99.999` | Clips less |
| AMSE | `--act-calib mse` | Activation scale minimizes the FP8 round-trip error over the histogram |
| WMSE | `--quant-algo mse` | Weight scales by MSE search (modelopt); activations stay max |
| WAMSE | `--quant-algo mse --act-calib mse` | Both |

All of these stay per-tensor, so they stay TensorRT-exportable.

**New code:**
- `quantize.py`: `--act-calib max|percentile|mse`, `--act-percentile`, and
  `--quant-algo mse`.
- `config.py`: `FixedMethodHistogramCalibrator`, which pins the histogram
  method for input quantizers so weights keep their own calibrator.
- `lingbot_va_parity.py`: `--bf16-cache-writes`, normalized-space drift
  stats per chunk, `--episode-start` for sharding, and multi-JSON
  `--resummarize` merging.

**modelopt 0.46.1 bug.** `HistogramCalibrator.compute_amax("mse")` crashes for
FP8 inputs: `scaled_e4m3()` is called without the `M` argument. Our
calibrator implements the FP8 MSE search itself, using torch
`float8_e4m3fn` (tested on a toy model).

Also noticed on the toy model: modelopt's weight MSE chose an amax 3.4× the
weight max. Worth checking on the real checkpoint.

### Round 2 results

**Drift noise floor** (bf16 vs bf16, 5 episodes): exactly 0 in every chunk.

**End-to-end action drift on C**, 50 episodes × K=4 (196 chunks). Normalized
[-1, 1] action space, FP8 vs bf16. Files:
`results/drift/{fp8,fp8cw}_merged.json`.

| Setup | Mean abs error | Median chunk worst | Chunks with a value > 0.1 | Worst value | Worst chunk rel_mean | Per-call gate |
|---|---|---|---|---|---|---|
| Thor bf16 TensorRT (reference) | 0.00044 | – | – | 0.0039 | – | – |
| FP8 | 0.0014 | 0.016 | 7 / 196 | **1.02** (2 chunks) | 2.7% | 13.6% |
| FP8 + bf16 cache writes | 0.0009 | 0.008 | 2 / 196 | 0.14 | 1.0% | **5.6%** (mean 1.0%) |

Drift by chunk (normalized mean abs error):

| Chunk | FP8 | FP8 + bf16 cache writes |
|---|---|---|
| 0 | 0.00049 | 0.00047 |
| 1 | 0.0022 | 0.0012 |
| 2 | 0.0013 | 0.0009 |
| 3 | 0.0015 | 0.0012 |

- **On average, FP8 actions stay close to bf16.** Mean error is about 0.1% of
  the action range, 2–3× the bf16 TensorRT error on Thor.
- **Drift doesn't compound across chunks.**
- **bf16 cache writes help:**
  - mean error 35% lower
  - typical worst value halved
  - both large outliers gone
  - per-call gate from 13.6% to 5.6%. The rest is video denoising, 1.6
    points over the gate.
- **Two large outliers in plain FP8.** Both are in chunk 1: episode 21
  (`place_a2b_right`, 1.02) and episode 37 (`press_stapler`, 1.01).
  - The physical and normalized errors are equal, which rules out the gripper
    (normalization doubles its differences) and points to a quaternion
    component.
  - **Resolved: both are harmless q/−q sign flips.** Both episodes were re-run
    with per-channel errors and raw actions saved
    (`results/drift/outlier_{21,37}.json`):

    | Episode | Worst channel | Component diff | Actual rotation diff | Position | Gripper |
    |---|---|---|---|---|---|
    | 21, chunk 1 | right-arm quaternion | 1.02 | 2.5° | 2.3 mm | 0 |
    | 37, chunk 1 | right-arm quaternion | 1.01 | 1.1° | 3.1 mm | 0.016 |

    The right arm's quaternion has every component near ±0.5. At one of the 32
    steps, FP8 emits −q where bf16 emits q. That's the same rotation, since the
    client goes through scipy `Rotation`.
  - **Metric fix to make:** compare quaternions by rotation angle
    (sign-invariant), not componentwise. The componentwise "worst value" for
    plain FP8 overstates the real difference. Without the flips its worst is
    about 0.15, similar to FP8 with bf16 cache writes.

**Recalibration variants** (fast parity: 10 episodes × K=2, 280 calls). The
C/max rows are computed from the drift runs' teacher-forced records, restricted
to the same episodes and chunks.

| Calibration | All FP8: worst / mean | bf16 KV cache retention: worst / mean |
|---|---|---|
| **max (C)** | 12.8% / 4.0% | **4.8% / 1.2%** |
| WMSE (weight MSE) | 12.8% / 4.0% | 5.4% / 1.2% |
| P99999 (percentile 99.999) | 44.0% / 9.2% | 6.5% / 2.1% |
| P9999 (percentile 99.99) | 81.1% / 15.8% | 14.4% / 4.2% |
| AMSE (activation MSE) | 12.9% / 4.0% | 5.1% / 1.2% |
| WAMSE (weight + activation MSE) | 12.8% / 4.0% | 5.1% / 1.2% |

- **max is still the best calibration.** No variant beats it.
- **Weight MSE, activation MSE and both together change nothing.** The MSE
  scales land close to the max scales.
- **Percentile is ruled out:** the more it clips, the worse it gets.

**Percentile clipping makes it much worse** (44% vs 12.8%). The activation
outliers carry real signal, as with the "massive activations" known from LLMs,
and clipping them destroys it. So FP8 needs to keep the full range there, and
the fix has to come from elsewhere: MSE-chosen scales, bf16 KV cache
retention, or finer granularity.

### bf16 KV cache retention: what it is

It's per **call**, not per layer. Of the ~79 transformer calls per chunk,
4 write the KV cache:
- the video commit (last video denoising step)
- the action commit (last action step)
- the 2 `compute_kv_cache` calls, for the observed frames and the executed
  actions

During those 4 calls, all 240 FP8 layers run in bf16. The other ~75 calls run
fully in FP8.

Switching only the K/V-producing layers (`attn1.to_k` / `to_v`) isn't enough.
Their input already carries FP8 error from every earlier layer, and the sweep
showed it: `to_v` or all attn1 Q/K/V back in bf16 only moves 12.8% to
11.9–12.3%.

### Current FP8 coverage

| | What | Precision |
|---|---|---|
| Blocks 3–26 | 24 blocks × 10 Linear layers = **240 layers** | FP8 |
| Blocks 0–2 and 27–29 | 60 Linear layers | bf16 |
| Other | Embedders, output heads, norms, attention softmax/BMM | bf16 |

That's 80% of the Linear layers in the transformer blocks.

### Status message sent to colleagues (2026-09-30)

> Small FP8 rounding errors were getting written into the model's memory of
> past frames (the KV cache) and carried forward. With bf16 KV cache retention,
> the few steps that write that memory (4 of ~79 per action chunk) run in full
> precision, and everything else stays FP8. That cuts the worst-case error from
> about 14% to about 6% (target 4%), and the robot's actions stay within about
> 0.05% of the original on average. I'm now tuning to bring the error down
> further before testing on Thor.

## Next steps

Superseded: see "Remaining work" under Current status at the top. Phases 2–4
are complete.
