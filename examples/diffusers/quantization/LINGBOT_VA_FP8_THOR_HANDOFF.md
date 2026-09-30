# LingBot-VA FP8: handoff to IGX Thor

What to build on Thor, from the GB200 recalibration (details in `calibration.md`).

## What passed on GB200

- **Checkpoint:** `results/C/transformer.pt` in this directory. It's a
  `mto.save` fake-quant checkpoint of about 9.6 GB, made with **modelopt
  0.46.1** and max calibration. Don't use `--compress`.
- **Weights:** calibrated on the `robbyant/lingbot-va-posttrain-robotwin`
  weights, the same RoboTwin model faster-wam runs.
- **FP8 layers:** 240 Linear layers in blocks 3–26. Blocks 0–2 and 27–29,
  the embedders, the heads and all attention math stay bf16.
- **Call routing:** some calls run the whole transformer in **bf16**, the
  rest in FP8:

  | Call | Precision |
  |---|---|
  | KV-cache writes: `update_cache` 1 (commit) or 2 (`compute_kv_cache`) | **bf16** |
  | Every video call of an episode's first chunk | **bf16** |
  | Last video denoising step (t ≤ 172.4) | **bf16** |
  | Last 2 action denoising steps (t ≤ 40) | **bf16** |
  | Everything else | FP8 |

- **Result on GB200** (50 episodes × 4 chunks, every call checked): worst
  per-call rel_mean **3.92%** (gate 4%), mean 0.92%. End-to-end action
  drift is **5.3e-4** normalized mean abs, vs 4.4e-4 for bf16 TensorRT on
  Thor.
- **Without the routing** (all calls FP8), the worst call is 13.6% or more.
  Drift is 1.4e-3.

## Build

Follow `faster-wam/lingbot-va/FP8_GUIDE.md` B0–B4 to build the FP8 trunk
engine from `results/C/transformer.pt`, and keep the existing bf16 trunk
engine. You need **both**:

| Engine | Weights (from FP8_GUIDE) |
|---|---|
| `trunk.plan` (bf16, existing) | 8.1 GiB |
| `trunk_fp8.plan` (new) | 4.9 GiB |

Loaded together that's about 13 GiB of weights: more memory than bf16 alone,
in exchange for about 1.21–1.25× transformer speed (estimate in
`calibration.md`).

## Fixes needed before exporting

1. **`mto.restore` overwrites the weights.** `export_onnx_fp8_test.py`
   restores the checkpoint onto the LeRobot policy, and that also loads the
   checkpoint's weights.
   - With checkpoint C that's intended, since C holds the RoboTwin weights.
     On GB200, C with all quantizers off matches bf16 exactly.
   - Still, **assert** that the restored block weights equal the policy's
     own weights before exporting. The first FP8 attempt silently ran
     `lingbot-va-base` weights this way.
2. **The bf16 cast.**
   - `WanTransformer3DModel.from_pretrained` keeps `_keep_in_fp32_modules`
     (the norms, `scale_shift_table` and `time_embedder`) in fp32. lingbot-va's
     `VA_Server` then casts the whole model to bf16.
   - On GB200, skipping that cast made two bf16 copies differ by up to 4.9%.
   - Check that the Thor policy's transformer parameters are all bf16, as
     `VA_Server` has them, both before calibration comparisons and before
     export.
3. **Calibrator class in the checkpoint.** Not an issue for C: max
   calibration uses modelopt's own calibrators. Checkpoints made with
   `--act-calib` store `config.FixedMethodHistogramCalibrator` and need it
   importable, or allowlisted with `torch.serialization.add_safe_globals`.

## Routing on Thor

`TrunkForward` (`trunk_wrapper.py`) already sees everything the router needs.
Give it both engines and pick one per call:

```python
class RoutedTrunkForward(TrunkForward):
    """FP8 trunk by default; bf16 trunk for cache writes, the first chunk's
    video, and the last low-noise denoising steps."""

    VIDEO_T_MAX = 172.414   # last video step (25 steps, snr_shift 5)
    ACTION_T_MAX = 40.0     # last 2 action steps (50 steps, snr_shift 1)

    def __init__(self, transformer, pool_mgr, trunk_fp8, trunk_bf16):
        super().__init__(transformer, pool_mgr, trunk_fp8)
        self.trunk_fp8, self.trunk_bf16 = trunk_fp8, trunk_bf16

    def _use_bf16(self, input_dict, update_cache, action_mode):
        if update_cache:                          # commit / compute_kv_cache
            return True
        t = float(input_dict["timesteps"].max())  # max: chunk 0 zeroes the cond frame's t
        if not action_mode and int(input_dict["grid_id"][0, 0].min()) == 0:
            return True                           # first chunk (frame_st_id == 0), video
        return t <= (self.ACTION_T_MAX if action_mode else self.VIDEO_T_MAX)

    def __call__(self, input_dict, update_cache=0, cache_name="pos", action_mode=False, train_mode=False):
        self.trunk_fn = self.trunk_bf16 if self._use_bf16(input_dict, update_cache, action_mode) else self.trunk_fp8
        return super().__call__(input_dict, update_cache, cache_name, action_mode, train_mode)
```

Notes:
- **Thresholds:** they're the scheduler's own timesteps, `timesteps[-1]` for
  video with 25 steps and `timesteps[-2]` for action with 50 steps. Recompute
  them if the step counts or SNR shifts change. On GB200 they were read from
  `VA_Server.scheduler` / `action_scheduler` after `set_timesteps`.
- **First-chunk test:** it assumes `grid_id[:, 0]` is the frame index with
  offset `frame_st_id`, as in lingbot-va's `get_mesh_id`. Check it on Thor
  with a quick print on chunk 0 vs chunk 1, or pass the chunk index in
  explicitly if the LeRobot policy exposes it.
- **Engine I/O:** both engines take the same inputs and write K/V into the
  same pool, so switching per call needs no data movement.

## Validation on Thor

1. **Torch parity (fixtures).** `parity_torch.py` with the routed trunk:
   calls 0/1/25/26/76 plus the pool snapshot. Calls 25 and 76 are cache
   writes, so they now run bf16 and should match the bf16 fixtures to within
   bf16 TensorRT noise (about 1%).
2. **TensorRT parity:** the same, with both TensorRT engines.
3. **Closed-loop RoboTwin success rate:** bf16 vs routed FP8. This is the
   real sign-off; GB200 only measured open-loop drift on recorded episodes.
4. **Timing:** per chunk and per episode, vs bf16. GB200's estimate is
   about 5.1 s vs 6.4 s per chunk after the first chunk, which is slower
   (about 5.7 s) because all its video runs in bf16.
