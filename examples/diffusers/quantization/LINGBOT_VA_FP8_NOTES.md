# lingbot-va FP8 Quantization: Integration Notes

## Goal

Adapt NVIDIA Model-Optimizer's FP8 PTQ quantization (`quantize.py`) — built for stock diffusers `WanPipeline`/Wan2.2 — to work with `lingbot-va-base`, a custom Wan2.2-derived video-action transformer, and measure memory/latency impact. Scope for this pass: get quantization running end-to-end and measure memory/latency, using synthetic calibration data (real RoboTwin calibration data deferred).

## Architecture background

- `lingbot-va-base` uses a custom `WanTransformer3DModel` (`wan_va/modules/model.py:569`) — a from-scratch reimplementation reusing diffusers building blocks, **not** diffusers' own class of the same name.
- Single shared backbone: the same 30 `blocks` are called twice per denoising step via an `action_mode` flag — `action_mode=False` routes through `patch_embedding_mlp`/`condition_embedder`/`proj_out` (video), `action_mode=True` routes through `action_embedder`/`condition_embedder_action`/`action_proj_out` (action). No `transformer_2` (unlike stock Wan2.2-14B's dual-transformer MoE).
- All layers are standard `nn.Linear`/`RMSNorm`/SDPA-based attention — quantization-hookable.
- No `WanPipeline`-style `__call__`; inference goes through `wan_va.wan_va_server.VA_Server`, which loads `vae`/`tokenizer`/`text_encoder`/`transformer` via its own loaders and runs a KV-cached, chunked, two-phase (video then action) denoising loop driven by an `obs` dict.
- Target config: `wan_va/configs/va_robotwin_cfg.py` (RoboTwin-2.0 eval) — 3 cameras (`cam_high`/`cam_left_wrist`/`cam_right_wrist`), 256×320 resolution, `frame_chunk_size=2`, 25 video + 50 action inference steps, `guidance_scale=5`.

## Files modified (Model-Optimizer repo, `examples/diffusers/quantization/`)

- **`models_utils.py`**: added `ModelType.LINGBOT_VA`, `MODEL_REGISTRY`/`MODEL_PIPELINE` entries (`MODEL_PIPELINE = None`, special-cased like `LTX2`), `MODEL_DEFAULTS[ModelType.LINGBOT_VA]` (backbone=`"transformer"`, mirrors `va_robotwin_cfg` values), wired into `_FILTER_FUNC_MAP`.
- **`utils.py`**: added `filter_func_lingbot_va` — excludes embedding/`proj_out`/action-head layers and boundary blocks (0,1,2,27,28,29 — 30 blocks, not stock Wan2.2's 40) from quantization.
- **`pipeline_manager.py`**: added `ModelType.LINGBOT_VA` special-case branch in `create_pipeline()` (mirrors `LTX2`'s pattern), `_create_lingbot_va_pipeline()` (constructs a real `VA_Server` from `va_robotwin_cfg`, never calls `init_distributed()` so the transformer stays unsharded/plain `nn.Module` — required for `mtq.quantize` to hook cleanly), `LingbotVAPipe` wrapper class (`generate(prompt, cam_images)` drives calibration through `VA_Server.infer()`), added to `setup_device()`'s skip-list. `iter_backbones()` needed **no** special case — the generic `getattr(self.pipe, "transformer")` path already works since `LingbotVAPipe.transformer` is a plain attribute.
- **`calibration.py`**: added `_LINGBOT_VA_DUMMY_PROMPT` / `_make_lingbot_va_dummy_obs` (synthetic `(480,640,3)` uint8 camera frames — arbitrary resolution is fine, `_encode_obs` resizes via `F.interpolate` internally), dispatch branch in `run_calibration()`, `_run_lingbot_va_calibration()`.
- **`check_memory.py`** (new standalone script, not part of `quantize.py`): GPU memory/latency probe. Builds a real `VA_Server` the same way `_create_lingbot_va_pipeline()` does, optionally `mto.restore()`s the FP8-compressed `transformer.pt` onto it, then runs one `infer()` denoise chunk while tracking `torch.cuda.max_memory_allocated()` and wall-clock latency. Runs the `baseline`/`quantized` cases as two isolated subprocesses (self-invoking via `--case`) so peak-memory stats from one case never leak into the other. Has hardcoded local paths (`LINGBOT_VA_REPO`, `MODEL_PATH`, `QUANTIZED_CKPT`) — needs editing before reuse elsewhere. Usage: `python check_memory.py`.

## Bugs found and fixed during review

1. `job_config.param_dtype = self.config.model_dtype` assigned a **dict** (`{"default": torch.bfloat16}`) instead of a plain `torch.dtype` — `VA_Server`/`wan_va` loaders don't support diffusers' per-component dtype-map convention. Fixed: `.get("transformer", self.config.model_dtype["default"])`.
2. Duplicate `pipeline_cls = MODEL_PIPELINE[...]` line left over from inserting the `LINGBOT_VA` branch — harmless, removed.
3. **`utils` module name collision**: `wan_va_server.py` does `from utils import (...)` expecting its own `wan_va/utils/` package, but Python's `sys.modules` cache already holds Model-Optimizer's own `quantization/utils.py` under the same name (imported at program startup by `models_utils.py`/`calibration.py`), and cache lookup happens before `sys.path` is ever consulted — so no `sys.path` reordering fixes it. Fixed by manually building the `wan_va/utils` package via `importlib.util.spec_from_file_location(..., submodule_search_locations=[...])` and injecting it into `sys.modules["utils"]` before triggering the `wan_va_server` import, then restoring our own module afterward.
4. `VA_Server._reset()` builds its save-directory name directly from the raw prompt string with no truncation (`wan_va_server.py:438-439`) — real OpenVid-1M captions are full paragraph-length, blowing past OS filename length limits (`ENAMETOOLONG`). Fixed by always using a short fixed dummy prompt in `_run_lingbot_va_calibration`, ignoring the loaded (real but irrelevant) caption batch.

## Environment

Reused the existing `lingbot-va` conda env (`/home/scratch.thannan_wwfo/miniforge-aarch64/envs/lingbot-va`, aarch64, matches the GB200 node's architecture) rather than creating a new one — `wan_va` and `modelopt` must be importable in the same process regardless. Already had `torch==2.9.0+cu128` (correct for GB200/Blackwell — `cu126` lacks `sm_100` kernels), `diffusers==0.36.0`, `transformers==4.55.2` matching lingbot-va's README pins. Added: `nvidia-modelopt[onnx,hf]`, `datasets`, `nvtx` (per Model-Optimizer's own `Pre-Requisites` README section).

## Command used

```sh
python quantize.py \
    --model lingbot-va \
    --override-model-path /home/thannan/scratch/robotics/model/lingbot-va-base \
    --extra-param lingbot_va_repo=/home/thannan/scratch/robotics/lingbot-va \
    --model-dtype BFloat16 \
    --format fp8 --batch-size 1 --calib-size 1 --collect-method default \
    --compress \
    --quantized-torch-ckpt-save-path ./lingbot_va_fp8_compressed.pt
```

Completed successfully (~52-66s for `--calib-size 1`).

## Results

**Disk size**: 5.9G (compressed) vs 9.6G (bf16 baseline) — **~39% reduction**. Less than the theoretical ~50% floor because `filter_func_lingbot_va` keeps several layers in bf16 (embeddings, boundary blocks, action heads).

**Root cause of missing real FP8 GEMM (resolved)**: Every quantized layer initially fell back to a dequantize-FP8-to-bf16-then-matmul path (`RealQuantLinear: No real-quant GEMM found` on every layer). Root-caused to a Model-Optimizer packaging bug, not a hardware/config issue: `modelopt/torch/quantization/backends/fp8_per_tensor_gemm.py` defines the FP8 per-tensor real-GEMM kernel and registers it via a module-level `gemm_registry.register(...)` call, but `backends/__init__.py` never imports that module (`from .gemm_registry import *` / `from .nvfp4_gemm import *` only — no `fp8_per_tensor_gemm`). So the kernel exists but is never registered in any normal code path; confirmed identical in both the source checkout (`0.48.0.dev17+g946639aa1`) and the installed `nvidia-modelopt==0.46.1` package. Not a Blackwell/`sm_100` gate (capability check only requires `>= (8,9)`), not a config mismatch, not a `modelopt_cuda_ext_fp8` gap (that extension backs the fake-quant simulation path, unrelated). Worth filing upstream.

**Workaround**: add `from modelopt.torch.quantization.backends import fp8_per_tensor_gemm  # noqa: F401` before running inference — executes the registration side effect. Confirmed working: zero `"No real-quant GEMM"` warnings after adding it.

**GPU memory and latency** (measured via a standalone `check_memory.py`, spawning a subprocess per case for clean isolation, driving a real forward pass through `VA_Server.infer()` with synthetic camera frames, `--batch-size 1`, `frame_chunk_size=2`, 256×320):

| | Cold | Warm avg (2 iters) | Peak load | Peak forward |
|---|---|---|---|---|
| Baseline (bf16) | 6.96 s | 6.36 s | 24.39 GB | 31.89 GB |
| Quantized, fallback path (before workaround) | 18.21 s | 14.38 s | 24.45 GB | 28.49 GB |
| Quantized, real FP8 GEMM (after workaround) | 28.95 s | 12.97 s | 24.45 GB | 28.29 GB |

Even with the real FP8 GEMM kernel confirmed active, the quantized model is still **~2x slower than baseline** (12.97s vs 6.36s warm) and load-time memory is unchanged (24.45 vs 24.39 GB — expected, since checkpoint loading happens before any forward-pass GEMM path is exercised). Forward-pass peak memory dropped modestly (~11%, 31.89→28.29 GB) but not the ~50% a full real-FP8 win would suggest. Cold latency got *worse* with the real kernel active (18.2s→28.95s) — likely one-time kernel-selection/JIT overhead the first time the real GEMM path runs.

Likely explanation: at `--batch-size 1` with short per-chunk sequences, FP8 GEMM kernels don't have enough matmul work to amortize their fixed overhead (scale application, kernel dispatch) against a well-optimized bf16 GEMM — real wins from FP8 typically show up at larger batch sizes where compute, not overhead, dominates. Additionally, the layers deliberately excluded from quantization (boundary blocks, action heads, embeddings) still run in bf16 regardless.

**Bottom line**: FP8 quantization for `lingbot-va` currently delivers disk savings only (~39%), with a real (now-registered) GEMM path available but not yet a net memory or latency win at this batch size/config.

## Not yet done

- Real RoboTwin calibration data (currently synthetic zero-frames + fixed dummy prompt) — needed before treating amax/quantizer scales as meaningful for accuracy.
- Round-trip / quality verification against a bf16 baseline using real inputs.
- Whether FP8 breaks even or wins over bf16 at larger batch sizes (untested).
- Filing the `fp8_per_tensor_gemm` registration gap as an upstream Model-Optimizer bug.
