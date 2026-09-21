# lingbot-va FP8 Quantization

Adapted ModelOpt's FP8 PTQ (`quantize.py`) — built for stock diffusers `WanPipeline`/Wan2.2 — to work with `robbyant/lingbot-va-base`. Added a `ModelType.LINGBOT_VA` and ran it end-to-end. Using synthetic calibration data for now, real RoboTwin data still needed.

Checkpoint: [`robbyant/lingbot-va-base`](https://huggingface.co/robbyant/lingbot-va-base).

```sh
python Model-Optimizer/examples/diffusers/quantization/quantize.py \
    --model lingbot-va \
    --override-model-path /home/thannan/scratch/robotics/model/lingbot-va-base \
    --extra-param lingbot_va_repo=/home/thannan/scratch/robotics/lingbot-va \
    --model-dtype BFloat16 \
    --format fp8 --batch-size 1 --calib-size 1 --collect-method default \
    --quantized-torch-ckpt-save-path ./lingbot_va_fp8_fakequant.pt
```

Don't add `--compress` here. It stores real low-precision weights and turns off `fake_quant` (`modelopt/torch/quantization/compress.py:91`), but the FP8 ONNX exporter (`onnx_utils/export.py:export_fp8`) needs that fake-quant round-trip live in the graph to emit `QuantizeLinear`/`DequantizeLinear` nodes for TensorRT. Found this out the hard way — once compressed, there's no going back.

### Files touched (`examples/diffusers/quantization/`)

- `models_utils.py` — registered lingbot-va as a known model type.
- `utils.py` — skip-list rule (embeddings, first/last blocks, action-output layers stay full precision).
- `pipeline_manager.py` — loads lingbot-va and feeds it calibration data, since it's not a standard model type.
- `calibration.py` — synthetic calibration inputs (dummy camera images + placeholder prompt).

Commit with the full diff: https://github.com/thannan-ctrl/Model-Optimizer/commit/91d79a2800931eb1744a52e8c082bf7a2569bcc1

### Environment

1. Follow lingbot-va's own install instructions first: https://github.com/thannan-ctrl/lingbot-va#installation (`torch==2.9.0+cu128` for Blackwell/`sm_100`, `diffusers==0.36.0`, `transformers==4.55.2`).
2. Then, in the same env:
   ```sh
   pip install nvidia-modelopt[onnx,hf]
   pip install -r /home/thannan/scratch/robotics/Model-Optimizer/examples/diffusers/requirements.txt
   pip install datasets
   ```

## TensorRT integration (lingbot_trt on igx-thor)

Trying to get a real FP8-on-TensorRT speedup number, combining this checkpoint with Maycon's TensorRT trunk port (`igx-thor:~/workspaces/tanveer/lingbot_trt`).

```sh
rsync -avz --progress lingbot_va_fp8_fakequant.pt igx-thor:~/workspaces/tanveer/lingbot_trt/
```

Restore it with `mto.restore()` onto the transformer loaded through `lerobot.policies.lingbot_va.modeling_lingbot_va.LingBotVAPolicy`.

One more wrinkle: `export_fp8` in `onnx_utils/export.py` is a legacy TorchScript ONNX symbolic function, only fires under `torch.onnx.export(..., dynamo=False)`. But `lingbot_trt/export_onnx.py` uses the dynamo exporter (`dynamo=True`). Good news — tested it, and the legacy tracer (`dynamo=False`, swap `dynamic_shapes` for `dynamic_axes`) still handles `TrunkWrapper`'s graph fine. So the plan is: legacy exporter + `export_fp8` symbolic hook + `convert_zp_fp8` graph-surgery pass, same as `modelopt_export_sd` does for other models — no need to write new dynamo-specific QDQ code.

### Still to do
- Check the fake-quant checkpoint actually produces real QDQ nodes through the legacy exporter, that the engine builds, and that it's actually faster than bf16 (in progress on igx-thor).
- Real RoboTwin calibration data.
- Quality check against bf16 with real inputs.
