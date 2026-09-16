# lingbot-va FP8 Quantization

Adapted NVIDIA Model-Optimizer's FP8 PTQ quantization (`quantize.py`) — built for stock diffusers `WanPipeline`/Wan2.2 — to work with `robbyant/lingbot-va-base`. Added a new `ModelType.LINGBOT_VA`, ran it end-to-end, and measured GPU memory/latency impact against a bf16 baseline. Synthetic calibration data was used for this pass (real RoboTwin calibration data deferred).

Checkpoint: [`robbyant/lingbot-va-base`](https://huggingface.co/robbyant/lingbot-va-base) on Hugging Face.

```sh
python Model-Optimizer/examples/diffusers/quantization/quantize.py \
    --model lingbot-va \
    --override-model-path /home/thannan/scratch/robotics/model/lingbot-va-base \
    --extra-param lingbot_va_repo=/home/thannan/scratch/robotics/lingbot-va \
    --model-dtype BFloat16 \
    --format fp8 --batch-size 1 --calib-size 1 --collect-method default \
    --compress \
    --quantized-torch-ckpt-save-path ./lingbot_va_fp8_compressed.pt
python Model-Optimizer/examples/diffusers/quantization/check_memory.py
```

## **Result**:

| | Cold | Warm avg (2 iters) | Model load | Peak forward |
|---|---|---|---|---|
| Baseline, BF16 | 6.96 s | 6.36 s | 24.39 GB | 31.89 GB |
| Quantized, FP8 | 28.95 s | 12.97 s | 24.45 GB | 28.29 GB |



## Details

### Files modified (`examples/diffusers/quantization/`)

- **`models_utils.py`**: registered lingbot-va as a new model the quantization tool knows about.
- **`utils.py`**: added a rule for which layers to skip when quantizing (keeps embeddings, the first/last few blocks, and the action-output layers at full precision, since those are most sensitive to precision loss).
- **`pipeline_manager.py`**: taught the tool how to actually load lingbot-va — it isn't a standard model type the tool already understands, so this builds it directly and feeds it calibration data.
- **`calibration.py`**: added fake input data (dummy camera images + a placeholder prompt) so the quantizer has something to run through the model during calibration.

See [this commit](https://github.com/thannan-ctrl/Model-Optimizer/commit/91d79a2800931eb1744a52e8c082bf7a2569bcc1) for the exact code changes.


### Environment

1. Follow [lingbot-va's own installation instructions](https://github.com/thannan-ctrl/lingbot-va#installation) first — gets you `torch==2.9.0+cu128` (required for Blackwell/`sm_100`), `diffusers==0.36.0`, `transformers==4.55.2`, etc.
2. Then add Model-Optimizer's deps on top, in the same env:
   ```sh
   pip install nvidia-modelopt[onnx,hf]
   pip install -r /home/thannan/scratch/robotics/Model-Optimizer/examples/diffusers/requirements.txt
   pip install datasets
   ```

### Not yet done
- Real RoboTwin calibration data (currently synthetic).
- Quality verification against bf16 with real inputs.
