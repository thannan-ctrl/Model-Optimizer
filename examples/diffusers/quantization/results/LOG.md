# LingBot-VA FP8 recalibration log

Parity: 50 fixed eval episodes (`eval_manifest.json`), K=4, strides 5/10 plus all
cache writes. The gate is worst call ≤ 4%, outliers explained (see
`../calibration.md`). "Raw" = rel_mean before near-zero handling.

| Run | Date | Calibration | Quantized blocks | Worst (judged) | Raw worst | Mean | Worst area | Result | Notes |
|---|---|---|---|---|---|---|---|---|---|
| A | 2026-09-30 | `dummy`, 50 runs | 24 (3–26) | 16.1% | 16.1% | 3.9% | Action cache writes: output 15–16%, K/V 12%. Video cache writes: K/V 13.8% (blocks 23, 29, 24). Denoising: video 5.4%, action 1.9% | FAIL | Re-ran with the gate script: identical numbers (deterministic), and 0 near-zero tensors, so the failures are real error. 835/2744 calls > 4%. Parity took 44 min |
| B | 2026-09-30 | `first_chunk`, C's 50 episodes | 24 (3–26) | 15.0% | 15.0% | 3.9% | Same as A: action cache-write outputs 15%, video V in blocks 23/24 13.6%. Denoising: video 5.4%, action 2.1% | FAIL | Real data barely changes anything vs A (means identical to 0.1 pt; 833 vs 835 calls > 4%). 0 near-zero tensors |
| C | 2026-09-30 | `replay`, 50 episodes, K=4 (198 chunks) | 24 (3–26) | 15.9% | 15.9% | 3.9% | Same as A/B: action cache-write outputs 15.9%, video V in blocks 23/24 13.6%. Denoising: video 5.6%, action 2.0% | FAIL | Multi-chunk and cache-write calibration changes nothing vs A/B (833 calls > 4%). Calibration data is not the cause |

## Localization sweep on C (2026-09-30, fast subset: 10 episodes × K=2, 280 calls; `loc/ranking.tsv`)

Baseline on the subset: worst 12.8%, mean 4.0%.

| Back in bf16 | Worst | Mean | Weight quantizers left |
|---|---|---|---|
| All input (activation) quantizers → weight-only FP8 | 8.5% | 2.7% | 240 |
| FFN up, all blocks | 10.1% | 3.4% | 216 |
| FFN down, all blocks | 10.8% | 3.5% | 216 |
| attn1 Q/K/V, all blocks | 11.9% | 3.8% | 168 |
| attn1 out, all blocks | 12.1% | 3.7% | 216 |
| All weight quantizers → activation-only FP8 | 12.3% | 3.4% | 0 |
| attn1 `to_v`, all blocks / blocks 23–24 | 12.3% / 12.4% | 3.9% | 216 / 238 |
| attn2 (Q/K/V or out) | 12.8–13.0% | 4.0% | 168–216 |
| Any single block 3–26 | 12.0–13.0% | 3.9–4.0% | 230 |

## Round 2 (2026-09-30)

Drift on C, 50 episodes × K=4, normalized [-1, 1] actions:
- **FP8:** mean abs 0.0014. Two componentwise outliers of about 1.0 are
  q/−q sign flips (really 1–2.5° of rotation).
- **FP8 + bf16 KV cache retention:** mean abs 0.0009, worst 0.14, per-call
  gate 5.6% worst / 1.0% mean.
- **bf16 vs bf16:** 0.

Recalibration variants, fast subset (10 episodes × K=2). Each cell is worst /
mean; "retention" means bf16 KV cache retention.

| Run | All FP8 | With retention |
|---|---|---|
| max (C) | 12.8 / 4.0 | **4.8 / 1.2** |
| WMSE | 12.8 / 4.0 | 5.4 / 1.2 |
| AMSE | 12.9 / 4.0 | 5.1 / 1.2 |
| WAMSE | 12.8 / 4.0 | 5.1 / 1.2 |
| P99999 | 44.0 / 9.2 | 6.5 / 2.1 |
| P9999 | 81.1 / 15.8 | 14.4 / 4.2 |

Checkpoints deleted per the keep-A/B/C-plus-best rule: dry, P9999, P99999,
WMSE, AMSE, WAMSE.

With max + retention, all 49/2744 calls over 4% are chunk-0 video denoising
(worst 5.6%). Chunks 1–3 are at most 3.3%.

## Every-call confirmation (2026-09-30)

Full eval set, 50 episodes × K=4, stride 1, 15,484 calls. All runs use
checkpoint C.

| Setup (bf16 calls) | Worst | Mean | Calls > 4% | Drift (normalized mean abs) | Result |
|---|---|---|---|---|---|
| Cache writes + chunk-0 video | 7.5% (action step 49) | 1.06% | 123 | 0.00088 | FAIL |
| Cache writes + chunk-0 video + last video step + last 2 action steps | **3.92%** (video step 23) | 0.92% | 0 | **0.00053** | **PASS** |
| Same, second eval seed (2000), 50 new episodes | **3.88%** (video step 23) | 0.93% | 0 | **0.00048** | **PASS** |
