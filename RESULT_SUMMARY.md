# LingBot-VA Inference: PyTorch vs TensorRT bf16 vs TensorRT FP8 on IGX Thor

**What is being compared:** the same LingBot-VA v1 (RoboTwin) model over 4-chunk episodes
(25 video + 50 action denoising steps per chunk, CFG batch 2), with three backends for the
30-block transformer trunk:
- **PyTorch:** eager bf16.
- **TRT bf16:** `engines/trunk.plan`.
- **TRT FP8 routed:** a new FP8 engine (`engines/trunk_fp8_C.plan`, calibrated on real RoboTwin
  data). Calls switch between it and the bf16 engine: memory writes, the first chunk's video
  and the last denoising steps stay bf16, and FP8 handles the other 84% of calls.

Inputs, prompt and seed are identical, all on one NVIDIA IGX Thor (MAXN, TensorRT 10.13).
Numbers are medians of 4 episodes, all measured on 2026-10-01.

| | **PyTorch** | **TRT bf16** | **TRT FP8 routed** | **FP8 vs bf16** |
|---|---:|---:|---:|---:|
| **Latency** (transformer) | | | | |
| Video call, 240 tokens | 176 ms | 93 ms | 72 ms | 1.29× faster |
| Action call, 32 tokens | 103 ms | 58 ms | 43 ms | 1.35× faster |
| Chunk 0 (77 calls) | 8.98 s | 5.14 s | 4.30 s | 1.20× faster ¹ |
| Each later chunk (79 calls) | 9.66–10.46 s | 5.36–5.83 s | 4.13–4.55 s | **1.28–1.30× faster** |
| Whole 4-chunk episode | 39.2 s | 21.9 s | 17.4 s | **1.26× faster** ² |
| **Accuracy** (vs PyTorch) | | | | |
| Chunk-0 actions, max / mean abs. deviation ([-1,1] range) | reference | 3.9e-3 / 4.4e-4 | 3.9e-3 / 3.9e-4 | same |
| Chunks 1–3 actions, mean abs. deviation ³ | reference | 6.0–7.4e-3 | 5.9–7.2e-3 | same |
| Per-call error of FP8 calls vs bf16, average / worst (limit 4%) | – | – | 1.14% / 3.10% | ✓ |
| **Memory** | | | | |
| TRT engine weights | – | 8.1 GiB | 13.0 GiB (both engines) | +4.9 GiB |
| Peak total, first episode (whole process) ⁴ | 31.2 GiB | 41.1 GiB | 47.1 GiB | +5.9 GiB |
| **Build** (trtexec) | – | 70 s | 137 s | |

¹ Chunk 0 gains less because all of its video steps run in bf16.

² PyTorch vs TRT bf16 is 1.79×, and PyTorch vs FP8 routed is 2.26×. July's summary measured
6.45 s per chunk for TRT bf16; today the same engine takes 5.1 s, while PyTorch is unchanged
(9.16 → 8.98 s). The cause of the earlier figure is unknown. Not included above: the UMT5 text
encoder runs on the **CPU** and takes about 13 s per episode with every backend. Moving it to
the GPU is the biggest end-to-end lever.

³ Each chunk's actions feed into the next, so later chunks drift more for every backend. FP8
routed is as close to PyTorch as TRT bf16.

⁴ CPU and GPU share one memory on Thor, so this includes the 11 GiB CPU-side text encoder.
The TRT setups also keep PyTorch's unused 8.1 GiB copy of the trunk weights; dropping it is an
easy saving. PyTorch's **reserved** memory also grows about 8 GiB per episode with every
backend. That's allocator fragmentation, and it needs fixing before long runs.

**External HIL deployment benchmark.** These come from a separate closed-loop benchmark that wasn't run
as part of this work. They're reported as received, 10 episodes per setup:

| Metric | Original bf16 ⁵ | TRT FP8 engine only | TRT FP8 routed |
|---|---:|---:|---:|
| Task successes | 10/10 (100%) | 9/10 (90%) | 8/10 (80%) |
| Observed success-rate difference | reference | −10 pp | −20 pp |
| Mean server inference per action chunk | 9.041 s | 4.434 s | 5.693 s |
| Median server inference per action chunk | 8.754 s | 4.159 s | 4.658 s |
| Mean KV-cache commit | 0.567 s | 0.474 s | 0.543 s |
| Total simulator steps | 1,857 | 2,202 | 2,970 |
| Setup / runtime failures | 0 / 0 | 0 / 0 | 0 / 0 |
| PyTorch allocation | 30.03 GiB peak | 29.95 GiB sampled max | 29.96 GiB sampled max |
| Additional native TRT weights + context | none | ≈5.87 GiB | ≈14.99 GiB |

⁵ As named in the report; its timing is close to our PyTorch numbers. Ten episodes per setup can't
separate 10/10, 9/10 and 8/10, so more episodes are needed.

**Bottom line:** TRT bf16 makes the transformer 1.8× faster than PyTorch with no accuracy loss.
Routed FP8 adds another **1.26×** (2.26× over PyTorch), for **+5.9 GiB**, with actions as close
to PyTorch as bf16's in offline tests. In an external closed-loop HIL benchmark (not run by us), both FP8 setups completed the
task in 8–9 of 10 episodes, against 10/10 for the original. That's too few episodes to tell
whether FP8 costs success rate, or whether routing helps.

---

FP8 method, reproduction steps and per-call details: [FP8_C_THOR_SESSION.md](./FP8_C_THOR_SESSION.md).
Original bf16 TRT design and validation: [COMPARISON.md](./COMPARISON.md)
