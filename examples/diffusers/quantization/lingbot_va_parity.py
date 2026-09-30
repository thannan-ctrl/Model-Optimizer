# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-call parity of a quantized lingbot-va transformer against bf16, on RoboTwin replays.

Port of faster-wam ``lingbot-va/parity_torch.py`` from one captured first chunk
on Thor to multi-chunk RoboTwin episodes on any GPU, through lingbot-va's own
VA_Server (no lerobot). Same metric:
``rel_mean = mean|test - ref| / mean|ref|``.

Teacher forcing: the bf16 VA_Server replays each eval episode
(robotwin_replay.py). For every selected transformer call, the bf16 KV cache is
copied into the test transformer, the test transformer runs on the same inputs,
then bf16 runs. So each rel_mean is that call's own error, with no drift from
earlier calls. Calls that write the cache (update_cache 1 = commit,
2 = compute_kv_cache) also compare the K/V they write.

Selected calls: every ``--video-stride``-th video denoising step, every
``--action-stride``-th action step, and all cache-writing calls.

Gate (worst call, outliers explained): each checked tensor (a call's output,
and each block's written K and V) is judged on rel_mean, unless its bf16
magnitude is near zero, where rel_mean blows up. A tensor is near zero if its
mean|bf16| is below ``--near-zero-frac`` of the median for its group (same
mode, kind, and for K/V the same block and tensor) over the whole run. Those
are judged on absolute error instead, as mean|test - bf16| over that group
median, and flagged in the report. A call passes if every tensor it checks is
within ``--threshold``. The outputs of cache-writing calls are recorded but not
gated: VA_Server discards them, and only the K/V they write is used.
``--resummarize parity.json`` re-applies the gate to a finished run.

``--test``:
  fp8           the quantized checkpoint (the gate)
  bf16          a second bf16 copy: the noise floor, expect ~0
  fp8-disabled  the checkpoint with all quantizers off: expect ~0, proves the
                restored weights are the bf16 weights

Example:
    python lingbot_va_parity.py \\
        --lingbot-va-repo /path/to/lingbot-va \\
        --model-path /path/to/lingbot-va-posttrain-robotwin \\
        --fp8-ckpt results/C/transformer.pt \\
        --eval-manifest results/eval_manifest.json \\
        --robotwin-data-dir /path/to/robotwin-clean-and-aug-lerobot \\
        --exclude-manifest results/C/calib_manifest.json \\
        --out results/C/parity.json
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

GATE = 0.04


def compare(ours, ref):
    """rel_mean as in faster-wam parity_torch.py, plus the magnitudes the gate needs."""
    ours, ref = ours.float(), ref.float()
    d = (ours - ref).abs()
    ref_scale = ref.abs().mean().item()
    mean_abs = d.mean().item()
    return {
        "rel_mean": mean_abs / max(ref_scale, 1e-8),
        "mean_abs": mean_abs,
        "max_abs": d.max().item(),
        "ref_scale": ref_scale,
    }


def _caches(transformer):
    return [block.attn1.attn_caches for block in transformer.blocks]


def sync_cache(ref, test, cache_name):
    """Make test's KV cache an exact copy of ref's (reusing test's buffers)."""
    for ref_c, test_c in zip(_caches(ref), _caches(test)):
        src = ref_c.get(cache_name)
        dst = test_c.get(cache_name)
        if src is None:
            test_c[cache_name] = None
        elif dst is None or any(dst[k].shape != src[k].shape for k in src):
            test_c[cache_name] = {k: v.clone() for k, v in src.items()}
        else:
            for k in src:
                dst[k].copy_(src[k])


class TeacherForcedParity:
    """Wraps ref.forward; runs test on the same inputs and cache for selected calls."""

    def __init__(self, ref, test, video_stride, action_stride):
        self.ref, self.test = ref, test
        self.video_stride, self.action_stride = video_stride, action_stride
        self.orig_forward = ref.forward
        self.records = []
        self.episode = None
        self.chunk = 0
        self.steps = defaultdict(int)

    def on_chunk(self, k):
        self.chunk = k
        self.steps.clear()

    def _selected(self, kind, action_mode, step):
        if kind != "denoise":
            return True
        return step % (self.action_stride if action_mode else self.video_stride) == 0

    def __call__(self, input_dict, update_cache=0, cache_name="pos", action_mode=False, train_mode=False):
        kind = {0: "denoise", 1: "commit", 2: "kv_cache"}[int(update_cache)]
        key = (bool(action_mode), kind)
        step = self.steps[key]
        self.steps[key] += 1
        if not self._selected(kind, action_mode, step):
            return self.orig_forward(input_dict, update_cache, cache_name, action_mode)

        sync_cache(self.ref, self.test, cache_name)
        ref_c0 = self.ref.blocks[0].attn1.attn_caches[cache_name]
        id_before, mask_before = ref_c0["id"].clone(), ref_c0["mask"].clone()

        test_in = {k: v.clone() for k, v in input_dict.items()}
        test_out = self.test(test_in, update_cache=update_cache, cache_name=cache_name, action_mode=action_mode)
        ref_out = self.orig_forward(input_dict, update_cache, cache_name, action_mode)

        rec = {
            "episode": self.episode,
            "chunk": self.chunk,
            "mode": "action" if action_mode else "video",
            "kind": kind,
            "step": step,
            "t": float(input_dict["timesteps"].flatten()[-1].item()),
            "out": compare(test_out, ref_out),
        }
        if update_cache:
            # Both caches started identical, so the written slots match; compare them per block.
            new = ref_c0["mask"] & (~mask_before | (ref_c0["id"] != id_before))
            rec["kv"] = [
                {"block": i, "tensor": name, **compare(test_c[cache_name][name][:, new], ref_c[cache_name][name][:, new])}
                for i, (ref_c, test_c) in enumerate(zip(_caches(self.ref), _caches(self.test)))
                for name in ("k", "v")
            ]
        self.records.append(rec)
        return ref_out


def load_transformer(model_path, dtype):
    """A second copy of the transformer, loaded exactly as VA_Server loads its own."""
    srv = sys.modules["wan_va.wan_va_server"]
    t = srv.load_transformer(
        str(Path(model_path) / "transformer"), torch_dtype=dtype, torch_device="cuda:0", attn_mode="torch"
    )
    # from_pretrained keeps _keep_in_fp32_modules (norms, scale_shift_table,
    # time_embedder) in fp32; VA_Server's _configure_model casts everything to
    # param_dtype, so do the same or the copies differ by ~1%.
    return t.eval().requires_grad_(False).to(dtype)


def quantizer_summary(model):
    weight = sum(
        1 for _, m in model.named_modules() if getattr(m, "weight_quantizer", None) is not None and m.weight_quantizer.is_enabled
    )
    inputs = sum(
        1 for _, m in model.named_modules() if getattr(m, "input_quantizer", None) is not None and m.input_quantizer.is_enabled
    )
    other = sorted(
        n
        for n, m in model.named_modules()
        if n.endswith(("bmm_quantizer", "softmax_quantizer")) and getattr(m, "is_enabled", False)
    )
    return {"weight": weight, "input": inputs, "attention": other}


def build_eval_manifest(args, job_config):
    from robotwin_episodes import plan_episodes, read_manifest, write_manifest
    from robotwin_replay import min_episode_length

    path = Path(args.eval_manifest)
    if path.exists():
        return read_manifest(path)
    if not args.robotwin_data_dir:
        raise ValueError(f"{path} doesn't exist; pass --robotwin-data-dir to create it")
    exclude = set()
    for m in args.exclude_manifest:
        exclude |= {(e["task_dir"], e["episode_index"]) for e in read_manifest(m)["episodes"]}
    min_length = min_episode_length(args.chunks, job_config.action_per_frame, job_config.frame_chunk_size)
    entries = plan_episodes(args.robotwin_data_dir, args.num_episodes, args.eval_seed, exclude, min_length)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(path, args.robotwin_data_dir, args.eval_seed, entries, chunks=args.chunks,
                   excluded_manifests=[str(m) for m in args.exclude_manifest])
    print(f"Wrote eval manifest ({len(entries)} episodes, {len(exclude)} excluded) to {path}")
    return read_manifest(path)


def _tensors(rec, include_discarded=False):
    """(group, stats) for every gated tensor of a call: its output, then each block's K and V.

    The outputs of cache-writing calls (commit, kv_cache) are recorded but not gated:
    VA_Server discards them (last video step with video_exec_step=-1, last action
    step, and both _compute_kv_cache calls). Only the K/V those calls write reaches
    later calls.
    """
    if rec["kind"] == "denoise" or include_discarded:
        yield (rec["mode"], rec["kind"], "out"), rec["out"]
    for kv in rec.get("kv", []):
        yield (rec["mode"], rec["kind"], kv["tensor"], kv["block"]), kv


def _raw_worst(rec):
    return max(st["rel_mean"] for _, st in _tensors(rec, include_discarded=True))


def judge(records, near_zero_frac):
    """Set rec["judged"] (the gated error), rec["rel_mean"] (raw worst) and rec["flags"] per call."""
    import statistics

    scales = defaultdict(list)
    for rec in records:
        for group, st in _tensors(rec):
            scales[group].append(st["ref_scale"])
    typical = {g: statistics.median(v) for g, v in scales.items()}

    for rec in records:
        rec["rel_mean"], rec["judged"], rec["flags"], rec["worst_tensor"] = 0.0, 0.0, [], "out"
        for group, st in _tensors(rec):
            name = "out" if group[2] == "out" else f"{group[2]}[{group[3]}]"
            rec["rel_mean"] = max(rec["rel_mean"], st["rel_mean"])
            judged = st["rel_mean"]
            if st["ref_scale"] < near_zero_frac * typical[group]:
                judged = st["mean_abs"] / max(typical[group], 1e-8)
                rec["flags"].append(
                    {"tensor": name, "rel_mean": st["rel_mean"], "abs_vs_typical": judged,
                     "ref_scale": st["ref_scale"], "typical_scale": typical[group]}
                )
            if judged > rec["judged"]:
                rec["judged"], rec["worst_tensor"] = judged, name


def summarize(records, threshold, near_zero_frac):
    judge(records, near_zero_frac)
    groups = defaultdict(list)
    for r in records:
        groups[(r["chunk"], r["mode"], r["kind"])].append(r)
    table = [
        {
            "chunk": c, "mode": m, "kind": k, "n": len(v),
            "worst": max(r["judged"] for r in v),
            "mean": sum(r["judged"] for r in v) / len(v),
            "worst_raw_rel": max(r["rel_mean"] for r in v),
        }
        for (c, m, k), v in sorted(groups.items())
    ]
    worst = max(r["judged"] for r in records)
    flagged = [r for r in records if r["flags"]]
    return {
        "num_calls": len(records),
        "worst": worst,
        "mean": sum(r["judged"] for r in records) / len(records),
        "worst_raw_rel_mean": max(r["rel_mean"] for r in records),
        "threshold": threshold,
        "near_zero_frac": near_zero_frac,
        "num_flagged_calls": len(flagged),
        "pass": worst <= threshold,
        "table": table,
        "top20": sorted(records, key=lambda r: -r["judged"])[:20],
    }


def free_running_summary(drift):
    """Action drift, physical units and the model's normalized [-1, 1] space, overall and per chunk."""
    by_chunk = defaultdict(list)
    for d in drift:
        by_chunk[d["chunk"]].append(d)
    return {
        "worst_rel_mean": max(d["rel_mean"] for d in drift),
        "worst_max_abs": max(d["max_abs"] for d in drift),
        "worst_norm_max_abs": max(d["norm"]["max_abs"] for d in drift),
        "mean_norm_mean_abs": sum(d["norm"]["mean_abs"] for d in drift) / len(drift),
        "by_chunk": {
            c: {
                "n": len(v),
                "mean_rel_mean": sum(d["rel_mean"] for d in v) / len(v),
                "worst_rel_mean": max(d["rel_mean"] for d in v),
                "mean_norm_mean_abs": sum(d["norm"]["mean_abs"] for d in v) / len(v),
                "worst_norm_max_abs": max(d["norm"]["max_abs"] for d in v),
            }
            for c, v in sorted(by_chunk.items())
        },
        "per_chunk": drift,
    }


class Bf16CallPolicy:
    """Run selected calls of a quantized model with every quantizer off (i.e. in bf16).

    - ``cache_writes``: calls that write the KV cache (update_cache 1 and 2), about
      4 of ~79 per chunk. "bf16 KV cache retention".
    - ``first_chunk_video``: every video call of chunk 0 (the first chunk of an
      episode, when the cache holds only the first frame).
    - ``video_t_max`` / ``action_t_max``: denoising calls whose timestep is at or
      below this (the last few, low-noise steps, where FP8 error is largest).

    All other calls stay quantized. Deployment equivalent: a bf16 engine for the
    selected calls and an FP8 engine for the rest. ``chunk`` must be kept current
    via ``on_chunk`` (replay_episode's hook).
    """

    def __init__(self, model, cache_writes=False, first_chunk_video=False, video_t_max=None, action_t_max=None):
        from modelopt.torch.quantization.nn import TensorQuantizer

        self.cache_writes, self.first_chunk_video = cache_writes, first_chunk_video
        self.video_t_max, self.action_t_max = video_t_max, action_t_max
        self.chunk = 0
        self.quantizers = [m for m in model.modules() if isinstance(m, TensorQuantizer) and m.is_enabled]
        self.orig_forward = model.forward
        model.forward = self.forward

    def on_chunk(self, k):
        self.chunk = k

    def wants_bf16(self, update_cache, action_mode, t=None):
        if self.cache_writes and update_cache:
            return True
        if self.first_chunk_video and self.chunk == 0 and not action_mode:
            return True
        t_max = self.action_t_max if action_mode else self.video_t_max
        return not update_cache and t_max is not None and t is not None and t <= t_max

    def forward(self, input_dict, update_cache=0, cache_name="pos", action_mode=False, train_mode=False):
        # max over frames: chunk 0's conditioning frame has its timestep zeroed.
        t = float(input_dict["timesteps"].max())
        if not self.wants_bf16(update_cache, action_mode, t):
            return self.orig_forward(input_dict, update_cache, cache_name, action_mode)
        for q in self.quantizers:
            q.disable()
        try:
            return self.orig_forward(input_dict, update_cache, cache_name, action_mode)
        finally:
            for q in self.quantizers:
                q.enable()


def report(summary, free_running=None):
    print(f"\n{'chunk':>5} {'mode':>6} {'kind':>8} {'n':>5} {'worst':>10} {'mean':>10} {'raw worst':>10}")
    for row in summary["table"]:
        print(f"{row['chunk']:>5} {row['mode']:>6} {row['kind']:>8} {row['n']:>5} "
              f"{row['worst']:>10.4e} {row['mean']:>10.4e} {row['worst_raw_rel']:>10.4e}")
    print("\nTop 5 worst calls (judged):")
    for r in summary["top20"][:5]:
        flags = f" flagged={[f['tensor'] for f in r['flags']][:4]}" if r["flags"] else ""
        print(f"  ep={r['episode']} chunk={r['chunk']} {r['mode']} {r['kind']} step={r['step']} t={r['t']:.1f} "
              f"judged={r['judged']:.4e} ({r['worst_tensor']}) raw={r['rel_mean']:.4e}{flags}")
    if free_running:
        fr = free_running
        print(f"\nFree-running action drift: worst rel_mean={fr['worst_rel_mean']:.4e}, "
              f"worst max_abs={fr['worst_max_abs']:.4e} (physical units)")
        if "worst_norm_max_abs" in fr:
            print(f"  normalized [-1,1]: worst max_abs={fr['worst_norm_max_abs']:.4e}, "
                  f"mean abs={fr['mean_norm_mean_abs']:.4e}  (Thor bf16 TRT: 3.9e-3 / 4.4e-4)")
            for c, v in fr["by_chunk"].items():
                print(f"  chunk {c}: n={v['n']} rel_mean mean={v['mean_rel_mean']:.4e} worst={v['worst_rel_mean']:.4e} "
                      f"norm mean abs={v['mean_norm_mean_abs']:.4e} norm worst max_abs={v['worst_norm_max_abs']:.4e}")
    print(f"\nWORST (judged): {summary['worst']:.4e}  mean: {summary['mean']:.4e}  "
          f"raw worst rel_mean: {summary['worst_raw_rel_mean']:.4e}  over {summary['num_calls']} calls, "
          f"{summary['num_flagged_calls']} with near-zero tensors judged on absolute error")
    print(f"{'PASS' if summary['pass'] else 'FAIL'} (threshold {summary['threshold']:.0%})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resummarize", nargs="+", metavar="JSON",
                   help="Re-apply the gate to finished parity JSON(s), then exit. Several JSONs "
                        "(episode shards) are merged into --out")
    p.add_argument("--lingbot-va-repo")
    p.add_argument("--model-path", help="bf16 RoboTwin weights (the calibration weights)")
    p.add_argument("--test", choices=["fp8", "bf16", "fp8-disabled"], default="fp8")
    p.add_argument("--fp8-ckpt", help="mto.save'd transformer checkpoint (transformer.pt)")
    p.add_argument("--eval-manifest", help="Read if it exists, else planned and written")
    p.add_argument("--robotwin-data-dir")
    p.add_argument("--exclude-manifest", action="append", default=[], help="Calibration manifests to keep disjoint from")
    p.add_argument("--num-episodes", type=int, default=50, help="Eval episodes (planned count, or cap on a manifest)")
    p.add_argument("--eval-seed", type=int, default=1000)
    p.add_argument("--chunks", type=int, default=4)
    p.add_argument("--video-stride", type=int, default=5)
    p.add_argument("--action-stride", type=int, default=10)
    p.add_argument("--threshold", type=float, default=GATE)
    p.add_argument("--near-zero-frac", type=float, default=0.1,
                   help="Tensors with mean|bf16| below this fraction of their group median are judged on absolute error")
    p.add_argument("--disable-quantizers", action="append", default=[], metavar="REGEX",
                   help="Turn off quantizers whose module name matches (re.search), e.g. 'blocks\\.2[34]\\.attn1\\.to_v'. "
                        "For localization: no recalibration needed. Repeatable")
    p.add_argument("--episode-start", type=int, default=0, help="First eval episode (for sharding across GPUs)")
    p.add_argument("--bf16-cache-writes", action="store_true",
                   help="Run the test model's cache-writing calls with quantizers off (bf16 KV cache retention)")
    p.add_argument("--bf16-first-chunk-video", action="store_true",
                   help="Also run every video call of chunk 0 (first chunk of an episode) with quantizers off")
    p.add_argument("--bf16-last-video-steps", type=int, default=0,
                   help="Also run the last N video denoising steps of every chunk with quantizers off")
    p.add_argument("--bf16-last-action-steps", type=int, default=0,
                   help="Also run the last N action denoising steps of every chunk with quantizers off")
    p.add_argument("--free-running", action="store_true", help="Also report end-to-end action drift (secondary)")
    p.add_argument("--save-actions", action="store_true", help="With --free-running: store both runs' raw actions in the JSON")
    p.add_argument("--out", help="Parity JSON path")
    args = p.parse_args()
    if args.resummarize:
        parts = [json.loads(Path(f).read_text()) for f in args.resummarize]
        result = parts[0]
        if len(parts) > 1:
            if not args.out:
                p.error("merging several JSONs needs --out")
            result["episodes"] = [e for r in parts for e in r["episodes"]]
            result["calls"] = [c for r in parts for c in r["calls"]]
            result["runtime_s"] = max(r["runtime_s"] for r in parts)
            result["merged_from"] = args.resummarize
            drift = [d for r in parts for d in r.get("free_running", {}).get("per_chunk", [])]
            if drift:
                result["free_running"] = free_running_summary(drift)
        result["summary"] = summarize(result["calls"], args.threshold, args.near_zero_frac)
        Path(args.out or args.resummarize[0]).write_text(json.dumps(result, indent=1))
        report(result["summary"], result.get("free_running"))
        return
    for name in ("lingbot_va_repo", "model_path", "eval_manifest", "out"):
        if not getattr(args, name):
            p.error(f"--{name.replace('_', '-')} is required")
    if args.test != "bf16" and not args.fp8_ckpt:
        p.error(f"--test {args.test} needs --fp8-ckpt")

    # Same RMSNorm swap as quantize.py main(), so the module tree matches the checkpoint.
    from diffusers.models.normalization import RMSNorm as DiffuserRMSNorm

    torch.nn.RMSNorm = DiffuserRMSNorm
    torch.nn.modules.normalization.RMSNorm = DiffuserRMSNorm

    from lingbot_va_utils import load_va_server
    from robotwin_episodes import RobotwinEpisode
    from robotwin_replay import episode_seed, replay_episode

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    va_server = load_va_server(
        args.lingbot_va_repo, args.model_path, torch.bfloat16,
        # VA_Server dumps latents/actions/obs every chunk; keep them off the shared disk if asked.
        save_root=os.environ.get("LINGBOT_VA_DUMP_DIR") or str(out_path.parent / "server_dumps"),
    )
    ref = va_server.transformer
    test = load_transformer(args.model_path, torch.bfloat16)
    if args.test != "bf16":
        import modelopt.torch.opt as mto
        import modelopt.torch.quantization as mtq

        # Checkpoints calibrated with --act-calib store our calibrator class in their modelopt state.
        from config import FixedMethodHistogramCalibrator

        torch.serialization.add_safe_globals([FixedMethodHistogramCalibrator])
        mto.restore(test, args.fp8_ckpt)
        if args.test == "fp8-disabled":
            mtq.disable_quantizer(test, lambda name: True)
        for pattern in args.disable_quantizers:
            import re

            mtq.disable_quantizer(test, lambda name, pattern=pattern: re.search(pattern, name) is not None)
    ref_dtypes = {n: p.dtype for n, p in ref.named_parameters()}
    mismatched = [n for n, p in test.named_parameters() if n in ref_dtypes and p.dtype != ref_dtypes[n]]
    if mismatched:
        raise SystemExit(f"Test transformer dtypes differ from VA_Server's: {mismatched[:5]}")
    quantizers = quantizer_summary(test)
    policy = None
    if args.bf16_cache_writes or args.bf16_first_chunk_video or args.bf16_last_video_steps or args.bf16_last_action_steps:
        # Timestep of the N-th last denoising step, from the server's own schedules.
        cfg = va_server.job_config

        def last_t(scheduler, n_steps, n_last):
            if not n_last:
                return None
            scheduler.set_timesteps(n_steps)
            return float(scheduler.timesteps[-n_last])

        video_t_max = last_t(va_server.scheduler, cfg.num_inference_steps, args.bf16_last_video_steps)
        action_t_max = last_t(va_server.action_scheduler, cfg.action_num_inference_steps, args.bf16_last_action_steps)
        policy = Bf16CallPolicy(test, args.bf16_cache_writes, args.bf16_first_chunk_video, video_t_max, action_t_max)
        print(f"bf16 calls: cache_writes={args.bf16_cache_writes} first_chunk_video={args.bf16_first_chunk_video} "
              f"video t<={video_t_max} action t<={action_t_max} "
              f"({len(policy.quantizers)} quantizers switched off for those calls)")
    print(f"test={args.test}: enabled quantizers {quantizers}")
    if quantizers["attention"]:
        raise SystemExit(f"Attention quantizers enabled (not TRT-deployable here): {quantizers['attention']}")

    manifest = build_eval_manifest(args, va_server.job_config)
    entries = manifest["episodes"][args.episode_start : args.episode_start + args.num_episodes]
    cam_keys = va_server.job_config.obs_cam_keys

    hook = TeacherForcedParity(ref, test, args.video_stride, args.action_stride)
    ref.forward = hook
    drift, actions_dump = [], []
    t0 = time.time()
    for i, entry in enumerate(entries, start=args.episode_start):
        episode = RobotwinEpisode(manifest["dataset_dir"], entry, cam_keys)
        seed = episode_seed(entry, manifest["seed"])
        hook.episode = i
        def on_chunk(k):
            hook.on_chunk(k)
            if policy is not None:
                policy.on_chunk(k)

        ref_run = replay_episode(va_server, episode, args.chunks, seed, on_chunk=on_chunk)
        ep_recs = [r for r in hook.records if r["episode"] == i]
        print(
            f"[{i + 1 - args.episode_start}/{len(entries)}] {entry['task_dir'].split('/')[-1]} ep={entry['episode_index']} "
            f"chunks={ref_run['chunks']} raw worst={max(_raw_worst(r) for r in ep_recs):.4e} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
        if args.free_running:
            va_server.transformer = test
            test_run = replay_episode(va_server, episode, args.chunks, seed,
                                      on_chunk=policy.on_chunk if policy is not None else None)
            va_server.transformer = ref
            # The model's normalized action space ([-1, 1] via the q01/q99 quantiles), as Thor's numbers.
            used = va_server.job_config.used_action_channel_ids
            q01, q99 = va_server.actions_q01[used], va_server.actions_q99[used]
            norm = lambda a: (torch.from_numpy(a) - q01) / (q99 - q01 + 1e-6) * 2 - 1  # noqa: E731
            for c, (a_ref, a_test) in enumerate(zip(ref_run["actions"], test_run["actions"])):
                drift.append({
                    "episode": i, "chunk": c,
                    **compare(torch.from_numpy(a_test), torch.from_numpy(a_ref)),
                    "norm": compare(norm(a_test), norm(a_ref)),
                    # Per channel (16: left xyz, quat xyzw, gripper, right ...), to spot e.g. quaternion sign flips.
                    "norm_max_abs_by_channel": (norm(a_test) - norm(a_ref)).abs().flatten(1).max(1).values.tolist(),
                })
                if args.save_actions:
                    actions_dump.append({"episode": i, "chunk": c, "ref": a_ref.tolist(), "test": a_test.tolist()})
    ref.forward = hook.orig_forward

    summary = summarize(hook.records, args.threshold, args.near_zero_frac)
    result = {
        "args": vars(args),
        "quantizers": quantizers,
        "episodes": entries,
        "runtime_s": time.time() - t0,
        "summary": summary,
        "calls": hook.records,
    }
    if drift:
        result["free_running"] = free_running_summary(drift)
    if actions_dump:
        result["actions"] = actions_dump
    out_path.write_text(json.dumps(result, indent=1))

    report(summary, result.get("free_running"))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
