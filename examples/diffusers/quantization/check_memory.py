import argparse
import os
import subprocess
import sys
import time

import torch

import modelopt.torch.opt as mto
from modelopt.torch.quantization.backends import fp8_per_tensor_gemm  # noqa: F401 — triggers FP8 GEMM registration

LINGBOT_VA_REPO = "/home/thannan/scratch/robotics/lingbot-va"
MODEL_PATH = "/home/thannan/scratch/robotics/model/lingbot-va-base"
QUANTIZED_CKPT = "./lingbot_va_fp8_compressed.pt/transformer.pt"


def _load_wan_va_utils_package():
    import importlib.util

    wan_va_utils_dir = os.path.join(LINGBOT_VA_REPO, "wan_va", "utils")
    spec = importlib.util.spec_from_file_location(
        "utils",
        os.path.join(wan_va_utils_dir, "__init__.py"),
        submodule_search_locations=[wan_va_utils_dir],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["utils"] = module
    spec.loader.exec_module(module)


def build_va_server():
    if LINGBOT_VA_REPO not in sys.path:
        sys.path.append(LINGBOT_VA_REPO)
    from wan_va.configs import VA_CONFIGS

    _load_wan_va_utils_package()
    from wan_va.wan_va_server import VA_Server

    job_config = VA_CONFIGS["robotwin"]
    job_config.wan22_pretrained_model_name_or_path = MODEL_PATH
    job_config.local_rank = 0
    job_config.param_dtype = torch.bfloat16

    return VA_Server(job_config)


def make_dummy_obs(job_config):
    import numpy as np

    return {
        cam_key: np.zeros((480, 640, 3), dtype=np.uint8)
        for cam_key in job_config.obs_cam_keys
    }


def run_case(quantized: bool) -> None:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    va_server = build_va_server()

    if quantized:
        mto.restore(va_server.transformer, QUANTIZED_CKPT)
        va_server.transformer = va_server.transformer.to(va_server.device)

    mem_after_load = torch.cuda.max_memory_allocated() / 1e9
    print(f"Peak GPU memory after model load: {mem_after_load:.2f} GB")

    prompt = "a robot arm manipulating objects on a table"
    obs = make_dummy_obs(va_server.job_config)

    num_iters = 3
    latencies = []
    for i in range(num_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        va_server.infer({"reset": True, "prompt": prompt})
        va_server.infer({"obs": [obs]})
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)
        print(f"Iteration {i}: {elapsed:.3f} s")

    mem_after_forward = torch.cuda.max_memory_allocated() / 1e9
    print(f"Peak GPU memory after forward pass: {mem_after_forward:.2f} GB")

    warm = latencies[1:] if len(latencies) > 1 else latencies
    print(f"Cold (first) latency: {latencies[0]:.3f} s")
    print(f"Warm avg latency ({len(warm)} iters): {sum(warm) / len(warm):.3f} s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["baseline", "quantized"], default=None)
    args = parser.parse_args()

    if args.case is not None:
        # Worker mode: run exactly one case in this process.
        run_case(quantized=(args.case == "quantized"))
        return

    # Driver mode: spawn each case as its own subprocess so peak-memory
    # stats from one case never leak into the other.
    for case in ("baseline", "quantized"):
        print(f"\n=== {case} ===")
        subprocess.run(
            [sys.executable, __file__, "--case", case],
            check=True,
        )


if __name__ == "__main__":
    main()