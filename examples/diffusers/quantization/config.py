# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
from calib.plugin_calib import PercentileCalibrator

from modelopt.torch.quantization.calib import HistogramCalibrator

from modelopt.torch.opt.config_loader import load_config
from modelopt.torch.quantization.config import QuantizeConfig

FP8_DEFAULT_CONFIG = load_config(
    "configs/ptq/presets/diffusers/fp8", schema_type=QuantizeConfig
).model_dump(exclude_unset=True)
INT8_DEFAULT_CONFIG = load_config(
    "configs/ptq/presets/diffusers/int8", schema_type=QuantizeConfig
).model_dump(exclude_unset=True)
NVFP4_DEFAULT_CONFIG = load_config(
    "configs/ptq/presets/diffusers/nvfp4", schema_type=QuantizeConfig
).model_dump(exclude_unset=True)
NVFP4_FP8_MHA_CONFIG = load_config(
    "configs/ptq/presets/diffusers/nvfp4_fp8_mha", schema_type=QuantizeConfig
).model_dump(exclude_unset=True)


def set_quant_config_attr(quant_config, trt_high_precision_dtype, quant_algo, **kwargs):
    algo_cfg = {"method": quant_algo}

    if quant_algo == "smoothquant" and "alpha" in kwargs:
        algo_cfg["alpha"] = kwargs["alpha"]
    elif quant_algo == "svdquant":
        if "lowrank" in kwargs:
            algo_cfg["lowrank"] = kwargs["lowrank"]
        # Layers excluded from the SVDQuant algorithm (no AWQ smoothing, no
        # low-rank branch); they stay quantized with plain max calibration.
        if kwargs.get("skip_layers"):
            algo_cfg["skip_layers"] = kwargs["skip_layers"]
    quant_config["algorithm"] = algo_cfg

    for entry in quant_config["quant_cfg"]:
        p = entry.get("cfg", {})
        if isinstance(p, dict) and "num_bits" in p and "trt_high_precision_dtype" not in p:
            p["trt_high_precision_dtype"] = trt_high_precision_dtype


class FixedMethodHistogramCalibrator(HistogramCalibrator):
    """Histogram calibrator with its amax method fixed at construction.

    max/mse calibration calls ``compute_amax()`` without a method; this pins it,
    so activation quantizers can use percentile or FP8-aware MSE while weight
    quantizers keep their own calibrator.
    """

    def __init__(self, method="percentile", percentile=99.99, **kwargs):
        super().__init__(**kwargs)
        self._method = method
        self._percentile = percentile

    def compute_amax(self, *args, **kwargs):
        if self._method == "mse":
            # modelopt 0.46.1's histogram MSE crashes for FP8 (scaled_e4m3 signature), so search here.
            return self._fp8_mse_amax()
        return super().compute_amax(self._method, percentile=self._percentile)

    def _fp8_mse_amax(self, start_bin=128):
        """amax minimizing the histogram-weighted E4M3 round-trip error of |x|."""
        import torch

        if self._calib_hist is None:  # never saw data (e.g. an unused Linear), like modelopt's calibrators
            return None

        edges = torch.as_tensor(self._calib_bin_edges, dtype=torch.float64)
        hist = torch.as_tensor(self._calib_hist, dtype=torch.float64).to(edges.device)
        centers = ((edges[:-1] + edges[1:]) / 2).float()
        candidates = edges[start_bin:].float()
        best, best_amax = None, candidates[-1]
        for chunk in candidates.split(256):
            scale = (chunk / 448.0)[:, None]
            q = (centers[None] / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale
            err = (((q - centers[None]) ** 2).double() * hist[None]).sum(1)
            i = int(err.argmin())
            if best is None or err[i] < best:
                best, best_amax = err[i], chunk[i]
        self._calib_amax = best_amax.clone().detach()
        return self._calib_amax


def fp8_input_calibrator_rule(method, percentile):
    """quant_cfg entry giving every input quantizer a per-tensor histogram calibrator."""
    return {
        "quantizer_name": "*input_quantizer",
        "cfg": {
            "num_bits": (4, 3),
            "axis": None,
            "calibrator": (
                FixedMethodHistogramCalibrator,
                (),
                {"num_bits": (4, 3), "axis": None, "method": method, "percentile": percentile},
            ),
        },
    }


def reset_set_int8_config(quant_config, percentile, n_steps, collect_method, backbone):
    """Add PercentileCalibrator to Conv2d input quantizers.

    Linear layers are left unchanged — their axis settings come from the base
    quant_config (e.g. INT8_SMOOTHQUANT_CFG or INT8_DEFAULT_CONFIG).

    Args:
        quant_config: The quantization configuration dictionary
        percentile: Percentile value for calibration
        n_steps: Number of calibration steps
        collect_method: Method for collecting calibration statistics
        backbone: The model backbone to analyze layer types
    """
    for name, module in backbone.named_modules():
        if isinstance(module, nn.Conv2d):
            aq_name = f"*{name}*input_quantizer*"
            quant_config["quant_cfg"].append(
                {
                    "quantizer_name": aq_name,
                    "cfg": {
                        "num_bits": 8,
                        "axis": None,
                        "calibrator": (
                            PercentileCalibrator,
                            (),
                            {
                                "num_bits": 8,
                                "axis": None,
                                "percentile": percentile,
                                "total_step": n_steps,
                                "collect_method": collect_method,
                            },
                        ),
                    },
                }
            )
