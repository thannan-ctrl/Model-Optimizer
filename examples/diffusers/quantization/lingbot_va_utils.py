# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load a lingbot-va VA_Server from an external lingbot-va checkout."""

import importlib.util
import os
import sys
from typing import Any

_VA_SERVER_CLS = None
_VA_CONFIGS = None


def import_va_server(lingbot_va_repo: str) -> tuple[Any, Any]:
    """Return (VA_CONFIGS, VA_Server) from the lingbot-va repo at ``lingbot_va_repo``."""
    global _VA_SERVER_CLS, _VA_CONFIGS
    if _VA_SERVER_CLS is not None:
        return _VA_CONFIGS, _VA_SERVER_CLS

    if lingbot_va_repo not in sys.path:
        sys.path.append(lingbot_va_repo)

    from wan_va.configs import VA_CONFIGS

    # wan_va_server.py does `from utils import (...)` expecting its own
    # wan_va/utils/ package. Python checks sys.modules (by name) before
    # ever consulting sys.path, and even a fresh lookup would find this
    # repo's own quantization/utils.py first via sys.path[0] (the running
    # script's own directory), regardless of what we append afterward. So
    # we manually build the wan_va/utils package module and inject it
    # into sys.modules["utils"] before triggering the import, then
    # restore our own utils.py afterward.
    wan_va_utils_dir = os.path.join(lingbot_va_repo, "wan_va", "utils")
    spec = importlib.util.spec_from_file_location(
        "utils",
        os.path.join(wan_va_utils_dir, "__init__.py"),
        submodule_search_locations=[wan_va_utils_dir],
    )
    wan_va_utils_module = importlib.util.module_from_spec(spec)

    our_utils_module = sys.modules.get("utils")
    sys.modules["utils"] = wan_va_utils_module
    try:
        spec.loader.exec_module(wan_va_utils_module)
        from wan_va.wan_va_server import VA_Server
    finally:
        if our_utils_module is not None:
            sys.modules["utils"] = our_utils_module
        else:
            sys.modules.pop("utils", None)

    _VA_CONFIGS, _VA_SERVER_CLS = VA_CONFIGS, VA_Server
    return _VA_CONFIGS, _VA_SERVER_CLS


def load_va_server(lingbot_va_repo: str, model_path: str, dtype, save_root: str | None = None):
    """Build a VA_Server with the RoboTwin job config on cuda:0.

    ``save_root`` redirects the server's debug dumps (obs/latents/actions per
    chunk). The config default is the relative path ``./train_out``.
    """
    va_configs, va_server_cls = import_va_server(lingbot_va_repo)
    job_config = va_configs["robotwin"]
    job_config.wan22_pretrained_model_name_or_path = model_path
    job_config.local_rank = 0
    job_config.param_dtype = dtype
    if save_root is not None:
        job_config.save_root = save_root
    return va_server_cls(job_config)
