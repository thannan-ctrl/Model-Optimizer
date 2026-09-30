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

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from models_utils import MODEL_DEFAULTS, ModelType
from pipeline_manager import PipelineManager
from quantize_config import CalibrationConfig
from tqdm import tqdm
from utils import load_calib_prompts

# lingbot-va calibration modes (--extra-param robotwin_calib_mode=...):
#   dummy        the original recipe: one fixed prompt, all-black cameras, first chunk.
#   first_chunk  RoboTwin episodes, first chunk only (frame 0, no compute_kv_cache).
#   replay       RoboTwin episodes, robotwin_chunks chunks each, including the
#                compute_kv_cache calls (robotwin_replay.py). Default.
_LINGBOT_VA_CALIB_MODES = ("dummy", "first_chunk", "replay")
_LINGBOT_VA_DUMMY_PROMPT = "a robot arm manipulating objects on a table"


def _make_lingbot_va_dummy_obs(job_config) -> dict:
    return {
        cam_key: np.zeros((480, 640, 3), dtype=np.uint8) for cam_key in job_config.obs_cam_keys
    }


class Calibrator:
    """Handles model calibration for quantization."""

    def __init__(
        self,
        pipeline_manager: PipelineManager,
        config: CalibrationConfig,
        model_type: ModelType,
        logger: logging.Logger,
    ):
        """
        Initialize calibrator.

        Args:
            pipeline_manager: Pipeline manager with main and upsampler pipelines
            config: Calibration configuration
            model_type: Type of model being calibrated
            logger: Logger instance
        """
        self.pipeline_manager = pipeline_manager
        self.pipe = pipeline_manager.pipe
        self.pipe_upsample = pipeline_manager.pipe_upsample
        self.config = config
        self.model_type = model_type
        self.logger = logger

    def load_and_batch_prompts(self) -> list[list[str]]:
        """
        Load calibration prompts from file.

        Returns:
            List of batched calibration prompts
        """
        if self.model_type == ModelType.LINGBOT_VA:
            # Prompts come from the RoboTwin episodes, paired with their cameras.
            return []
        self.logger.info(f"Loading calibration prompts from {self.config.prompts_dataset}")
        if isinstance(self.config.prompts_dataset, Path):
            return load_calib_prompts(
                self.config.batch_size,
                self.config.prompts_dataset,
            )

        return load_calib_prompts(
            self.config.batch_size,
            self.config.prompts_dataset["name"],
            self.config.prompts_dataset["split"],
            self.config.prompts_dataset["column"],
        )

    def run_calibration(self, batched_prompts: list[list[str]]) -> None:
        """
        Run calibration steps on the pipeline.

        Args:
            batched_prompts: List of batched calibration prompts
        """
        if self.model_type == ModelType.LINGBOT_VA:
            self._run_lingbot_va_calibration()
            return

        self.logger.info(f"Starting calibration with {self.config.num_batches} batches")
        extra_args = MODEL_DEFAULTS.get(self.model_type, {}).get("inference_extra_args", {})

        with tqdm(total=self.config.num_batches, desc="Calibration", unit="batch") as pbar:
            for i, prompt_batch in enumerate(batched_prompts):
                if i >= self.config.num_batches:
                    break

                if self.model_type == ModelType.LTX2:
                    self._run_ltx2_calibration(prompt_batch, extra_args)
                elif self.model_type == ModelType.LTX_VIDEO_DEV:
                    # Special handling for LTX-Video
                    self._run_ltx_video_calibration(prompt_batch, extra_args)
                elif self.model_type in [ModelType.WAN22_T2V_14b, ModelType.WAN22_T2V_5b]:
                    # Special handling for WAN video models
                    self._run_wan_video_calibration(prompt_batch, extra_args)
                else:
                    common_args = {
                        "prompt": prompt_batch,
                        "num_inference_steps": self.config.n_steps,
                    }
                    self.pipe(**common_args, **extra_args).images
                pbar.update(1)
                self.logger.debug(f"Completed calibration batch {i + 1}/{self.config.num_batches}")
        self.logger.info("Calibration completed successfully")

    def _run_wan_video_calibration(
        self, prompt_batch: list[str], extra_args: dict[str, Any]
    ) -> None:
        extra_params = self.pipeline_manager.config.extra_params
        kwargs = {}
        kwargs["negative_prompt"] = extra_args["negative_prompt"]
        kwargs["height"] = extra_params.get("height", extra_args["height"])
        kwargs["width"] = extra_params.get("width", extra_args["width"])
        kwargs["num_frames"] = extra_params.get("num_frames", extra_args["num_frames"])
        kwargs["guidance_scale"] = extra_args["guidance_scale"]
        if "guidance_scale_2" in extra_args:
            kwargs["guidance_scale_2"] = extra_args["guidance_scale_2"]
        kwargs["num_inference_steps"] = self.config.n_steps

        self.pipe(prompt=prompt_batch, **kwargs).frames

    def _run_lingbot_va_calibration(self) -> None:
        """Calibrate on ``--calib-size`` lingbot-va runs (episodes, or dummy runs).

        ``--batch-size`` doesn't apply: each run is one episode through VA_Server.
        """
        extra_params = self.pipeline_manager.config.extra_params
        mode = extra_params.get("robotwin_calib_mode", "replay")
        if mode not in _LINGBOT_VA_CALIB_MODES:
            raise ValueError(f"robotwin_calib_mode={mode!r}, expected one of {_LINGBOT_VA_CALIB_MODES}")
        va_server = self.pipe.va_server
        num_runs = self.config.calib_size

        if mode == "dummy":
            self.logger.info(f"Calibrating lingbot-va on {num_runs} dummy runs")
            obs = _make_lingbot_va_dummy_obs(va_server.job_config)
            for _ in tqdm(range(num_runs), desc="Calibration", unit="run"):
                self.pipe.generate(_LINGBOT_VA_DUMMY_PROMPT, obs)
            return

        from robotwin_episodes import RobotwinEpisode
        from robotwin_replay import episode_seed, replay_episode

        num_chunks = 1 if mode == "first_chunk" else int(extra_params.get("robotwin_chunks", 4))
        data_dir, entries, seed = self._plan_robotwin_episodes(num_runs, num_chunks)
        self.logger.info(
            f"Calibrating lingbot-va on {len(entries)} RoboTwin episodes, mode={mode}, chunks={num_chunks}"
        )
        chunks_used = []
        for entry in tqdm(entries, desc="Calibration", unit="episode"):
            episode = RobotwinEpisode(data_dir, entry, va_server.job_config.obs_cam_keys)
            result = replay_episode(
                va_server,
                episode,
                num_chunks,
                seed=episode_seed(entry, seed),
                final_kv_cache=mode == "replay",
            )
            chunks_used.append(result["chunks"])
            self.logger.info(
                f"Episode {entry['sample']}: {entry['task_dir']} ep={entry['episode_index']} "
                f"len={entry['length']} chunks={result['chunks']} task={entry['prompt']!r}"
            )
        self.logger.info(
            f"Replayed {sum(chunks_used)} chunks; {sum(c < num_chunks for c in chunks_used)} "
            f"episodes were too short for {num_chunks}"
        )

    def _plan_robotwin_episodes(self, num_episodes: int, num_chunks: int):
        """Episode list for calibration: replayed from a manifest, or planned and saved."""
        from robotwin_episodes import plan_episodes, read_manifest, write_manifest
        from robotwin_replay import min_episode_length

        extra_params = self.pipeline_manager.config.extra_params
        replay_path = extra_params.get("robotwin_calib_manifest")
        if replay_path:
            manifest = read_manifest(replay_path)
            entries = manifest["episodes"]
            if len(entries) < num_episodes:
                raise ValueError(
                    f"Manifest {replay_path} has {len(entries)} episodes, calibration needs {num_episodes}"
                )
            self.logger.info(f"Replaying {num_episodes} calibration episodes from {replay_path}")
            # robotwin_data_dir, if given, overrides the manifest's (absolute) dataset path.
            data_dir = extra_params.get("robotwin_data_dir") or manifest["dataset_dir"]
            return data_dir, entries[:num_episodes], manifest["seed"]

        data_dir = extra_params.get("robotwin_data_dir")
        if not data_dir:
            raise ValueError(
                "Missing required extra_param: robotwin_data_dir "
                "(pass --extra-param robotwin_data_dir=/path/to/robotwin-clean-and-aug-lerobot)"
            )
        seed = int(extra_params.get("robotwin_seed", 0))
        job_config = self.pipe.va_server.job_config
        min_length = min_episode_length(num_chunks, job_config.action_per_frame, job_config.frame_chunk_size)
        entries = plan_episodes(data_dir, num_episodes, seed, min_length=min_length)

        ckpt_dir = self.config.manifest_dir
        if ckpt_dir is not None:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = ckpt_dir / "calib_manifest.json"
            write_manifest(manifest_path, data_dir, seed, entries, chunks=num_chunks)
            self.logger.info(f"Wrote calibration manifest to {manifest_path}")
        else:
            self.logger.warning("No checkpoint save path set; calibration manifest not written")
        return data_dir, entries, seed

    def _run_ltx2_calibration(self, prompt_batch: list[str], extra_args: dict[str, Any]) -> None:
        warnings.warn(
            "LTX-2 packages (ltx-core, ltx-pipelines, ltx-trainer) are provided by Lightricks and are NOT "
            "covered by the Apache 2.0 license governing NVIDIA Model Optimizer. You MUST comply "
            "with the LTX Community License Agreement when installing and using LTX-2 with NVIDIA "
            "Model Optimizer. Any derivative models or fine-tuned weights from LTX-2 remain "
            "subject to the LTX Community License Agreement, not Apache 2.0. "
            "See: https://github.com/Lightricks/LTX-2/blob/main/LICENSE",
            UserWarning,
            stacklevel=2,
        )
        from ltx_core.model.video_vae import TilingConfig
        from ltx_pipelines.utils.constants import (
            DEFAULT_AUDIO_GUIDER_PARAMS,
            DEFAULT_VIDEO_GUIDER_PARAMS,
        )

        prompt = prompt_batch[0]
        extra_params = self.pipeline_manager.config.extra_params
        kwargs = {
            "negative_prompt": extra_args.get(
                "negative_prompt", "worst quality, inconsistent motion, blurry, jittery, distorted"
            ),
            "seed": extra_params.get("seed", 0),
            "height": extra_params.get("height", extra_args.get("height", 1024)),
            "width": extra_params.get("width", extra_args.get("width", 1536)),
            "num_frames": extra_params.get("num_frames", extra_args.get("num_frames", 121)),
            "frame_rate": extra_params.get("frame_rate", extra_args.get("frame_rate", 24.0)),
            "num_inference_steps": self.config.n_steps,
            "video_guider_params": DEFAULT_VIDEO_GUIDER_PARAMS,
            "audio_guider_params": DEFAULT_AUDIO_GUIDER_PARAMS,
            "images": extra_params.get("images", []),
            "tiling_config": extra_params.get("tiling_config", TilingConfig.default()),
        }
        decoded_video, decoded_audio = self.pipe(prompt=prompt, **kwargs)
        # vae_decode_video returns a lazy generator — consume it so the
        # video decoder's forward() actually runs during calibration.
        for _ in decoded_video:
            pass

    def _run_ltx_video_calibration(
        self, prompt_batch: list[str], extra_args: dict[str, Any]
    ) -> None:
        """
        Run calibration for LTX-Video model using the full multi-stage pipeline.

        Args:
            prompt_batch: Batch of prompts
            extra_args: Model-specific arguments
        """
        # Extract specific args for LTX-Video
        expected_height = extra_args.get("height", 512)
        expected_width = extra_args.get("width", 704)
        num_frames = extra_args.get("num_frames", 121)
        negative_prompt = extra_args.get(
            "negative_prompt", "worst quality, inconsistent motion, blurry, jittery, distorted"
        )

        def round_to_nearest_resolution_acceptable_by_vae(height, width):
            height = height - (height % self.pipe.vae_spatial_compression_ratio)
            width = width - (width % self.pipe.vae_spatial_compression_ratio)
            return height, width

        downscale_factor = 2 / 3
        # Part 1: Generate video at smaller resolution
        downscaled_height, downscaled_width = (
            int(expected_height * downscale_factor),
            int(expected_width * downscale_factor),
        )
        downscaled_height, downscaled_width = round_to_nearest_resolution_acceptable_by_vae(
            downscaled_height, downscaled_width
        )

        # Generate initial latents at lower resolution
        latents = self.pipe(
            conditions=None,
            prompt=prompt_batch,
            negative_prompt=negative_prompt,
            width=downscaled_width,
            height=downscaled_height,
            num_frames=num_frames,
            num_inference_steps=self.config.n_steps,
            output_type="latent",
        ).frames

        # Part 2: Upscale generated video using latent upsampler (if available)
        if self.pipe_upsample is not None:
            _ = self.pipe_upsample(latents=latents, output_type="latent").frames

            # Part 3: Denoise the upscaled video with few steps to improve texture
            # However, in this example code, we will omit the upscale step since its optional.
