# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Episode-level access to a local RoboTwin LeRobot (v2.1) dataset.

The robotwin-clean-and-aug-lerobot download is a collection of per-task
LeRobot datasets (``<root>/<collection>/<task>/meta/info.json``). Each
per-task dataset's own ``tasks`` field holds hundreds of paraphrased
instructions for the same task, so stratification is done over tasks, not
over instruction strings.

Reads parquet + AV1 video directly (pandas + PyAV) rather than through
``lerobot``, which is not installed in the lingbot-va environment.

The download holds the same 50 tasks in two collections (randomized ``aug_500``
and ``clean_50``); planning stratifies over task names, not directories.

Episodes are planned up front from ``meta/episodes.jsonl`` alone and written
to a JSON manifest, so any calibration or parity run can be replayed exactly.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

MANIFEST_VERSION = 2


def find_task_dirs(dataset_dir: str | Path) -> list[Path]:
    root = Path(dataset_dir)
    task_dirs = sorted(
        p.parent.parent for p in root.glob("**/meta/info.json") if ".cache" not in p.parts
    )
    if not task_dirs:
        raise ValueError(f"No LeRobot datasets (*/meta/info.json) found under {dataset_dir}")
    return task_dirs


def _read_episodes(task_dir: Path) -> list[dict]:
    with open(task_dir / "meta" / "episodes.jsonl") as f:
        episodes = [json.loads(line) for line in f if line.strip()]
    return sorted(episodes, key=lambda ep: ep["episode_index"])


def task_name(task_dir: Path) -> str:
    """``adjust_bottle-aloha-agilex_randomized_500-1000`` -> ``adjust_bottle``.

    The download holds each task twice (``lerobot_robotwin_eef_aug_500`` and
    ``lerobot_robotwin_eef_clean_50``), so stratification groups by this name.
    """
    return task_dir.name.split("-", 1)[0]


def plan_episodes(
    dataset_dir: str | Path,
    num_episodes: int,
    seed: int = 0,
    exclude: set[tuple[str, int]] | None = None,
    min_length: int = 0,
) -> list[dict]:
    """Round-robin across tasks, alternating collections within a task.

    With 50 tasks, ``num_episodes=50`` gives one episode per task. Within a task,
    successive draws alternate between its collections (aug / clean), starting
    from a seeded random one, so both are represented.

    ``exclude`` holds ``(task_dir, episode_index)`` pairs that must not be drawn,
    used to keep the parity eval set disjoint from calibration. ``min_length``
    is a preference, not a filter: episodes at least that long (enough frames
    for the requested number of chunks) are drawn first. Tasks with none fall
    back to their longest episodes, which then replay fewer chunks.
    """
    root = Path(dataset_dir)
    exclude = exclude or set()

    rng = random.Random(seed)
    by_task: dict[str, list[list[tuple[str, dict]]]] = defaultdict(list)
    for task_dir in find_task_dirs(root):
        rel = str(task_dir.relative_to(root))
        eps = [(rel, ep) for ep in _read_episodes(task_dir) if (rel, ep["episode_index"]) not in exclude]
        rng.shuffle(eps)
        # Long-enough episodes first (random order), then the rest longest-first,
        # so tasks with no long episode still get their longest ones.
        long_eps = [e for e in eps if e[1]["length"] >= min_length]
        short_eps = sorted((e for e in eps if e[1]["length"] < min_length), key=lambda e: -e[1]["length"])
        by_task[task_name(task_dir)].append(long_eps + short_eps)

    # Interleave each task's collections: [aug0, clean0, aug1, clean1, ...],
    # rotated by a random start so the first draw isn't always the same collection.
    candidates = {}
    for name, pools in sorted(by_task.items()):
        start = rng.randrange(len(pools))
        pools = pools[start:] + pools[:start]
        interleaved = []
        for i in range(max(len(p) for p in pools)):
            interleaved.extend(p[i] for p in pools if i < len(p))
        candidates[name] = interleaved

    names = sorted(candidates)
    cursor: dict[str, int] = defaultdict(int)
    planned = []
    for i in range(num_episodes):
        name = names[i % len(names)]
        if cursor[name] >= len(candidates[name]):
            raise ValueError(f"Ran out of episodes for task {name}")
        rel, ep = candidates[name][cursor[name]]
        cursor[name] += 1
        planned.append(
            {
                "sample": i,
                "task_dir": rel,
                "episode_index": ep["episode_index"],
                "length": ep["length"],
                "prompt": ep["tasks"][0],
            }
        )
    return planned


def write_manifest(path: str | Path, dataset_dir: str | Path, seed: int, episodes: list[dict], **extra) -> None:
    manifest = {
        "version": MANIFEST_VERSION,
        "dataset_dir": str(dataset_dir),
        "seed": seed,
        **extra,
        "num_episodes": len(episodes),
        "episodes": episodes,
    }
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def read_manifest(path: str | Path) -> dict:
    with open(path) as f:
        manifest = json.load(f)
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError(f"Unsupported manifest version in {path}")
    return manifest


def get_relative_pose(pose: np.ndarray) -> np.ndarray:
    """Pose (N, 7: xyz + xyzw quat) relative to its first row.

    Same math as lingbot-va ``wan_va/dataset/lerobot_latent_dataset.py:55-68``
    (used for RoboTwin post-training), and the inverse of the eval client's
    ``add_init_pose``. Not imported from there because that module imports
    ``lerobot``.
    """
    from scipy.spatial.transform import Rotation as R

    rot = R.from_quat(pose[:, 3:7])
    first_rot = R.from_quat(np.tile(pose[:1, 3:7], (pose.shape[0], 1)))
    relative_trans = pose[:, :3] - pose[0:1, :3]
    relative_quat = (first_rot.inv() * rot).as_quat()
    return np.concatenate([relative_trans, relative_quat], axis=1)


class RobotwinEpisode:
    """One episode: prompt, relative actions, and on-demand decoded camera frames."""

    def __init__(self, dataset_dir: str | Path, entry: dict, cam_keys: list[str]):
        import pandas as pd

        self.entry = entry
        self.task_dir = Path(dataset_dir) / entry["task_dir"]
        self.episode_index = entry["episode_index"]
        self.prompt = entry["prompt"]
        self.cam_keys = cam_keys

        with open(self.task_dir / "meta" / "info.json") as f:
            self.info = json.load(f)
        chunk = self.episode_index // self.info["chunks_size"]
        data_path = self.task_dir / self.info["data_path"].format(
            episode_chunk=chunk, episode_index=self.episode_index
        )
        df = pd.read_parquet(data_path)
        frame_index = df["frame_index"].to_numpy()
        if not np.array_equal(frame_index, np.arange(len(df))):
            raise ValueError(f"{data_path}: frame_index is not 0..N-1")
        self.length = len(df)

        # Absolute EE actions (16: left xyz+quat, left gripper, right xyz+quat,
        # right gripper) -> relative to the episode's first frame, as in
        # post-training (lerobot_latent_dataset.py:260-263).
        action = np.stack(df["action"].to_numpy()).astype(np.float64)
        self.relative_action = np.concatenate(
            [
                get_relative_pose(action[:, :7]),
                action[:, 7:8],
                get_relative_pose(action[:, 8:15]),
                action[:, 15:16],
            ],
            axis=1,
        ).astype(np.float32)

        self._video_paths = {
            k: self.task_dir
            / self.info["video_path"].format(
                episode_chunk=chunk, video_key=k, episode_index=self.episode_index
            )
            for k in cam_keys
        }
        self._frames: dict[str, np.ndarray] | None = None

    def _decode_all(self) -> dict[str, np.ndarray]:
        import av

        frames = {}
        for k, path in self._video_paths.items():
            with av.open(str(path)) as container:
                decoded = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
            if len(decoded) < self.length:
                raise ValueError(f"{path}: {len(decoded)} frames < episode length {self.length}")
            frames[k] = np.stack(decoded[: self.length])
        return frames

    def obs(self, frame_idx: int) -> dict[str, np.ndarray]:
        """Camera dict for one frame: {cam_key: HxWx3 uint8}, the format VA_Server expects."""
        if self._frames is None:
            self._frames = self._decode_all()
        return {k: self._frames[k][frame_idx] for k in self.cam_keys}
