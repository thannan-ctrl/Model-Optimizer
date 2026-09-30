# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-chunk replay of a recorded RoboTwin episode through a lingbot-va VA_Server.

Mirrors the RoboTwin eval client's chunk loop
(lingbot-va ``evaluation/robotwin/eval_polict_client_openpi.py:545-590``), with
the recorded dataset standing in for the simulator:

- chunk 0: ``infer(obs=frame 0)``. Its first latent frame is the conditioning
  frame, so only the second one is executed: 16 env steps.
- chunk k >= 1: ``infer(obs=...)`` again; both latent frames are executed: 32 steps.
- after each chunk: ``infer(obs=key_frames, compute_kv_cache=True, state=...)``,
  with a key frame every ``action_per_frame // 4`` executed steps (4 for chunk 0,
  8 later) and ``state`` = the chunk's actions.

Dataset frame t is the observation after t executed steps, so the executed
steps of a chunk that starts at step s are dataset actions ``s .. s+n-1``, and its
key frames are dataset frames ``s+4, s+8, ..., s+n``.

``state`` is teacher-forced: the model's own action output with the executed
steps replaced by the dataset's relative actions (the client feeds back its own
predictions; here the recorded trajectory is the ground truth that the key
frames actually show). The unexecuted conditioning frame of chunk 0 keeps the
model's value, as in the client.
"""

import numpy as np
import torch


def chunk_schedule(length: int, num_chunks: int, action_per_frame: int, frame_chunk_size: int):
    """``[(start_step, num_steps), ...]`` for the chunks that fit in ``length`` frames.

    A chunk fits if its last key frame (dataset frame ``start + num_steps``) exists.
    """
    schedule = []
    start = 0
    for k in range(num_chunks):
        n = (frame_chunk_size - 1 if k == 0 else frame_chunk_size) * action_per_frame
        if start + n > length - 1:
            break
        schedule.append((start, n))
        start += n
    return schedule


def min_episode_length(num_chunks: int, action_per_frame: int, frame_chunk_size: int) -> int:
    """Frames needed to replay ``num_chunks`` chunks."""
    steps = (num_chunks * frame_chunk_size - 1) * action_per_frame
    return steps + 1


def replay_episode(
    va_server,
    episode,
    num_chunks: int,
    seed: int,
    final_kv_cache: bool = True,
    on_chunk=None,
) -> dict:
    """Replay ``episode`` for up to ``num_chunks`` chunks. Returns per-chunk actions.

    ``final_kv_cache=False`` skips the ``compute_kv_cache`` after the last chunk
    (first-chunk-only calibration). ``on_chunk(k)`` is called before chunk k's
    ``infer``, so a transformer hook can tag calls with their chunk.

    Seeds torch and numpy with ``seed`` right before the first chunk, so two
    servers replaying the same episode start from identical noise.
    """
    cfg = va_server.job_config
    apf, fcs = cfg.action_per_frame, cfg.frame_chunk_size
    key_every = apf // 4
    schedule = chunk_schedule(episode.length, num_chunks, apf, fcs)
    if not schedule:
        raise ValueError(f"Episode too short ({episode.length} frames) for one chunk")

    va_server.infer({"reset": True, "prompt": episode.prompt})
    torch.manual_seed(seed)
    np.random.seed(seed % 2**32)

    first_obs = episode.obs(0)
    actions = []
    for k, (start, n) in enumerate(schedule):
        if on_chunk is not None:
            on_chunk(k)
        # Only chunk 0 encodes this obs (as the conditioning frame); later
        # chunks ignore it, as with the client.
        action = va_server.infer({"obs": first_obs})["action"]  # (C=16, F, H=apf)
        actions.append(action)

        if k == len(schedule) - 1 and not final_kv_cache:
            break

        executed = episode.relative_action[start : start + n]  # (n, 16)
        exec_frames = n // apf
        state = action.copy()
        state[:, fcs - exec_frames :, :] = executed.reshape(exec_frames, apf, -1).transpose(2, 0, 1)
        key_frames = [episode.obs(start + j) for j in range(key_every, n + 1, key_every)]
        va_server.infer({"obs": key_frames, "compute_kv_cache": True, "state": state})

    return {"chunks": len(schedule), "actions": actions}


def episode_seed(entry: dict, base_seed: int = 0) -> int:
    """Stable per-episode seed from a manifest entry (identical across runs and processes)."""
    import zlib

    key = f"{entry['task_dir']}:{entry['episode_index']}:{base_seed}"
    return zlib.crc32(key.encode())
