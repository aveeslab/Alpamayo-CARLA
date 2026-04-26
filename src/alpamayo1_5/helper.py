# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from transformers import AutoProcessor, AutoTokenizer

from typing import Any

import torch
import collections.abc

MIN_PIXELS = 163840
MAX_PIXELS = 196608
BASE_PROCESSOR_NAME = "Qwen/Qwen3-VL-2B-Instruct"

CAMERA_DISPLAY_NAMES = {
    0: "Front left camera",
    1: "Front camera",
    2: "Front right camera",
    3: "Rear left camera",
    4: "Rear camera",
    5: "Rear right camera",
    6: "Front telephoto camera",
}


def _build_image_content(
    frames: torch.Tensor,
    camera_indices: torch.Tensor | None = None,
    num_frames_per_camera: int = 4,
) -> list[dict[str, Any]]:
    """Build the image portion of the user message content.

    When ``camera_indices`` is provided, each image is annotated with
    a camera display name (on the first frame of each camera group) and
    a frame index, matching the format used during model training.

    Args:
        frames: Flattened camera frames, shape ``(N_total, C, H, W)``.
        camera_indices: Per-camera indices, shape ``(N_cameras,)``.
            When provided, ``N_total`` must equal
            ``N_cameras * num_frames_per_camera``.
        num_frames_per_camera: Number of temporal frames per camera.
    """
    if camera_indices is None:
        return [{"type": "image", "image": frame} for frame in frames]

    expanded_cam_ids = camera_indices.repeat_interleave(num_frames_per_camera)
    content: list[dict[str, Any]] = []
    prev_cam_id = None
    frame_idx = 0
    for i, frame in enumerate(frames):
        cam_id = expanded_cam_ids[i].item()
        if prev_cam_id is not None and cam_id != prev_cam_id:
            frame_idx = 0
        if frame_idx == 0:
            cam_name = CAMERA_DISPLAY_NAMES.get(cam_id, f"Camera {cam_id}")
            content.append({"type": "text", "text": f"{cam_name}: "})
        content.append({"type": "text", "text": f"frame {frame_idx} "})
        content.append({"type": "image", "image": frame})
        prev_cam_id = cam_id
        frame_idx += 1
    return content


def create_message(
    frames: torch.Tensor,
    camera_indices: torch.Tensor | None = None,
    num_frames_per_camera: int = 4,
    nav_text: str | None = None,
    use_nav_prompt: bool = False,
):
    """Construct the chat message for model inference.

    Args:
        frames: Camera image tensors, shape ``(N_total, C, H, W)``
            (typically ``data["image_frames"].flatten(0, 1)``).
        camera_indices: Per-camera indices from the dataset, shape
            ``(N_cameras,)``. When provided, camera display names and
            frame numbers are included before each image to match the
            training format. Pass ``data["camera_indices"]``.
        num_frames_per_camera: Number of temporal frames per camera
            (default 4, matching the dataset loader).
        nav_text: Optional navigation instruction string, e.g.
            ``"Turn left onto De La Cruz Boulevard in 40m"``.
            When provided, the model conditions its trajectory prediction
            on this instruction.
        use_nav_prompt: When ``True``, use the nav-style prompt
            (``"output the future trajectory."``) even if ``nav_text``
            is ``None``. Useful for constructing a fair no-nav baseline
            in nav-conditioned comparisons.
    """
    assert frames.ndim == 4, f"{frames.ndim=}, expected (N, C, H, W)"

    num_traj_token = 48
    hist_traj_placeholder = (
        f"<|traj_history_start|>{'<|traj_history|>' * num_traj_token}<|traj_history_end|>"
    )

    route_section = ""
    if nav_text is not None:
        route_section = f"<|route_start|>{nav_text}<|route_end|>"

    prompt_text = (
        "output the chain-of-thought reasoning of the driving process, "
        "then output the future trajectory."
    )

    user_text = f"{hist_traj_placeholder}{route_section}{prompt_text}"

    image_content = _build_image_content(frames, camera_indices, num_frames_per_camera)

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a driving assistant that generates safe and accurate actions.",
                }
            ],
        },
        {
            "role": "user",
            "content": image_content + [{"type": "text", "text": user_text}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "<|cot_start|>"}],
        },
    ]


def create_vqa_message(
    frames: torch.Tensor,
    question: str,
    camera_indices: torch.Tensor | None = None,
    num_frames_per_camera: int = 4,
):
    """Construct the chat message for model inference.

    Args:
        frames: Camera image tensors, shape ``(N_total, C, H, W)``
            (typically ``data["image_frames"].flatten(0, 1)``).
        question: The question string.
        camera_indices: Per-camera indices from the dataset, shape
            ``(N_cameras,)``. When provided, camera display names and
            frame numbers are included before each image to match the
            training format. Pass ``data["camera_indices"]``.
        num_frames_per_camera: Number of temporal frames per camera
            (default 4, matching the dataset loader).
    """
    assert frames.ndim == 4, f"{frames.ndim=}, expected (N, C, H, W)"
    user_text = f"<|question_start|>{question}<|question_end|>"

    image_content = _build_image_content(frames, camera_indices, num_frames_per_camera)

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a driving assistant that generates safe and accurate actions.",
                }
            ],
        },
        {
            "role": "user",
            "content": image_content + [{"type": "text", "text": user_text}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "<|answer_start|>"}],
        },
    ]


def get_processor(tokenizer: AutoTokenizer) -> AutoProcessor:
    """Get the processor for the Qwen3-VL-2B-Instruct model."""
    processor_kwargs = {
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
    }

    processor = AutoProcessor.from_pretrained(BASE_PROCESSOR_NAME, **processor_kwargs)
    processor.tokenizer = tokenizer
    return processor


def to_device(
    data: Any,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Any:
    """Recursively cast data into the specified device, dtype."""
    if isinstance(data, torch.Tensor):
        data = data.to(
            device=device,
            dtype=dtype,
        )
        return data
    elif isinstance(data, collections.abc.Mapping):
        return {key: to_device(data[key], device=device, dtype=dtype) for key in data}
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, (str, bytes)):
        return [to_device(elem, device=device, dtype=dtype) for elem in data]
    else:
        return data
