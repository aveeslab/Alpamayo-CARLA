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

"""Navigation-conditioned trajectory utilities.

This module provides helpers for conditioning trajectory predictions on
navigation instructions:

- :func:`compare_nav_conditions` -- run the model under three conditions
  (with nav, without nav, with counterfactual nav) in a single call.
- :func:`swap_direction` -- flip left/right in a navigation instruction.

All functions are self-contained with no internal training dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer

from alpamayo1_5 import helper
from alpamayo1_5.models.base_model import SPECIAL_TOKENS

ROUTE_START_TOKEN = SPECIAL_TOKENS["route_start"]
ROUTE_END_TOKEN = SPECIAL_TOKENS["route_end"]


@dataclass
class NavComparisonResult:
    """Results from :func:`compare_nav_conditions`.

    Attributes:
        pred_with_nav: Predicted trajectories conditioned on the nav instruction.
            Shape ``[B, num_traj_sets, num_traj_samples, T, 3]``.
        pred_no_nav: Predicted trajectories without any nav instruction.
        pred_counterfactual: Predicted trajectories with direction-swapped nav.
        nav_text: The original navigation instruction.
        nav_text_swapped: The direction-swapped navigation instruction.
        extra_with_nav: Extra outputs (e.g., CoT) from the with-nav condition.
        extra_no_nav: Extra outputs from the no-nav condition.
        extra_counterfactual: Extra outputs from the counterfactual condition.
    """

    pred_with_nav: torch.Tensor
    pred_no_nav: torch.Tensor
    pred_counterfactual: torch.Tensor
    nav_text: str
    nav_text_swapped: str
    extra_with_nav: dict | None = None
    extra_no_nav: dict | None = None
    extra_counterfactual: dict | None = None


def compare_nav_conditions(
    model,
    processor,
    data: dict,
    nav_text: str,
    num_traj_samples: int = 16,
    top_p: float = 0.98,
    temperature: float = 0.6,
    max_generation_length: int = 256,
    return_extra: bool = True,
    nav_inference_fn=None,
    inference_fn=None,
    additional_nav_inference_kwargs: dict | None = None,
) -> NavComparisonResult:
    """Run trajectory inference under three navigation conditions.

    This is the recommended entry point for exploring how the model responds
    to navigation instructions. It builds three variants of the input
    (with nav, without nav, with direction-swapped nav) and calls
    ``model.sample_trajectories_from_data_with_vlm_rollout`` for each.

    Args:
        model: An Alpamayo1_5 model instance (on CUDA).
        processor: The chat-template processor from ``helper.get_processor(model.tokenizer)``.
        data: Output from ``load_physical_aiavdataset``, containing
            ``image_frames``, ``ego_history_xyz``, ``ego_history_rot``, etc.
        nav_text: Navigation instruction, e.g. ``"Turn left onto Main St in 40m"``.
        num_traj_samples: Number of trajectory samples per condition.
        top_p: Top-p sampling parameter.
        temperature: Sampling temperature.
        max_generation_length: Max VLM generation tokens.
        return_extra: Whether to return extra outputs (CoT traces, etc.).
        nav_inference_fn: Optional callable ``(data, **kwargs) -> tuple``
        additional_nav_inference_kwargs: Additional keyword arguments to pass to the nav_inference_fn function.

    Returns:
        A :class:`NavComparisonResult` with predictions from all three conditions.

    Example::

        from alpamayo1_5 import helper, nav_utils
        from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset

        data = load_physical_aiavdataset(clip_id)
        result = nav_utils.compare_nav_conditions(
            model, processor, data,
            nav_text="Turn left onto De La Cruz Boulevard in 40m",
        )
        # result.pred_with_nav, result.pred_no_nav, result.pred_counterfactual
    """
    inference_fn = model.sample_trajectories_from_data_with_vlm_rollout
    if nav_inference_fn is None:
        nav_inference_fn = model.sample_trajectories_from_data_with_vlm_rollout

    frames = data["image_frames"].flatten(0, 1)
    camera_indices = data.get("camera_indices")
    nav_text_swapped = swap_direction(nav_text)

    inference_kwargs = dict(
        top_p=top_p,
        temperature=temperature,
        num_traj_samples=num_traj_samples,
        max_generation_length=max_generation_length,
        return_extra=return_extra,
    )

    def _build_inputs(nav: str | None, use_nav_prompt: bool = False) -> dict:
        messages = helper.create_message(
            frames,
            camera_indices=camera_indices,
            nav_text=nav,
            use_nav_prompt=use_nav_prompt,
        )
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = {
            "tokenized_data": inputs,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        }
        return helper.to_device(model_inputs, "cuda")

    def _run(model_inputs: dict) -> tuple[torch.Tensor, dict | None]:
        _inference_kwargs = inference_kwargs.copy()
        outputs = inference_fn(
            data=model_inputs,
            **_inference_kwargs,
        )
        pred_xyz = outputs[0]
        extra = outputs[2] if return_extra and len(outputs) > 2 else None
        return pred_xyz.cpu(), extra

    def _run_nav(model_inputs: dict) -> tuple[torch.Tensor, dict | None]:
        _inference_kwargs = inference_kwargs.copy()
        if isinstance(additional_nav_inference_kwargs, dict):
            _inference_kwargs.update(**additional_nav_inference_kwargs)
        outputs = nav_inference_fn(
            data=model_inputs,
            **_inference_kwargs,
        )
        pred_xyz = outputs[0]
        extra = outputs[2] if return_extra and len(outputs) > 2 else None
        return pred_xyz.cpu(), extra

    inputs_with_nav = _build_inputs(nav_text)
    inputs_no_nav = _build_inputs(None, use_nav_prompt=True)
    inputs_counterfactual = _build_inputs(nav_text_swapped)

    pred_with_nav, extra_with = _run_nav(inputs_with_nav)
    pred_no_nav, extra_no = _run(inputs_no_nav)
    pred_counter, extra_counter = _run_nav(inputs_counterfactual)

    return NavComparisonResult(
        pred_with_nav=pred_with_nav,
        pred_no_nav=pred_no_nav,
        pred_counterfactual=pred_counter,
        nav_text=nav_text,
        nav_text_swapped=nav_text_swapped,
        extra_with_nav=extra_with,
        extra_no_nav=extra_no,
        extra_counterfactual=extra_counter,
    )


def swap_direction(nav_text: str) -> str:
    """Swap left/right direction words in a navigation instruction.

    Uses a placeholder to avoid double-swapping.

    Examples::

        >>> swap_direction("Turn left onto Main St")
        'Turn right onto Main St'
        >>> swap_direction("Turn right onto Oak Ave")
        'Turn left onto Oak Ave'
        >>> swap_direction("Continue straight")
        'Continue straight'
    """
    PH = "___PH___"
    result = re.sub(r"\bleft\b", PH, nav_text, flags=re.IGNORECASE)
    result = re.sub(
        r"\bright\b",
        lambda m: "left" if m.group()[0].islower() else "Left",
        result,
        flags=re.IGNORECASE,
    )
    result = result.replace(PH, "right")
    return result


def get_nav_token_span(
    input_ids: torch.Tensor, tokenizer: AutoTokenizer, batch_idx: int = 0
) -> tuple[int, int]:
    """Find the positions of route_start and route_end tokens in input_ids.

    Args:
        input_ids: Token IDs, shape ``[B, L]``.
        tokenizer: The model's tokenizer (used to resolve special token IDs).
        batch_idx: Which batch element to inspect.

    Returns:
        ``(start_pos, end_pos)`` -- indices of the route delimiter tokens.

    Raises:
        ValueError: If route tokens are not found in the sequence.
    """
    route_start_id = tokenizer.convert_tokens_to_ids(ROUTE_START_TOKEN)
    route_end_id = tokenizer.convert_tokens_to_ids(ROUTE_END_TOKEN)
    ids = input_ids[batch_idx].tolist()
    if route_start_id not in ids:
        raise ValueError(f"{ROUTE_START_TOKEN} not found in input_ids")
    if route_end_id not in ids:
        raise ValueError(f"{ROUTE_END_TOKEN} not found in input_ids")
    return ids.index(route_start_id), ids.index(route_end_id)


def remove_nav_text(
    input_ids: torch.Tensor, tokenizer: AutoTokenizer, batch_idx: int = 0
) -> torch.Tensor:
    """Remove the entire navigation span from input_ids.

    Strips everything from route_start through route_end (inclusive).

    Args:
        input_ids: Token IDs, shape ``[B, L]``.
        tokenizer: The model's tokenizer.
        batch_idx: Which batch element to modify.

    Returns:
        New input_ids tensor with shape ``[1, new_len]``.
    """
    ids = input_ids[batch_idx].tolist()
    start, end = get_nav_token_span(input_ids, tokenizer, batch_idx)
    new_ids = ids[:start] + ids[end + 1 :]
    return torch.tensor([new_ids], dtype=input_ids.dtype, device=input_ids.device)
