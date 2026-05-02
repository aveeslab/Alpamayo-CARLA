"""Trajectory cache coordinate transforms for CARLA closed-loop reuse."""

import math

import numpy as np


def _state_xyzyaw(state):
    return (
        float(state["x"]),
        float(state["y"]),
        float(state.get("z", 0.0)),
        math.radians(float(state["yaw"])),
    )


def alpamayo_local_to_world(points, ego_state):
    """Convert Alpamayo local points (x forward, y left) to CARLA world points."""

    arr = np.asarray(points, dtype=np.float32)
    x0, y0, z0, yaw = _state_xyzyaw(ego_state)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    out = np.empty_like(arr, dtype=np.float32)
    local_x = arr[..., 0]
    carla_local_y = -arr[..., 1]
    out[..., 0] = x0 + cos_yaw * local_x - sin_yaw * carla_local_y
    out[..., 1] = y0 + sin_yaw * local_x + cos_yaw * carla_local_y
    out[..., 2] = z0 + arr[..., 2]
    return out


def world_to_alpamayo_local(points, ego_state):
    """Convert CARLA world points to Alpamayo local points for the current ego pose."""

    arr = np.asarray(points, dtype=np.float32)
    x0, y0, z0, yaw = _state_xyzyaw(ego_state)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    dx = arr[..., 0] - x0
    dy = arr[..., 1] - y0
    carla_local_x = cos_yaw * dx + sin_yaw * dy
    carla_local_y = -sin_yaw * dx + cos_yaw * dy

    out = np.empty_like(arr, dtype=np.float32)
    out[..., 0] = carla_local_x
    out[..., 1] = -carla_local_y
    out[..., 2] = arr[..., 2] - z0
    return out
