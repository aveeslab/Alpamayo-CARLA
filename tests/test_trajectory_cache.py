import numpy as np

from module.trajectory_cache import alpamayo_local_to_world, world_to_alpamayo_local


def test_trajectory_cache_round_trips_alpamayo_local_points():
    state = {"x": 10.0, "y": 20.0, "z": 1.0, "yaw": 90.0}
    local = np.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 2.0, 0.0],
            [10.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )

    world = alpamayo_local_to_world(local, state)
    restored = world_to_alpamayo_local(world, state)

    assert np.allclose(restored, local, atol=1e-5)


def test_cached_world_trajectory_reprojects_after_ego_moves_forward():
    anchor = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}
    current = {"x": 2.0, "y": 0.0, "z": 0.0, "yaw": 0.0}
    local = np.array(
        [
            [3.0, 0.0, 0.0],
            [6.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    world = alpamayo_local_to_world(local, anchor)
    reprojected = world_to_alpamayo_local(world, current)

    assert np.allclose(reprojected[:, 0], [1.0, 4.0])
    assert np.allclose(reprojected[:, 1], [0.0, 1.0])


def test_world_to_alpamayo_local_handles_batched_trajectory_samples():
    state = {"x": 1.0, "y": 2.0, "z": 0.5, "yaw": 0.0}
    batched_world = np.array(
        [
            [[2.0, 1.0, 0.5], [3.0, 2.0, 0.5]],
            [[1.0, 3.0, 0.5], [4.0, 0.0, 0.5]],
        ],
        dtype=np.float32,
    )

    local = world_to_alpamayo_local(batched_world, state)

    assert local.shape == batched_world.shape
    assert np.allclose(local[0], [[1.0, 1.0, 0.0], [2.0, -0.0, 0.0]])
    assert np.allclose(local[1], [[0.0, -1.0, 0.0], [3.0, 2.0, 0.0]])
