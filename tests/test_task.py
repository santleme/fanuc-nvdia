import json
from pathlib import Path

import numpy as np

from fanuc_cell import make_demo_episode


def test_reference_episode_reaches_pick_and_place_targets():
    episode = make_demo_episode()
    for waypoint in episode.waypoints:
        actual, _ = episode.arm.forward(waypoint.joints)
        _, orientation = episode.arm.forward_pose(waypoint.joints)
        if waypoint.label in {"approach", "grasp", "hold", "lift", "transfer", "release", "retreat"}:
            assert np.linalg.norm(actual - expected_target(episode, waypoint.label)) < 0.005
        assert np.linalg.norm(orientation - np.eye(3)) < 0.02


def expected_target(episode, label):
    config = episode.config
    grasp = np.array([config.pick_x, config.belt_y,
                      config.belt_surface_z + config.part_size / 2])
    offset = np.asarray(config.tool_to_part_offset)
    place = np.asarray(config.place_xyz)
    targets = {
        "approach": grasp - offset + [0.0, 0.0, config.clearance],
        "grasp": grasp - offset,
        "hold": grasp - offset,
        "lift": grasp - offset + [0.0, 0.0, config.clearance],
        "transfer": place - offset + [0.0, 0.0, config.clearance],
        "release": place - offset,
        "retreat": place - offset + [0.0, 0.0, config.clearance],
    }
    return targets[label]


def test_part_reaches_intercept_and_is_carried_then_released():
    episode = make_demo_episode()
    grasp_time = next(point.time for point in episode.waypoints if point.label == "grasp")
    release_time = next(point.time for point in episode.waypoints if point.label == "release")
    _, at_grasp, closed, _, _ = episode.sample(grasp_time)
    assert np.isclose(at_grasp[0], episode.config.pick_x, atol=1e-3)
    target_center = np.array([episode.config.pick_x, episode.config.belt_y,
                              episode.config.belt_surface_z + episode.config.part_size / 2])
    assert np.linalg.norm(at_grasp - target_center) < 0.001
    assert closed
    _, while_held, _, _, _ = episode.sample((grasp_time + release_time) / 2)
    assert np.linalg.norm(while_held - at_grasp) > 0.1
    _, at_release, closed, _, success = episode.sample(release_time)
    assert not closed
    assert np.allclose(at_release, episode.config.place_xyz)
    assert success


def test_cell_asset_belt_and_place_surface_match_task_coordinates():
    episode = make_demo_episode()
    asset_path = Path(__file__).parents[1] / "assets" / "cell_geometry.json"
    geometry = json.loads(asset_path.read_text())["obstacles"]
    belt = geometry["conveyor_belt"]
    belt_top = belt["center"][2] + belt["half_size"][2]
    pedestal = geometry["place_pedestal"]
    pedestal_top = pedestal["center"][2] + pedestal["half_size"][2]
    assert np.isclose(belt_top, episode.config.belt_surface_z)
    assert np.isclose(episode.config.place_xyz[2], pedestal_top + episode.config.part_size / 2)


def test_teacher_gripper_action_two_stage_grasp():
    from fanuc_cell.task import teacher_gripper_action

    # Reference open: always open, whatever the phase/capture state.
    assert teacher_gripper_action("approach", False, False) == -1.0
    assert teacher_gripper_action("approach", False, True) == -1.0
    assert teacher_gripper_action("release", False, True) == -1.0
    # Grasp phase: descend open until the belt capture stops the part.
    assert teacher_gripper_action("grasp", True, False) == -1.0
    assert teacher_gripper_action("grasp", True, True) == -0.04
    # Hold on: close regardless (a capture that never fires degrades to
    # the old timed behavior instead of deadlocking).
    assert teacher_gripper_action("hold", True, False) == -0.04
    assert teacher_gripper_action("lift", True, True) == -0.04
    assert teacher_gripper_action("transfer", True, True) == -0.04
