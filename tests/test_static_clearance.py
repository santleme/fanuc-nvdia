"""Geometric clearance checks for the scripted proxy trajectory."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fanuc_cell import SerialArm, make_demo_episode

ROOT = Path(__file__).resolve().parents[1]


def _asset_geometry() -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    geometry = json.loads((ROOT / "assets/cell_geometry.json").read_text())
    obstacles = {
        name: (
            np.asarray(item["center"]) - np.asarray(item["half_size"]),
            np.asarray(item["center"]) + np.asarray(item["half_size"]),
        )
        for name, item in geometry["obstacles"].items()
    }
    return obstacles, geometry["robot"]


def _segment_clearance(
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    obstacle: tuple[np.ndarray, np.ndarray],
) -> float:
    lo, hi = obstacle
    alpha = np.linspace(0.0, 1.0, 101)[:, None]
    points = start[None, :] + alpha * (end - start)[None, :]
    closest = np.clip(points, lo, hi)
    return float(np.linalg.norm(points - closest, axis=1).min() - radius)


def _aabb_clearance(
    center: np.ndarray,
    half_size: np.ndarray,
    obstacle: tuple[np.ndarray, np.ndarray],
) -> float:
    lo, hi = obstacle
    robot_lo, robot_hi = center - half_size, center + half_size
    gap = np.maximum(np.maximum(lo - robot_hi, robot_lo - hi), 0.0)
    return float(np.linalg.norm(gap))


def test_scripted_robot_envelope_clears_static_cell_geometry() -> None:
    obstacles, geoms = _asset_geometry()
    arm = SerialArm()
    episode = make_demo_episode(arm)
    capsule_names = (
        "shoulder_column",
        "upper_arm",
        "forearm",
        "wrist_1",
        "wrist_2",
    )
    radii = [float(geoms["capsule_radii"][name]) for name in capsule_names]
    minimum_clearance = {name: float("inf") for name in obstacles}
    sample_count = int(np.ceil(episode.duration / 0.01)) + 1

    for time in np.linspace(0.0, episode.duration, sample_count):
        joints, _, gripper_closed, _, _ = episode.sample(time)
        points = arm.forward(joints)[1][1:]
        tool_center, rotation = arm.forward_pose(joints)
        link6_origin = points[-2]

        for index, radius in enumerate(radii):
            for name, bounds in obstacles.items():
                clearance = _segment_clearance(points[index], points[index + 1], radius, bounds)
                minimum_clearance[name] = min(minimum_clearance[name], clearance)

        # Conservative AABBs include the wrist cylinder and rectangular tool.
        wrist_radius, wrist_half_length = geoms["wrist_cylinder"]
        wrist_center = link6_origin + rotation @ np.array([0.04, 0.0, 0.0])
        wrist_half_extents = np.abs(rotation) @ np.array(
            [wrist_half_length, wrist_radius, wrist_radius]
        )
        tool_half_extents = np.asarray(geoms["tool_half_size"])
        tool_half_extents = np.abs(rotation) @ tool_half_extents
        boxes = [(wrist_center, wrist_half_extents), (tool_center, tool_half_extents)]

        finger_slide = 0.025 if gripper_closed else 0.0
        finger_half_size = np.asarray(geoms["finger_half_size"])
        for side in (-1.0, 1.0):
            finger_local = np.array([0.055, side * (0.055 - finger_slide), 0.0])
            finger_center = tool_center + rotation @ finger_local
            finger_extents = np.abs(rotation) @ finger_half_size
            boxes.append((finger_center, finger_extents))

        for center, half_size in boxes:
            for name, bounds in obstacles.items():
                clearance = _aabb_clearance(center, half_size, bounds)
                minimum_clearance[name] = min(minimum_clearance[name], clearance)

    assert minimum_clearance["conveyor_belt"] > 0.005, minimum_clearance
    assert minimum_clearance["place_pedestal"] > 0.005, minimum_clearance
