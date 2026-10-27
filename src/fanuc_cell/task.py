"""Deterministic conveyor interception and pick/place reference behavior."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .kinematics import SerialArm, inverse_kinematics_pose

Vector = NDArray[np.float64]


@dataclass(frozen=True)
class CellConfig:
    # Chosen so the part reaches pick_x when the reference motion reaches grasp.
    belt_start_x: float = 0.828
    belt_end_x: float = 1.25
    belt_y: float = 0.0
    belt_surface_z: float = 0.11
    belt_speed: float = 0.08
    pick_x: float = 0.98
    place_xyz: tuple[float, float, float] = (0.80, -0.58, 0.28)
    clearance: float = 0.20
    part_size: float = 0.08
    # Keep the wrist above the conveyor/platform while the fingers overlap the
    # upper half of the workpiece. The model is an approximate side-grip tool.
    tool_to_part_offset: tuple[float, float, float] = (0.095, 0.0, -0.02)
    approach_duration: float = 1.2
    pick_duration: float = 0.7
    gripper_close_duration: float = 0.3
    transfer_duration: float = 1.5
    place_duration: float = 0.6
    release_duration: float = 0.5


@dataclass(frozen=True)
class Waypoint:
    time: float
    joints: Vector
    gripper_closed: bool
    label: str


class PickPlaceEpisode:
    """A timed kinematic episode; it does not model contact dynamics."""

    def __init__(self, arm: SerialArm, config: CellConfig, waypoints: list[Waypoint]):
        if len(waypoints) < 2 or waypoints[0].time != 0.0:
            raise ValueError("Episode needs at least two waypoints and must start at t=0")
        if any(b.time <= a.time for a, b in zip(waypoints, waypoints[1:])):
            raise ValueError("Waypoint times must be strictly increasing")
        self.arm = arm
        self.config = config
        self.waypoints = waypoints

    @property
    def duration(self) -> float:
        return self.waypoints[-1].time

    def sample(self, time: float) -> tuple[Vector, Vector, bool, str, bool]:
        """Return joints, part position, gripper state, phase, and held flag."""
        t = float(np.clip(time, 0.0, self.duration))
        index = max(i for i in range(len(self.waypoints) - 1) if self.waypoints[i].time <= t)
        start, end = self.waypoints[index], self.waypoints[index + 1]
        u = (t - start.time) / (end.time - start.time)
        eased = u * u * (3.0 - 2.0 * u)
        joints = (1.0 - eased) * start.joints + eased * end.joints

        # The part rides the belt until the grasp waypoint. While carried, it is
        # attached to the tool with a fixed local vertical offset.
        grasp_time = next(point.time for point in self.waypoints if point.label == "grasp")
        place_time = next(point.time for point in self.waypoints if point.label == "release")
        held = grasp_time <= t < place_time
        if t < grasp_time:
            part_x = min(
                self.config.belt_start_x + self.config.belt_speed * t,
                self.config.pick_x,
            )
            part = np.array([part_x, self.config.belt_y, self.config.belt_surface_z + self.config.part_size / 2])
        elif held:
            tool, rotation = self.arm.forward_pose(joints)
            part = tool + rotation @ np.asarray(self.config.tool_to_part_offset)
        else:
            part = np.asarray(self.config.place_xyz, dtype=np.float64)

        gripper_closed = start.gripper_closed if u < 1.0 else end.gripper_closed
        phase = start.label if u < 1.0 else end.label
        success = t >= place_time and not held and np.linalg.norm(part - self.config.place_xyz) < 1e-8
        return joints, part, gripper_closed, phase, success


def make_demo_episode(arm: SerialArm | None = None, config: CellConfig | None = None) -> PickPlaceEpisode:
    """Construct a repeatable baseline sequence with position-only IK."""
    arm = arm or SerialArm()
    config = config or CellConfig()
    grasp_part = np.array([config.pick_x, config.belt_y,
                           config.belt_surface_z + config.part_size / 2])
    place_part = np.asarray(config.place_xyz, dtype=np.float64)
    tool_to_part = np.asarray(config.tool_to_part_offset, dtype=np.float64)
    grasp = grasp_part - tool_to_part
    place = place_part - tool_to_part
    poses = [
        ("home", np.array([1.05, 0.0, 0.82]), False),
        ("approach", grasp + [0.0, 0.0, config.clearance], False),
        ("grasp", grasp, True),
        ("hold", grasp, True),
        ("lift", grasp + [0.0, 0.0, config.clearance], True),
        ("transfer", place + [0.0, 0.0, config.clearance], True),
        ("release", place, False),
        ("retreat", place + [0.0, 0.0, config.clearance], False),
    ]
    times = [0.0]
    for duration in (
        config.approach_duration,
        config.pick_duration,
        config.gripper_close_duration,
        config.pick_duration,
        config.transfer_duration,
        config.place_duration,
        config.release_duration,
    ):
        times.append(times[-1] + duration)

    waypoints: list[Waypoint] = []
    seed = np.zeros(6, dtype=np.float64)
    for (label, target, closed), time in zip(poses, times):
        joints = inverse_kinematics_pose(arm, target, np.eye(3), seed=seed)
        waypoints.append(Waypoint(time=time, joints=joints, gripper_closed=closed, label=label))
        seed = joints
    return PickPlaceEpisode(arm, config, waypoints)
