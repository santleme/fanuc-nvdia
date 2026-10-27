"""Approximate six-axis industrial arm kinematics.

The link offsets are a configurable proxy sized to the M-10iD/12 published
reach. They are not FANUC CAD dimensions and do not model dynamics or accuracy.
All lengths are metres and all angles are radians.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]


@dataclass(frozen=True)
class ArmModel:
    name: str
    base_height: float
    link_offsets: tuple[tuple[float, float, float], ...]
    tool_offset: tuple[float, float, float]
    joint_axes: tuple[tuple[float, float, float], ...]
    joint_limits: tuple[tuple[float, float], ...]
    published_reach: float


M10ID12_APPROX = ArmModel(
    name="FANUC M-10iD/12-inspired proxy (approximate geometry)",
    base_height=0.45,
    # Link vectors after J1 ... J6; the tool offset is specified separately.
    link_offsets=((0.0, 0.0, 0.15), (0.32, 0.0, 0.0), (0.47, 0.0, 0.0),
                  (0.31, 0.0, 0.0), (0.08, 0.0, 0.0), (0.0, 0.0, 0.0)),
    tool_offset=(0.10, 0.0, 0.0),
    joint_axes=((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0),
                (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
    # Symmetric ranges are a visualization proxy; FANUC's published ranges are
    # asymmetric in some axes and must replace these before controller transfer.
    joint_limits=tuple((np.deg2rad(-r), np.deg2rad(r)) for r in (170, 117.5, 227.5, 190, 180, 450)),
    published_reach=1.441,
)


def _rotation(axis: Vector, angle: float) -> Matrix:
    """Rodrigues rotation for a normalized axis."""
    x, y, z = axis / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


class SerialArm:
    """Forward kinematics and geometric Jacobian for the six-axis proxy."""

    def __init__(self, model: ArmModel = M10ID12_APPROX) -> None:
        if len(model.joint_axes) != 6 or len(model.joint_limits) != 6:
            raise ValueError("The arm model must define exactly six revolute joints")
        self.model = model

    def validate_joints(self, joints: Vector) -> Vector:
        joints = np.asarray(joints, dtype=np.float64)
        if joints.shape != (6,):
            raise ValueError(f"Expected six joint values, got shape {joints.shape}")
        limits = np.asarray(self.model.joint_limits)
        if np.any(joints < limits[:, 0]) or np.any(joints > limits[:, 1]):
            raise ValueError("Joint command exceeds the configured proxy limits")
        return joints

    def forward(self, joints: Vector) -> tuple[Vector, list[Vector]]:
        """Return tool position and all joint origins in world coordinates."""
        tool, _, origins, _, _ = self._pose_components(joints)
        return tool, origins

    def forward_pose(self, joints: Vector) -> tuple[Vector, Matrix]:
        """Return tool position and orientation as a 3x3 rotation matrix."""
        position, rotation, _, _, _ = self._pose_components(joints)
        return position, rotation

    def _pose_components(
        self, joints: Vector
    ) -> tuple[Vector, Matrix, list[Vector], list[Vector], list[Vector]]:
        q = self.validate_joints(joints)
        rotation = np.eye(3)
        position = np.array([0.0, 0.0, self.model.base_height], dtype=np.float64)
        origins = [np.zeros(3, dtype=np.float64), position.copy()]
        joint_origins: list[Vector] = []
        axes_world: list[Vector] = []
        for index, angle in enumerate(q):
            axis_local = np.asarray(self.model.joint_axes[index], dtype=np.float64)
            joint_origins.append(position.copy())
            axes_world.append(rotation @ axis_local)
            rotation = rotation @ _rotation(axis_local, angle)
            link_offset = np.asarray(self.model.link_offsets[index])
            position = position + rotation @ link_offset
            if np.linalg.norm(link_offset) > 0.0:
                origins.append(position.copy())
        tool = position + rotation @ np.asarray(self.model.tool_offset)
        origins.append(tool.copy())
        return tool, rotation, origins, joint_origins, axes_world

    def jacobian(self, joints: Vector) -> Matrix:
        """Position-only geometric Jacobian, shape (3, 6)."""
        tool, _, _, joint_origins, axes_world = self._pose_components(joints)
        return np.column_stack(
            [np.cross(axis, tool - origin) for axis, origin in zip(axes_world, joint_origins)]
        )

    def pose_jacobian(self, joints: Vector) -> Matrix:
        """Geometric position/orientation Jacobian in world coordinates (6, 6)."""
        tool, _, _, joint_origins, axes_world = self._pose_components(joints)
        linear = [np.cross(axis, tool - origin) for axis, origin in zip(axes_world, joint_origins)]
        return np.vstack((np.column_stack(linear), np.column_stack(axes_world)))


def inverse_kinematics(
    arm: SerialArm,
    target: Vector,
    seed: Vector | None = None,
    *,
    tolerance: float = 0.004,
    max_iterations: int = 180,
    damping: float = 0.035,
) -> Vector:
    """Damped least-squares position IK; tool orientation is unconstrained."""
    goal = np.asarray(target, dtype=np.float64)
    if goal.shape != (3,) or not np.all(np.isfinite(goal)):
        raise ValueError("Target must be a finite 3D position")
    q = np.zeros(6, dtype=np.float64) if seed is None else arm.validate_joints(seed).copy()
    lower, upper = np.asarray(arm.model.joint_limits).T
    for _ in range(max_iterations):
        current, _ = arm.forward(q)
        error = goal - current
        if np.linalg.norm(error) <= tolerance:
            return q
        jacobian = arm.jacobian(q)
        # DLS in task space is stable near singular arm configurations.
        delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping**2 * np.eye(3), error
        )
        q = np.clip(q + np.clip(delta, -0.18, 0.18), lower, upper)
    residual = np.linalg.norm(arm.forward(q)[0] - goal)
    raise ValueError(f"IK did not converge: residual {residual:.4f} m exceeds {tolerance:.4f} m")


def inverse_kinematics_pose(
    arm: SerialArm,
    target_position: Vector,
    target_orientation: Matrix,
    seed: Vector | None = None,
    *,
    position_tolerance: float = 0.001,
    orientation_tolerance: float = 0.02,
    max_iterations: int = 240,
    damping: float = 0.04,
    orientation_scale: float = 0.25,
) -> Vector:
    """Damped least-squares IK for TCP position and 3D orientation."""
    goal_position = np.asarray(target_position, dtype=np.float64)
    goal_rotation = np.asarray(target_orientation, dtype=np.float64)
    if goal_position.shape != (3,) or not np.all(np.isfinite(goal_position)):
        raise ValueError("Target position must be a finite 3D vector")
    if goal_rotation.shape != (3, 3) or not np.all(np.isfinite(goal_rotation)):
        raise ValueError("Target orientation must be a finite 3x3 matrix")
    if not np.allclose(goal_rotation.T @ goal_rotation, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(goal_rotation), 1.0, atol=1e-5
    ):
        raise ValueError("Target orientation must be a proper rotation matrix")

    q = np.zeros(6, dtype=np.float64) if seed is None else arm.validate_joints(seed).copy()
    lower, upper = np.asarray(arm.model.joint_limits).T
    for _ in range(max_iterations):
        current_position, current_rotation = arm.forward_pose(q)
        position_error = goal_position - current_position
        orientation_error = _rotation_vector(goal_rotation @ current_rotation.T)
        if (np.linalg.norm(position_error) <= position_tolerance
                and np.linalg.norm(orientation_error) <= orientation_tolerance):
            return q
        error = np.concatenate((position_error, orientation_scale * orientation_error))
        jacobian = arm.pose_jacobian(q).copy()
        jacobian[3:, :] *= orientation_scale
        delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping**2 * np.eye(6), error
        )
        q = np.clip(q + np.clip(delta, -0.18, 0.18), lower, upper)

    position_residual = np.linalg.norm(goal_position - arm.forward_pose(q)[0])
    orientation_residual = np.linalg.norm(_rotation_vector(goal_rotation @ arm.forward_pose(q)[1].T))
    raise ValueError(
        "Pose IK did not converge: "
        f"position residual {position_residual:.4f} m, "
        f"orientation residual {orientation_residual:.4f} rad"
    )


def _rotation_vector(rotation: Matrix) -> Vector:
    """Map a rotation matrix to its shortest axis-angle vector."""
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    skew = np.array(
        [rotation[2, 1] - rotation[1, 2],
         rotation[0, 2] - rotation[2, 0],
         rotation[1, 0] - rotation[0, 1]],
        dtype=np.float64,
    )
    if angle < 1e-7:
        return 0.5 * skew
    if np.pi - angle < 1e-5:
        eigenvalues, eigenvectors = np.linalg.eig(rotation)
        axis = np.real(eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))])
        axis /= np.linalg.norm(axis)
        return angle * axis
    return (angle / (2.0 * np.sin(angle))) * skew
