from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from fanuc_cell import M10ID12_APPROX, SerialArm, inverse_kinematics, inverse_kinematics_pose


def test_model_has_six_revolute_joint_limits():
    arm = SerialArm()
    assert len(arm.model.joint_axes) == 6
    assert len(arm.model.joint_limits) == 6
    assert np.isclose(sum(np.linalg.norm(link) for link in M10ID12_APPROX.link_offsets)
                      + np.linalg.norm(M10ID12_APPROX.tool_offset),
                      M10ID12_APPROX.published_reach, atol=0.02)


def test_forward_kinematics_and_ik_round_trip():
    arm = SerialArm()
    known_joints = np.array([0.25, 0.45, -0.7, 0.15, -0.4, 0.1])
    target, _ = arm.forward(known_joints)
    recovered = inverse_kinematics(arm, target)
    assert np.linalg.norm(arm.forward(recovered)[0] - target) < 0.004


def test_full_pose_ik_reaches_a_level_gripper_pose():
    arm = SerialArm()
    target_position = np.array([0.98, 0.0, 0.15])
    target_orientation = np.eye(3)
    joints = inverse_kinematics_pose(arm, target_position, target_orientation)
    actual_position, actual_orientation = arm.forward_pose(joints)
    assert np.linalg.norm(actual_position - target_position) < 0.004
    assert np.linalg.norm(actual_orientation - target_orientation) < 0.02


def test_pose_jacobian_matches_finite_difference():
    arm = SerialArm()
    joints = np.array([0.2, 0.4, -0.6, 0.3, -0.2, 0.1])
    jacobian = arm.pose_jacobian(joints)
    epsilon = 1e-6
    for index in range(6):
        delta = np.zeros(6)
        delta[index] = epsilon
        plus_position, plus_rotation = arm.forward_pose(joints + delta)
        minus_position, minus_rotation = arm.forward_pose(joints - delta)
        linear = (plus_position - minus_position) / (2 * epsilon)
        rotation_dot = (plus_rotation - minus_rotation) / (2 * epsilon)
        angular_matrix = rotation_dot @ arm.forward_pose(joints)[1].T
        angular = np.array([
            angular_matrix[2, 1] - angular_matrix[1, 2],
            angular_matrix[0, 2] - angular_matrix[2, 0],
            angular_matrix[1, 0] - angular_matrix[0, 1],
        ]) / 2
        assert np.allclose(jacobian[:3, index], linear, atol=2e-6)
        assert np.allclose(jacobian[3:, index], angular, atol=2e-6)


def test_joint_limit_and_target_validation():
    arm = SerialArm()
    with pytest.raises(ValueError, match="six joint"):
        arm.forward(np.zeros(5))
    with pytest.raises(ValueError, match="limits"):
        arm.forward(np.full(6, 10.0))
    with pytest.raises(ValueError, match="finite 3D"):
        inverse_kinematics(arm, np.array([np.nan, 0.0, 0.0]))


def test_unreachable_target_is_reported():
    with pytest.raises(ValueError, match="IK did not converge"):
        inverse_kinematics(SerialArm(), np.array([4.0, 0.0, 1.0]))


def test_urdf_asset_has_six_revolute_arm_joints_and_two_gripper_slides():
    asset = Path(__file__).parents[1] / "assets" / "m10id12_robot.urdf"
    root = ET.parse(asset).getroot()
    joints = {joint.attrib["name"]: joint for joint in root.findall("./joint")}
    assert [joints[f"J{index}"].attrib["type"] for index in range(1, 7)] == ["revolute"] * 6
    assert joints["gripper_left"].attrib["type"] == "prismatic"
    assert joints["gripper_right"].attrib["type"] == "prismatic"
    assert len(root.findall("./link")) == 10
    assert "part" not in joints


def test_urdf_gripper_has_clearance_for_workpiece_when_open():
    asset = Path(__file__).parents[1] / "assets" / "m10id12_robot.urdf"
    root = ET.parse(asset).getroot()
    left = root.find("./joint[@name='gripper_left']/origin")
    right = root.find("./joint[@name='gripper_right']/origin")
    left_finger = root.find("./link[@name='left_finger']/collision/geometry/box")
    assert left is not None and right is not None and left_finger is not None
    left_y = float(left.attrib["xyz"].split()[1])
    right_y = float(right.attrib["xyz"].split()[1])
    half_width = float(left_finger.attrib["size"].split()[1]) / 2
    open_gap = left_y - right_y - 2 * half_width
    max_closure = float(root.find("./joint[@name='gripper_left']/limit").attrib["upper"])
    closed_gap = open_gap - 2 * max_closure
    workpiece_width = 0.08
    assert open_gap > workpiece_width
    assert closed_gap < workpiece_width
