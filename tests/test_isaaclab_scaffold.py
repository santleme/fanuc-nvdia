"""Static contracts for the external Isaac Lab project scaffold.

These checks run without Isaac Lab installed. They protect the package wiring;
the Kaggle smoke still owns runtime/API validation.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "isaaclab_overlay/src/fanuc_cell_lab/tasks/conveyor_pick_place"


def test_external_task_is_imported_and_registered_with_an_rl_agent() -> None:
    tasks_init = (OVERLAY.parents[0] / "__init__.py").read_text()
    task_init = (OVERLAY / "__init__.py").read_text()
    config_init = (OVERLAY / "config/__init__.py").read_text()
    robot_init = (OVERLAY / "config/m10id12/__init__.py").read_text()
    agent_cfg = (OVERLAY / "config/m10id12/agents/rsl_rl_ppo_cfg.py").read_text()

    assert "from . import conveyor_pick_place" in tasks_init
    assert "from . import config, mdp" in task_init
    assert "from . import m10id12" in config_init
    assert "gym.register(" in robot_init
    assert 'id="Fanuc-Conveyor-PickPlace-M10id12-v0"' in robot_init
    assert "env_cfg_entry_point" in robot_init
    assert "rsl_rl_cfg_entry_point" in robot_init
    assert "class M10id12PPORunnerCfg" in agent_cfg
    assert "RslRlPpoAlgorithmCfg" in agent_cfg


def test_task_uses_partner_filtered_contact_on_both_gripper_fingers() -> None:
    env_cfg = (OVERLAY / "config/m10id12/env_cfg.py").read_text()
    mdp = (OVERLAY / "mdp/cell.py").read_text()

    assert env_cfg.count("ContactSensorCfg(") == 2
    assert env_cfg.count('filter_prim_paths_expr=["{ENV_REGEX_NS}/Part"]') == 2
    assert "left_finger_contact" in mdp and "right_finger_contact" in mdp
    assert "normal_force_matrix_w.torch" in mdp
    assert "grasp_now = gripper_closed & bilateral_contact & lifted_near_tool" in mdp
    assert "env._part_was_grasped & gripper_open" in mdp


def test_task_spawns_dynamic_part_and_collidable_cell_surfaces() -> None:
    env_cfg = (OVERLAY / "config/m10id12/env_cfg.py").read_text()
    assert env_cfg.count("collision_props=sim_utils.CollisionPropertiesCfg()") >= 3
    assert "rigid_props=sim_utils.RigidBodyPropertiesCfg()" in env_cfg
    assert "mass_props=sim_utils.MassPropertiesCfg(mass=0.35)" in env_cfg
    assert '"J3": (-3.97062, 3.97062)' in env_cfg
    assert '"J[1-6]": (-1.0, 1.0)' not in env_cfg
    assert "stiffness=15000.0" in env_cfg
    assert '"J[1-2]": 450.0' in env_cfg


def test_reset_and_interval_terms_accept_isaaclab_full_environment_slice() -> None:
    mdp = (OVERLAY / "mdp/cell.py").read_text()
    baseline = (ROOT / "scripts/run_scripted_demo.py").read_text()
    assert "def _resolve_env_ids(env, env_ids: torch.Tensor | slice)" in mdp
    assert "env_ids = _resolve_env_ids(env, env_ids)" in mdp
    assert "count = env_ids.numel()" in mdp
    assert "env.scene.env_origins.torch" not in mdp
    assert "scene.env_origins.torch" not in baseline
    assert "env._belt_origin_x[env_ids] = state[:, 0] - origins[:, 0]" in mdp
    assert "env._belt_start_steps[env_ids] = int(env.common_step_counter)" in mdp
    assert "int(env.common_step_counter) - env._belt_start_steps[env_ids]" in mdp
    assert "env._belt_speeds[env_ids] * elapsed_time" in mdp
    assert "velocity[on_belt, 0] = 0.0" in mdp
    assert "capture_attempt = gripper_pinched & close_to_pick" in mdp
    assert "env._belt_capture_paused[env_ids] |= capture_attempt" in mdp
    assert "on_belt &= ~env._belt_capture_paused[env_ids]" in mdp
    assert "env._belt_capture_paused[env_ids] = False" in mdp
    assert "closed_joint_position: float = 0.009" in mdp
    assert "tool_distance: float = 0.14" in mdp
    assert "as physics_cfg" in baseline
    assert "cfg.sim.physics = physics_cfg" in baseline
    assert "measured_part = (" in baseline
    assert "command_joints[1] += 0.018" in baseline
    assert "command_joints[2] += 0.057" in baseline
    assert 'phase in {"approach", "grasp", "hold", "lift"}' in baseline
    assert 'if last_phase != "lift":' in baseline
    assert "lift_origin_part = measured_part.copy()" in baseline
    assert '"closed_bilateral_contact_steps": closed_bilateral_contact_steps' in baseline


def test_success_and_failure_terms_return_from_their_own_functions() -> None:
    tree = ast.parse((OVERLAY / "mdp/cell.py").read_text())
    functions = {
        node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    for name in ("object_at_goal", "object_lost"):
        assert any(isinstance(statement, ast.Return) for statement in functions[name].body)


def test_placement_success_metric_is_exported_for_environment_config() -> None:
    mdp_init = (OVERLAY / "mdp/__init__.py").read_text()
    assert "ObjectPlacementSuccessRate," in mdp_init
    assert '"ObjectPlacementSuccessRate"' in mdp_init


def test_placement_success_is_logged_for_isaaclab_benchmark() -> None:
    env_cfg = (OVERLAY / "config/m10id12/env_cfg.py").read_text()
    mdp_source = (OVERLAY / "mdp/cell.py").read_text()
    builder = (ROOT / "scripts/build_kaggle_notebook.py").read_text()

    assert "success_rate = RewardTerm(func=mdp.ObjectPlacementSuccessRate, weight=0.0)" in env_cfg
    assert "self._env.extras.setdefault(\"log\", {})[\"Metrics/success_rate\"]" in mdp_source
    assert "policy_success_rate'] is None" in builder


def test_notebook_requests_t4_and_physical_policy_evidence() -> None:
    metadata = json.loads((ROOT / "kaggle/kernel-metadata.json").read_text())
    notebook = json.loads((ROOT / "kaggle/fanuc-conveyor-pick-place.ipynb").read_text())
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["enable_gpu"] is True and metadata["enable_tpu"] is False
    assert "kaggle_t4_detected" in source
    assert "scripts/run_scripted_demo.py" in source
    assert "'--headless', '--device', 'cuda:0'" not in source
    assert "--max_iterations', '500" in source
    assert "--benchmark_formatter', 'schema,summary" in source
    assert "--video_length', '400" in source
    assert "'newton_gl', '--headless', '--video" in source
    assert "did not report Metrics/success_rate" in source
    assert "baseline.returncode != 0 or not baseline_metrics.is_file()" in source
    assert "PPO training is gated until it passes" in source
    assert "physics=ovphysx" in source
    assert "task_smoke = run(['uv', 'run', '--extra', 'ovphysx', 'isaaclab', 'zero_agent'" in source
    assert "convert_urdf.py" in source
    assert "convert_urdf_ovphysx.py" in source


def test_notebook_initializes_report_before_command_logger_runs() -> None:
    notebook = json.loads((ROOT / "kaggle/fanuc-conveyor-pick-place.ipynb").read_text())
    preflight = "".join(notebook["cells"][1]["source"])
    assert preflight.index("report = {'commands': []}") < preflight.index("def run(")
