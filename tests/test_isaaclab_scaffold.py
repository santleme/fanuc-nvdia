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
    # Clone-mode diagnosis (fanuc-diagnose-clone-modes v1): the root is fixed
    # by the spawner, not by the importer's --fix-base joint, and physics
    # replication stays on so OvPhysX contact binding resolves every env.
    assert "fix_root_link=True" in env_cfg
    assert "replicate_physics = False" not in env_cfg


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
    assert "capture_attempt = close_to_pick" in mdp
    assert "gripper_pinched & close_to_pick" not in mdp
    assert "env._belt_capture_paused[env_ids] |= capture_attempt" in mdp
    assert "on_belt &= ~env._belt_capture_paused[env_ids]" in mdp
    assert "env._belt_capture_paused[env_ids] = False" in mdp
    assert "closed_joint_position: float = 0.004" in mdp
    assert "tool_distance: float = 0.14" in mdp
    assert "as physics_cfg" in baseline
    assert "cfg.sim.physics = physics_cfg" in baseline
    assert "measured_part = (" in baseline
    # The manual J2/J3 compensation was removed after the v46 analysis showed
    # it displaced the TCP by ~4 cm after IK; the scripted baseline must not
    # reintroduce it.
    assert "command_joints[1] +=" not in baseline
    assert "command_joints[2] +=" not in baseline
    assert 'phase in {"approach", "grasp", "hold", "lift"}' in baseline
    assert 'if last_phase != "lift":' in baseline
    assert "lift_origin_part = (" in baseline
    assert "capture_anchor_part.copy()" in baseline
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
    assert "'newton_gl', '--video" in source
    assert "'--headless'" not in source  # Isaac Lab 3.0 play CLI rejects it (v66 verdict)
    # Protocol v2 (founder decision 26/10/2026): train and probe at 64 envs,
    # same 500 PPO iterations/updates as v66's 8-env run.
    assert "'--num_envs', '64', '--max_iterations', '500'" in source
    assert "'--num-envs', '64', '--steps', '24'" in source
    assert "did not report Metrics/success_rate" in source
    assert "baseline.returncode != 0 or not baseline_metrics.is_file()" in source
    assert "PPO training is gated until it passes" in source
    assert "physics=ovphysx" in source
    # Redundant backend/task smokes were removed from the debug path (v43+);
    # the Cartpole smoke stays available behind the RUN_BACKEND_SMOKES gate.
    assert "RUN_BACKEND_SMOKES = False" in source
    assert "smoke = run(['uv', 'run', '--extra', 'ovphysx', 'isaaclab', 'zero_agent'" in source
    assert "diagnose_training_nan.py" in source
    assert "replicate_physics" not in source  # env-side fix lives in env_cfg, not the notebook
    assert "'--fix-base'" not in source  # root is fixed via spawn fix_root_link (mode D verdict)
    assert "convert_urdf.py" in source
    assert "convert_urdf_ovphysx.py" in source


def test_notebook_initializes_report_before_command_logger_runs() -> None:
    notebook = json.loads((ROOT / "kaggle/fanuc-conveyor-pick-place.ipynb").read_text())
    preflight = "".join(notebook["cells"][1]["source"])
    assert preflight.index("report = {'commands': []}") < preflight.index("def run(")


def test_notebook_parses_list_shaped_benchmark_summary() -> None:
    notebook = json.loads((ROOT / "kaggle/fanuc-conveyor-pick-place.ipynb").read_text())
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    # v67 verdict: the benchmark summary is a LIST of phase records whose
    # measurements carry 'benchmark_play play success_rate'; a dict .get()
    # crashed the notebook after every stage had already passed.
    assert "evaluation_data.get('success_rate')" not in source
    assert "play success_rate" in source


def test_v68_reward_is_staged_and_gated() -> None:
    """v68 (issue #1): reach no ponto de grasp, place/lift travados no segurar."""
    mdp = (OVERLAY / "mdp/cell.py").read_text()
    env_cfg = (OVERLAY / "config/m10id12/env_cfg.py").read_text()

    # reach mira o ponto de grasp (peça - offset), kernel tanh largo
    assert "tool_to_part_offset: tuple[float, float, float] = (0.095, 0.0, -0.02)" in mdp
    assert "grasp_point = part - offset" in mdp
    assert "1.0 - torch.tanh(distance / 0.1)" in mdp
    # degrau intermediário: segurar (contato bilateral + garra fechada)
    assert "def object_held(" in mdp
    # place condicionado ao segurar — sem ótimo local da peça parada
    assert "return tracking * object_held(env, robot_cfg, part_cfg)" in mdp
    # lift binário por passo, travado no segurar
    assert "lifted = above_belt & (object_held(env, robot_cfg, part_cfg) > 0.0)" in mdp
    # cfg: termo novo + pesos staged
    assert "grasp_contact = RewardTerm(func=mdp.object_held, weight=2.0)" in env_cfg
    assert "lift = RewardTerm(func=mdp.object_lifted, weight=6.0)" in env_cfg


def test_v68_contact_observations_and_curriculum() -> None:
    """v68 (issue #1): política sente o toque; currículo de esteira em 3 fases."""
    mdp = (OVERLAY / "mdp/cell.py").read_text()
    env_cfg = (OVERLAY / "config/m10id12/env_cfg.py").read_text()

    assert "def finger_contact_forces(env)" in mdp
    assert "finger_contact = ObservationTerm(func=mdp.finger_contact_forces)" in env_cfg
    assert "def belt_difficulty_curriculum(" in mdp
    assert "speed_levels: tuple = ((0.0, 0.0), (0.02, 0.04), (0.05, 0.10))" in mdp
    assert 'getattr(env, "_belt_speed_range", (0.05, 0.10))' in mdp
    assert 'getattr(env, "_belt_x_jitter", 0.10)' in mdp
    assert "class CurriculumCfg:" in env_cfg
    assert "belt_difficulty = CurriculumTerm(func=mdp.belt_difficulty_curriculum)" in env_cfg
    assert "curriculum: CurriculumCfg = CurriculumCfg()" in env_cfg


def test_newton_preset_is_available_but_inert() -> None:
    """Issue #2: preset Newton/MJWarp existe; o default segue OvPhysX."""
    env_cfg = (OVERLAY / "config/m10id12/env_cfg.py").read_text()

    assert "from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg" in env_cfg
    assert "newton_mjwarp: NewtonCfg = NewtonCfg(" in env_cfg
    assert 'integrator="implicitfast"' in env_cfg
    assert "nconmax=512" in env_cfg
    assert "default: OvPhysxCfg = ovphysx" in env_cfg


def test_notebook_builder_parameterizes_experiment() -> None:
    """Builder: física/escala/slug viram parâmetros; defaults = histórico."""
    builder = (ROOT / "scripts/build_kaggle_notebook.py").read_text()

    assert '"--physics"' in builder
    assert 'choices=("ovphysx", "newton_mjwarp")' in builder
    assert '"--train-num-envs"' in builder and '"--train-max-iterations"' in builder
    assert '"--kernel-slug"' in builder
    assert "args.train_num_envs" in builder and "args.physics" in builder
    # com defaults, nenhuma troca é aplicada (saída byte a byte a de sempre)
    assert 'if args.physics != "ovphysx" or args.train_num_envs != 64' in builder


def test_env_cfg_mdp_references_are_all_exported() -> None:
    """Regressão da v68: o probe morreu com AttributeError porque
    `mdp/__init__.py` tem lista explícita de exports e as funções novas
    (finger_contact_forces, object_held, belt_difficulty_curriculum)
    ficaram fora. Todo `mdp.<nome>` usado no env_cfg precisa estar
    exportado — e definido em cell.py."""
    import re

    env_cfg = (OVERLAY / "config/m10id12/env_cfg.py").read_text()
    mdp_init = (OVERLAY / "mdp/__init__.py").read_text()
    cell = (OVERLAY / "mdp/cell.py").read_text()

    referenced = set(re.findall(r"(?<![\w])mdp\.(\w+)", env_cfg))
    assert referenced, "env_cfg deveria referenciar termos mdp"
    for name in sorted(referenced):
        assert f"    {name},\n" in mdp_init or f"    {name}\n" in mdp_init, \
            f"mdp.{name} usado no env_cfg mas não exportado em mdp/__init__.py"
        assert f"def {name}(" in cell or f"class {name}(" in cell, \
            f"mdp.{name} exportado mas não definido em cell.py"


def test_builder_newton_mode_drops_uv_extra() -> None:
    """Regressão do spike v1: não existe extra `newton` no Isaac Lab
    ("Extra 'newton' is not defined" na conversão). Fora do default,
    o builder REMOVE o par '--extra', 'ovphysx' em vez de renomear."""
    builder = (ROOT / "scripts/build_kaggle_notebook.py").read_text()

    assert "source.replace(\"'--extra', 'ovphysx', \", \"\")" in builder
    assert "'--extra', '{args.physics}'" not in builder


def test_scripts_take_backend_from_env_and_baseline_skips_curriculum() -> None:
    """Vereditos v69/spike v2 (27/10/2026): (1) probe/baseline tinham
    args.physics hardcoded em "ovphysx" — no spike eles testariam o
    backend errado; agora leem FANUC_CELL_PHYSICS, que o notebook
    define por experimento. (2) O baseline morreu na v69 sem testar
    fisica: o curriculo da v68 poe a fase 1 estatica (pick em x=0,80)
    e o IK do professor diverge nesse pick — reproduzido localmente
    com o residuo exato do kernel (1,0006 m / 1,8201 rad). O baseline
    e o portao fisico historico: roda SEM curriculo."""
    demo = (ROOT / "scripts/run_scripted_demo.py").read_text()
    probe = (ROOT / "scripts/diagnose_training_nan.py").read_text()
    builder = (ROOT / "scripts/build_kaggle_notebook.py").read_text()

    assert 'os.environ.get("FANUC_CELL_PHYSICS", "ovphysx")' in demo
    assert 'os.environ.get("FANUC_CELL_PHYSICS", "ovphysx")' in probe
    assert "cfg.curriculum = None" in demo
    assert "baseline_env['FANUC_CELL_PHYSICS'] = 'ovphysx'" in builder
    assert "'FANUC_CELL_PHYSICS'] = '{args.physics}'" in builder


def test_bc_train_script_handles_deprecated_rsl_rl_cfg() -> None:
    # Regressao da BC v4: o treino oficial chama
    # handle_deprecated_rsl_rl_cfg antes do OnPolicyRunner; sem isso o
    # rsl-rl >= 4 morre com KeyError 'class_name' em cfg["actor"].
    train_bc = (ROOT / "scripts/train_ppo_bc.py").read_text()

    assert "handle_deprecated_rsl_rl_cfg(" in train_bc
    assert 'metadata.version("rsl-rl-lib")' in train_bc
    # Injecao compativel com a API nova (alg.actor/mlp) e a antiga.
    assert "def _actor_linears" in train_bc
    assert 'getattr(alg, "actor", None)' in train_bc
    assert "BC_WARMSTART_APPLIED" in train_bc
