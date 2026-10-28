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
    assert "_rate is None" in builder  # B3: extracao por seed do benchmark


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


def test_newton_preset_contact_retune_attempt_1() -> None:
    """Spike v3 (issue #2): portão 0 verde, portão 1 reprovado por
    comportamento — contato unilateral (picos 63,6 × 250,8 N),
    `closed_bilateral_contact_steps = 0` e peça ejetada da esteira
    com o cone `pyramidal` e o `impratio` default (= 1). O protocolo
    do spike permite ≤ 3 retunes de contato (nconmax/atrito/impratio);
    a tentativa 1 reforça a formulação de atrito no preset Newton:
    cone `elliptic` + `impratio=10`. Regressão: se alguém reverter o
    retune (ou mexer nas capacidades), este teste quebra antes do
    kernel. O preset OvPhysX e o default seguem intocados."""
    env_cfg = (OVERLAY / "config/m10id12/env_cfg.py").read_text()

    assert 'cone="elliptic"' in env_cfg
    assert "impratio=10.0" in env_cfg
    assert 'cone="pyramidal"' not in env_cfg
    assert "nconmax=512" in env_cfg and "njmax=1024" in env_cfg
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


def test_bundled_project_zip_contains_every_fanuc_cell_module() -> None:
    # Regressao da BC v5: o PR #7 adicionou src/fanuc_cell/rerun_log.py
    # e o __init__ passou a importa-lo, mas o manifesto do builder nao
    # acompanhou — o kernel morreu no estagio de testes locais com
    # ModuleNotFoundError antes de qualquer fisica.
    import base64
    import importlib.util
    import io
    import zipfile

    builder_path = ROOT / "scripts/build_kaggle_notebook.py"
    spec = importlib.util.spec_from_file_location("builder", builder_path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    archive = zipfile.ZipFile(io.BytesIO(base64.b64decode(builder.bundled_project())))
    names = set(archive.namelist())
    on_disk = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src/fanuc_cell").glob("*.py")
    }
    assert on_disk <= names, f"modulos fora do zip do kernel: {on_disk - names}"


def test_notebook_benchmark_is_multi_seed_b3() -> None:
    # B3 (issue #10): o veredito oficial sai de 3 seeds fixas, com
    # media +- desvio no runtime-report — seed unica nao separa
    # sinal de ruido.
    builder = (ROOT / "scripts/build_kaggle_notebook.py").read_text()
    assert "benchmark_seeds = (101, 202, 303)" in builder
    assert "runtime_report['policy_success_rates'] = seed_rates" in builder
    assert "runtime_report['policy_success_rate_std']" in builder
    assert "'--seed', str(_seed)," in builder



def _class_block(source: str, start: str, end: str) -> str:
    i = source.index(start)
    return source[i:source.index(end, i)]


def _build_notebook_source(tmp_path, monkeypatch, extra_argv) -> str:
    """Roda o builder de verdade (saida em tmp) e devolve o fonte."""
    import importlib.util
    import sys

    builder_path = ROOT / "scripts/build_kaggle_notebook.py"
    spec = importlib.util.spec_from_file_location("builder_cinematic",
                                                  builder_path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    monkeypatch.setattr(builder, "KAGGLE_DIR", tmp_path)
    monkeypatch.setattr(
        sys, "argv",
        ["build_kaggle_notebook.py", "--kaggle-user", "tester",
         *extra_argv])
    builder.main()
    notebook = json.loads((tmp_path / builder.NOTEBOOK_NAME).read_text())
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


def _statement(source: str, start_marker: str, end_marker: str) -> str:
    return source[source.index(start_marker):source.index(end_marker)]


def test_builder_cinematic_video_flag_is_opt_in() -> None:
    """Issue #9 fatia 1: a flag existe e o default não muda nada."""
    builder = (ROOT / "scripts/build_kaggle_notebook.py").read_text()

    assert '"--cinematic-video", action="store_true"' in builder
    assert "if args.cinematic_video:" in builder
    assert "'--viz', 'newton_rtx'" in builder
    assert "'--extra', 'ovrtx'" in builder
    assert "cinematic_env['FANUC_CELL_CINEMATIC'] = '1'" in builder


def test_builder_default_notebook_has_no_cinematic_traces(
        tmp_path, monkeypatch) -> None:
    source = _build_notebook_source(tmp_path, monkeypatch, [])

    assert "'newton_gl', '--video" in source
    assert "newton_rtx" not in source
    assert "ovrtx" not in source
    assert "FANUC_CELL_CINEMATIC" not in source
    assert "cinematic_env" not in source


def test_builder_cinematic_only_changes_playback(
        tmp_path, monkeypatch) -> None:
    default = _build_notebook_source(tmp_path / "d", monkeypatch, [])
    cinematic = _build_notebook_source(
        tmp_path / "c", monkeypatch, ["--cinematic-video"])

    # Fora do playback, os dois notebooks são idênticos: o que antecede
    # a definição do env cinematic e o que segue o fim do comando de
    # playback não podem diferir (treino, benchmark, conversão...).
    head_d, tail_d = default.split("    playback = run(", 1)
    head_c, tail_c = cinematic.split("    cinematic_env = os.environ.copy()", 1)
    assert head_d == head_c
    end = "runtime_report['policy_playback_exit']"
    assert tail_d[tail_d.index(end):] == tail_c[tail_c.index(end):]

    playback = _statement(cinematic, "playback = run(", end)
    assert "'--viz', 'newton_rtx', '--video', '--video_length', '400'" in playback
    assert "'--extra', 'video', '--extra', 'ovrtx', 'isaaclab', 'play'" in playback
    assert "cwd=project_dir, env=cinematic_env" in playback
    assert "cinematic_env['FANUC_CELL_CINEMATIC'] = '1'" in cinematic

    training = _statement(cinematic, "training = run(",
                          "runtime_report['ppo_training_exit']")
    assert "newton_rtx" not in training and "ovrtx" not in training
    evaluation = _statement(cinematic, "evaluation = run(",
                            "runtime_report['policy_evaluation_exit']")
    assert "'--viz', 'none'" in evaluation
    assert "newton_rtx" not in evaluation and "ovrtx" not in evaluation
    # O token do path tracer aparece uma única vez no notebook inteiro.
    assert cinematic.count("newton_rtx") == 1


def test_cinematic_scene_variant_keeps_physics_and_mdp() -> None:
    """Issue #9 fatia 1: a variante cinematic é camada visual pura.

    - A cena default não ganha NENHUM material (byte a byte a mesma).
    - belt/pedestal/part da variante: mesma geometria/colisão/massa
      do default, só com visual_material a mais.
    - Todo prim novo da variante é decorativo: sem collision_props e
      sem rigid_props (visual-only por construção do spawner).
    - MDP/actions/rewards/events não sabem que a variante existe.
    """
    env_cfg = (OVERLAY / "config/m10id12/env_cfg.py").read_text()

    base = _class_block(env_cfg, "class CellSceneCfg(",
                        "def _cinematic_enabled")
    assert "visual_material" not in base
    assert "CellSceneCfgCinematic" not in base

    assert 'os.environ.get("FANUC_CELL_CINEMATIC", "0") == "1"' in env_cfg
    cine = _class_block(env_cfg, "class CellSceneCfgCinematic(",
                        "@configclass\nclass ActionsCfg")

    for name, size in (("belt", "(0.53, 0.84, 0.11)"),
                       ("place_pedestal", "(0.36, 0.32, 0.04)"),
                       ("part", "(0.08, 0.08, 0.08)")):
        block = _class_block(cine, f"    {name} = ", "\n    )\n")
        assert f"size={size}" in block
        assert "collision_props=sim_utils.CollisionPropertiesCfg()" in block
        assert "visual_material=" in block
    part_block = _class_block(cine, "    part = ", "\n    )\n")
    assert "MassPropertiesCfg(mass=0.35)" in part_block
    assert "RigidBodyPropertiesCfg()" in part_block

    decor = ("belt_top", "belt_rail_left", "belt_rail_right",
             "belt_roller_in", "belt_roller_out",
             "safety_stripe_front", "safety_stripe_back",
             "safety_stripe_left", "safety_stripe_right",
             "fence_panel", "fence_post_left", "fence_post_right",
             "fence_top_rail", "key_light")
    for name in decor:
        block = _class_block(cine, f"    {name} = ", "\n    )\n")
        assert "collision_props" not in block, name
        assert "rigid_props" not in block, name

    # Seleção só no __post_init__ da env cfg, sob a variável de ambiente.
    post_init = env_cfg[env_cfg.index("def __post_init__"):]
    assert "if _cinematic_enabled():" in post_init
    assert "CellSceneCfgCinematic(" in post_init

    for cfg_cls in ("ActionsCfg", "ObservationsCfg", "EventsCfg",
                    "RewardsCfg", "CurriculumCfg", "TerminationsCfg",
                    "CellPhysicsCfg"):
        block = _class_block(env_cfg, f"class {cfg_cls}",
                             "@configclass\nclass ")
        assert "Cinematic" not in block, cfg_cls
    physics = _class_block(env_cfg, "class CellPhysicsCfg", "@configclass")
    assert "default: OvPhysxCfg = ovphysx" in physics
