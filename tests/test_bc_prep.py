"""Issue #3 (BC warm-start): núcleo puro, portões e fiação do builder.

Os testes de demos/BC rodam com dados sintéticos na VM (teste de
código, não experimento — o treino de verdade é no Kaggle). Os
estruturais seguem o padrão da suíte: protegem a fiação sem Isaac.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fanuc_cell.bc import ActorMLP, train_bc  # noqa: E402
from fanuc_cell.demos import DemoRecorder, DemoSet  # noqa: E402


def _recorder_with_episodes(n_episodes=3, steps=10, obs_dim=4, act_dim=2,
                            success_upto=2):
    rng = np.random.default_rng(0)
    rec = DemoRecorder(metadata={"teacher": "synthetic"})
    for ep in range(n_episodes):
        rec.begin_episode()
        for _ in range(steps):
            rec.record(rng.standard_normal(obs_dim),
                       rng.standard_normal(act_dim))
        rec.end_episode(success=ep < success_upto, grasped=True)
    return rec


def test_demo_recorder_accounting_and_roundtrip(tmp_path) -> None:
    demos = _recorder_with_episodes().to_demoset()
    assert demos.num_steps == 30 and demos.num_episodes == 3
    assert demos.grasp_rate == 1.0
    assert demos.success_rate == pytest.approx(2 / 3)
    path = tmp_path / "demos.npz"
    demos.save_npz(path)
    loaded = DemoSet.load_npz(path)
    assert np.allclose(loaded.observations, demos.observations)
    assert np.allclose(loaded.actions, demos.actions)
    assert loaded.metadata["teacher"] == "synthetic"
    assert loaded.summary()["num_episodes"] == 3


def test_demo_recorder_rejects_misuse() -> None:
    rec = DemoRecorder()
    with pytest.raises(RuntimeError):
        rec.record(np.zeros(2), np.zeros(2))
    rec.begin_episode()
    with pytest.raises(RuntimeError):
        rec.begin_episode()
    with pytest.raises(RuntimeError):
        rec.to_demoset()


def test_bc_reduces_loss_on_synthetic_teacher() -> None:
    rng = np.random.default_rng(0)
    teacher = ActorMLP(6, 3, hidden_dims=(32, 16), seed=1)
    # Saída do professor em escala realista: com o ganho de saída
    # default (0.01) a perda nasce no piso de ruído e o teste não
    # mede aprendizado.
    teacher.weights[-1] *= 30.0
    x = rng.standard_normal((600, 6), dtype=np.float32)
    y = teacher.predict(x) + 0.01 * rng.standard_normal((600, 3),
                                                       dtype=np.float32)
    _, history = train_bc(x, y, hidden_dims=(32, 16), epochs=150,
                          lr=2e-3, batch_size=128, seed=2)
    assert history[-1] < 0.25 * history[0]


def test_actor_export_matches_rsl_rl_architecture() -> None:
    mlp = ActorMLP(24, 8)  # obs da tarefa, ação da tarefa
    shapes = [(256, 24), (128, 256), (64, 128), (8, 64)]
    assert [tuple(w.shape) for w in mlp.weights] == shapes
    arrays = mlp.state_arrays()
    for i in range(4):
        assert f"actor.{2 * i}.weight" in arrays
        assert tuple(arrays[f"actor.{2 * i}.weight"].shape) == shapes[i]
        assert tuple(arrays[f"layer{i}_bias"].shape) == (shapes[i][0],)


def test_pretrain_gate_rejects_weak_demos(tmp_path) -> None:
    weak = _recorder_with_episodes(n_episodes=4, success_upto=1).to_demoset()
    weak_path = tmp_path / "weak.npz"
    weak.save_npz(weak_path)
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "pretrain_bc.py"),
         "--demos", str(weak_path), "--out", str(tmp_path / "a.npz")],
        capture_output=True, text=True,
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 3
    assert "PORTÃO BC" in result.stdout


def test_builder_bc_flag_is_opt_in_and_wired() -> None:
    builder = (ROOT / "scripts/build_kaggle_notebook.py").read_text()
    assert '"--bc-warmstart", action="store_true"' in builder
    assert '"--bc-episodes"' in builder
    # BC v1 é só na linha principal: combinação com newton é erro.
    assert 'parser.error("--bc-warmstart v1 é só na linha principal (ovphysx)")' in builder
    assert "if args.bc_warmstart:" in builder
    assert "str(project_dir / 'scripts' / 'train_ppo_bc.py'), " in builder
    assert "'--bc-weights', '/kaggle/working/bc-actor.npz'" in builder
    # bundle carrega o núcleo e os scripts BC
    for name in ("src/fanuc_cell/demos.py", "src/fanuc_cell/bc.py",
                 "scripts/collect_demos.py", "scripts/pretrain_bc.py",
                 "scripts/train_ppo_bc.py"):
        assert f'"{name}"' in builder


def test_bc_scripts_and_spec_protocol_exist() -> None:
    collect = (ROOT / "scripts/collect_demos.py").read_text()
    train = (ROOT / "scripts/train_ppo_bc.py").read_text()
    spec = (ROOT / "SPEC.md").read_text()
    assert "DemoRecorder" in collect
    assert "BC_WARMSTART_APPLIED" in train
    assert "inject_bc_weights" in train
    assert "Protocolo do experimento BC" in spec
    assert "success ≥ max(0,30; S68 + 0,10)" in spec
    assert "fanuc-bc-warmstart" in spec


def test_collect_demos_disables_training_curriculum() -> None:
    """Pos-v69: com o curriculo ativo, a fase 1 estatica (pick x=0,80)
    diverge o IK do professor e a coleta morre no episodio 1; as demos
    precisam das condicoes historicas da esteira, como o baseline."""
    collector = (ROOT / "scripts/collect_demos.py").read_text()
    assert "cfg.curriculum = None" in collector


def test_pose_ik_fallback_seed_escapes_local_minimum() -> None:
    """Armadilha real da coleta BC v1 (episódio 76/200): o DLS de pose
    trava a partir das juntas de referência do approach (resíduo
    estacionário em 240 ou 5000 iterações), mas o MESMO alvo é
    alcançável da seed zero. O fallback só age onde hoje levanta."""
    from fanuc_cell.kinematics import SerialArm, inverse_kinematics_pose
    from fanuc_cell.task import CellConfig, make_demo_episode

    cfg = CellConfig(
        belt_start_x=0.80, belt_speed=0.05, pick_x=0.95,
        approach_duration=1.8, pick_duration=1.2,
        gripper_close_duration=0.5, transfer_duration=2.8,
        place_duration=0.8, release_duration=0.6,
    )
    episode = make_demo_episode(SerialArm(), cfg)
    seed = episode.sample(1.8)[0]  # juntas do waypoint approach
    target = np.array([0.795, 0.0, 0.37])  # peça medida no início do approach
    with pytest.raises(ValueError):
        inverse_kinematics_pose(SerialArm(), target, np.eye(3), seed=seed)
    solved = inverse_kinematics_pose(
        SerialArm(), target, np.eye(3), seed=seed,
        fallback_seeds=(np.zeros(6),))
    direct = inverse_kinematics_pose(
        SerialArm(), target, np.eye(3), seed=np.zeros(6))
    assert np.allclose(solved, direct)
    pos, _ = SerialArm().forward_pose(solved)
    assert np.linalg.norm(pos - target) <= 0.001


def test_pose_ik_fallback_does_not_change_converged_solutions() -> None:
    """Com a seed primária convergindo, o fallback nunca é consultado:
    o resultado é bit-idêntico ao solve de seed única."""
    from fanuc_cell.kinematics import SerialArm, inverse_kinematics_pose

    target = np.array([0.90, 0.0, 0.37])
    plain = inverse_kinematics_pose(
        SerialArm(), target, np.eye(3), seed=np.zeros(6))
    with_fallback = inverse_kinematics_pose(
        SerialArm(), target, np.eye(3), seed=np.zeros(6),
        fallback_seeds=(np.ones(6) * 0.3,))
    assert np.array_equal(plain, with_fallback)


def test_recorder_failed_episode_counts_without_pairs() -> None:
    """Episódio abortado: fora dos pares do BC, dentro do denominador
    do portão — success_rate mede o professor inteiro."""
    rec = DemoRecorder()
    rec.begin_episode()
    for _ in range(3):
        rec.record(np.zeros(4), np.zeros(2))
    rec.end_episode(success=True, grasped=True)
    rec.begin_episode()
    rec.record(np.ones(4), np.ones(2))
    rec.register_failed_episode()
    demos = rec.to_demoset()
    assert demos.num_episodes == 2
    assert demos.success_rate == 0.5
    assert len(demos.observations) == 3  # só os pares do episódio bom


def test_drivers_use_ik_fallback() -> None:
    for script in ("scripts/run_scripted_demo.py", "scripts/collect_demos.py"):
        assert "fallback_seeds" in (ROOT / script).read_text()
    collector = (ROOT / "scripts/collect_demos.py").read_text()
    assert "register_failed_episode" in collector


def test_builder_bc_v7_guards_scale_and_explicit_checkpoint() -> None:
    # Regressoes da BC v6: (1) build BC sem flags treinou 64x500 em
    # vez dos 256x1500 do protocolo; (2) playback/benchmark com
    # '--checkpoint latest' exigem manifesto que o runner direto nao
    # escreve — a v7 usa o checkpoint explicito do glob da celula.
    builder = (ROOT / "scripts/build_kaggle_notebook.py").read_text()
    assert "--bc-warmstart exige a escala explicita do" in builder
    assert "'--checkpoint', str(checkpoints[-1])," in builder
    train_bc = (ROOT / "scripts/train_ppo_bc.py").read_text()
    assert 'Path("bc-warmstart.json").write_text(' in train_bc


def test_builder_exp1_wiring() -> None:
    # Experimento 1 (issue #8): flag --bc-exp1 exige --bc-warmstart,
    # copia o avaliador do ator puro, roda o Estágio A (seeds 2000+)
    # e liga o Estágio B protegido no comando de treino.
    builder = (ROOT / "scripts/build_kaggle_notebook.py").read_text()
    assert '"--bc-exp1", action="store_true"' in builder
    assert '"--bc-exp1 exige --bc-warmstart"' in builder
    assert '"scripts/eval_bc_actor.py"' in builder
    assert "eval_bc_actor.py')," in builder
    assert "'--seed', '2000'," in builder
    assert "'--protected', " in builder
    assert "'--warmup-iters', '50'" in builder
    eval_script = (ROOT / "scripts/eval_bc_actor.py").read_text()
    assert "ActorMLP.load_npz" in eval_script
    assert "default=2000" in eval_script  # fora das seeds das demos


def test_train_ppo_bc_protected_mode() -> None:
    # Estágio B (issue #8): entropia zerada na cfg e warmup do crítico
    # com o ator congelado via requires_grad (o otimizador do rsl_rl
    # 5.x é único p/ ator+crítico), com learn() em duas chamadas que
    # somam o budget total.
    src = (ROOT / "scripts/train_ppo_bc.py").read_text()
    assert '"--protected", action="store_true"' in src
    assert '"--warmup-iters", type=int, default=50' in src
    assert "agent_cfg.algorithm.entropy_coef = 0.0" in src
    assert "p.requires_grad_(False)" in src
    assert "p.requires_grad_(True)" in src
    assert "BC_PROTECTED_WARMUP_DONE" in src
    assert "total_iters - args.warmup_iters" in src
    assert "runner.current_learning_iteration += 1" in src  # off-by-one do learn() em 2 chamadas
