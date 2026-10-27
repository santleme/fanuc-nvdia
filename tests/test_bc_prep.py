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
