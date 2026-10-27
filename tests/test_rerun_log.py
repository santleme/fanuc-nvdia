"""Issue #6 (Rerun): módulo de logging de rollouts, no-op e fiação.

Os testes rodam sem rerun-sdk, sem rede e sem display: o SDK é
substituído por um stub que grava as chamadas (o módulo aceita SDK
injetado justamente para isso). Os estruturais seguem o padrão da
suíte: protegem a fiação dos scripts sem precisar do Isaac.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fanuc_cell import (
    RolloutLogger,
    make_rollout_logger,
    rerun_available,
    rerun_log,
)


class _FakeRerun:
    """Stub do rerun-sdk: registra chamadas em vez de gravar .rrd."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def init(self, app_id):
        self.calls.append(("init", app_id))

    def save(self, path):
        self.calls.append(("save", path))

    def disconnect(self):
        self.calls.append(("disconnect",))

    def set_time_sequence(self, name, value):
        self.calls.append(("set_time_sequence", name, value))

    def set_time_seconds(self, name, value):
        self.calls.append(("set_time_seconds", name, value))

    def log(self, path, archetype):
        self.calls.append(("log", path, archetype))

    # Construtores de arquétipos: devolvem tuplas identificáveis.
    def Transform3D(self, **kwargs):
        return ("Transform3D", kwargs)

    def Boxes3D(self, **kwargs):
        return ("Boxes3D", kwargs)

    def Scalars(self, values):
        return ("Scalars", list(values))

    def TextLog(self, text):
        return ("TextLog", text)

    def logged(self, path):
        return [c[2] for c in self.calls if c[0] == "log" and c[1] == path]


def test_noop_without_sdk(monkeypatch, tmp_path) -> None:
    """Sem rerun-sdk, tudo vira no-op silencioso (aceitação da issue)."""
    monkeypatch.setattr(rerun_log, "_rerun_sdk", None)
    assert rerun_available() is False
    logger = make_rollout_logger(True, tmp_path / "rollout.rrd")
    assert logger is not None and logger.active is False
    logger.start_episode(0)
    logger.log_step(step=0, phase="approach", tcp_pos=[0, 0, 0],
                    part_pos=[1, 0, 0], left_force_n=1.0,
                    right_force_n=2.0, gripper_joints=[0.0, 0.0])
    logger.log_event("grasp")
    logger.finish_episode(success=True, grasped=True)
    assert logger.finish() is None
    assert not (tmp_path / "rollout.rrd").exists()


def test_default_off_and_disabled_logger_stays_silent(tmp_path) -> None:
    """Sem a flag, a fábrica devolve None; desabilitado não chama o SDK."""
    assert make_rollout_logger(False, tmp_path / "x.rrd") is None
    fake = _FakeRerun()
    logger = RolloutLogger(tmp_path / "x.rrd", enabled=False, sdk=fake)
    assert logger.active is False
    logger.start_episode(0)
    logger.log_step(step=0, tcp_pos=[0, 0, 0], part_pos=[0, 0, 0])
    logger.finish()
    assert fake.calls == []


def test_functional_logging_with_stub_sdk(tmp_path) -> None:
    fake = _FakeRerun()
    path = tmp_path / "nested" / "rollout.rrd"
    logger = RolloutLogger(path, sdk=fake)
    assert logger.active is True
    assert ("init", "fanuc_cell_rollout") in fake.calls
    assert ("save", str(path)) in fake.calls

    logger.start_episode(3)
    logger.log_step(step=0, time_s=0.0, phase="approach",
                    tcp_pos=[0.0, 0.0, 0.0], part_pos=[0.3, 0.4, 0.0],
                    left_force_n=0.0, right_force_n=0.0,
                    gripper_joints=[0.0, 0.0])
    logger.log_step(step=1, time_s=0.02, phase="approach",
                    tcp_pos=[0.1, 0.0, 0.0], part_pos=[0.3, 0.4, 0.0],
                    left_force_n=1.5, right_force_n=2.5,
                    gripper_joints=[0.005, 0.006])
    logger.log_event("grasp")
    logger.log_step(step=2, time_s=0.04, phase="grasp",
                    tcp_pos=[0.2, 0.0, 0.0], part_pos=[0.3, 0.4, 0.0])
    logger.finish_episode(success=True, grasped=True)
    assert logger.finish() == path
    assert ("disconnect",) in fake.calls

    assert fake.logged("world/tcp") and fake.logged("world/part")
    distances = [a[1][0] for a in fake.logged("tcp_part/distance_m")]
    assert distances[0] == pytest.approx(0.5)
    left = [a[1][0] for a in fake.logged("contact/left_force_n")]
    assert left == [0.0, 1.5]
    gripper = [a[1][0] for a in fake.logged("gripper/right_joint_rad")]
    assert gripper == [0.0, 0.006]
    # Fase só é logada na troca (2 fases em 3 passos).
    phases = [a[1] for a in fake.logged("phase")]
    assert phases == ["approach", "grasp"]
    events = [a[1] for a in fake.logged("events")]
    assert any(e.startswith("episode_start") for e in events)
    assert "grasp" in events
    assert any(e.startswith("episode_end") for e in events)
    steps = [c[2] for c in fake.calls if c[0] == "set_time_sequence"]
    assert steps == [0, 1, 2]


def test_scripts_and_packaging_wiring() -> None:
    """Estrutural: flag nos dois scripts, import protegido, extra opcional."""
    collect = (ROOT / "scripts/collect_demos.py").read_text()
    baseline = (ROOT / "scripts/run_scripted_demo.py").read_text()
    for source in (collect, baseline):
        assert '"--rerun"' in source
        assert '"--rerun-path"' in source
        assert "make_rollout_logger" in source
        assert "log_step" in source
    module = (ROOT / "src/fanuc_cell/rerun_log.py").read_text()
    assert "except ImportError" in module
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert 'dependencies = ["numpy>=2.0"]' in pyproject  # SDK fora do núcleo
    assert 'rerun = ["rerun-sdk>=0.40"]' in pyproject  # só como extra opcional
