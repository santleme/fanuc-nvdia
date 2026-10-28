"""Contratos da issue #23: PPO ancorado em BC (receita DAPG/RFT).

Três camadas, nenhuma com treino/simulação (regra do projeto):
(1) funcionais sobre o núcleo puro ``fanuc_cell.anchored_ppo``
(cronogramas e KL com valores conhecidos; buffer de demos
determinístico); (2) estruturais sobre a cola em
``scripts/train_ppo_bc.py`` (flags desligadas por default, embrulho
do update, instalação antes do learn); (3) builder executado de
verdade — notebook default sem nenhum traço da receita, notebook
ancorado com a fiação completa e composição com o OBS-2.
"""

from __future__ import annotations

import ast
import base64
import io
import itertools
import zipfile
from pathlib import Path

import numpy as np
import pytest

from fanuc_cell.anchored_ppo import (
    AnchorSchedule,
    DemoBuffer,
    anchor_weight_at,
    gate_beta,
    gaussian_kl_diag,
)
from fanuc_cell.demos import DemoSet

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/build_kaggle_notebook.py"
TRAIN = ROOT / "scripts/train_ppo_bc.py"
PURE = ROOT / "src/fanuc_cell/anchored_ppo.py"

ANCHOR_TOKENS = ("--anchor-weight", "--freeze-std", "bc_anchor",
                 "BC_ANCHOR", "anchor-demos")


# ------------------------------------------------------------------
# (1) Núcleo puro — cronogramas e contas
# ------------------------------------------------------------------

def test_pure_module_has_no_torch_or_isaac() -> None:
    source = PURE.read_text()
    assert "import torch" not in source
    assert "isaaclab" not in source
    tree = ast.parse(source)
    imports = [node for node in ast.walk(tree)
               if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert imports  # sanidade: o módulo importa numpy/demos


def test_gate_beta_known_values() -> None:
    assert gate_beta(0, 50) == 0.0
    assert gate_beta(50, 50) == pytest.approx(float(np.tanh(1.0)))
    assert gate_beta(10_000, 50) == pytest.approx(1.0)
    # Desligado: T <= 0 não atenua nada.
    assert gate_beta(0, 0) == 1.0
    assert gate_beta(123, -5) == 1.0
    with pytest.raises(ValueError):
        gate_beta(-1, 50)


def test_anchor_weight_linear_decay_endpoints() -> None:
    total = 1450
    assert anchor_weight_at(0, total, 0.02, 0.0) == pytest.approx(0.02)
    assert anchor_weight_at(total - 1, total, 0.02, 0.0) == 0.0
    assert anchor_weight_at(total - 1, total, 0.02, 0.5) \
        == pytest.approx(0.01)
    weights = [anchor_weight_at(t, total, 0.02, 0.0)
               for t in range(0, total, 100)]
    assert all(a >= b for a, b in itertools.pairwise(weights))
    # Corrida de uma iteração só: peso cheio.
    assert anchor_weight_at(0, 1, 0.02, 0.0) == pytest.approx(0.02)
    with pytest.raises(ValueError):
        anchor_weight_at(0, total, -0.1, 0.0)
    with pytest.raises(ValueError):
        anchor_weight_at(0, total, 0.02, 1.5)


def test_gaussian_kl_diag_known_values() -> None:
    zeros = np.zeros((4, 8))
    assert gaussian_kl_diag(zeros, 0.0, zeros, 0.0) == 0.0
    # Deslocamento de 1 em todas as 8 dims, σ=1 dos dois lados:
    # KL/dim = 0,5 → 4,0.
    assert gaussian_kl_diag(zeros + 1.0, 0.0, zeros, 0.0) \
        == pytest.approx(4.0)
    # Mesma média, σ_estudante=1, σ_professor=2:
    # KL/dim = log 2 + 1/8 − 1/2.
    per_dim = float(np.log(2.0) + 0.125 - 0.5)
    assert gaussian_kl_diag(zeros, 0.0, zeros, float(np.log(2.0))) \
        == pytest.approx(8 * per_dim)
    with pytest.raises(ValueError):
        gaussian_kl_diag(zeros, 0.0, np.zeros((4, 7)), 0.0)


def test_anchor_schedule_state_and_validation() -> None:
    schedule = AnchorSchedule(beta0=0.02, mode="kl", gate_warmup=50,
                              total_iterations=1450)
    state0 = schedule.state(0)
    assert state0["anchor_weight"] == pytest.approx(0.02)
    assert state0["gate_beta"] == 0.0
    assert schedule.state(1449)["anchor_weight"] == 0.0
    assert schedule.enabled
    assert not AnchorSchedule(beta0=0.0).enabled
    for kwargs in ({"beta0": -1.0}, {"mode": "ppo"}, {"final_fraction": 2.0},
                   {"gate_warmup": -1}, {"total_iterations": 0},
                   {"teacher_std": 0.0}, {"steps_per_iteration": 0},
                   {"batch_size": 0}):
        with pytest.raises(ValueError):
            AnchorSchedule(**kwargs)


def _demo_set(n: int = 10, obs_dim: int = 4, act_dim: int = 2,
              offset: float = 0.0) -> DemoSet:
    obs = (np.arange(n * obs_dim, dtype=np.float32).reshape(n, obs_dim)
           + offset)
    act = np.zeros((n, act_dim), dtype=np.float32)
    return DemoSet(
        observations=obs, actions=act,
        episode_index=np.zeros(n, dtype=np.int64),
        episode_success=np.array([True]),
        episode_grasped=np.array([True]))


def test_demo_buffer_deterministic_batches() -> None:
    buffer = DemoBuffer(_demo_set(), seed=7)
    obs_a, _ = buffer.batch(3, 4)
    obs_b, _ = buffer.batch(3, 4)
    assert np.array_equal(obs_a, obs_b)
    obs_c, _ = buffer.batch(4, 4)
    assert not np.array_equal(obs_a, obs_c)
    # Batch maior que o buffer: com reposição, sem erro.
    obs_big, act_big = buffer.batch(0, 25)
    assert obs_big.shape == (25, 4) and act_big.shape == (25, 2)


def test_demo_buffer_append_is_the_dagger_hook() -> None:
    buffer = DemoBuffer(_demo_set(n=10), seed=0)
    assert buffer.num_steps == 10 and buffer.num_sets == 1
    buffer.append(_demo_set(n=5, offset=1000.0))
    assert buffer.num_steps == 15 and buffer.num_sets == 2
    with pytest.raises(ValueError):
        buffer.append(_demo_set(n=3, obs_dim=9))
    with pytest.raises(ValueError):
        DemoBuffer(_demo_set(n=0))


def test_package_exports_anchor_names() -> None:
    import fanuc_cell

    for name in ("AnchorSchedule", "DemoBuffer", "anchor_weight_at",
                 "gate_beta", "gaussian_kl_diag"):
        assert name in fanuc_cell.__all__
        assert hasattr(fanuc_cell, name)


# ------------------------------------------------------------------
# (2) Cola no train_ppo_bc.py — estrutural
# ------------------------------------------------------------------

def test_train_script_anchor_flags_default_off() -> None:
    source = TRAIN.read_text()

    assert '"--anchor-weight", type=float, default=0.0' in source
    assert '"--anchor-mode", choices=("kl", "mse")' in source
    assert '"--anchor-final-frac", type=float, default=0.0' in source
    assert '"--gate-warmup", type=int, default=0' in source
    assert '"--freeze-std", action="store_true"' in source
    assert '"--anchor-demos", type=Path, default=None' in source
    assert "ANCHORED = args.anchor_weight > 0" in source
    # Validações altas, antes de qualquer treino.
    assert 'parser.error("--anchor-weight > 0 exige --anchor-demos "' in source
    assert "0.0005" in source  # teto de entropia da receita


def test_train_script_anchor_wiring() -> None:
    source = TRAIN.read_text()

    # Embrulho do update do PPO com gating + passos de âncora.
    assert "def anchored_update():" in source
    assert "orig_update = self.alg.update" in source
    assert "self.alg.update = anchored_update" in source
    assert "param.mul_(beta).add_(pre, alpha=1.0 - beta)" in source
    assert 'self.phase != "joint"' in source
    # Marcadores duráveis (a lição da BC v6: stdout some, JSON fica).
    assert "BC_ANCHOR_APPLIED" in source
    assert "BC_ANCHOR_STEP" in source
    assert "BC_ANCHOR_DONE" in source
    assert 'warmstart_info["anchored"] = anchor_info' in source
    # Instalação/validacao ANTES do primeiro learn().
    install_at = source.index("install_anchor(")
    first_learn = source.index("runner.learn(")
    assert install_at < first_learn
    # Fase conjunta começa explicitamente depois do warmup.
    assert 'anchor_ctl.phase = "joint"' in source
    # Entropia: explícita vence; protegido/ancorado zeram.
    assert "if args.entropy_coef is not None:" in source
    assert "elif args.protected or ANCHORED:" in source
    # O warmup não é mais exclusivo do --protected (a receita
    # ancorada roda sobre ele) e o comportamento protegido puro
    # fica idêntico: warmup_iters só é > 0 com protected/ancorado.
    assert "if args.protected and args.warmup_iters > 0:" not in source
    assert "if warmup_iters > 0:" in source
    assert 'if (args.protected or ANCHORED) else 0' in source


# ------------------------------------------------------------------
# (3) Builder — flags, default limpo, fiação e composição
# ------------------------------------------------------------------

def _load_builder():
    import importlib.util

    spec = importlib.util.spec_from_file_location("builder_anchor",
                                                  BUILDER)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    return builder


def _build(tmp_path, monkeypatch, extra_argv) -> str:
    import json
    import sys

    builder = _load_builder()
    monkeypatch.setattr(builder, "KAGGLE_DIR", tmp_path)
    monkeypatch.setattr(
        sys, "argv",
        ["build_kaggle_notebook.py", "--kaggle-user", "tester",
         *extra_argv])
    builder.main()
    notebook = json.loads((tmp_path / builder.NOTEBOOK_NAME).read_text())
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


BC_SCALE = ["--bc-warmstart", "--train-num-envs", "256",
            "--train-max-iterations", "1500"]


def test_builder_anchor_flags_are_opt_in() -> None:
    builder = BUILDER.read_text()

    assert '"--bc-anchor", action="store_true"' in builder
    assert '"--anchor-weight", type=float, default=0.02' in builder
    assert '"--anchor-mode", choices=("kl", "mse")' in builder
    assert '"--gate-warmup", type=int, default=50' in builder
    assert 'parser.error("--bc-anchor exige --bc-warmstart")' in builder
    assert 'parser.error("--bc-anchor exige --bc-exp1' in builder


def test_builder_default_notebook_has_no_anchor_traces(
        tmp_path, monkeypatch) -> None:
    source = _build(tmp_path, monkeypatch, [])
    for token in ANCHOR_TOKENS:
        assert token not in source


def test_builder_bc_notebook_without_anchor_has_no_traces(
        tmp_path, monkeypatch) -> None:
    source = _build(tmp_path, monkeypatch, [*BC_SCALE, "--bc-exp1"])
    assert "'--protected', '--warmup-iters', '50'" in source
    for token in ANCHOR_TOKENS:
        assert token not in source


def test_builder_anchor_notebook_wiring(tmp_path, monkeypatch) -> None:
    source = _build(tmp_path, monkeypatch,
                    [*BC_SCALE, "--bc-exp1", "--bc-anchor"])

    # Sobre o Estágio B protegido, sem substituí-lo.
    assert "'--protected', '--warmup-iters', '50'," in source
    assert "'--anchor-weight', '0.02'" in source
    assert "'--anchor-mode', 'kl'" in source
    assert "'--anchor-final-frac', '0.0'" in source
    assert "'--gate-warmup', '50'" in source
    # A âncora amostra das demos DESTA corrida.
    assert "'--anchor-demos', '/kaggle/working/bc-demos.npz'" in source
    assert "'--freeze-std'" in source
    # Configuração durável no runtime-report.
    assert "runtime_report['bc_anchor'] = {" in source
    assert "'weight': 0.02" in source
    assert "'gate_warmup': 50" in source


def test_builder_anchor_composes_with_obs_history(
        tmp_path, monkeypatch) -> None:
    source = _build(tmp_path, monkeypatch,
                    [*BC_SCALE, "--bc-exp1", "--bc-anchor",
                     "--obs-history", "8"])
    assert "os.environ['FANUC_CELL_OBS_HISTORY'] = '8'" in source
    assert "runtime_report['bc_anchor'] = {" in source
    assert "'--anchor-weight', '0.02'" in source


def test_builder_anchor_custom_values_render(
        tmp_path, monkeypatch) -> None:
    source = _build(tmp_path, monkeypatch,
                    [*BC_SCALE, "--bc-exp1", "--bc-anchor",
                     "--anchor-weight", "0.01", "--anchor-mode", "mse",
                     "--anchor-final-frac", "0.25", "--gate-warmup", "100"])
    assert "'--anchor-weight', '0.01'" in source
    assert "'--anchor-mode', 'mse'" in source
    assert "'--anchor-final-frac', '0.25'" in source
    assert "'--gate-warmup', '100'" in source
    assert "'weight': 0.01" in source


def test_builder_anchor_requires_warmstart_and_exp1(
        tmp_path, monkeypatch) -> None:
    with pytest.raises(SystemExit):
        _build(tmp_path / "a", monkeypatch, ["--bc-anchor"])
    with pytest.raises(SystemExit):
        _build(tmp_path / "b", monkeypatch, [*BC_SCALE, "--bc-anchor"])
    with pytest.raises(SystemExit):
        _build(tmp_path / "c", monkeypatch,
               [*BC_SCALE, "--bc-exp1", "--bc-anchor",
                "--anchor-weight", "0"])


def test_builder_bundle_includes_anchor_module_and_tests() -> None:
    builder = _load_builder()
    archive = zipfile.ZipFile(
        io.BytesIO(base64.b64decode(builder.bundled_project())))
    names = set(archive.namelist())
    assert "src/fanuc_cell/anchored_ppo.py" in names
    assert "tests/test_anchored_ppo.py" in names
