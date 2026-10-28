"""Contratos do Experimento OBS-2 (issue #8): histórico de observações.

O gate FANUC_CELL_OBS_HISTORY empilha frames na observação da
política pelo mecanismo nativo do Isaac Lab (history_length do
grupo). Estes testes rodam sem Isaac Lab: estruturais sobre o
env_cfg/builder, funcionais sobre o helper extraído por AST e sobre
o builder executado de verdade (default byte a byte sem a flag).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "isaaclab_overlay/src/fanuc_cell_lab/tasks/conveyor_pick_place"
ENV_CFG = OVERLAY / "config/m10id12/env_cfg.py"
BUILDER = ROOT / "scripts/build_kaggle_notebook.py"


def _env_cfg_source() -> str:
    return ENV_CFG.read_text()


def _policy_post_init_block(source: str) -> str:
    start = source.index("class PolicyCfg(ObservationGroup):")
    end = source.index("policy: PolicyCfg = PolicyCfg()", start)
    return source[start:end]


def test_env_cfg_obs_history_gate_is_opt_in_and_native() -> None:
    source = _env_cfg_source()

    assert "def _obs_history_length() -> int:" in source
    assert 'os.environ.get("FANUC_CELL_OBS_HISTORY", "1")' in source

    block = _policy_post_init_block(source)
    # Default (1) não toca nenhum atributo de história: history_length
    # segue None no cfg e o ObservationManager não cria buffers.
    assert "if history > 1:" in block
    assert "self.history_length = history" in block
    assert "self.flatten_history_dim = True" in block
    # Os 5 termos e a concatenação seguem os mesmos de sempre.
    assert "self.concatenate_terms = True" in block
    assert block.count("ObservationTerm(") == 5


def _load_history_helper(tmp_path):
    """Importa SO o helper do env_cfg (sem importar o Isaac Lab).

    Extrai o fonte da função por AST, grava num módulo temporário e
    importa — o módulo só depende de ``os``.
    """
    import importlib.util

    tree = ast.parse(_env_cfg_source())
    func = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_obs_history_length"
    )
    module_path = tmp_path / "obs_history_helper.py"
    module_path.write_text(
        "import os\n\n\n" + ast.unparse(func) + "\n")
    spec = importlib.util.spec_from_file_location(
        "obs_history_helper", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._obs_history_length


def test_obs_history_helper_defaults_to_single_frame(
        tmp_path, monkeypatch) -> None:
    helper = _load_history_helper(tmp_path)
    monkeypatch.delenv("FANUC_CELL_OBS_HISTORY", raising=False)
    assert helper() == 1
    monkeypatch.setenv("FANUC_CELL_OBS_HISTORY", "1")
    assert helper() == 1


def test_obs_history_helper_parses_and_validates(
        tmp_path, monkeypatch) -> None:
    helper = _load_history_helper(tmp_path)
    monkeypatch.setenv("FANUC_CELL_OBS_HISTORY", "8")
    assert helper() == 8
    monkeypatch.setenv("FANUC_CELL_OBS_HISTORY", "0")
    with pytest.raises(ValueError):
        helper()
    monkeypatch.setenv("FANUC_CELL_OBS_HISTORY", "abc")
    with pytest.raises(ValueError):
        helper()


def _build_notebook_source(tmp_path, monkeypatch, extra_argv) -> str:
    """Roda o builder de verdade (saida em tmp) e devolve o fonte."""
    import importlib.util
    import json
    import sys

    spec = importlib.util.spec_from_file_location("builder_obs_history",
                                                  BUILDER)
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


def test_builder_obs_history_flag_is_opt_in() -> None:
    builder = BUILDER.read_text()

    assert '"--obs-history", type=int, default=1' in builder
    assert 'parser.error("--obs-history deve ser >= 1' in builder
    assert "if args.obs_history > 1:" in builder
    # A variável precisa chegar a TODOS os subprocessos que criam a
    # tarefa: processo do notebook (herdado pelo isaaclab train) e
    # baseline_env (coleta, eval do BC, baseline).
    assert "os.environ['FANUC_CELL_OBS_HISTORY']" in builder
    assert "baseline_env['FANUC_CELL_OBS_HISTORY']" in builder
    assert "runtime_report['obs_history']" in builder


def test_builder_default_notebook_has_no_obs_history_traces(
        tmp_path, monkeypatch) -> None:
    source = _build_notebook_source(tmp_path, monkeypatch, [])

    assert "FANUC_CELL_OBS_HISTORY" not in source
    assert "obs_history" not in source


def test_builder_obs_history_injects_env_everywhere(
        tmp_path, monkeypatch) -> None:
    source = _build_notebook_source(
        tmp_path, monkeypatch, ["--obs-history", "8"])

    assert "os.environ['FANUC_CELL_OBS_HISTORY'] = '8'" in source
    assert "baseline_env['FANUC_CELL_OBS_HISTORY'] = '8'" in source
    assert "runtime_report['obs_history'] = 8" in source
    # Injeção única: a âncora do baseline_env aparece uma vez só.
    assert source.count("FANUC_CELL_OBS_HISTORY") == 2
