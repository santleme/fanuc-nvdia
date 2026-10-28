"""Modo eval-only do notebook (issue #11, B2): testes estruturais.

Padrão da casa: o builder roda de verdade (saída em tmp) e as
afirmações são sobre o fonte do notebook gerado e o metadata —
nenhum treino/simulação roda aqui.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

CKPT_ARG = "fanuc-checkpoint-teste/model_1499.pt"


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "builder_eval", ROOT / "scripts/build_kaggle_notebook.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    return builder


def _build(tmp_path, monkeypatch, extra_argv):
    builder = _load_builder()
    monkeypatch.setattr(builder, "KAGGLE_DIR", tmp_path)
    monkeypatch.setattr(
        sys, "argv",
        ["build_kaggle_notebook.py", "--kaggle-user", "tester",
         *extra_argv])
    builder.main()
    notebook = json.loads((tmp_path / builder.NOTEBOOK_NAME).read_text())
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    metadata = json.loads((tmp_path / "kernel-metadata.json").read_text())
    return source, metadata


def _statement(source: str, start_marker: str, end_marker: str) -> str:
    return source[source.index(start_marker):source.index(end_marker)]


def test_builder_eval_only_flag_is_opt_in() -> None:
    builder = (ROOT / "scripts/build_kaggle_notebook.py").read_text()

    assert '"--eval-only", action="store_true"' in builder
    assert '"--eval-checkpoint", default=None' in builder
    assert "if args.eval_only:" in builder
    assert "dataset_sources" in builder


def test_builder_default_notebook_has_no_eval_traces(tmp_path, monkeypatch):
    source, metadata = _build(tmp_path, monkeypatch, [])

    assert "eval_checkpoint" not in source
    assert "training = run(" in source
    assert source.count("'--checkpoint', 'latest',") == 2
    assert "dataset_sources" not in metadata


def test_builder_eval_only_skips_training_and_uses_explicit_checkpoint(
        tmp_path, monkeypatch):
    source, metadata = _build(
        tmp_path, monkeypatch,
        ["--kernel-slug", "fanuc-eval-only", "--eval-only",
         "--eval-checkpoint", CKPT_ARG])

    # Sem treino: o comando e os portões dele sumiram.
    assert "isaaclab', 'train'" not in source
    assert "training = run(" not in source
    assert "RSL-RL PPO training failed" not in source
    assert "ppo_training_stdout" not in source
    # Checkpoint explícito do dataset, resolvido com portão de existência.
    assert f"Path('/kaggle/input') / '{CKPT_ARG}'" in source
    assert "_eval_ckpt_path.is_file()" in source
    assert "runtime_report['eval_only'] = True" in source
    # Playback e benchmark usam a variável — 'latest' não sobra em
    # nenhum comando de checkpoint.
    assert "'--checkpoint', 'latest'" not in source
    assert source.count("'--checkpoint', eval_checkpoint,") == 2
    playback = _statement(source, "playback = run(",
                          "runtime_report['policy_playback_exit']")
    assert "'--checkpoint', eval_checkpoint," in playback
    evaluation = _statement(source, "evaluation = run(",
                            "runtime_report['policy_evaluation_exit']")
    assert "'--checkpoint', eval_checkpoint," in evaluation
    # Sanidade mantida: probe de NaN e baseline scriptado intactos.
    assert "ppo_nan_probe_exit" in source
    assert "scripted_baseline_exit" in source
    # O dataset do checkpoint é a fonte do kernel.
    assert metadata["dataset_sources"] == ["tester/fanuc-checkpoint-teste"]
    assert metadata["id"] == "tester/fanuc-eval-only"


def test_builder_eval_only_composes_with_cinematic(tmp_path, monkeypatch):
    """O caso da issue #9: vídeo cinematic de checkpoint já treinado."""
    source, _ = _build(
        tmp_path, monkeypatch,
        ["--eval-only", "--eval-checkpoint", CKPT_ARG, "--cinematic-video"])

    playback = _statement(source, "playback = run(",
                          "runtime_report['policy_playback_exit']")
    assert "'--checkpoint', eval_checkpoint," in playback
    assert "'--viz', 'newton_rtx'" in playback
    assert "cinematic_env['FANUC_CELL_CINEMATIC'] = '1'" in source


@pytest.mark.parametrize("argv", [
    ["--eval-only"],                                        # sem checkpoint
    ["--eval-checkpoint", CKPT_ARG],                        # sem a flag
    ["--eval-only", "--eval-checkpoint", "latest"],         # latest puro
    ["--eval-only", "--eval-checkpoint", "ds/latest"],      # latest de arquivo
    ["--eval-only", "--eval-checkpoint", "/abs/model.pt"],  # caminho absoluto
    ["--eval-only", "--eval-checkpoint", "model_1499.pt"],  # sem dataset
    ["--eval-only", "--eval-checkpoint", "ds/model.bin"],   # não é .pt
    ["--eval-only", "--eval-checkpoint", CKPT_ARG,
     "--bc-warmstart", "--train-num-envs", "256",
     "--train-max-iterations", "1500"],                     # incompatível c/ BC
])
def test_builder_eval_only_rejects_bad_invocations(tmp_path, monkeypatch,
                                                   argv):
    builder = _load_builder()
    monkeypatch.setattr(builder, "KAGGLE_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["build_kaggle_notebook.py", *argv])
    with pytest.raises(SystemExit):
        builder.main()


def test_push_rewrites_dataset_source_owner(tmp_path, monkeypatch):
    """--usuario do push troca o dono das dataset_sources, como no id."""
    spec = importlib.util.spec_from_file_location(
        "kaggle_push", ROOT / "scripts/kaggle_push_kernel.py")
    push = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(push)
    (tmp_path / "kernel-metadata.json").write_text(json.dumps({
        "id": "INSERT_KAGGLE_USERNAME/fanuc-eval-only",
        "code_file": "nb.ipynb",
        "dataset_sources": ["INSERT_KAGGLE_USERNAME/fanuc-checkpoint-teste"],
    }))
    (tmp_path / "nb.ipynb").write_text("{}")
    monkeypatch.setattr(push, "NOTEBOOKS", tmp_path)

    meta, _ = push.load_inputs("pascalcoimbra")
    assert meta["id"] == "pascalcoimbra/fanuc-eval-only"
    assert meta["dataset_sources"] == ["pascalcoimbra/fanuc-checkpoint-teste"]
    payload = push.build_payload(meta, "{}")
    assert payload["datasetDataSources"] == [
        "pascalcoimbra/fanuc-checkpoint-teste"]


def _load_packager():
    spec = importlib.util.spec_from_file_location(
        "packager", ROOT / "scripts/package_checkpoint_dataset.py")
    packager = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(packager)
    return packager


def test_package_checkpoint_dataset(tmp_path):
    packager = _load_packager()
    ckpt = tmp_path / "model_1499.pt"
    ckpt.write_bytes(b"fake-weights" * 128)
    out = tmp_path / "dataset-dir"

    manifest = packager.package(
        ckpt, out, kaggle_user="pascalcoimbra",
        dataset="fanuc-checkpoint-teste", title=None,
        source_note="corrida de teste")

    assert (out / "model_1499.pt").read_bytes() == ckpt.read_bytes()
    assert manifest["checkpoint_file"] == "model_1499.pt"
    assert manifest["bytes"] == ckpt.stat().st_size
    assert len(manifest["sha256"]) == 64
    assert manifest["eval_checkpoint_arg"] == \
        "fanuc-checkpoint-teste/model_1499.pt"
    dataset_meta = json.loads((out / "dataset-metadata.json").read_text())
    assert dataset_meta["id"] == "pascalcoimbra/fanuc-checkpoint-teste"
    on_disk = json.loads((out / "checkpoint-manifest.json").read_text())
    assert on_disk["sha256"] == manifest["sha256"]


def test_package_checkpoint_rejects_bad_inputs(tmp_path):
    packager = _load_packager()
    with pytest.raises(SystemExit):
        packager.package(tmp_path / "missing.pt", tmp_path / "o",
                         kaggle_user="u", dataset="d", title=None,
                         source_note=None)
    not_pt = tmp_path / "model.bin"
    not_pt.write_bytes(b"x")
    with pytest.raises(SystemExit):
        packager.package(not_pt, tmp_path / "o2", kaggle_user="u",
                         dataset="d", title=None, source_note=None)
    full = tmp_path / "o3"
    full.mkdir()
    (full / "keep.txt").write_text("x")
    ckpt = tmp_path / "model_1.pt"
    ckpt.write_bytes(b"y")
    with pytest.raises(SystemExit):
        packager.package(ckpt, full, kaggle_user="u", dataset="d",
                         title=None, source_note=None)
