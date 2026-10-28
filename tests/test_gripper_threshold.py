"""Probe do limiar da garra (issue #13, B6): análise pura + fiação.

A geometria das amostras sintéticas é conhecida por construção:
pinch sustentado num valor dado, perda de contato noutro, platô
vazio em 0,0120 m — a recomendação da regra pré-registrada tem que
sair exatamente a esperada para cada regime (PROVA/PARCIAL/MATA/
SEM_SINAL). Nenhuma simulação roda aqui: a medição é do kernel
``fanuc-probe-gripper-threshold`` no Kaggle.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import zipfile
from pathlib import Path

from fanuc_cell.gripper_threshold import (
    analyze_threshold_samples,
    recommend_threshold,
    sustained_contact_mask,
    sweep_thresholds,
)

ROOT = Path(__file__).resolve().parents[1]


def _sample(rep, scenario, phase, joint_left, joint_right,
            force_left=0.0, force_right=0.0):
    return {
        "rep": rep,
        "scenario": scenario,
        "phase": phase,
        "joint_left": joint_left,
        "joint_right": joint_right,
        "force_left_n": force_left,
        "force_right_n": force_right,
    }


def _pinch_rep(rep: int, pinch_joint: float, loss_max_joint: float) -> list:
    samples = [_sample(rep, "pinch", "approach", 0.0, 0.0) for _ in range(3)]
    samples.append(_sample(rep, "pinch", "grasp", 0.0020, 0.0021))
    # Travel dentro da banda ambígua atual, ainda sem contato.
    samples.append(_sample(rep, "pinch", "grasp", 0.0045, 0.0046))
    for _ in range(10):  # pinch sustentado
        samples.append(_sample(rep, "pinch", "hold",
                               pinch_joint, pinch_joint + 0.0002, 1.5, 1.7))
    for _ in range(3):  # lift, ainda segurando
        samples.append(_sample(rep, "pinch", "lift",
                               pinch_joint, pinch_joint + 0.0002, 1.4, 1.6))
    samples.append(_sample(rep, "pinch", "release",
                           pinch_joint - 0.0001, pinch_joint + 0.0001, 1.2, 1.3))
    # Último passo com contato: é a perda de contato medida.
    samples.append(_sample(rep, "pinch", "release",
                           loss_max_joint - 0.0002, loss_max_joint, 0.8, 0.9))
    for i in range(12):  # abertura confirmada, sem contato
        samples.append(_sample(rep, "pinch", "retreat",
                               0.0040 - 0.0002 * i, 0.0042 - 0.0002 * i))
    return samples


def _empty_samples() -> list:
    samples = []
    for i in range(10):
        value = 0.0012 * (i + 1)
        samples.append(_sample(0, "empty", "empty_close", value, value + 0.0001))
    for _ in range(10):  # platô do fechamento vazio
        samples.append(_sample(0, "empty", "empty_close", 0.0120, 0.0121))
    for _ in range(5):
        samples.append(_sample(0, "empty", "empty_open", 0.0060, 0.0061))
    return samples


def test_sustained_mask_requires_a_run_of_five() -> None:
    short = [_sample(0, "pinch", "hold", 0.006, 0.006, 1.0, 1.0) for _ in range(3)]
    long = [_sample(0, "pinch", "hold", 0.006, 0.006, 1.0, 1.0) for _ in range(5)]
    assert sustained_contact_mask(short) == [False, False, False]
    assert sustained_contact_mask(long) == [True] * 5


def test_sweep_rates_against_measured_contact() -> None:
    samples = [
        _sample(0, "pinch", "hold", 0.006, 0.006, 1.0, 1.0),      # contato
        _sample(0, "pinch", "grasp", 0.0045, 0.0046),             # banda, sem contato
        _sample(0, "empty", "empty_open", 0.001, 0.001),          # sem contato
        _sample(0, "pinch", "hold", 0.003, 0.003, 1.0, 1.0),      # contato abaixo de 0,004
    ]
    rows = {row["threshold_m"]: row for row in sweep_thresholds(samples)}
    at_0004 = rows[0.004]
    assert at_0004["false_positive_rate"] == 0.5   # a banda conta como fechada
    assert at_0004["false_negative_rate"] == 0.5   # o pinch em 0,003 escapa
    at_0005 = rows[0.005]
    assert at_0005["false_positive_rate"] == 0.0
    assert at_0005["false_negative_rate"] == 0.5


def test_prova_when_measured_geometry_clears_the_band() -> None:
    samples = _pinch_rep(0, pinch_joint=0.0060, loss_max_joint=0.0057)
    samples += _empty_samples()
    report = analyze_threshold_samples(samples)

    assert report["verdict"] == "PROVA"
    recommendation = report["recommendation"]
    assert recommendation["pinch_p05_m"] == 0.0060
    assert recommendation["contact_loss_p05_m"] == 0.0057
    assert recommendation["admissible_max_m"] == 0.0057
    # Maior ponto da grade (passo 0,00025) sob o teto admissível.
    assert recommendation["threshold_m"] == 0.0055
    assert recommendation["current_closed_ok"] is True
    # Abertura em 0,005 só dispara depois que a junta cai abaixo dele —
    # e o contato já se perdeu em 0,0057: o literal atual é seguro
    # NESTA geometria; quem a medição condena é a coexistência dos dois.
    assert recommendation["current_open_ok"] is True
    assert report["empty_close_stall_min_joint_m"] == 0.0120


def test_band_occupancy_counts_simultaneous_closed_and_open() -> None:
    samples = _pinch_rep(0, pinch_joint=0.0060, loss_max_joint=0.0057)
    report = analyze_threshold_samples(samples)
    occupancy = report["current_band_occupancy"]
    # 1 passo de travel em grasp + 1 de retreat têm min >= 0,004 e
    # max < 0,005; nenhum passo COM contato cai na banda.
    assert occupancy["all_steps"] == 2 / 32
    assert occupancy["contact_steps"] == 0.0
    assert occupancy["no_contact_steps"] == 2 / 17


def test_parcial_keeps_0004_and_aligns_only_the_open_literal() -> None:
    samples = _pinch_rep(0, pinch_joint=0.0043, loss_max_joint=0.0042)
    report = analyze_threshold_samples(samples)
    assert report["verdict"] == "PARCIAL"
    assert report["recommendation"]["threshold_m"] == 0.004


def test_mata_when_even_the_code_value_misses_real_pinches() -> None:
    samples = _pinch_rep(0, pinch_joint=0.0035, loss_max_joint=0.0034)
    report = analyze_threshold_samples(samples)
    assert report["verdict"] == "MATA"
    assert report["recommendation"]["threshold_m"] < 0.004


def test_sem_sinal_when_no_contact_was_measured() -> None:
    samples = _empty_samples()
    report = analyze_threshold_samples(samples)
    assert report["verdict"] == "SEM_SINAL"
    assert report["recommendation"]["threshold_m"] is None


def test_recommend_threshold_boundaries() -> None:
    assert recommend_threshold(None, 0.0057)["verdict"] == "SEM_SINAL"
    assert recommend_threshold(0.0060, None)["verdict"] == "SEM_SINAL"
    assert recommend_threshold(0.0060, 0.0057)["threshold_m"] == 0.0055
    assert recommend_threshold(0.0010, 0.0010)["verdict"] == "MATA"


def test_analyze_rejects_empty_or_malformed_samples() -> None:
    for bad in ([], [{"scenario": "pinch"}]):
        try:
            analyze_threshold_samples(bad)
        except ValueError:
            continue
        raise AssertionError(f"deveria rejeitar: {bad!r}")


def _load_builder(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _zip_names(builder) -> set:
    archive = zipfile.ZipFile(io.BytesIO(base64.b64decode(builder.bundled_project())))
    return set(archive.namelist())


def test_probe_wiring_and_bundles() -> None:
    # Este teste viaja nos bundles dos dois kernels; dentro de um
    # bundle extraído, os arquivos que só existem no repo (script do
    # probe no bundle principal, builders cruzados, artefatos gerados
    # kaggle-probe/) não estão lá — essas checagens pulam nesse
    # contexto e rodam inteiras no repo. A regressão do manifesto
    # (módulo no zip do kernel principal) roda sempre que o builder
    # principal está presente. (Correção 28/10/2026: a suíte local do
    # kernel eval-only morreu nestas leituras — 14 falhas FileNotFound.)
    probe_script = ROOT / "scripts/probe_gripper_threshold.py"
    if probe_script.exists():
        probe = probe_script.read_text()
        assert "VERDICT_JSON:" in probe
        assert "analyze_threshold_samples" in probe
        assert 'os.environ.get("FANUC_CELL_PHYSICS", "ovphysx")' in probe
        assert '"pinch"' in probe and '"empty"' in probe
        assert "teacher_gripper_action" in probe  # mesmo fechamento do baseline

    # O zip do kernel principal carrega o módulo novo (regressão da
    # BC v5: módulo fora do manifesto = kernel morto no import).
    if (ROOT / "scripts/build_kaggle_notebook.py").exists():
        main_names = _zip_names(_load_builder("build_kaggle_notebook"))
        assert "src/fanuc_cell/gripper_threshold.py" in main_names

    # O kernel do probe é separado e carrega script + módulo + teste.
    probe_meta = ROOT / "kaggle-probe/kernel-metadata.json"
    if (ROOT / "scripts/build_probe_notebook.py").exists() \
            and probe_meta.exists():
        probe_names = _zip_names(_load_builder("build_probe_notebook"))
        assert "scripts/probe_gripper_threshold.py" in probe_names
        assert "src/fanuc_cell/gripper_threshold.py" in probe_names
        assert "tests/test_gripper_threshold.py" in probe_names

        metadata = json.loads(probe_meta.read_text())
        assert metadata["id"].endswith("/fanuc-probe-gripper-threshold")
        assert metadata["enable_gpu"] is True
        notebook = json.loads(
            (ROOT / "kaggle-probe/fanuc-probe-gripper-threshold.ipynb").read_text())
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        assert "probe_gripper_threshold.py" in source
        assert "VERDICT_JSON:" in source


def test_protocol_is_preregistered_in_spec() -> None:
    spec = (ROOT / "SPEC.md").read_text()
    assert "fanuc-gripper-threshold-b6" in spec
    assert "Protocolo do probe do limiar da garra (B6, issue #13" in spec
