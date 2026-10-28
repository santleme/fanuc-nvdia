"""Spike B5 (issue #14): probe+baseline num único boot do Isaac Sim.

Testes estruturais e do núcleo puro de orquestração — sem Isaac
instalado e sem simulação. Quem prova a reentrada do
``launch_simulation`` é a corrida do spike no Kaggle; aqui provamos
que os estágios são invocados intactos (mesmo arquivo, como
``__main__``, com o argv certo), que os vereditos são combinados
pelo critério pré-registrado e que o builder só funde os comandos
sob a flag opt-in.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_spike():
    spec = importlib.util.spec_from_file_location(
        "probe_and_baseline", ROOT / "scripts/probe_and_baseline.py")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve o modulo via sys.modules durante o exec.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_stage(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body)
    return path


def test_run_stage_captures_system_exit_and_forwards_argv(tmp_path) -> None:
    spike = _load_spike()
    stage = _fake_stage(
        tmp_path, "fake_probe.py",
        "import json, sys\n"
        "from pathlib import Path\n"
        "Path(sys.argv[1]).write_text(json.dumps(sys.argv[2:]))\n"
        "raise SystemExit(3)\n")
    marker = tmp_path / "argv.json"

    result = spike.run_stage("probe", stage, [str(marker), "--seed", "7"])

    assert result.exit_code == 3
    assert result.error is None
    assert json.loads(marker.read_text()) == ["--seed", "7"]


def test_run_stage_success_string_exit_and_exception(tmp_path) -> None:
    spike = _load_spike()

    ok = _fake_stage(tmp_path, "ok.py", "print('fine')\n")
    assert spike.run_stage("baseline", ok, []).exit_code == 0

    msg = _fake_stage(tmp_path, "msg.py", "raise SystemExit('boom')\n")
    assert spike.run_stage("baseline", msg, []).exit_code == 1

    crash = _fake_stage(tmp_path, "crash.py",
                        "raise RuntimeError('reentry broke')\n")
    crashed = spike.run_stage("baseline", crash, [])
    assert crashed.exit_code is None
    assert "RuntimeError: reentry broke" in crashed.error


def test_run_stage_restores_sys_argv(tmp_path) -> None:
    spike = _load_spike()
    stage = _fake_stage(tmp_path, "any.py", "raise SystemExit(0)\n")
    before = list(sys.argv)
    spike.run_stage("probe", stage, ["--x"])
    assert sys.argv == before


def test_stage_argv_matches_the_separate_invocations() -> None:
    """Os argv do boot único são os das invocações separadas do kernel:
    probe --device cuda:0 --num-envs 64 --steps 24 --seed 7; baseline
    --device cuda:0 --seed 23 --output <json>."""
    spike = _load_spike()

    assert spike.probe_stage_argv("cuda:0", 64, 24, 7) == [
        "--device", "cuda:0", "--num-envs", "64",
        "--steps", "24", "--seed", "7"]
    assert spike.baseline_stage_argv(
        "cuda:0", 23, Path("/kaggle/working/isaaclab-baseline.json")) == [
        "--device", "cuda:0", "--seed", "23",
        "--output", "/kaggle/working/isaaclab-baseline.json"]
    assert spike.baseline_stage_argv(
        "cuda:0", 23, Path("out.json"), rerun=True,
        rerun_path=Path("out.rrd"))[-3:] == [
        "--rerun", "--rerun-path", "out.rrd"]


def _stage(spike, name, exit_code=0, error=None):
    return spike.StageResult(name=name, script=f"{name}.py", argv=[],
                             exit_code=exit_code, error=error)


def test_evaluate_spike_adopt_and_reject_reasons() -> None:
    spike = _load_spike()
    good_report = {"success": True, "scripted_grasp_sensor_passed": True}

    verdict = spike.evaluate_spike(
        _stage(spike, "probe"), _stage(spike, "baseline"), good_report)
    assert verdict.verdict == "adopt" and verdict.reasons == []

    verdict = spike.evaluate_spike(
        _stage(spike, "probe", exit_code=1), _stage(spike, "baseline"),
        good_report)
    assert verdict.verdict == "reject"
    assert any("probe" in reason for reason in verdict.reasons)

    verdict = spike.evaluate_spike(
        _stage(spike, "probe"),
        _stage(spike, "baseline", exit_code=None,
               error="Traceback...\nRuntimeError: launch again"),
        good_report)
    assert verdict.verdict == "reject"
    assert any("reentrada" in reason for reason in verdict.reasons)

    verdict = spike.evaluate_spike(
        _stage(spike, "probe"), _stage(spike, "baseline"), None)
    assert verdict.verdict == "reject"
    assert any("não produziu" in reason for reason in verdict.reasons)

    verdict = spike.evaluate_spike(
        _stage(spike, "probe"), _stage(spike, "baseline"),
        {"success": False, "scripted_grasp_sensor_passed": True})
    assert verdict.verdict == "reject"
    assert any("success=False" in reason for reason in verdict.reasons)


def _write_fake_stages(scripts_dir: Path, probe_body: str,
                       baseline_body: str) -> None:
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / "diagnose_training_nan.py").write_text(probe_body)
    (scripts_dir / "run_scripted_demo.py").write_text(baseline_body)


_BASELINE_OK = (
    "import json, sys\n"
    "from pathlib import Path\n"
    "out = Path(sys.argv[sys.argv.index('--output') + 1])\n"
    "out.write_text(json.dumps({'success': True,\n"
    "    'scripted_grasp_sensor_passed': True}))\n")


def test_main_end_to_end_adopt(tmp_path) -> None:
    spike = _load_spike()
    scripts_dir = tmp_path / "scripts"
    _write_fake_stages(scripts_dir,
                       "print('{\"finite\": true}')\n", _BASELINE_OK)
    baseline_out = tmp_path / "baseline.json"
    report_path = tmp_path / "spike.json"

    code = spike.main(["--baseline-output", str(baseline_out),
                       "--report", str(report_path)],
                      scripts_dir=scripts_dir)

    assert code == 0
    report = json.loads(report_path.read_text())
    assert report["verdict"] == "adopt"
    assert report["probe_exit_code"] == 0
    assert report["baseline_exit_code"] == 0
    assert report["baseline_report"]["success"] is True
    assert report["probe"]["argv"] == [
        "--device", "cuda:0", "--num-envs", "64",
        "--steps", "24", "--seed", "7"]


def test_main_end_to_end_probe_fails_but_baseline_still_runs(
        tmp_path) -> None:
    """Mesmo com o probe reprovado, o baseline roda no mesmo processo:
    a reentrada é o objeto do spike, não o veredito do probe."""
    spike = _load_spike()
    scripts_dir = tmp_path / "scripts"
    _write_fake_stages(scripts_dir, "raise SystemExit(1)\n", _BASELINE_OK)
    baseline_out = tmp_path / "baseline.json"
    report_path = tmp_path / "spike.json"

    code = spike.main(["--baseline-output", str(baseline_out),
                       "--report", str(report_path)],
                      scripts_dir=scripts_dir)

    assert code == 1
    report = json.loads(report_path.read_text())
    assert report["verdict"] == "reject"
    assert report["probe_exit_code"] == 1
    assert report["baseline_exit_code"] == 0  # rodou: reentrada provada


def test_combined_script_never_imports_isaac_itself() -> None:
    source = (ROOT / "scripts/probe_and_baseline.py").read_text()
    assert "launch_simulation(" not in source
    assert "from isaaclab" not in source and "import isaaclab" not in source
    assert 'runpy.run_path(str(script), run_name="__main__")' in source
    assert 'PROBE_SCRIPT = "diagnose_training_nan.py"' in source
    assert 'BASELINE_SCRIPT = "run_scripted_demo.py"' in source


# ---------------------------------------------------------------- builder


def _build_notebook_source(tmp_path, monkeypatch, extra_argv) -> str:
    builder_path = ROOT / "scripts/build_kaggle_notebook.py"
    spec = importlib.util.spec_from_file_location("builder_spike",
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


def test_builder_spike_flag_is_opt_in() -> None:
    builder = (ROOT / "scripts/build_kaggle_notebook.py").read_text()
    assert '"--single-boot-spike", action="store_true"' in builder
    assert "if args.single_boot_spike:" in builder
    assert '"scripts/probe_and_baseline.py",' in builder  # manifesto do zip


def test_builder_default_notebook_keeps_separate_boots(
        tmp_path, monkeypatch) -> None:
    source = _build_notebook_source(tmp_path, monkeypatch, [])
    assert "nan_probe = run(['uv', 'run', '--extra', 'ovphysx', 'python'," in source
    assert "baseline = run(['uv', 'run', '--extra', 'ovphysx', 'python'," in source
    assert "probe_and_baseline" not in source
    assert "single_boot_spike" not in source


def test_builder_spike_notebook_fuses_the_two_stages(
        tmp_path, monkeypatch) -> None:
    source = _build_notebook_source(
        tmp_path, monkeypatch, ["--single-boot-spike"])

    # Um comando só, com os parâmetros históricos dos dois estágios.
    assert "spike = run(['uv', 'run', '--extra', 'ovphysx', 'python'," in source
    assert "str(project_dir / 'scripts' / 'probe_and_baseline.py')" in source
    assert "'--probe-num-envs', '64', '--probe-steps', '24', '--probe-seed', '7'" in source
    assert "'--baseline-seed', '23'" in source
    assert "'--baseline-output', '/kaggle/working/isaaclab-baseline.json'" in source
    # Os comandos separados somem (as cópias dos estágios continuam:
    # o entrypoint do spike os executa a partir da mesma pasta).
    assert "nan_probe = run(" not in source
    assert "baseline = run(" not in source
    assert "project_dir / 'scripts' / 'probe_and_baseline.py')" in source
    # Relatório do spike + chaves que o restante do kernel lê.
    assert "runtime_report['single_boot_spike']" in source
    assert "runtime_report['ppo_nan_probe_exit']" in source
    assert "runtime_report['scripted_baseline']" in source
    # Os portões continuam: exit != 0 aborta e o gate do baseline é o mesmo.
    assert "Single-boot spike (B5, issue #14)" in source
    assert "PPO training is gated until it passes" in source
    # O treino segue intacto depois da fusão.
    assert "training = run(['uv', 'run', '--extra', 'ovphysx', 'isaaclab', 'train'" in source


def test_builder_spike_rejects_bc_combination(tmp_path, monkeypatch) -> None:
    builder_path = ROOT / "scripts/build_kaggle_notebook.py"
    spec = importlib.util.spec_from_file_location("builder_spike_bc",
                                                  builder_path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    monkeypatch.setattr(builder, "KAGGLE_DIR", tmp_path)
    monkeypatch.setattr(
        sys, "argv",
        ["build_kaggle_notebook.py", "--kaggle-user", "tester",
         "--single-boot-spike", "--bc-warmstart",
         "--train-num-envs", "256", "--train-max-iterations", "1500"])
    try:
        builder.main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("spike + BC deveria ser recusado pelo builder")


def test_bundled_zip_contains_the_spike_entrypoint() -> None:
    import base64
    import io
    import zipfile

    builder_path = ROOT / "scripts/build_kaggle_notebook.py"
    spec = importlib.util.spec_from_file_location("builder_zip_spike",
                                                  builder_path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    archive = zipfile.ZipFile(
        io.BytesIO(base64.b64decode(builder.bundled_project())))
    assert "scripts/probe_and_baseline.py" in archive.namelist()
