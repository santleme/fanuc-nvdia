"""Spike B5 (issue #14): probe + baseline num único boot do Isaac Sim.

Hoje o probe (`diagnose_training_nan.py`) e o baseline
(`run_scripted_demo.py`) rodam como processos separados, e cada um
paga um boot do Isaac Sim. Este entrypoint roda os DOIS no mesmo
processo: cada estágio executa o seu arquivo intacto, como
``__main__`` (via ``runpy``), com o próprio argv — ou seja, duas
entradas sequenciais em ``launch_simulation`` sob um único boot.
É exatamente a reentrada que a v67 adiou por não provada; o spike
existe para prová-la ou matá-la.

Contrato:
- Os estágios não mudam em nada: rodados separados, comportamento
  idêntico ao histórico; rodados daqui, é o mesmo código, na mesma
  ordem (probe primeiro, baseline depois), com os mesmos defaults
  do pipeline (probe 64 envs/24 steps/seed 7; baseline seed 23).
- Os artefatos são os mesmos: o baseline escreve o seu JSON no
  ``--baseline-output`` e o probe imprime o veredito JSON dele.
- Este arquivo NÃO importa isaaclab: quem importa é cada estágio,
  dentro do próprio ``runpy``. O núcleo de orquestração fica puro
  e testável sem o Isaac instalado.
- Veredito do spike (pré-registrado na SPEC): ADOTA se os dois
  estágios completarem no mesmo processo com os mesmos vereditos
  dos boots separados (probe exit 0; baseline success + grasp
  sensor aprovados); MATA caso contrário. Exit deste processo:
  0 = ADOTA, 1 = MATA.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROBE_SCRIPT = "diagnose_training_nan.py"
BASELINE_SCRIPT = "run_scripted_demo.py"


@dataclass
class StageResult:
    name: str
    script: str
    argv: list[str]
    exit_code: int | None = None
    error: str | None = None
    duration_seconds: float = 0.0


@dataclass
class SpikeVerdict:
    verdict: str  # "adopt" | "reject"
    reasons: list[str] = field(default_factory=list)


def _system_exit_code(code: object) -> int:
    """Normaliza o payload de um SystemExit como o SO faria."""
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    print(code)
    return 1


def run_stage(name: str, script: Path, argv: list[str]) -> StageResult:
    """Roda um estágio neste processo, exatamente como ``__main__``.

    O argv do estágio é instalado em ``sys.argv`` durante a execução
    e restaurado ao sair — o estágio parseia os próprios argumentos
    (incluindo os do launcher) como se tivesse sido invocado direto.
    ``SystemExit`` vira exit_code; qualquer outra exceção é registrada
    (é o sintoma esperado se a reentrada do ``launch_simulation``
    quebrar o segundo estágio) e NÃO derruba o orquestrador.
    """
    result = StageResult(name=name, script=script.name, argv=list(argv))
    started = time.perf_counter()
    old_argv = sys.argv
    sys.argv = [str(script), *argv]
    try:
        runpy.run_path(str(script), run_name="__main__")
        result.exit_code = 0
    except SystemExit as exc:
        result.exit_code = _system_exit_code(exc.code)
    except Exception:  # noqa: BLE001 — fronteira do orquestrador: a
        # exceção do estágio É o dado do spike (reentrada quebrada) e
        # precisa virar registro, nunca derrubar quem coleta o veredito.
        result.exit_code = None
        result.error = traceback.format_exc()
    finally:
        sys.argv = old_argv
        result.duration_seconds = round(time.perf_counter() - started, 3)
    return result


def probe_stage_argv(device: str, num_envs: int, steps: int, seed: int) -> list[str]:
    return ["--device", device, "--num-envs", str(num_envs),
            "--steps", str(steps), "--seed", str(seed)]


def baseline_stage_argv(device: str, seed: int, output: Path,
                        rerun: bool = False,
                        rerun_path: Path | None = None) -> list[str]:
    argv = ["--device", device, "--seed", str(seed),
            "--output", str(output)]
    if rerun:
        argv.append("--rerun")
        if rerun_path is not None:
            argv += ["--rerun-path", str(rerun_path)]
    return argv


def evaluate_spike(probe: StageResult, baseline: StageResult,
                   baseline_report: dict | None) -> SpikeVerdict:
    """Veredito pré-registrado: mesmos vereditos, um só processo."""
    reasons: list[str] = []
    if probe.error is not None:
        reasons.append(f"probe levantou exceção no boot compartilhado: "
                       f"{probe.error.strip().splitlines()[-1]}")
    elif probe.exit_code != 0:
        reasons.append(f"probe saiu com código {probe.exit_code} "
                       "(nos boots separados ele reprovaria o portão)")
    if baseline.error is not None:
        reasons.append(f"baseline levantou exceção no boot compartilhado "
                       f"(sintoma de reentrada do launch_simulation): "
                       f"{baseline.error.strip().splitlines()[-1]}")
    elif baseline.exit_code != 0:
        reasons.append(f"baseline saiu com código {baseline.exit_code}")
    if baseline_report is None:
        reasons.append("baseline não produziu o relatório JSON")
    else:
        if not baseline_report.get("success"):
            reasons.append("baseline success=False no boot compartilhado")
        if not baseline_report.get("scripted_grasp_sensor_passed"):
            reasons.append("baseline scripted_grasp_sensor_passed=False "
                           "no boot compartilhado")
    return SpikeVerdict(verdict="adopt" if not reasons else "reject",
                        reasons=reasons)


def build_report(probe: StageResult, baseline: StageResult,
                 baseline_report: dict | None,
                 verdict: SpikeVerdict) -> dict:
    return {
        "spike": "B5-probe-baseline-single-boot",
        "issue": 14,
        "probe": asdict(probe),
        "baseline": asdict(baseline),
        "probe_exit_code": probe.exit_code,
        "baseline_exit_code": baseline.exit_code,
        "baseline_report": baseline_report,
        "verdict": verdict.verdict,
        "verdict_reasons": verdict.reasons,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--probe-num-envs", type=int, default=64)
    parser.add_argument("--probe-steps", type=int, default=24)
    parser.add_argument("--probe-seed", type=int, default=7)
    parser.add_argument("--baseline-seed", type=int, default=23)
    parser.add_argument("--baseline-output", type=Path,
                        default=Path("outputs/isaaclab-baseline.json"))
    parser.add_argument("--baseline-rerun", action="store_true",
                        help="Repassa --rerun ao baseline (issue #6).")
    parser.add_argument("--baseline-rerun-path", type=Path, default=None)
    parser.add_argument("--report", type=Path,
                        default=Path("outputs/probe-baseline-spike.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None,
         scripts_dir: Path | None = None) -> int:
    args = parse_args(argv)
    scripts_dir = scripts_dir or Path(__file__).resolve().parent

    probe = run_stage(
        "probe", scripts_dir / PROBE_SCRIPT,
        probe_stage_argv(args.device, args.probe_num_envs,
                         args.probe_steps, args.probe_seed))
    baseline = run_stage(
        "baseline", scripts_dir / BASELINE_SCRIPT,
        baseline_stage_argv(args.device, args.baseline_seed,
                            args.baseline_output, args.baseline_rerun,
                            args.baseline_rerun_path))

    baseline_report = None
    if args.baseline_output.is_file():
        baseline_report = json.loads(args.baseline_output.read_text())
    verdict = evaluate_spike(probe, baseline, baseline_report)
    report = build_report(probe, baseline, baseline_report, verdict)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if verdict.verdict == "adopt" else 1


if __name__ == "__main__":
    raise SystemExit(main())
