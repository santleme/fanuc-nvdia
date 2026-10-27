"""Pré-treino BC do ator a partir das demonstrações (issue #3).

Roda em qualquer lugar (numpy puro, sem Isaac): lê o ``.npz`` do
coletor, treina o MLP espelho do ator RSL-RL e salva os pesos +
``bc-report.json`` (estatísticas das demos + perda inicial/final).
No kernel, este script roda entre a coleta e o PPO warm-start.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fanuc_cell.bc import train_bc
from fanuc_cell.demos import DemoSet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demos", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="Arquivo .npz dos pesos do ator treinado.")
    parser.add_argument("--report", type=Path, default=None,
                        help="JSON com estatísticas (default: ao lado do --out).")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-demo-success", type=float, default=0.8,
                        help="Portão: abaixo disso as demos não prestam "
                             "para BC e o script falha (exit 3).")
    args = parser.parse_args()

    demos = DemoSet.load_npz(args.demos)
    summary = demos.summary()
    print(json.dumps({"demos": summary}, indent=2))
    if demos.success_rate < args.min_demo_success:
        print(f"PORTÃO BC: success das demos {demos.success_rate:.3f} < "
              f"{args.min_demo_success} — professor não está ensinando "
              "o ciclo completo; abortando antes do treino.")
        return 3

    mlp, history = train_bc(
        demos.observations, demos.actions,
        epochs=args.epochs, lr=args.lr, seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mlp.save_npz(args.out, metadata={
        "source_demos": str(args.demos),
        "demo_summary": summary,
        "bc_epochs": args.epochs,
        "bc_final_loss": history[-1] if history else None,
    })
    report = {
        "demos": summary,
        "bc": {
            "epochs": args.epochs,
            "lr": args.lr,
            "loss_first_epoch": history[0] if history else None,
            "loss_last_epoch": history[-1] if history else None,
            "weights_file": str(args.out),
        },
    }
    report_path = args.report or args.out.with_name("bc-report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
