"""Empacota um checkpoint treinado como dataset Kaggle (issue #11, B2).

O modo eval-only do notebook (builder com --eval-only) avalia um
checkpoint existente sem retreino; para isso o checkpoint precisa
chegar ao kernel como dataset anexado, montado em /kaggle/input.
Este script prepara a pasta do dataset a partir de um arquivo .pt
EXPLÍCITO (nada de 'latest' nem de glob — lição da BC v6):

  python3 scripts/package_checkpoint_dataset.py \
      --checkpoint /caminho/model_1499.pt \
      --kaggle-user pascalcoimbra --dataset fanuc-checkpoint-bc-v7 \
      --out /tmp/fanuc-checkpoint-bc-v7 \
      --source-note "BC v7, model_1499, corrida <kernel/version>"

Saída na pasta: o .pt (nome preservado), dataset-metadata.json
(formato do Kaggle) e checkpoint-manifest.json (sha256, bytes, origem
declarada) — o manifesto viaja com o dataset e o kernel eval pode
conferir o que está avaliando.

Upload (fluxo verificado do conector custom.kaggle):

  python3 ~/workspace/custom-connectors/custom.kaggle/bin/kaggle_api.py \
      create-dataset /tmp/fanuc-checkpoint-bc-v7

Depois, montar e empurrar o kernel eval:

  python3 scripts/build_kaggle_notebook.py --kaggle-user <user> \
      --kernel-slug fanuc-eval-only --eval-only \
      --eval-checkpoint fanuc-checkpoint-bc-v7/model_1499.pt
  python3 scripts/kaggle_push_kernel.py --usuario <user>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package(checkpoint: Path, out: Path, *, kaggle_user: str,
            dataset: str, title: str | None,
            source_note: str | None) -> dict:
    if not checkpoint.is_file():
        raise SystemExit(f"ERRO: checkpoint não existe: {checkpoint}")
    if checkpoint.suffix != ".pt":
        raise SystemExit(
            f"ERRO: checkpoint deve ser o arquivo .pt do RSL-RL: {checkpoint}")
    if "/" in dataset or dataset == "latest":
        raise SystemExit(
            f"ERRO: --dataset deve ser um slug simples: {dataset!r}")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(
            f"ERRO: pasta de saída já existe e não está vazia: {out}")
    out.mkdir(parents=True, exist_ok=True)

    target = out / checkpoint.name
    shutil.copy2(checkpoint, target)
    manifest = {
        "checkpoint_file": checkpoint.name,
        "sha256": sha256_of(target),
        "bytes": target.stat().st_size,
        "dataset": f"{kaggle_user}/{dataset}",
        "packaged_at_utc": datetime.now(UTC).isoformat(),
        "source_note": source_note,
        "eval_checkpoint_arg": f"{dataset}/{checkpoint.name}",
    }
    (out / "checkpoint-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    dataset_metadata = {
        "title": title or f"FANUC checkpoint {dataset}",
        "id": f"{kaggle_user}/{dataset}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (out / "dataset-metadata.json").write_text(
        json.dumps(dataset_metadata, indent=2) + "\n")
    return manifest


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True,
                    help="arquivo .pt EXATO do checkpoint (sem 'latest')")
    ap.add_argument("--out", required=True, help="pasta do dataset a criar")
    ap.add_argument("--kaggle-user", required=True,
                    help="dono do dataset no Kaggle")
    ap.add_argument("--dataset", required=True,
                    help="slug do dataset (ex.: fanuc-checkpoint-bc-v7)")
    ap.add_argument("--title", help="título do dataset (default derivado do slug)")
    ap.add_argument("--source-note",
                    help="origem do checkpoint (corrida/kernel/versão), "
                         "registrada no manifesto")
    args = ap.parse_args(argv[1:])

    manifest = package(Path(args.checkpoint), Path(args.out),
                       kaggle_user=args.kaggle_user, dataset=args.dataset,
                       title=args.title, source_note=args.source_note)
    print(json.dumps(manifest, indent=2))
    print("\npróximos passos:")
    print("  1) subir o dataset: kaggle_api.py create-dataset "
          f"{args.out}")
    print("  2) montar o kernel eval: build_kaggle_notebook.py "
          "--eval-only --eval-checkpoint "
          f"{manifest['eval_checkpoint_arg']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
