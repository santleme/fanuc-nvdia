"""Prepara o cache da instalação do Isaac Lab para virar dataset (issue #12).

Roda DENTRO do kernel construtor (``fanuc-isaaclab-cache-builder``),
depois de uma instalação normal (clone + ``uv sync`` com
``UV_LINK_MODE=copy`` e os extras aquecidos): deixa a árvore instalada
+ o cache do uv num diretório com o layout do dataset e o manifesto
de ``fanuc_cell.isaac_cache``. O diretório resultante é a saída do
kernel construtor — o dataset Kaggle nasce dele (passo de corrida,
fora deste script; ver ``docs/isaac-cache.md``).

O snapshot é por MOVIMENTO, não por cópia: a tentativa v2 do
construtor (28/10/2026) instalou tudo e morreu na célula do snapshot
com ``OSError: [Errno 28] No space left on device`` — copiar a árvore
instalada + o cache do uv duplicava vários GB no disco da sessão
(saída parcial > 1,3 GB) e o dataset nunca nasceu. Agora o prepare:

1. MEDE as partes e imprime os tamanhos (registro permanente);
2. aplica o portão ``snapshot_size_problems`` — medição acima do
   limiar declarado recusa o snapshot ANTES de tocar em qualquer
   árvore, com mensagem clara;
3. PODA o regenerável (``__pycache__``; ``.git``/``logs`` da raiz da
   árvore) — a sonda de import já rodou sobre a instalação viva no
   aquecimento dos extras, e a poda de bytecode não a invalida;
4. MOVE (``os.replace``, mesma partição) cada parte para o layout
   de saída e grava o manifesto com os tamanhos medidos no destino;
5. o chamador roda ``--verify``/``verify_cache`` sobre o layout já
   movido — o mesmo portão estrutural do consumidor.

Uso no kernel::

    python scripts/prepare_isaac_cache.py \
        --source-dir /kaggle/working/IsaacLab \
        --uv-cache-dir /kaggle/working/uv-cache \
        --out /kaggle/working/isaaclab-cache

Verificação (roda em qualquer lugar, sem Isaac)::

    python scripts/prepare_isaac_cache.py --verify <dir-do-cache>

    --verify confere manifesto válido, partes do manifesto presentes
    em disco e a .venv dentro da árvore — o mesmo portão que o kernel
    consumidor aplica antes de restaurar (mais a sonda de import).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path

from fanuc_cell.isaac_cache import (
    ISAACLAB_PART,
    MANIFEST_NAME,
    SNAPSHOT_MAX_TOTAL_BYTES,
    UV_CACHE_PART,
    build_cache_manifest,
    cache_snapshot_plan,
    read_manifest,
    snapshot_size_problems,
    validate_cache_manifest,
)


def tree_bytes(path: Path) -> int:
    """Tamanho total de uma árvore em bytes (arquivos)."""
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _move_part(source: Path, dest: Path) -> None:
    """Move (rename) uma parte para o layout de saída.

    NUNCA copia: copiar foi o `Errno 28` da tentativa v2. Se o
    rename falhar (ex.: partições diferentes), a recusa é explícita
    — origem e destino do snapshot nascem na mesma partição por
    construção do construtor (ver ``cache_snapshot_plan``).
    """
    if dest.exists():
        shutil.rmtree(dest)
    try:
        os.replace(source, dest)
    except OSError as exc:
        raise RuntimeError(
            f"mover {source} -> {dest} falhou ({exc}); o snapshot não "
            "copia (a cópia estourou o disco na tentativa v2) — origem "
            "e destino precisam estar na mesma partição."
        ) from exc


def _prune_tree(
    root: Path,
    *,
    dir_names,
    drop_root_names=(),
) -> dict:
    """Poda segura de uma árvore antes do movimento.

    Remove os ``drop_root_names`` da raiz e todo diretório de nome
    em ``dir_names`` em qualquer profundidade (bytecode regenerável:
    some no snapshot e renasce no primeiro import do consumidor).
    Devolve o que foi removido (entradas e bytes) para o log.
    """
    removed = {"entries": 0, "bytes": 0}
    if not root.is_dir():
        return removed
    for name in drop_root_names:
        target = root / name
        if target.is_dir():
            removed["bytes"] += tree_bytes(target)
            removed["entries"] += 1
            shutil.rmtree(target)
    prune_names = set(dir_names)
    for dirpath, dirnames, _filenames in os.walk(root, topdown=True):
        for name in list(dirnames):
            if name in prune_names:
                target = Path(dirpath) / name
                removed["bytes"] += tree_bytes(target)
                removed["entries"] += 1
                shutil.rmtree(target)
                dirnames.remove(name)
    return removed


def venv_external_symlinks(venv: Path, uv_cache: Path) -> int:
    """Symlinks da .venv apontando para dentro do cache do uv.

    Com ``UV_LINK_MODE=copy`` o esperado é 0; >0 não invalida o cache
    (o plano de restauração repõe o cache do uv no mesmo lugar), mas
    fica registrado no relatório para diagnóstico.
    """
    if not venv.is_dir():
        return 0
    cache_root = str(uv_cache)
    count = 0
    for entry in venv.rglob("*"):
        if entry.is_symlink() and str(entry.readlink()).startswith(cache_root):
            count += 1
    return count


def measure_snapshot_inputs(
    *,
    source_dir: Path,
    uv_cache_dir: Path,
) -> dict[str, int]:
    """Medição das partes ANTES do snapshot (parte → bytes).

    É a medição que o construtor imprime no log e que alimenta o
    portão de tamanho: se um dia a árvore/cache crescer demais, o
    problema aparece dimensionado aqui, não como Errno 28.
    """
    measurements = {ISAACLAB_PART: tree_bytes(source_dir)}
    if uv_cache_dir.is_dir():
        measurements[UV_CACHE_PART] = tree_bytes(uv_cache_dir)
    return measurements


def snapshot_cache(
    *,
    source_dir: Path,
    uv_cache_dir: Path,
    out_dir: Path,
    created_by: str,
    max_total_bytes: int = SNAPSHOT_MAX_TOTAL_BYTES,
) -> dict:
    """Move a instalação para ``out_dir`` e devolve o manifesto.

    Ordem: medir → portão de tamanho (recusa ANTES de tocar nas
    árvores) → podar → mover → manifesto com os tamanhos do destino.
    """
    venv = source_dir / ".venv"
    if not venv.is_dir():
        raise RuntimeError(
            f"{source_dir} não tem .venv — o snapshot exige a árvore "
            "DEPOIS do `uv sync` (é a instalação que está sendo cacheada)."
        )
    measurements = measure_snapshot_inputs(
        source_dir=source_dir, uv_cache_dir=uv_cache_dir)
    for part, size in measurements.items():
        print(f"[cache] medição pré-snapshot: {part} = {size} bytes")
    problems = snapshot_size_problems(
        measurements, max_total_bytes=max_total_bytes)
    if problems:
        raise RuntimeError(
            "snapshot recusado ANTES de mover qualquer árvore: "
            + "; ".join(problems)
        )

    plan = cache_snapshot_plan(
        source_dir=source_dir,
        uv_cache_dir=uv_cache_dir,
        out_dir=out_dir,
        uv_cache_exists=UV_CACHE_PART in measurements,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for step in plan:
        if step["action"] == "prune":
            removed = _prune_tree(
                Path(step["root"]),
                dir_names=step["dir_names"],
                drop_root_names=step["drop_root_names"],
            )
            print(f"[cache] poda {step['part']}: "
                  f"{removed['entries']} entradas, "
                  f"{removed['bytes']} bytes removidos")
        elif step["action"] == "move":
            _move_part(Path(step["source"]), Path(step["dest"]))
            print(f"[cache] movido {step['part']}: "
                  f"{step['source']} -> {step['dest']}")

    parts = {
        step["part"]: tree_bytes(Path(step["dest"]))
        for step in plan
        if step["action"] == "move"
    }
    manifest = build_cache_manifest(
        parts=parts,
        python_version=platform.python_version(),
        platform=platform.platform(),
        created_by=created_by,
    )
    manifest["venv_uv_cache_symlinks"] = venv_external_symlinks(
        out_dir / ISAACLAB_PART / ".venv", uv_cache_dir
    )
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_cache(cache_dir: Path) -> list[str]:
    """Problemas de um cache preparado (``[]`` = pronto para dataset)."""
    manifest = read_manifest(cache_dir)
    if manifest is None:
        return [f"sem {MANIFEST_NAME} legível em {cache_dir}"]
    problems = validate_cache_manifest(manifest)
    for part in manifest.get("parts", {}):
        if not (cache_dir / part).is_dir():
            problems.append(f"parte {part!r} do manifesto ausente em disco")
    isaaclab = cache_dir / ISAACLAB_PART
    if isaaclab.is_dir() and not (isaaclab / ".venv").is_dir():
        problems.append("árvore isaaclab sem .venv (sync não fotografado)")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path,
                        default=Path("/kaggle/working/IsaacLab"),
                        help="Árvore do Isaac Lab já instalada (com .venv).")
    parser.add_argument("--uv-cache-dir", type=Path,
                        default=Path(os.environ.get("UV_CACHE_DIR")
                                     or Path.home() / ".cache" / "uv"),
                        help="Cache do uv a incluir (default: $UV_CACHE_DIR "
                             "ou ~/.cache/uv).")
    parser.add_argument("--out", type=Path,
                        default=Path("/kaggle/working/isaaclab-cache"),
                        help="Diretório de saída com o layout do dataset.")
    parser.add_argument("--created-by", default="fanuc-isaaclab-cache-builder",
                        help="Identidade gravada no manifesto.")
    parser.add_argument("--max-snapshot-bytes", type=int,
                        default=SNAPSHOT_MAX_TOTAL_BYTES,
                        help="Limiar do portão de tamanho: medição acima "
                             "disto recusa o snapshot antes de mover.")
    parser.add_argument("--verify", type=Path, default=None,
                        help="Só verifica um cache preparado e sai.")
    parser.add_argument("--describe", action="store_true",
                        help="Imprime o layout esperado do dataset e sai.")
    args = parser.parse_args()

    if args.describe:
        print(
            "Layout do dataset do cache (docs/isaac-cache.md):\n"
            f"  {MANIFEST_NAME}  manifesto (formato, ref, partes, bytes)\n"
            f"  {ISAACLAB_PART}/              checkout do Isaac Lab + .venv "
            "(UV_LINK_MODE=copy, sem .git)\n"
            f"  {UV_CACHE_PART}/              cache do uv (~/.cache/uv)\n"
            "Snapshot por MOVIMENTO (rename) das árvores instaladas —\n"
            "nada é copiado; medição e portão de tamanho antes de mover.\n"
        )
        return 0
    if args.verify is not None:
        problems = verify_cache(args.verify)
        if problems:
            print("CACHE INVÁLIDO:")
            for problem in problems:
                print(f"  - {problem}")
            return 1
        manifest = read_manifest(args.verify)
        total = sum(manifest["parts"].values())
        print(f"CACHE OK: partes={manifest['parts']} total={total} bytes "
              f"ref={manifest['isaaclab_ref']}")
        return 0

    try:
        manifest = snapshot_cache(
            source_dir=args.source_dir,
            uv_cache_dir=args.uv_cache_dir,
            out_dir=args.out,
            created_by=args.created_by,
            max_total_bytes=args.max_snapshot_bytes,
        )
    except RuntimeError as exc:
        print(f"SNAPSHOT RECUSADO: {exc}")
        return 2
    problems = verify_cache(args.out)
    if problems:
        print("snapshot gerado mas NÃO verifica:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    total = sum(manifest["parts"].values())
    print(f"cache preparado em {args.out}: {total} bytes "
          f"(partes: {manifest['parts']})")
    print("próximo passo: criar o dataset Kaggle a partir desta saída "
          "(docs/isaac-cache.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
