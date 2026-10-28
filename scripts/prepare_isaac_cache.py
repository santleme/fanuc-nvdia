"""Prepara o cache da instalação do Isaac Lab para virar dataset (issue #12).

Roda DENTRO do kernel construtor (``fanuc-isaaclab-cache-builder``),
depois de uma instalação normal (clone + ``uv sync`` com
``UV_LINK_MODE=copy`` e os extras aquecidos): fotografa a árvore
instalada + o cache do uv num diretório com o layout do dataset e o
manifesto de ``fanuc_cell.isaac_cache``. O diretório resultante é a
saída do kernel construtor — o dataset Kaggle nasce dele (passo de
corrida, fora deste script; ver ``docs/isaac-cache.md``).

Uso no kernel::

    python scripts/prepare_isaac_cache.py \
        --source-dir /kaggle/working/IsaacLab \
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
import platform
import shutil
import sys
from pathlib import Path

from fanuc_cell.isaac_cache import (
    ISAACLAB_PART,
    MANIFEST_NAME,
    UV_CACHE_PART,
    build_cache_manifest,
    read_manifest,
    validate_cache_manifest,
)

#: Subárvores fora do cache: o .git do clone (inútil para os kernels —
#: nada roda git sobre a árvore restaurada) e logs de corridas.
_EXCLUDED_NAMES = {".git", "logs"}


def tree_bytes(path: Path) -> int:
    """Tamanho total de uma árvore em bytes (arquivos)."""
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _copy_part(source: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(
        source,
        dest,
        symlinks=True,
        ignore=shutil.ignore_patterns(*_EXCLUDED_NAMES),
    )


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


def snapshot_cache(
    *,
    source_dir: Path,
    uv_cache_dir: Path,
    out_dir: Path,
    created_by: str,
) -> dict:
    """Fotografa a instalação em ``out_dir`` e devolve o manifesto."""
    venv = source_dir / ".venv"
    if not venv.is_dir():
        raise RuntimeError(
            f"{source_dir} não tem .venv — o snapshot exige a árvore "
            "DEPOIS do `uv sync` (é a instalação que está sendo cacheada)."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    parts: dict[str, int] = {}
    _copy_part(source_dir, out_dir / ISAACLAB_PART)
    parts[ISAACLAB_PART] = tree_bytes(out_dir / ISAACLAB_PART)
    if uv_cache_dir.is_dir():
        _copy_part(uv_cache_dir, out_dir / UV_CACHE_PART)
        parts[UV_CACHE_PART] = tree_bytes(out_dir / UV_CACHE_PART)
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
                        default=Path.home() / ".cache" / "uv",
                        help="Cache do uv a incluir (default: ~/.cache/uv).")
    parser.add_argument("--out", type=Path,
                        default=Path("/kaggle/working/isaaclab-cache"),
                        help="Diretório de saída com o layout do dataset.")
    parser.add_argument("--created-by", default="fanuc-isaaclab-cache-builder",
                        help="Identidade gravada no manifesto.")
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

    manifest = snapshot_cache(
        source_dir=args.source_dir,
        uv_cache_dir=args.uv_cache_dir,
        out_dir=args.out,
        created_by=args.created_by,
    )
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
