"""Cache da instalação do Isaac Lab em dataset Kaggle (issue #12, B4).

Cada kernel paga clone + ``uv sync`` do Isaac Lab do zero (~4–6 min,
medidos no veredito v67). O desenho registrado na SPEC: um kernel
construtor gera o ambiente instalado uma vez e o publica como dataset;
os kernels seguintes montam o dataset e restauram a árvore instalada
em vez de instalar — com fallback para a instalação completa se a
restauração falhar por qualquer motivo.

Este módulo é o núcleo PURO (só stdlib) desse contrato — nenhuma
decisão de restauração vive copiada no notebook:

- ``build_cache_manifest`` / ``validate_cache_manifest``: o manifesto
  ``isaac-cache-manifest.json`` na raiz do dataset amarra formato,
  ref do Isaac Lab, partes presentes e tamanhos. Restaurar cache de
  outra versão do Isaac Lab é exatamente o erro que o manifesto
  existe para impedir.
- ``find_cache_mount``: dado os caminhos candidatos (override por
  ambiente, mount do Kaggle), devolve o primeiro com manifesto válido.
- ``cache_restore_plan``: o plano de restauração como DADOS — pares
  (origem no mount, destino absoluto, merge?) — para o notebook só
  executar cópias e os testes conferirem a montagem sem Kaggle.

Layout do dataset (ver ``docs/isaac-cache.md``)::

    isaac-cache-manifest.json   manifesto (este módulo)
    isaaclab/                   checkout v3.0.0 + .venv do uv sync
                                (sem .git; venv com UV_LINK_MODE=copy)
    uv-cache/                   cache do uv (~/.cache/uv) para os
                                `uv run --extra ...` resolverem offline

A sonda de importação pós-restauração (``import isaaclab, torch`` via
``uv run --no-sync``) é quem decide no kernel: manifesto válido +
cópia ok + sonda ok = cache; qualquer falha = instalação completa.
"""

from __future__ import annotations

import json
from pathlib import Path

#: Versão do formato do cache. Sobe quando o layout/manifesto muda de
#: forma incompatível — caches antigos são recusados, nunca adaptados.
CACHE_FORMAT_VERSION = 1

#: Pin do Isaac Lab: o mesmo ref clonado pelo builder dos kernels.
ISAACLAB_REF = "v3.0.0"
ISAACLAB_REPO = "https://github.com/isaac-sim/IsaacLab.git"

MANIFEST_NAME = "isaac-cache-manifest.json"

#: Nome da parte da árvore instalada dentro do dataset.
ISAACLAB_PART = "isaaclab"
#: Nome da parte do cache do uv dentro do dataset.
UV_CACHE_PART = "uv-cache"

#: Extras do workspace que os kernels do projeto resolvem via
#: ``uv run --extra ...`` — o construtor os aquece no sync para a
#: primeira corrida restaurada não baixar nada.
WARMED_EXTRAS = ("ovphysx", "video", "importers", "ovrtx")

#: Modo de link exigido no sync do construtor: ``copy`` deixa a .venv
#: autossuficiente (sem hardlinks/symlinks para o cache do uv, que
#: mora em outro filesystem no mount do dataset).
UV_LINK_MODE = "copy"


def build_cache_manifest(
    *,
    parts: dict[str, int],
    python_version: str,
    platform: str,
    created_by: str,
    isaaclab_ref: str = ISAACLAB_REF,
) -> dict:
    """Monta o manifesto de um cache preparado.

    ``parts`` mapeia nome da parte → tamanho total em bytes
    (``isaaclab`` sempre; ``uv-cache`` quando incluído).
    """
    return {
        "format": CACHE_FORMAT_VERSION,
        "isaaclab_repo": ISAACLAB_REPO,
        "isaaclab_ref": isaaclab_ref,
        "python": python_version,
        "platform": platform,
        "created_by": created_by,
        "uv_link_mode": UV_LINK_MODE,
        "warmed_extras": list(WARMED_EXTRAS),
        "parts": dict(parts),
    }


def validate_cache_manifest(
    data: object,
    *,
    expected_ref: str = ISAACLAB_REF,
) -> list[str]:
    """Lista de problemas do manifesto (``[]`` = válido para restaurar)."""
    if not isinstance(data, dict):
        return ["manifesto não é um objeto JSON"]
    problems: list[str] = []
    if data.get("format") != CACHE_FORMAT_VERSION:
        problems.append(
            f"format={data.get('format')!r} != {CACHE_FORMAT_VERSION} "
            "(cache de outro formato: reconstruir, não adaptar)"
        )
    if data.get("isaaclab_ref") != expected_ref:
        problems.append(
            f"isaaclab_ref={data.get('isaaclab_ref')!r} != {expected_ref!r} "
            "(cache de outra versão do Isaac Lab)"
        )
    parts = data.get("parts")
    if not isinstance(parts, dict) or ISAACLAB_PART not in parts:
        problems.append(f"manifesto sem a parte obrigatória {ISAACLAB_PART!r}")
    return problems


def read_manifest(mount: str | Path) -> dict | None:
    """Lê o manifesto de um mount candidato; ``None`` se ilegível."""
    try:
        data = json.loads((Path(mount) / MANIFEST_NAME).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def find_cache_mount(candidates) -> Path | None:
    """Primeiro candidato com manifesto presente E válido, ou ``None``.

    Candidato ausente, sem manifesto ou com manifesto inválido é
    simplesmente ignorado — quem chama cai na instalação completa.
    """
    for candidate in candidates:
        if candidate is None:
            continue
        mount = Path(candidate)
        manifest = read_manifest(mount)
        if manifest is None:
            continue
        if validate_cache_manifest(manifest):
            continue
        if not (mount / ISAACLAB_PART).is_dir():
            continue
        return mount
    return None


def cache_restore_plan(
    mount: str | Path,
    *,
    working_dir: str | Path,
    home_dir: str | Path,
) -> list[dict]:
    """Plano de restauração como dados: o que copiar, para onde.

    - ``isaaclab/`` → ``<working>/IsaacLab``: o destino NÃO pode
      existir (o clone criaria a árvore do zero; merge=False) — quem
      executa apaga um resto anterior antes de copiar.
    - ``uv-cache/`` → ``~/.cache/uv``: merge sobre o que já houver.

    A parte ``uv-cache`` só entra no plano se existir no mount.
    """
    mount = Path(mount)
    plan = [
        {
            "source": str(mount / ISAACLAB_PART),
            "dest": str(Path(working_dir) / "IsaacLab"),
            "merge": False,
        }
    ]
    if (mount / UV_CACHE_PART).is_dir():
        plan.append(
            {
                "source": str(mount / UV_CACHE_PART),
                "dest": str(Path(home_dir) / ".cache" / "uv"),
                "merge": True,
            }
        )
    return plan
