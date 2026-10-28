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
- ``cache_snapshot_plan``: o plano de snapshot do construtor como
  DADOS — poda seguida de MOVIMENTOS (rename), nunca cópias. A
  tentativa v2 do construtor (28/10/2026) morreu com ``Errno 28``
  porque o snapshot COPIAVA a árvore instalada + o cache do uv para
  a saída, duplicando vários GB no disco da sessão. Mover na mesma
  partição tem pegada extra ~zero.
- ``snapshot_size_problems``: o portão de tamanho do snapshot —
  medição acima de ``SNAPSHOT_MAX_TOTAL_BYTES`` recusa o snapshot
  ANTES de qualquer movimento, com mensagem clara.

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

#: Diretórios podados de qualquer árvore antes do snapshot: caches
#: de bytecode são regeneráveis no primeiro import e a poda não
#: invalida a sonda (ela roda sobre a instalação viva, no aquecimento,
#: antes do snapshot).
SNAPSHOT_PRUNE_DIR_NAMES = ("__pycache__",)

#: Nomes removidos da RAIZ da árvore do Isaac Lab antes do snapshot
#: (na cópia antiga eram só excluídos; no movimento, saem do disco):
#: o .git do clone (nada roda git sobre a árvore restaurada) e logs
#: de corridas do construtor.
SNAPSHOT_DROP_ROOT_NAMES = (".git", "logs")

#: Limiar declarado do snapshot: medição (árvore + cache do uv) acima
#: disto ameaça o limite de saída/dataset do Kaggle e o prepare
#: recusa ANTES de mover qualquer coisa — a lição da tentativa v2 é
#: falhar cedo e claro, nunca com Errno 28 no meio de uma operação.
SNAPSHOT_MAX_TOTAL_BYTES = 32 * 1024**3


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


def cache_snapshot_plan(
    *,
    source_dir: str | Path,
    uv_cache_dir: str | Path,
    out_dir: str | Path,
    uv_cache_exists: bool = True,
) -> list[dict]:
    """Plano de snapshot do construtor como dados: podar, depois MOVER.

    O snapshot NÃO copia (a cópia estourou o disco da sessão na
    tentativa v2 — ``Errno 28`` com a saída parcial já > 1,3 GB):
    cada parte é renomeada (``os.replace``) da origem para o layout
    de saída, o que exige origem e destino na mesma partição e tem
    pegada extra ~zero. O construtor garante a mesma partição por
    construção: a árvore nasce em ``/kaggle/working/IsaacLab`` e o
    cache do uv nasce sob ``/kaggle/working`` (``UV_CACHE_DIR``).

    Passos, em ordem:

    - ``prune`` por parte: remover ``SNAPSHOT_PRUNE_DIR_NAMES`` em
      qualquer profundidade e, na árvore do Isaac Lab, também
      ``SNAPSHOT_DROP_ROOT_NAMES`` da raiz.
    - ``move`` por parte: ``source`` → ``out_dir/<parte>``.

    A parte ``uv-cache`` só entra se ``uv_cache_exists`` — espelho
    do plano de restauração, que só a repõe se estiver no dataset.
    """
    source_dir = Path(source_dir)
    uv_cache_dir = Path(uv_cache_dir)
    out_dir = Path(out_dir)
    plan: list[dict] = [
        {
            "action": "prune",
            "part": ISAACLAB_PART,
            "root": str(source_dir),
            "dir_names": list(SNAPSHOT_PRUNE_DIR_NAMES),
            "drop_root_names": list(SNAPSHOT_DROP_ROOT_NAMES),
        }
    ]
    if uv_cache_exists:
        plan.append(
            {
                "action": "prune",
                "part": UV_CACHE_PART,
                "root": str(uv_cache_dir),
                "dir_names": list(SNAPSHOT_PRUNE_DIR_NAMES),
                "drop_root_names": [],
            }
        )
    plan.append(
        {
            "action": "move",
            "part": ISAACLAB_PART,
            "source": str(source_dir),
            "dest": str(out_dir / ISAACLAB_PART),
        }
    )
    if uv_cache_exists:
        plan.append(
            {
                "action": "move",
                "part": UV_CACHE_PART,
                "source": str(uv_cache_dir),
                "dest": str(out_dir / UV_CACHE_PART),
            }
        )
    return plan


def snapshot_size_problems(
    measurements: dict[str, int],
    *,
    max_total_bytes: int = SNAPSHOT_MAX_TOTAL_BYTES,
) -> list[str]:
    """Portão de tamanho do snapshot (``[]`` = pode mover).

    ``measurements`` é o mapa parte → bytes medidos ANTES do
    snapshot (a mesma medição que o construtor imprime no log).
    Medição acima do limiar declarado recusa o snapshot antes de
    qualquer movimento: mesmo sem duplicar, um volume grande demais
    ameaça o limite de saída do Kaggle — e a recusa é uma mensagem,
    não um ``Errno 28`` no meio de uma cópia.
    """
    problems: list[str] = []
    if ISAACLAB_PART not in measurements:
        problems.append(
            f"medição sem a parte obrigatória {ISAACLAB_PART!r} "
            "(snapshot de quê?)"
        )
    total = sum(measurements.values())
    if total > max_total_bytes:
        problems.append(
            f"snapshot medido com {total} bytes > limiar declarado "
            f"de {max_total_bytes} bytes — recusado ANTES de mover; "
            "investigar o crescimento da árvore/cache em vez de "
            "forçar a saída"
        )
    return problems
