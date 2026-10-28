"""Bundle the Isaac Lab install-cache CONSTRUCTOR into its own kernel.

Issue #12 (B4): every project kernel pays clone + ``uv sync`` of
Isaac Lab v3.0.0 (~4–6 min). This constructor kernel pays it ONCE,
warms the extras the project kernels resolve, snapshots the installed
tree + the uv cache into the dataset layout (``scripts/
prepare_isaac_cache.py`` + ``fanuc_cell.isaac_cache``) and leaves the
snapshot as its output — the Kaggle dataset is created from that
output (run step, documented in ``docs/isaac-cache.md``), and the
training kernels then build with ``--isaac-cache-dataset``.

Separate kernel (pascalcoimbra/fanuc-isaaclab-cache-builder) so the
training kernel's version history stays clean, same pattern as the
clone-modes diagnostic. No GPU: install/sync is download + wheel
unpacking; nothing here runs the simulator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_kaggle_notebook import bundled_project

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "kaggle-cache"
NOTEBOOK_NAME = "fanuc-isaaclab-cache-builder.ipynb"
KERNEL_SLUG = "fanuc-isaaclab-cache-builder"


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": hashlib.sha1(source.encode("utf-8")).hexdigest()[:8],
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": hashlib.sha1(source.encode("utf-8")).hexdigest()[:8],
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle-user", default="INSERT_KAGGLE_USERNAME")
    args = parser.parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    notebook = {
        "cells": [
            markdown(
                "# FANUC — construtor do cache da instalação do Isaac Lab\n\n"
                "Issue #12 (B4): instala o Isaac Lab 3.0.0 UMA vez "
                "(clone + `uv sync` com `UV_LINK_MODE=copy` + extras "
                "aquecidos), fotografa a árvore instalada e o cache do "
                "uv no layout do dataset e deixa o snapshot como saída. "
                "O dataset Kaggle nasce desta saída; os kernels do "
                "projeto passam a montar com `--isaac-cache-dataset` e "
                "restauram em vez de instalar — com fallback automático "
                "para a instalação completa. Desenho: `docs/isaac-cache.md` "
                "no repo.\n"
            ),
            code(
                "from __future__ import annotations\n"
                "import json, os, platform, subprocess, sys, time\n"
                "from pathlib import Path\n\n"
                "working_dir = Path('/kaggle/working')\n"
                "report_path = working_dir / 'cache-builder-report.json'\n"
                "report = {'commands': []}\n"
                "def save_report():\n"
                "    report_path.write_text(json.dumps(report, indent=2))\n"
                "def run(command, **kwargs):\n"
                "    started = time.perf_counter()\n"
                "    result = subprocess.run(command, text=True, capture_output=True, check=False, **kwargs)\n"
                "    elapsed = round(time.perf_counter() - started, 3)\n"
                "    report['commands'].append({'argv': command, 'exit_code': result.returncode,\n"
                "        'duration_seconds': elapsed,\n"
                "        'stdout_tail': result.stdout[-4000:], 'stderr_tail': result.stderr[-4000:]})\n"
                "    print(f'[PERF] command_seconds={elapsed} exit={result.returncode} command={command[0]}')\n"
                "    save_report()\n"
                "    return result\n"
                "report.update({'python': sys.version, 'platform': platform.platform()})\n"
                "print(json.dumps(report, indent=2))\n"
                "save_report()\n"
            ),
            code(
                "import base64, io, zipfile\n"
                f"project_zip_b64 = {bundled_project()!r}\n"
                "with zipfile.ZipFile(io.BytesIO(base64.b64decode(project_zip_b64))) as archive:\n"
                "    archive.extractall('/kaggle/working/fanuc-cell')\n"
                "%cd /kaggle/working/fanuc-cell\n"
            ),
            code(
                "# Instalação completa, do zero — é ESTA instalação que\n"
                "# vira cache. UV_LINK_MODE=copy deixa a .venv\n"
                "# autossuficiente (sem links para o cache do uv).\n"
                "os.environ['UV_LINK_MODE'] = 'copy'\n"
                "uv_probe = run(['uv', '--version'])\n"
                "if uv_probe.returncode != 0:\n"
                "    install_uv = run([sys.executable, '-m', 'pip', 'install', 'uv'])\n"
                "    if install_uv.returncode != 0:\n"
                "        raise RuntimeError(install_uv.stderr[-4000:])\n"
                "source_dir = Path('/kaggle/working/IsaacLab')\n"
                "clone = run(['git', 'clone', '--depth', '1', '--branch', 'v3.0.0',\n"
                "             'https://github.com/isaac-sim/IsaacLab.git', str(source_dir)])\n"
                "if clone.returncode != 0:\n"
                "    raise RuntimeError('Could not clone Isaac Lab v3.0.0: ' + clone.stderr[-4000:])\n"
                "sync = run(['uv', 'sync'], cwd=source_dir)\n"
                "if sync.returncode != 0:\n"
                "    raise RuntimeError('Isaac Lab dependency installation failed: ' + sync.stderr[-6000:])\n"
                "# Aquece os extras que os kernels do projeto resolvem\n"
                "# via `uv run --extra ...` (ovphysx/video/importers/\n"
                "# ovrtx): com a .venv e o cache do uv restaurados, a\n"
                "# primeira corrida não baixa nada.\n"
                "warm = run(['uv', 'run', '--extra', 'ovphysx', '--extra', 'video',\n"
                "            '--extra', 'importers', '--extra', 'ovrtx',\n"
                "            'python', '-c', 'import isaaclab, torch; print(\"warm ok\")'],\n"
                "           cwd=source_dir)\n"
                "report['warm_extras_exit'] = warm.returncode\n"
                "save_report()\n"
                "if warm.returncode != 0:\n"
                "    raise RuntimeError('Warm-up dos extras falhou: ' + warm.stderr[-6000:])\n"
                "print(warm.stdout[-2000:])\n"
            ),
            code(
                "# Snapshot no layout do dataset + verificação pelo\n"
                "# mesmo portão que o consumidor aplica.\n"
                "prep_env = os.environ.copy()\n"
                "prep_env['PYTHONPATH'] = '/kaggle/working/fanuc-cell/src' + os.pathsep + prep_env.get('PYTHONPATH', '')\n"
                "prep = run([sys.executable, '/kaggle/working/fanuc-cell/scripts/prepare_isaac_cache.py',\n"
                "            '--source-dir', '/kaggle/working/IsaacLab',\n"
                "            '--out', '/kaggle/working/isaaclab-cache',\n"
                "            '--created-by', 'fanuc-isaaclab-cache-builder'], env=prep_env)\n"
                "report['prepare_exit'] = prep.returncode\n"
                "save_report()\n"
                "print(prep.stdout[-4000:])\n"
                "if prep.returncode != 0:\n"
                "    raise RuntimeError('prepare_isaac_cache falhou: ' + prep.stderr[-4000:])\n"
                "manifest = json.loads(Path('/kaggle/working/isaaclab-cache/isaac-cache-manifest.json').read_text())\n"
                "report['cache_manifest'] = manifest\n"
                "report['cache_total_bytes'] = sum(manifest['parts'].values())\n"
                "save_report()\n"
                "print(json.dumps(manifest, indent=2))\n"
                "print('PROXIMO PASSO (manual, uma vez): criar o dataset '\n"
                "'fanuc-isaaclab-cache a partir da saida isaaclab-cache/ '\n"
                "'deste kernel — docs/isaac-cache.md.')\n"
            ),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path = CACHE_DIR / NOTEBOOK_NAME
    notebook_path.write_text(json.dumps(notebook, indent=1) + "\n")
    metadata = {
        "id": f"{args.kaggle_user}/{KERNEL_SLUG}",
        "title": "FANUC IsaacLab Cache Builder",
        "code_file": NOTEBOOK_NAME,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
    }
    (CACHE_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(notebook_path)


if __name__ == "__main__":
    main()
