"""Bundle the clone-mode diagnostic into its own Kaggle notebook/kernel.

Separate kernel (pascalcoimbra/fanuc-diagnose-clone-modes) so the main
training kernel's version history stays clean. The notebook installs the
pinned Isaac Lab, converts the URDF twice (with and without --fix-base),
then runs scripts/diagnose_clone_modes.py modes A-E as subprocesses and
saves clone-modes-report.json with one verdict per mode.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAG_DIR = ROOT / "kaggle-diag"
NOTEBOOK_NAME = "fanuc-diagnose-clone-modes.ipynb"
KERNEL_SLUG = "fanuc-diagnose-clone-modes"


def bundled_project() -> str:
    payload = io.BytesIO()
    included = [
        "pyproject.toml",
        "assets/cell_geometry.json",
        "assets/m10id12_robot.urdf",
        "src/fanuc_cell/__init__.py",
        "src/fanuc_cell/kinematics.py",
        "src/fanuc_cell/task.py",
        "src/fanuc_cell/render.py",
        "scripts/render_demo.py",
        "scripts/run_scripted_demo.py",
        "scripts/diagnose_training_nan.py",
        "scripts/diagnose_clone_modes.py",
        "tests/test_kinematics.py",
        "tests/test_task.py",
        "tests/test_static_clearance.py",
    ]
    included.extend(
        path.relative_to(ROOT).as_posix()
        for path in sorted((ROOT / "isaaclab_overlay").rglob("*.py"))
        if path.is_file()
    )
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in included:
            archive.write(ROOT / relative, arcname=relative)
    return base64.b64encode(payload.getvalue()).decode("ascii")


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


PREFLIGHT = (
    "from __future__ import annotations\n"
    "import json, os, platform, re, subprocess, sys, time\n"
    "from pathlib import Path\n\n"
    "working_dir = Path('/kaggle/working')\n"
    "working_dir.mkdir(exist_ok=True)\n"
    "report_path = working_dir / 'clone-modes-report.json'\n"
    "report = {'commands': []}\n"
    "def save_report():\n"
    "    report_path.write_text(json.dumps(report, indent=2))\n\n"
    "def run(command, **kwargs):\n"
    "    started = time.perf_counter()\n"
    "    result = subprocess.run(command, text=True, capture_output=True, check=False, **kwargs)\n"
    "    elapsed = round(time.perf_counter() - started, 3)\n"
    "    report.setdefault('commands', []).append({'argv': command, 'exit_code': result.returncode,\n"
    "        'duration_seconds': elapsed,\n"
    "        'stdout_tail': result.stdout[-6000:], 'stderr_tail': result.stderr[-6000:]})\n"
    "    print(f'[PERF] command_seconds={elapsed} exit={result.returncode} command={command[0]}')\n"
    "    save_report()\n"
    "    return result\n\n"
    "gpu = run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'])\n"
    "driver = run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'])\n"
    "report.update({'python': sys.version, 'platform': platform.platform(),\n"
    "          'gpus': [line.strip() for line in gpu.stdout.splitlines() if line.strip()],\n"
    "          'driver_versions': sorted(set(line.strip() for line in driver.stdout.splitlines() if line.strip()))})\n"
    "if gpu.returncode != 0 or not report['gpus']:\n"
    "    raise RuntimeError('Kaggle session has no visible NVIDIA GPU; select GPU accelerator.')\n"
    "caps = []\n"
    "try:\n"
    "    import torch\n"
    "    report['torch'] = torch.__version__\n"
    "    for device in range(torch.cuda.device_count()):\n"
    "        caps.append(torch.cuda.get_device_capability(device))\n"
    "    report['compute_capabilities'] = [list(cap) for cap in caps]\n"
    "except Exception as exc:\n"
    "    report['torch_probe_error'] = repr(exc)\n"
    "driver_text = report['driver_versions'][0] if report['driver_versions'] else '0'\n"
    "match = re.match(r'(\\d+)\\.(\\d+)(?:\\.(\\d+))?', driver_text)\n"
    "driver_tuple = tuple(map(int, match.groups(default='0'))) if match else (0, 0, 0)\n"
    "report['isaaclab_linux_driver_compatible'] = driver_tuple >= (580, 65, 6)\n"
    "report['isaaclab_gpu_arch_compatible'] = bool(caps) and all(cap >= (7, 5) for cap in caps)\n"
    "print(json.dumps(report, indent=2))\n"
    "save_report()\n"
)

BUNDLE = (
    "import base64, io, zipfile\n"
    "from pathlib import Path\n"
    "PROJECT_ZIP_B64 = '{bundle}'\n"
    "with zipfile.ZipFile(io.BytesIO(base64.b64decode(PROJECT_ZIP_B64))) as archive:\n"
    "    archive.extractall('/kaggle/working/fanuc-cell')\n"
    "print(sorted(p.name for p in Path('/kaggle/working/fanuc-cell/scripts').glob('*.py')))\n"
)

SETUP = (
    "import shutil\n"
    "from pathlib import Path\n"
    "if not (report['isaaclab_gpu_arch_compatible'] and report['isaaclab_linux_driver_compatible']):\n"
    "    raise RuntimeError('GPU/driver below the Isaac Lab 3.0 minimum: ' + str(report))\n"
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
    "project_dir = Path('/kaggle/working/fanuc_cell_lab')\n"
    "generate = run(['uv', 'run', 'isaaclab', '--new', '--non_interactive',\n"
    "               '--task_type', 'external', '--project_path', '/kaggle/working',\n"
    "               '--name', 'fanuc_cell_lab', '--author', 'Pascal Coimbra',\n"
    "               '--task_name', 'conveyor_pick_place', '--robot_name', 'm10id12',\n"
    "               '--workflow', 'manager-based:single-agent',\n"
    "               '--rl_library', 'rsl_rl', '--rl_algorithm', 'ppo'], cwd=source_dir)\n"
    "if generate.returncode != 0 or not project_dir.is_dir():\n"
    "    raise RuntimeError('Isaac Lab project generation failed: ' + generate.stderr[-6000:])\n"
    "overlay = Path('/kaggle/working/fanuc-cell/isaaclab_overlay/src/fanuc_cell_lab')\n"
    "project_package = project_dir / 'src' / 'fanuc_cell_lab'\n"
    "shutil.copytree(overlay, project_package, dirs_exist_ok=True)\n"
    "shutil.copytree('/kaggle/working/fanuc-cell/src/fanuc_cell',\n"
    "                project_dir / 'src' / 'fanuc_cell', dirs_exist_ok=True)\n"
    "shutil.copy2('/kaggle/working/fanuc-cell/scripts/diagnose_clone_modes.py',\n"
    "             project_dir / 'scripts' / 'diagnose_clone_modes.py')\n"
    "asset_dir = project_package / 'assets' / 'data'\n"
    "asset_dir.mkdir(parents=True, exist_ok=True)\n"
    "official_converter = (source_dir / 'scripts/tools/convert_urdf.py').read_text()\n"
    "converted_source, replacements = re.subn(\n"
    "    r'(args_cli\\.physics\\s*=\\s*\\\"isaacsim_physx\\\"\\s+if\\s+args_cli\\.require_kit\\s+else\\s+)\\\"[^\\\"]+\\\"',\n"
    "    r'\\1\"ovphysx\"', official_converter, count=1)\n"
    "if replacements != 1:\n"
    "    raise RuntimeError('Pinned Isaac Lab URDF converter changed its backend selection; inspect before conversion.')\n"
    "converter_script = source_dir / 'scripts/tools/convert_urdf_ovphysx.py'\n"
    "converter_script.write_text(converted_source)\n"
    "contact_code = (\n"
    "    'import json,sys; from pxr import Usd,UsdPhysics,Sdf; '\n"
    "    'stage=Usd.Stage.Open(sys.argv[1]); '\n"
    "    'bodies=[p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]; '\n"
    "    \"[(p.AddAppliedSchema('PhysxRigidBodyAPI'), p.AddAppliedSchema('PhysxContactReportAPI'), \"\n"
    "    \" p.CreateAttribute('physxRigidBody:sleepThreshold', Sdf.ValueTypeNames.Float, custom=False).Set(0.0), \"\n"
    "    \" p.CreateAttribute('physxContactReport:threshold', Sdf.ValueTypeNames.Float, custom=False).Set(0.0)) \"\n"
    "    \" for p in bodies]; \"\n"
    "    \"stage.GetRootLayer().Save(); print(json.dumps({'contact_report_bodies':len(bodies)}))\"\n"
    ")\n"
    "inspect_code = (\n"
    "    'import json,sys; from pxr import Usd,UsdPhysics; '\n"
    "    'stage=Usd.Stage.Open(sys.argv[1]); ps=list(stage.Traverse()); '\n"
    "    'r=sum(p.IsA(UsdPhysics.RevoluteJoint) for p in ps); '\n"
    "    'j=sum(p.IsA(UsdPhysics.PrismaticJoint) for p in ps); '\n"
    "    'b=[str(p.GetPath()) for p in ps if p.HasAPI(UsdPhysics.RigidBodyAPI)]; '\n"
    "    'print(json.dumps({\\\"revolute_joints\\\":int(r),\\\"prismatic_joints\\\":int(j),\\\"rigid_bodies\\\":b}))'\n"
    ")\n"
    "def convert(extra_args, out_dir, label):\n"
    "    conversion = run(['uv', 'run', '--extra', 'importers', '--extra', 'ovphysx', 'isaaclab', '-p',\n"
    "                     str(converter_script),\n"
    "                     '/kaggle/working/fanuc-cell/assets/m10id12_robot.urdf',\n"
    "                     str(out_dir), '--joint-stiffness', '0.0',\n"
    "                     '--joint-damping', '0.0', '--joint-target-type', 'none', *extra_args], cwd=source_dir)\n"
    "    if conversion.returncode != 0:\n"
    "        raise RuntimeError(f'URDF conversion ({label}) failed: ' + conversion.stderr[-6000:])\n"
    "    candidates = sorted(out_dir.rglob('m10id12_robot.usd*'))\n"
    "    if not candidates:\n"
    "        raise RuntimeError(f'URDF conversion ({label}) produced no USD under ' + str(out_dir))\n"
    "    usd = candidates[0]\n"
    "    patch = run(['uv', 'run', 'python', '-c', contact_code, str(usd)], cwd=source_dir)\n"
    "    if patch.returncode != 0:\n"
    "        raise RuntimeError(f'Contact patch ({label}) failed: ' + patch.stderr[-4000:])\n"
    "    inspect = run(['uv', 'run', 'python', '-c', inspect_code, str(usd)], cwd=source_dir)\n"
    "    if inspect.returncode != 0:\n"
    "        raise RuntimeError(f'USD inspect ({label}) failed: ' + inspect.stderr[-4000:])\n"
    "    summary = json.loads(inspect.stdout.strip().splitlines()[-1])\n"
    "    report[f'usd_{label}'] = {'path': str(usd), 'joints': {k: summary[k] for k in ('revolute_joints', 'prismatic_joints')}}\n"
    "    save_report()\n"
    "    return usd\n"
    "fix_usd = convert(['--fix-base'], asset_dir, 'fixbase')\n"
    "expected_asset = asset_dir / 'm10id12_robot' / 'm10id12_robot.usd'\n"
    "if fix_usd.resolve() != expected_asset.resolve():\n"
    "    expected_asset.parent.mkdir(parents=True, exist_ok=True)\n"
    "    shutil.copy2(fix_usd, expected_asset)\n"
    "nofix_usd = convert([], asset_dir / 'nofix', 'nofix')\n"
    "report['nofix_usd_path'] = str(nofix_usd)\n"
    "save_report()\n"
    "print(json.dumps({k: report[k] for k in ('usd_fixbase', 'usd_nofix')}, indent=2))\n"
)

MODES = (
    "import subprocess\n"
    "from pathlib import Path\n"
    "mode_env = os.environ.copy()\n"
    "mode_env['PYTHONPATH'] = str(project_dir / 'src') + os.pathsep + mode_env.get('PYTHONPATH', '')\n"
    "verdicts = {}\n"
    "for mode in ['A', 'B', 'C', 'D', 'E']:\n"
    "    command = ['uv', 'run', '--extra', 'ovphysx', 'python',\n"
    "               str(project_dir / 'scripts' / 'diagnose_clone_modes.py'),\n"
    "               '--device', 'cuda:0', '--mode', mode, '--num-envs', '8', '--steps', '24', '--seed', '7']\n"
    "    if mode == 'D':\n"
    "        command += ['--nofix-usd', report['nofix_usd_path']]\n"
    "    try:\n"
    "        result = subprocess.run(command, text=True, capture_output=True, timeout=1200, cwd=source_dir, env=mode_env)\n"
    "        lines = [line for line in result.stdout.splitlines() if line.startswith('VERDICT_JSON:')]\n"
    "        if lines:\n"
    "            verdicts[mode] = json.loads(lines[-1][len('VERDICT_JSON:'):])\n"
    "        else:\n"
    "            verdicts[mode] = {'mode': mode, 'stage': 'subprocess', 'error': 'no verdict line',\n"
    "                              'stdout_tail': result.stdout[-3000:], 'stderr_tail': result.stderr[-3000:],\n"
    "                              'exit_code': result.returncode}\n"
    "    except subprocess.TimeoutExpired:\n"
    "        verdicts[mode] = {'mode': mode, 'stage': 'subprocess', 'error': 'timeout 1200s'}\n"
    "    print(f\"mode {mode}: stage={verdicts[mode].get('stage')} error={str(verdicts[mode].get('error'))[:200]}\")\n"
    "report['verdicts'] = verdicts\n"
    "save_report()\n"
    "for mode, verdict in verdicts.items():\n"
    "    bad_bodies = []\n"
    "    for name, info in (verdict.get('bodies_after_steps') or {}).items():\n"
    "        if not info['finite_all']:\n"
    "            bad_bodies.append(f\"{name}@envs{info['bad_envs']}\")\n"
    "    print(f\"{mode}: stage={verdict.get('stage')} reward_finite={verdict.get('reward_finite')} \"\n"
    "          f\"bad_bodies_after_steps={bad_bodies} error={str(verdict.get('error'))[:160]}\")\n"
    "print(json.dumps({m: {'stage': v.get('stage'), 'error': (v.get('error') or '')[:300]} for m, v in verdicts.items()}, indent=2))\n"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle-user", default="pascalcoimbra")
    args = parser.parse_args()
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    notebook = {
        "cells": [
            markdown(
                "# FANUC clone-mode diagnosis\n\n"
                "Runs scripts/diagnose_clone_modes.py modes A-E against the pinned "
                "Isaac Lab 3.0 OvPhysX stack to find a multi-env configuration where "
                "contact sensors bind in every env AND body transforms stay finite. "
                "Diagnostic only: no baseline, no PPO training.\n"
            ),
            code(PREFLIGHT),
            code(BUNDLE.format(bundle=bundled_project())),
            code(SETUP),
            code(MODES),
            code(
                "from IPython.display import FileLink, display\n"
                "display(FileLink('/kaggle/working/clone-modes-report.json'))\n"
                "print('verdicts:', list(report.get('verdicts', {})))\n"
            ),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path = DIAG_DIR / NOTEBOOK_NAME
    notebook_path.write_text(json.dumps(notebook, indent=1) + "\n")
    metadata = {
        "id": f"{args.kaggle_user}/{KERNEL_SLUG}",
        "title": "FANUC Diagnose Clone Modes",
        "code_file": NOTEBOOK_NAME,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": True,
    }
    (DIAG_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(notebook_path)


if __name__ == "__main__":
    main()
