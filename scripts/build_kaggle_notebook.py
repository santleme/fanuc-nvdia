"""Bundle the local project into a self-contained Kaggle notebook."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_DIR = ROOT / "kaggle"
NOTEBOOK_NAME = "fanuc-conveyor-pick-place.ipynb"
KERNEL_SLUG = "fanuc-conveyor-pick-place"


def bundled_project() -> str:
    payload = io.BytesIO()
    included = [
        "pyproject.toml",
        "assets/cell_geometry.json",
        "assets/m10id12_robot.urdf",
        "src/fanuc_cell/__init__.py",
        "src/fanuc_cell/benchmark_stats.py",
        "src/fanuc_cell/kinematics.py",
        "src/fanuc_cell/task.py",
        "src/fanuc_cell/render.py",
        "src/fanuc_cell/demos.py",
        "src/fanuc_cell/bc.py",
        "src/fanuc_cell/rerun_log.py",
        "scripts/render_demo.py",
        "scripts/run_scripted_demo.py",
        "scripts/diagnose_training_nan.py",
        "scripts/collect_demos.py",
        "scripts/pretrain_bc.py",
        "scripts/train_ppo_bc.py",
        "scripts/eval_bc_actor.py",
        "tests/test_kinematics.py",
        "tests/test_benchmark_stats.py",
        "tests/test_task.py",
        "tests/test_static_clearance.py",
        "tests/test_rerun_log.py",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle-user", default="INSERT_KAGGLE_USERNAME")
    parser.add_argument(
        "--accelerator",
        choices=("NvidiaL4", "NvidiaTeslaT4"),
        default="NvidiaTeslaT4",
        help="Kaggle GPU shape; L4 is optional when the account receives it.",
    )
    parser.add_argument(
        "--physics",
        choices=("ovphysx", "newton_mjwarp"),
        default="ovphysx",
        help="Backend de física do Isaac Lab 3.0 (preset da tarefa). "
             "ovphysx = linha principal; newton = spike MuJoCo-Warp (issue #2).",
    )
    parser.add_argument("--train-num-envs", type=int, default=64)
    parser.add_argument("--train-max-iterations", type=int, default=500)
    parser.add_argument(
        "--benchmark-seeds",
        default="101,202,303",
        help="Seeds do benchmark oficial multi-seed (B3, issue #10), "
             "separadas por vírgula. Default = o protocolo pré-"
             "registrado (101,202,303). Uma seed só reproduz a "
             "leitura legada de seed única (std/IC95 = 0).",
    )
    parser.add_argument(
        "--kernel-slug",
        default=KERNEL_SLUG,
        help="Slug do kernel no Kaggle; o spike Newton usa slug próprio "
             "para não virar versão do kernel principal.",
    )
    parser.add_argument("--kernel-title", default="FANUC Conveyor Pick Place")
    parser.add_argument(
        "--bc-warmstart", action="store_true",
        help="Experimento BC (issue #3): coleta demos do professor, "
             "pré-treina o ator e treina o PPO pela entrada warm-start. "
             "Default desligado — o kernel normal não muda.",
    )
    parser.add_argument("--bc-episodes", type=int, default=200,
                        help="Episódios de demonstração coletados (com --bc-warmstart).")
    parser.add_argument("--bc-exp1", action="store_true",
                        help="Experimento 1 da issue #8 (exige "
                             "--bc-warmstart): Estágio A avalia o ator "
                             "BC puro em 200 episódios com seeds novas "
                             "(2000+); Estágio B roda o PPO warm-start "
                             "PROTEGIDO (entropia 0, crítico aquecido "
                             "com o ator congelado por 50 iterações).")
    parser.add_argument(
        "--obs-history", type=int, default=1,
        help="Experimento OBS-2 (issue #8): empilha os ultimos N "
             "frames na observacao da politica "
             "(FANUC_CELL_OBS_HISTORY=N no processo do notebook e no "
             "baseline_env; observacao vira 24*N dims). Default 1 = "
             "frame unico — notebook byte a byte o de sempre. Coleta, "
             "BC, treino e benchmark do mesmo kernel compartilham o N "
             "por construcao; demos/checkpoints nao sao "
             "intercambiaveis entre valores de N.",
    )
    parser.add_argument(
        "--cinematic-video", action="store_true",
        help="Vídeo industrial fatia 1 (issue #9): SO o playback muda "
             "— viz newton_gl -> newton_rtx (path tracer OVRTX; extra "
             "'ovrtx' do workspace pinado v3.0.0, mesmo mecanismo dos "
             "extras 'ovphysx'/'video') e FANUC_CELL_CINEMATIC=1 no "
             "env do play (cena cinematic do env_cfg). Treino e "
             "benchmark não mudam. Default desligado.",
    )
    args = parser.parse_args()
    if args.bc_warmstart and args.physics != "ovphysx":
        parser.error("--bc-warmstart v1 é só na linha principal (ovphysx)")
    if args.bc_exp1 and not args.bc_warmstart:
        parser.error("--bc-exp1 exige --bc-warmstart")
    if args.obs_history < 1:
        parser.error("--obs-history deve ser >= 1 (1 = frame unico)")
    if args.bc_warmstart and args.train_num_envs == 64 \
            and args.train_max_iterations == 500:
        # Regressao da BC v6: sem as flags, o treino warm-start saiu
        # 64x500 (defaults da linha principal) em vez dos 256x1500 do
        # protocolo — a corrida inteira ficou fora do protocolo.
        parser.error("--bc-warmstart exige a escala explicita do "
                     "protocolo: --train-num-envs 256 "
                     "--train-max-iterations 1500")
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)

    notebook = {
        "cells": [
            markdown(
                "# FANUC-style conveyor pick-and-place\n\n"
                "Reproducible kinematic baseline for a six-axis M-10iD/12-inspired arm. "
                "The link geometry is an explicitly approximate proxy, not FANUC CAD or a "
                "dynamics-validated digital twin. This kernel records the Kaggle GPU/driver, "
                "runs focused package checks, and then attempts the pinned "
                "Isaac Lab 3.0 OvPhysX smoke test when the runtime driver meets the "
                "documented CUDA 13 minimum.\n"
            ),
            code(
                "from __future__ import annotations\n"
                "import json, os, platform, re, subprocess, sys, time\n"
                "from pathlib import Path\n\n"
                "working_dir = Path('/kaggle/working')\n"
                "working_dir.mkdir(exist_ok=True)\n"
                "report_path = working_dir / 'runtime-report.json'\n"
                "report = {'commands': []}\n"
                "RUN_BACKEND_SMOKES = False\n"
                "stage_started = {}\n"
                "def stage_start(name):\n"
                "    stage_started[name] = time.perf_counter()\n"
                "def stage_end(name):\n"
                "    elapsed = round(time.perf_counter() - stage_started.pop(name), 3)\n"
                "    timings = report.setdefault('timings_seconds', {})\n"
                "    timings[name] = elapsed\n"
                "    print(f'[PERF] stage={name} seconds={elapsed}')\n"
                "    latest = globals().get('runtime_report')\n"
                "    if isinstance(latest, dict):\n"
                "        latest.setdefault('timings_seconds', {}).update(timings)\n"
                "    save_report()\n"
                "def save_report():\n"
                "    latest = globals().get('runtime_report')\n"
                "    if isinstance(latest, dict):\n"
                "        report.update(latest)\n"
                "    report_path.write_text(json.dumps(report, indent=2))\n\n"
                "def run(command, **kwargs):\n"
                "    started = time.perf_counter()\n"
                "    result = subprocess.run(command, text=True, capture_output=True, check=False, **kwargs)\n"
                "    elapsed = round(time.perf_counter() - started, 3)\n"
                "    report.setdefault('commands', []).append({'argv': command, 'exit_code': result.returncode,\n"
                "        'duration_seconds': elapsed,\n"
                "        'stdout_tail': result.stdout[-8000:], 'stderr_tail': result.stderr[-8000:]})\n"
                "    print(f'[PERF] command_seconds={elapsed} exit={result.returncode} command={command[0]}')\n"
                "    save_report()\n"
                "    return result\n\n"
                "stage_start('gpu_preflight')\n"
                "gpu = run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'])\n"
                "driver = run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'])\n"
                "report.update({'python': sys.version, 'platform': platform.platform(), 'gpu_query_exit': gpu.returncode,\n"
                "          'gpus': [line.strip() for line in gpu.stdout.splitlines() if line.strip()],\n"
                "          'driver_versions': sorted(set(line.strip() for line in driver.stdout.splitlines() if line.strip()))})\n"
                "report['gpu_count'] = len(report['gpus'])\n"
                "report['kaggle_t4_detected'] = any('t4' in line.lower() for line in report['gpus'])\n"
                "print(json.dumps(report, indent=2))\n"
                "save_report()\n"
                "if gpu.returncode != 0 or not report['gpus']:\n"
                "    raise RuntimeError('Kaggle session has no visible NVIDIA GPU; select GPU accelerator.')\n"
                "caps = []\n"
                "try:\n"
                "    import torch\n"
                "    report['torch'] = torch.__version__\n"
                "    for device in range(torch.cuda.device_count()):\n"
                "        caps.append(torch.cuda.get_device_capability(device))\n"
                "    report['torch_cuda_devices'] = torch.cuda.device_count()\n"
                "    report['compute_capabilities'] = [list(cap) for cap in caps]\n"
                "except Exception as exc:\n"
                "    report['torch_probe_error'] = repr(exc)\n"
                "driver_text = report['driver_versions'][0] if report['driver_versions'] else '0'\n"
                "match = re.match(r'(\\d+)\\.(\\d+)(?:\\.(\\d+))?', driver_text)\n"
                "driver_tuple = tuple(map(int, match.groups(default='0'))) if match else (0, 0, 0)\n"
                "report['isaaclab_linux_driver_compatible'] = driver_tuple >= (580, 65, 6)\n"
                "report['isaaclab_gpu_arch_compatible'] = bool(caps) and all(cap >= (7, 5) for cap in caps)\n"
                "Path('/kaggle/working').mkdir(exist_ok=True)\n"
                "Path('/kaggle/working/runtime-report.json').write_text(json.dumps(report, indent=2))\n"
                "stage_end('gpu_preflight')\n"
            ),
            code(
                "import base64, io, zipfile\n"
                "import importlib.util, os, shutil\n"
                "from pathlib import Path\n"
                f"project_zip_b64 = {bundled_project()!r}\n"
                "with zipfile.ZipFile(io.BytesIO(base64.b64decode(project_zip_b64))) as archive:\n"
                "    archive.extractall('/kaggle/working/fanuc-cell')\n"
                "%cd /kaggle/working/fanuc-cell\n"
                "stage_start('local_setup')\n"
                "project_src = str(Path.cwd() / 'src')\n"
                "os.environ['PYTHONPATH'] = project_src + os.pathsep + os.environ.get('PYTHONPATH', '')\n"
                "if importlib.util.find_spec('pytest') is None:\n"
                "    pytest_install = run([sys.executable, '-m', 'pip', 'install', 'pytest>=8'])\n"
                "    if pytest_install.returncode != 0:\n"
                "        raise RuntimeError('Could not install pytest: ' + pytest_install.stderr[-4000:])\n"
                "stage_end('local_setup')\n"
            ),
            code(
                "import json, subprocess\n"
                "from pathlib import Path\n"
                "stage_start('local_tests')\n"
                "test_env = os.environ.copy()\n"
                "test_env['PYTHONPATH'] = str(Path.cwd() / 'src') + os.pathsep + test_env.get('PYTHONPATH', '')\n"
                "subprocess.run([sys.executable, '-m', 'pytest', '-q'], check=True, env=test_env)\n"
                "stage_end('local_tests')\n"
            ),
            markdown(
                "## Isaac Lab custom task\n\n"
                "On a compatible T4 runtime, this installs Isaac Lab 3.0.0, runs the "
                "official kit-less PhysX smoke, generates the official external manager-based "
                "PPO project, imports the six-axis arm URDF into USD, and steps the conveyor "
                "task with a zero-action agent before training PPO for 500 iterations. "
                "It then plays the checkpoint for one episode while recording a headless "
                "headless video. The report keeps startup, conversion, training, playback, "
                "and episode evidence distinct; video alone is not a success-rate claim.\n"
            ),
            code(
                "import shutil\n"
                "from pathlib import Path\n"
                "runtime_report = json.loads(Path('/kaggle/working/runtime-report.json').read_text())\n"
                "if runtime_report['isaaclab_gpu_arch_compatible'] and runtime_report['isaaclab_linux_driver_compatible']:\n"
                "    uv_probe = run(['uv', '--version'])\n"
                "    if uv_probe.returncode != 0:\n"
                "        install_uv = run([sys.executable, '-m', 'pip', 'install', 'uv'])\n"
                "        if install_uv.returncode != 0:\n"
                "            raise RuntimeError(install_uv.stderr[-4000:])\n"
                "    source_dir = Path('/kaggle/working/IsaacLab')\n"
                "    clone = run(['git', 'clone', '--depth', '1', '--branch', 'v3.0.0',\n"
                "                 'https://github.com/isaac-sim/IsaacLab.git', str(source_dir)])\n"
                "    if clone.returncode != 0:\n"
                "        raise RuntimeError('Could not clone Isaac Lab v3.0.0: ' + clone.stderr[-4000:])\n"
                "    sync = run(['uv', 'sync'], cwd=source_dir)\n"
                "    if sync.returncode != 0:\n"
                "        raise RuntimeError('Isaac Lab dependency installation failed: ' + sync.stderr[-6000:])\n"
                "    if RUN_BACKEND_SMOKES:\n"
                "        smoke = run(['uv', 'run', '--extra', 'ovphysx', 'isaaclab', 'zero_agent', '--task', 'Isaac-Cartpole-Direct',\n"
                "                     '--num_envs', '1', '--max_steps', '64', '--viz', 'none',\n"
                "                     'physics=ovphysx'], cwd=source_dir)\n"
                "        runtime_report['isaaclab_smoke_exit'] = smoke.returncode\n"
                "        runtime_report['isaaclab_smoke_stdout'] = smoke.stdout[-6000:]\n"
                "        runtime_report['isaaclab_smoke_stderr'] = smoke.stderr[-6000:]\n"
                "        print(smoke.stdout[-6000:])\n"
                "        if smoke.returncode != 0:\n"
                "            raise RuntimeError('Isaac Lab OvPhysX smoke test failed; see runtime report.')\n"
                "    else:\n"
                "        runtime_report['isaaclab_smoke_exit'] = None\n"
                "        runtime_report['isaaclab_smoke_skip'] = 'Previously validated; full custom baseline runs below.'\n"
                "    project_dir = Path('/kaggle/working/fanuc_cell_lab')\n"
                "    generate = run(['uv', 'run', 'isaaclab', '--new', '--non_interactive',\n"
                "                   '--task_type', 'external', '--project_path', '/kaggle/working',\n"
                "                   '--name', 'fanuc_cell_lab', '--author', 'Pascal Coimbra',\n"
                "                   '--task_name', 'conveyor_pick_place', '--robot_name', 'm10id12',\n"
                "                   '--workflow', 'manager-based:single-agent',\n"
                "                   '--rl_library', 'rsl_rl', '--rl_algorithm', 'ppo'], cwd=source_dir)\n"
                "    if generate.returncode != 0 or not project_dir.is_dir():\n"
                "        raise RuntimeError('Isaac Lab project generation failed: ' + generate.stderr[-6000:])\n"
                "    overlay = Path('/kaggle/working/fanuc-cell/isaaclab_overlay/src/fanuc_cell_lab')\n"
                "    project_package = project_dir / 'src' / 'fanuc_cell_lab'\n"
                "    shutil.copytree(overlay, project_package, dirs_exist_ok=True)\n"
                "    shutil.copytree('/kaggle/working/fanuc-cell/src/fanuc_cell',\n"
                "                    project_dir / 'src' / 'fanuc_cell', dirs_exist_ok=True)\n"
                "    shutil.copy2('/kaggle/working/fanuc-cell/scripts/run_scripted_demo.py',\n"
                "                 project_dir / 'scripts' / 'run_scripted_demo.py')\n"
                "    shutil.copy2('/kaggle/working/fanuc-cell/scripts/diagnose_training_nan.py',\n"
                "                 project_dir / 'scripts' / 'diagnose_training_nan.py')\n"
                "    asset_dir = project_package / 'assets' / 'data'\n"
                "    asset_dir.mkdir(parents=True, exist_ok=True)\n"
                "    converted_asset = asset_dir / 'm10id12_robot' / 'm10id12_robot.usd'\n"
                "    official_converter = (source_dir / 'scripts/tools/convert_urdf.py').read_text()\n"
                "    converted_source, replacements = re.subn(\n"
                "        r'(args_cli\\.physics\\s*=\\s*\\\"isaacsim_physx\\\"\\s+if\\s+args_cli\\.require_kit\\s+else\\s+)\\\"[^\\\"]+\\\"',\n"
                "        r'\\1\"ovphysx\"', official_converter, count=1)\n"
                "    if replacements != 1:\n"
                "        raise RuntimeError('Pinned Isaac Lab URDF converter changed its backend selection; inspect before conversion.')\n"
                "    converter_script = source_dir / 'scripts/tools/convert_urdf_ovphysx.py'\n"
                "    converter_script.write_text(converted_source)\n"
                "    conversion = run(['uv', 'run', '--extra', 'importers', '--extra', 'ovphysx', 'isaaclab', '-p',\n"
                "                     str(converter_script),\n"
                "                     '/kaggle/working/fanuc-cell/assets/m10id12_robot.urdf',\n"
                "                     str(asset_dir), '--joint-stiffness', '0.0',\n"
                "                     '--joint-damping', '0.0', '--joint-target-type', 'none'], cwd=source_dir)\n"
                "    runtime_report['urdf_conversion_exit'] = conversion.returncode\n"
                "    runtime_report['urdf_conversion_stdout'] = conversion.stdout[-6000:]\n"
                "    runtime_report['urdf_conversion_stderr'] = conversion.stderr[-6000:]\n"
                "    if conversion.returncode != 0 or not converted_asset.is_file():\n"
                "        candidates = sorted(asset_dir.rglob('m10id12_robot.usd*'))\n"
                "        if conversion.returncode != 0 or not candidates:\n"
                "            raise RuntimeError('Isaac Lab URDF-to-USD conversion failed: ' + conversion.stderr[-6000:])\n"
                "        source_usd = candidates[0]\n"
                "        converted_asset.parent.mkdir(parents=True, exist_ok=True)\n"
                "        if source_usd.resolve() != converted_asset.resolve():\n"
                "            shutil.copy2(source_usd, converted_asset)\n"
                "    contact_code = (\n"
                "        'import json,sys; from pxr import Usd,UsdPhysics,Sdf; '\n"
                "        'stage=Usd.Stage.Open(sys.argv[1]); '\n"
                "        'bodies=[p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]; '\n"
                "        \"[(p.AddAppliedSchema('PhysxRigidBodyAPI'), p.AddAppliedSchema('PhysxContactReportAPI'), \"\n"
                "        \" p.CreateAttribute('physxRigidBody:sleepThreshold', Sdf.ValueTypeNames.Float, custom=False).Set(0.0), \"\n"
                "        \" p.CreateAttribute('physxContactReport:threshold', Sdf.ValueTypeNames.Float, custom=False).Set(0.0)) \"\n"
                "        \" for p in bodies]; \"\n"
                "        \"stage.GetRootLayer().Save(); print(json.dumps({'contact_report_bodies':len(bodies)}))\"\n"
                "    )\n"
                "    contact_patch = run(['uv', 'run', 'python', '-c', contact_code, str(converted_asset)], cwd=source_dir)\n"
                "    if contact_patch.returncode != 0:\n"
                "        raise RuntimeError('Failed to enable contact reporting in converted robot USD: ' + contact_patch.stderr[-4000:])\n"
                "    runtime_report['usd_contact_patch'] = json.loads(contact_patch.stdout.strip().splitlines()[-1])\n"
                "    if runtime_report['usd_contact_patch']['contact_report_bodies'] < 1:\n"
                "        raise AssertionError('No rigid bodies found while enabling USD contact reporting.')\n"
                "    inspect_code = (\n"
                "        'import json,sys; from pxr import Usd,UsdPhysics; '\n"
                "        'stage=Usd.Stage.Open(sys.argv[1]); ps=list(stage.Traverse()); '\n"
                "        'r=sum(p.IsA(UsdPhysics.RevoluteJoint) for p in ps); '\n"
                "        'j=sum(p.IsA(UsdPhysics.PrismaticJoint) for p in ps); '\n"
                "        'b=[str(p.GetPath()) for p in ps if p.HasAPI(UsdPhysics.RigidBodyAPI)]; '\n"
                "        'print(json.dumps({\\\"revolute_joints\\\":int(r),\\\"prismatic_joints\\\":int(j),\\\"rigid_bodies\\\":b}))'\n"
                "    )\n"
                "    inspect = run(['uv', 'run', 'python', '-c', inspect_code, str(converted_asset)], cwd=source_dir)\n"
                "    if inspect.returncode != 0:\n"
                "        raise RuntimeError('USD articulation inspection failed: ' + inspect.stderr[-4000:])\n"
                "    runtime_report['converted_usd'] = str(converted_asset)\n"
                "    usd_summary = json.loads(inspect.stdout.strip().splitlines()[-1])\n"
                "    runtime_report['usd_joint_counts'] = {key: usd_summary[key] for key in ('revolute_joints', 'prismatic_joints')}\n"
                "    runtime_report['usd_rigid_bodies'] = usd_summary['rigid_bodies']\n"
                "    if runtime_report['usd_joint_counts'] != {'revolute_joints': 6, 'prismatic_joints': 2}:\n"
                "        raise AssertionError('Converted articulation must contain 6 revolute + 2 gripper slide joints: ' + str(runtime_report['usd_joint_counts']))\n"
                "    if not any('tool0' in path.lower() for path in runtime_report['usd_rigid_bodies']):\n"
                "        raise AssertionError('Converted USD is missing the tool0 rigid body: ' + str(runtime_report['usd_rigid_bodies']))\n"
                "    baseline_env = os.environ.copy()\n"
                "    baseline_env['PYTHONPATH'] = str(project_dir / 'src') + os.pathsep + baseline_env.get('PYTHONPATH', '')\n"
                "    baseline_env['FANUC_CELL_PHYSICS'] = 'ovphysx'\n"
                "    nan_probe = run(['uv', 'run', '--extra', 'ovphysx', 'python',\n"
                "                    str(project_dir / 'scripts' / 'diagnose_training_nan.py'),\n"
                "                    '--device', 'cuda:0', '--num-envs', '64', '--steps', '24', '--seed', '7'],\n"
                "                   cwd=source_dir, env=baseline_env)\n"
                "    runtime_report['ppo_nan_probe_exit'] = nan_probe.returncode\n"
                "    runtime_report['ppo_nan_probe_stdout'] = nan_probe.stdout[-8000:]\n"
                "    runtime_report['ppo_nan_probe_stderr'] = nan_probe.stderr[-4000:]\n"
                "    print(nan_probe.stdout[-8000:])\n"
                "    if nan_probe.returncode != 0:\n"
                "        save_report()\n"
                "        raise RuntimeError('PPO environment probe found non-finite observations. stdout:\\n' + nan_probe.stdout[-8000:] + '\\nstderr:\\n' + nan_probe.stderr[-4000:])\n"
                "    baseline = run(['uv', 'run', '--extra', 'ovphysx', 'python',\n"
                "                    str(project_dir / 'scripts' / 'run_scripted_demo.py'),\n"
                "                    '--device', 'cuda:0', '--seed', '23',\n"
                "                    '--output', '/kaggle/working/isaaclab-baseline.json'],\n"
                "                   cwd=source_dir, env=baseline_env)\n"
                "    runtime_report['scripted_baseline_exit'] = baseline.returncode\n"
                "    runtime_report['scripted_baseline_stdout'] = baseline.stdout[-8000:]\n"
                "    runtime_report['scripted_baseline_stderr'] = baseline.stderr[-8000:]\n"
                "    baseline_metrics = Path('/kaggle/working/isaaclab-baseline.json')\n"
                "    if baseline_metrics.is_file():\n"
                "        runtime_report['scripted_baseline'] = json.loads(baseline_metrics.read_text())\n"
                "    if baseline.returncode != 0 or not baseline_metrics.is_file():\n"
                "        save_report()\n"
                "        raise RuntimeError('Scripted physical baseline did not complete. stdout:\\n' + baseline.stdout[-12000:] + '\\nstderr head:\\n' + baseline.stderr[:8000] + '\\nstderr tail:\\n' + baseline.stderr[-2000:])\n"
                "    baseline_result = runtime_report['scripted_baseline']\n"
                "    if not baseline_result.get('success') or not baseline_result.get('scripted_grasp_sensor_passed'):\n"
                "        save_report()\n"
                "        raise RuntimeError('Isaac Lab physical baseline failed grasp/place acceptance; PPO training is gated until it passes: ' + str(baseline_result))\n"
                "    task_sync = run(['uv', 'sync'], cwd=project_dir)\n"
                "    if task_sync.returncode != 0:\n"
                "        raise RuntimeError('Custom Isaac Lab project installation failed: ' + task_sync.stderr[-6000:])\n"
                "    task_query = run(['uv', 'run', 'python', '-c',\n"
                "                      'import json, gymnasium as gym; import fanuc_cell_lab.tasks; '\n"
                "                      'print(json.dumps([k for k in gym.registry if \\\"conveyor\\\" in k.lower() and \\\"m10id12\\\" in k.lower()]))'],\n"
                "                     cwd=project_dir)\n"
                "    if task_query.returncode != 0:\n"
                "        raise RuntimeError('Custom task registration failed: ' + task_query.stderr[-6000:])\n"
                "    task_ids = json.loads(task_query.stdout.strip().splitlines()[-1])\n"
                "    if len(task_ids) != 1:\n"
                "        raise RuntimeError('Expected one registered conveyor task, got: ' + str(task_ids))\n"
                "    task_id = task_ids[0]\n"
                "    runtime_report['cell_task_id'] = task_id\n"
                "    training = run(['uv', 'run', '--extra', 'ovphysx', 'isaaclab', 'train', '--rl_library', 'rsl_rl',\n"
                "                    '--task', task_id, '--num_envs', '64', '--max_iterations', '500',\n"
                "                    '--seed', '7', 'physics=ovphysx'], cwd=project_dir)\n"
                "    runtime_report['ppo_training_exit'] = training.returncode\n"
                "    runtime_report['ppo_training_stdout'] = training.stdout[-12000:]\n"
                "    runtime_report['ppo_training_stderr'] = training.stderr[-12000:]\n"
                "    checkpoints = sorted(project_dir.glob('logs/rsl_rl/**/*.pt'), key=lambda p: p.stat().st_mtime)\n"
                "    runtime_report['ppo_checkpoints'] = [{'path': str(p), 'bytes': p.stat().st_size} for p in checkpoints[-3:]]\n"
                "    if training.returncode != 0:\n"
                "        save_report()\n"
                "        raise RuntimeError('RSL-RL PPO training failed. stdout:\\n' + training.stdout[-8000:] + '\\nstderr:\\n' + training.stderr[-12000:])\n"
                "    if not checkpoints:\n"
                "        raise RuntimeError('PPO exited without producing an RSL-RL checkpoint.')\n"
                "    playback = run(['uv', 'run', '--extra', 'ovphysx', '--extra', 'video', 'isaaclab', 'play',\n"
                "                    '--rl_library', 'rsl_rl', '--task', task_id,\n"
                "                    '--checkpoint', 'latest', '--num_envs', '1', '--seed', '107',\n"
                "                    '--viz', 'newton_gl', '--video', '--video_length', '400',\n"
                "                    'physics=ovphysx'], cwd=project_dir)\n"
                "    runtime_report['policy_playback_exit'] = playback.returncode\n"
                "    runtime_report['policy_playback_stdout'] = playback.stdout[-8000:]\n"
                "    runtime_report['policy_playback_stderr'] = playback.stderr[-8000:]\n"
                "    if playback.returncode != 0:\n"
                "        save_report()\n"
                "        raise RuntimeError('Trained-policy playback/video failed: ' + playback.stderr[-8000:])\n"
                "    policy_videos = sorted(project_dir.glob('logs/rsl_rl/**/videos/**/*.mp4'),\n"
                "                           key=lambda p: p.stat().st_mtime)\n"
                "    if not policy_videos:\n"
                "        save_report()\n"
                "        raise RuntimeError('Trained-policy playback completed without an MP4 artifact.')\n"
                "    policy_video = Path('/kaggle/working/fanuc-cell-isaaclab-policy.mp4')\n"
                "    shutil.copy2(policy_videos[-1], policy_video)\n"
                "    runtime_report['policy_video'] = str(policy_video)\n"
                "    runtime_report['policy_video_bytes'] = policy_video.stat().st_size\n"
                "    # Benchmark multi-seed (B3, issue #10): seed unica nao\n"
                "    # separa sinal de ruido — o veredito oficial passa a ser\n"
                "    # media +- desvio das seeds fixas (default do protocolo:\n"
                "    # 101, 202, 303; --benchmark-seeds troca a lista). Por\n"
                "    # seed, success_rate (portao) e grasp_rate (degrau\n"
                "    # intermediario, quando o benchmark o reporta).\n"
                "    benchmark_seeds = (101, 202, 303)\n"
                "    seed_rates = []\n"
                "    seed_grasp_rates = []\n"
                "    runtime_report['policy_evaluation_runs'] = []\n"
                "    for _seed in benchmark_seeds:\n"
                "        evaluation_dir = Path(f'/kaggle/working/policy-evaluation-{_seed}')\n"
                "        evaluation = run(['uv', 'run', '--extra', 'ovphysx', 'isaaclab', 'benchmark', 'play',\n"
                "                          '--rl_library', 'rsl_rl', '--task', task_id,\n"
                "                          '--checkpoint', 'latest', '--num_envs', '16',\n"
                "                          '--num_steps', '800', '--seed', str(_seed),\n"
                "                          '--viz', 'none', '--benchmark_formatter', 'schema,summary',\n"
                "                          '--output_path', str(evaluation_dir), 'physics=ovphysx'],\n"
                "                         cwd=project_dir)\n"
                "        runtime_report['policy_evaluation_exit'] = evaluation.returncode\n"
                "        runtime_report['policy_evaluation_runs'].append({\n"
                "            'seed': _seed, 'exit': evaluation.returncode,\n"
                "            'stdout_tail': evaluation.stdout[-3000:],\n"
                "            'stderr_tail': evaluation.stderr[-3000:]})\n"
                "        if evaluation.returncode != 0:\n"
                "            save_report()\n"
                "            raise RuntimeError('Trained-policy evaluation failed: ' + evaluation.stderr[-10000:])\n"
                "        evaluation_results = sorted(evaluation_dir.rglob('*.json'), key=lambda p: p.stat().st_mtime)\n"
                "        if not evaluation_results:\n"
                "            runtime_report['policy_evaluation_artifact_missing'] = True\n"
                "            save_report()\n"
                "            raise RuntimeError('Isaac Lab benchmark did not produce a JSON evaluation result.')\n"
                "        evaluation_data = [json.loads(path.read_text()) for path in evaluation_results]\n"
                "        runtime_report['policy_evaluation'] = evaluation_data[-1]\n"
                "        _rate = None\n"
                "        _grasp = None\n"
                "        for _data in evaluation_data:\n"
                "            for _record in (_data if isinstance(_data, list) else [_data]):\n"
                "                if not isinstance(_record, dict):\n"
                "                    continue\n"
                "                if _record.get('success_rate') is not None:\n"
                "                    _rate = _record['success_rate']\n"
                "                if _record.get('grasp_rate') is not None:\n"
                "                    _grasp = _record['grasp_rate']\n"
                "                for _m in (_record.get('measurements') or []):\n"
                "                    if isinstance(_m, dict) and str(_m.get('name', '')).endswith('play success_rate'):\n"
                "                        _rate = _m.get('value')\n"
                "                    if isinstance(_m, dict) and str(_m.get('name', '')).endswith('play grasp_rate') and _m.get('value') is not None:\n"
                "                        _grasp = _m.get('value')\n"
                "        if _rate is None:\n"
                "            save_report()\n"
                "            raise RuntimeError('Isaac Lab benchmark did not report Metrics/success_rate for completed episodes.')\n"
                "        seed_rates.append(_rate)\n"
                "        seed_grasp_rates.append(_grasp)\n"
                "        runtime_report['policy_evaluation_runs'][-1]['success_rate'] = _rate\n"
                "        runtime_report['policy_evaluation_runs'][-1]['grasp_rate'] = _grasp\n"
                "    # Agregacao B3: matematica unica e testada em\n"
                "    # fanuc_cell.benchmark_stats (media, desvio amostral\n"
                "    # ddof=1, IC95 da media). O src do projeto gerado ja\n"
                "    # foi copiado para project_dir/src nesta celula.\n"
                "    import sys as _sys\n"
                "    if str(project_dir / 'src') not in _sys.path:\n"
                "        _sys.path.insert(0, str(project_dir / 'src'))\n"
                "    from fanuc_cell.benchmark_stats import format_rate_summary, summarize_rates\n"
                "    _summary = summarize_rates(seed_rates)\n"
                "    runtime_report['policy_benchmark_seeds'] = list(benchmark_seeds)\n"
                "    runtime_report['policy_success_rates'] = seed_rates\n"
                "    runtime_report['policy_success_rate'] = _summary['mean']\n"
                "    runtime_report['policy_success_rate_std'] = _summary['std']\n"
                "    runtime_report['policy_success_rate_ci95'] = _summary['ci95_half']\n"
                "    print('BENCHMARK B3 success_rate: ' + format_rate_summary(_summary))\n"
                "    runtime_report['policy_grasp_rates'] = seed_grasp_rates\n"
                "    _grasp_known = [g for g in seed_grasp_rates if g is not None]\n"
                "    if _grasp_known:\n"
                "        _gsummary = summarize_rates(_grasp_known)\n"
                "        runtime_report['policy_grasp_rate'] = _gsummary['mean']\n"
                "        runtime_report['policy_grasp_rate_std'] = _gsummary['std']\n"
                "        runtime_report['policy_grasp_rate_ci95'] = _gsummary['ci95_half']\n"
                "        print('BENCHMARK B3 grasp_rate: ' + format_rate_summary(_gsummary))\n"
                "    else:\n"
                "        runtime_report['policy_grasp_rate'] = None\n"
                "        print('BENCHMARK B3 grasp_rate: nao reportado pelo benchmark (Metrics/grasp_rate ausente); success_rate segue como portao.')\n"
                "else:\n"
                "    runtime_report['isaaclab_smoke_exit'] = None\n"
                "    runtime_report['isaaclab_smoke_skip'] = 'GPU capability or documented Linux driver minimum failed.'\n"
                "    print(runtime_report['isaaclab_smoke_skip'])\n"
                "Path('/kaggle/working/runtime-report.json').write_text(json.dumps(runtime_report, indent=2))\n"
            ),
            code(
                "from IPython.display import FileLink, display\n"
                "for artifact in ['/kaggle/working/fanuc-cell-isaaclab-policy.mp4', '/kaggle/working/runtime-report.json']:\n"
                "    display(FileLink(artifact))\n"
                "print(json.loads(Path('/kaggle/working/runtime-report.json').read_text()))\n"
            ),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    # Parametrização do experimento (v68 / spike Newton): as células
    # nascem com os defaults históricos (ovphysx, 64 envs, 500 it) e
    # os tokens são trocados aqui, num lugar só — com os defaults, a
    # saída é byte a byte a de sempre. O token de física cobre os
    # comandos isaaclab e o backend forçado no conversor URDF→USD
    # (no caminho Newton, o conversor usa o preset newton).
    # ATENÇÃO (spike v1, erro registrado na SPEC): NÃO existe extra
    # `newton` no optional-dependencies do Isaac Lab — renomear o
    # extra quebrou a conversão ("Extra 'newton' is not defined").
    # O backend Newton é dependência padrão do workspace 3.0 (o viz
    # newton_gl já roda nos kernels OvPhysX), então fora do default
    # o par '--extra', 'ovphysx' é REMOVIDO dos comandos, não trocado.
    if args.physics != "ovphysx" or args.train_num_envs != 64 \
            or args.train_max_iterations != 500:
        for cell in notebook["cells"]:
            source = "".join(cell["source"])
            source = source.replace("physics=ovphysx",
                                    f"physics={args.physics}")
            source = source.replace("'FANUC_CELL_PHYSICS'] = 'ovphysx'",
                                    f"'FANUC_CELL_PHYSICS'] = '{args.physics}'")
            if args.physics != "ovphysx":
                source = source.replace("'--extra', 'ovphysx', ", "")
            source = source.replace('\\1"ovphysx"',
                                    f'\\1"{args.physics}"')
            source = source.replace(
                "'--num_envs', '64', '--max_iterations', '500'",
                f"'--num_envs', '{args.train_num_envs}', "
                f"'--max_iterations', '{args.train_max_iterations}'")
            cell["source"] = source.splitlines(keepends=True)
    # Experimento OBS-2 (issue #8), só com --obs-history > 1: a
    # variável entra no processo do notebook E no baseline_env, para
    # TODOS os subprocessos que criam a tarefa (coleta, eval do BC,
    # treino por qualquer entrada — o isaaclab train herda o environ
    # do notebook, não o baseline_env —, playback e benchmark) verem
    # a mesma história. Dimensão de observação divergente entre
    # coleta e treino quebraria o BC/warm-start silenciosamente.
    # Com o default (1) nada aqui roda e o notebook é o de sempre.
    if args.obs_history > 1:
        anchor = ("    baseline_env['PYTHONPATH'] = str(project_dir / 'src')"
                  " + os.pathsep + baseline_env.get('PYTHONPATH', '')\n")
        injection = (
            anchor
            + f"    os.environ['FANUC_CELL_OBS_HISTORY'] = "
              f"'{args.obs_history}'\n"
            + f"    baseline_env['FANUC_CELL_OBS_HISTORY'] = "
              f"'{args.obs_history}'\n"
            + f"    runtime_report['obs_history'] = {args.obs_history}\n"
        )
        found = False
        for cell in notebook["cells"]:
            source = "".join(cell["source"])
            if anchor in source:
                source = source.replace(anchor, injection, 1)
                found = True
            cell["source"] = source.splitlines(keepends=True)
        if not found:
            raise RuntimeError(
                "anchor do baseline_env nao encontrado para injetar "
                "FANUC_CELL_OBS_HISTORY")
    # Seeds do benchmark multi-seed (B3, issue #10): o default do
    # protocolo (101, 202, 303) já é o literal da célula; outra
    # lista troca o token, num lugar só — mesmo padrão da física.
    # Uma seed só reproduz a leitura legada (std/IC95 = 0).
    try:
        benchmark_seeds = tuple(
            int(part) for part in args.benchmark_seeds.split(",")
            if part.strip()
        )
    except ValueError:
        parser.error(
            "--benchmark-seeds aceita inteiros separados por vírgula.")
    if not benchmark_seeds:
        parser.error("--benchmark-seeds exige ao menos uma seed.")
    if benchmark_seeds != (101, 202, 303):
        rendered = "(" + ", ".join(str(seed) for seed in benchmark_seeds)
        if len(benchmark_seeds) == 1:
            rendered += ","
        rendered += ")"
        for cell in notebook["cells"]:
            source = "".join(cell["source"])
            source = source.replace(
                "benchmark_seeds = (101, 202, 303)",
                f"benchmark_seeds = {rendered}")
            cell["source"] = source.splitlines(keepends=True)
    # Experimento BC (issue #3), só com --bc-warmstart: copia os
    # scripts BC para o projeto gerado, insere coleta+pré-treino
    # antes do treino e troca o comando de treino pela entrada
    # warm-start (scripts/train_ppo_bc.py). Sem a flag, nada aqui roda
    # e o notebook é o de sempre.
    if args.bc_warmstart:
        bc_block = (
            "    bc_collect = run(['uv', 'run', '--extra', 'ovphysx', 'python',\n"
            "                      str(project_dir / 'scripts' / 'collect_demos.py'),\n"
            f"                      '--episodes', '{args.bc_episodes}', '--seed', '1000',\n"
            "                      '--output', '/kaggle/working/bc-demos.npz'],\n"
            "                     cwd=source_dir, env=baseline_env)\n"
            "    runtime_report['bc_collect_exit'] = bc_collect.returncode\n"
            "    runtime_report['bc_collect_stdout'] = bc_collect.stdout[-8000:]\n"
            "    runtime_report['bc_collect_stderr'] = bc_collect.stderr[-8000:]\n"
            "    print(bc_collect.stdout[-8000:])\n"
            "    if bc_collect.returncode != 0:\n"
            "        save_report()\n"
            "        raise RuntimeError('BC demo collection failed: ' + bc_collect.stderr[-4000:])\n"
            "    bc_env = os.environ.copy()\n"
            "    bc_env['PYTHONPATH'] = str(project_dir / 'src') + os.pathsep + bc_env.get('PYTHONPATH', '')\n"
            "    bc_train = run([sys.executable, str(project_dir / 'scripts' / 'pretrain_bc.py'),\n"
            "                    '--demos', '/kaggle/working/bc-demos.npz',\n"
            "                    '--out', '/kaggle/working/bc-actor.npz',\n"
            "                    '--report', '/kaggle/working/bc-report.json'],\n"
            "                   cwd=project_dir, env=bc_env)\n"
            "    runtime_report['bc_pretrain_exit'] = bc_train.returncode\n"
            "    runtime_report['bc_pretrain_stdout'] = bc_train.stdout[-8000:]\n"
            "    runtime_report['bc_pretrain_stderr'] = bc_train.stderr[-8000:]\n"
            "    print(bc_train.stdout[-8000:])\n"
            "    if bc_train.returncode != 0:\n"
            "        save_report()\n"
            "        raise RuntimeError('BC pretrain failed or demo-quality gate rejected the demos.')\n"
            "    runtime_report['bc_report'] = json.loads(Path('/kaggle/working/bc-report.json').read_text())\n"
        )
        copies = (
            "    shutil.copy2('/kaggle/working/fanuc-cell/scripts/collect_demos.py',\n"
            "                 project_dir / 'scripts' / 'collect_demos.py')\n"
            "    shutil.copy2('/kaggle/working/fanuc-cell/scripts/pretrain_bc.py',\n"
            "                 project_dir / 'scripts' / 'pretrain_bc.py')\n"
            "    shutil.copy2('/kaggle/working/fanuc-cell/scripts/train_ppo_bc.py',\n"
            "                 project_dir / 'scripts' / 'train_ppo_bc.py')\n"
            "    shutil.copy2('/kaggle/working/fanuc-cell/scripts/eval_bc_actor.py',\n"
            "                 project_dir / 'scripts' / 'eval_bc_actor.py')\n"
        )
        if args.bc_exp1:
            # Estágio A do Experimento 1 (issue #8): o ator BC puro,
            # sem PPO, em 200 episódios com seeds 2000+ (fora das
            # demos). Diagnóstico, não portão: o resultado entra no
            # runtime-report e o Estágio B roda de qualquer forma.
            bc_block += (
                "    bc_eval = run(['uv', 'run', '--extra', 'ovphysx', 'python',\n"
                "                    str(project_dir / 'scripts' / 'eval_bc_actor.py'),\n"
                "                    '--bc-weights', '/kaggle/working/bc-actor.npz',\n"
                "                    '--episodes', '200', '--seed', '2000',\n"
                "                    '--report', '/kaggle/working/bc-actor-eval.json'],\n"
                "                   cwd=source_dir, env=baseline_env)\n"
                "    runtime_report['bc_actor_eval_exit'] = bc_eval.returncode\n"
                "    runtime_report['bc_actor_eval_stdout'] = bc_eval.stdout[-8000:]\n"
                "    runtime_report['bc_actor_eval_stderr'] = bc_eval.stderr[-8000:]\n"
                "    print(bc_eval.stdout[-8000:])\n"
                "    if bc_eval.returncode != 0:\n"
                "        save_report()\n"
                "        raise RuntimeError('BC actor eval (Exp. 1 stage A) failed: ' + bc_eval.stderr[-4000:])\n"
                "    runtime_report['bc_actor_eval'] = json.loads(Path('/kaggle/working/bc-actor-eval.json').read_text())\n"
            )
        for cell in notebook["cells"]:
            source = "".join(cell["source"])
            if "diagnose_training_nan.py')" in source and "collect_demos" not in source:
                anchor = ("project_dir / 'scripts' / 'diagnose_training_nan.py')\n")
                source = source.replace(anchor, anchor + copies, 1)
            if "task_sync = run(['uv', 'sync'], cwd=project_dir)" in source:
                anchor = "    task_sync = run(['uv', 'sync'], cwd=project_dir)\n"
                source = source.replace(anchor, bc_block + anchor, 1)
                source = re.sub(
                    r"\['uv', 'run', '--extra', 'ovphysx', 'isaaclab', 'train',"
                    r" '--rl_library', 'rsl_rl',",
                    "['uv', 'run', '--extra', 'ovphysx', 'python', "
                    "str(project_dir / 'scripts' / 'train_ppo_bc.py'), "
                    "'--bc-weights', '/kaggle/working/bc-actor.npz',",
                    source, count=1)
                source = re.sub(
                    r"'--seed', '7', 'physics=ovphysx'\], cwd=project_dir\)",
                    "'--seed', '7'], cwd=project_dir)",
                    source, count=1)
                if args.bc_exp1:
                    # Estágio B protegido (issue #8): entropia 0 e
                    # crítico aquecido com o ator congelado.
                    source = source.replace(
                        "'--seed', '7'], cwd=project_dir)",
                        "'--seed', '7', '--protected', "
                        "'--warmup-iters', '50'], cwd=project_dir)", 1)
                # Playback e benchmark com o checkpoint EXPLICITO: o
                # '--checkpoint latest' do entrypoint unificado exige
                # um manifesto de run que so o `isaaclab train`
                # oficial escreve; o treino warm-start (runner direto)
                # nao escreve — foi o que matou o playback da BC v6
                # depois do treino completo. `checkpoints` ja esta
                # calculado na celula (glob ordenado por mtime).
                source = source.replace(
                    "'--checkpoint', 'latest',",
                    "'--checkpoint', str(checkpoints[-1]),")
            cell["source"] = source.splitlines(keepends=True)
    # Vídeo industrial fatia 1 (issue #9), só com --cinematic-video:
    # mexe EXCLUSIVAMENTE no comando de playback da célula principal —
    # viz newton_rtx + extra ovrtx + env FANUC_CELL_CINEMATIC=1 para o
    # subprocesso do play (a cena cinematic do env_cfg só acorda com a
    # variável). As âncoras foram escolhidas para sobreviver aos
    # outros modos do builder (física newton remove o par ovphysx e
    # troca o token physics=; o BC troca o checkpoint): por isso o
    # alvo é '--extra', 'video', 'isaaclab', 'play' (único do
    # playback) e o env entra por posição (primeiro cwd=project_dir
    # depois do 'playback = run('), sem depender do rabo do comando.
    # Sem a flag, nada aqui roda e o notebook é o de sempre.
    if args.cinematic_video:
        env_def = (
            "    cinematic_env = os.environ.copy()\n"
            "    cinematic_env['FANUC_CELL_CINEMATIC'] = '1'\n"
        )
        for cell in notebook["cells"]:
            source = "".join(cell["source"])
            if "playback = run(" not in source:
                continue
            source = source.replace(
                "    playback = run(",
                env_def + "    playback = run(", 1)
            source = source.replace(
                "'--extra', 'video', 'isaaclab', 'play'",
                "'--extra', 'video', '--extra', 'ovrtx', "
                "'isaaclab', 'play'", 1)
            source = source.replace(
                "'--viz', 'newton_gl'", "'--viz', 'newton_rtx'", 1)
            start = source.index("playback = run(")
            end = source.index("cwd=project_dir)", start) \
                + len("cwd=project_dir")
            source = source[:end] + ", env=cinematic_env" + source[end:]
            cell["source"] = source.splitlines(keepends=True)
    notebook_path = KAGGLE_DIR / NOTEBOOK_NAME
    notebook_path.write_text(json.dumps(notebook, indent=1) + "\n")
    metadata = {
        "id": f"{args.kaggle_user}/{args.kernel_slug}",
        "title": args.kernel_title,
        "code_file": NOTEBOOK_NAME,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "machine_shape": args.accelerator,
        "enable_internet": True,
    }
    (KAGGLE_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(notebook_path)


if __name__ == "__main__":
    main()
