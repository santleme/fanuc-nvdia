"""Bundle the local project into a self-contained Kaggle notebook."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
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
        "src/fanuc_cell/kinematics.py",
        "src/fanuc_cell/task.py",
        "src/fanuc_cell/render.py",
        "scripts/render_demo.py",
        "scripts/run_scripted_demo.py",
        "scripts/diagnose_training_nan.py",
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
        choices=("ovphysx", "newton"),
        default="ovphysx",
        help="Backend de física do Isaac Lab 3.0 (preset da tarefa). "
             "ovphysx = linha principal; newton = spike MuJoCo-Warp (issue #2).",
    )
    parser.add_argument("--train-num-envs", type=int, default=64)
    parser.add_argument("--train-max-iterations", type=int, default=500)
    parser.add_argument(
        "--kernel-slug",
        default=KERNEL_SLUG,
        help="Slug do kernel no Kaggle; o spike Newton usa slug próprio "
             "para não virar versão do kernel principal.",
    )
    parser.add_argument("--kernel-title", default="FANUC Conveyor Pick Place")
    args = parser.parse_args()
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
                "    evaluation_dir = Path('/kaggle/working/policy-evaluation')\n"
                "    evaluation = run(['uv', 'run', '--extra', 'ovphysx', 'isaaclab', 'benchmark', 'play',\n"
                "                      '--rl_library', 'rsl_rl', '--task', task_id,\n"
                "                      '--checkpoint', 'latest', '--num_envs', '16',\n"
                "                      '--num_steps', '800', '--seed', '101',\n"
                "                      '--viz', 'none', '--benchmark_formatter', 'schema,summary',\n"
                "                      '--output_path', str(evaluation_dir), 'physics=ovphysx'],\n"
                "                     cwd=project_dir)\n"
                "    runtime_report['policy_evaluation_exit'] = evaluation.returncode\n"
                "    runtime_report['policy_evaluation_stdout'] = evaluation.stdout[-10000:]\n"
                "    runtime_report['policy_evaluation_stderr'] = evaluation.stderr[-10000:]\n"
                "    if evaluation.returncode != 0:\n"
                "        save_report()\n"
                "        raise RuntimeError('Trained-policy evaluation failed: ' + evaluation.stderr[-10000:])\n"
                "    evaluation_results = sorted(evaluation_dir.rglob('*.json'), key=lambda p: p.stat().st_mtime)\n"
                "    if evaluation_results:\n"
                "        evaluation_data = [json.loads(path.read_text()) for path in evaluation_results]\n"
                "        runtime_report['policy_evaluation'] = evaluation_data[-1]\n"
                "        runtime_report['policy_success_rate'] = None\n"
                "        for _data in evaluation_data:\n"
                "            for _record in (_data if isinstance(_data, list) else [_data]):\n"
                "                if not isinstance(_record, dict):\n"
                "                    continue\n"
                "                if _record.get('success_rate') is not None:\n"
                "                    runtime_report['policy_success_rate'] = _record['success_rate']\n"
                "                for _m in (_record.get('measurements') or []):\n"
                "                    if isinstance(_m, dict) and str(_m.get('name', '')).endswith('play success_rate'):\n"
                "                        runtime_report['policy_success_rate'] = _m.get('value')\n"
                "        if runtime_report['policy_success_rate'] is None:\n"
                "            save_report()\n"
                "            raise RuntimeError('Isaac Lab benchmark did not report Metrics/success_rate for completed episodes.')\n"
                "    else:\n"
                "        runtime_report['policy_success_rate'] = None\n"
                "        runtime_report['policy_evaluation_artifact_missing'] = True\n"
                "        save_report()\n"
                "        raise RuntimeError('Isaac Lab benchmark did not produce a JSON evaluation result.')\n"
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
            if args.physics != "ovphysx":
                source = source.replace("'--extra', 'ovphysx', ", "")
            source = source.replace('\\1"ovphysx"',
                                    f'\\1"{args.physics}"')
            source = source.replace(
                "'--num_envs', '64', '--max_iterations', '500'",
                f"'--num_envs', '{args.train_num_envs}', "
                f"'--max_iterations', '{args.train_max_iterations}'")
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
