"""Gera o notebook do spike cuRobo V2 (instalação + IK em lote no Kaggle T4).

Pergunta única do spike (pré-registrada): o cuRobo V2 instala via pip num
kernel Kaggle padrão e resolve IK em lote na GPU? Portões graduados:
- CUROBO_INSTALL_OK/FAIL (pip install nvidia-curobo)
- CUROBO_IMPORT_OK/FAIL
- CUROBO_IK_OK + taxa de sucesso + segundos (API clássica wrap.reacher);
  se a API mudou na V2, o notebook registra o erro exato e lista os
  submódulos do pacote com 'ik'/'solver' no nome — diagnóstico suficiente
  para a próxima iteração do spike.
Relatório estruturado em /kaggle/working/curobo-spike-report.json.
"""
import json
from pathlib import Path

OUT = Path("/tmp/curobo-spike/kaggle")
OUT.mkdir(parents=True, exist_ok=True)


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(True)}


CELL_ENV = """
import json, platform, subprocess, sys, time
report = {'stages': {}}
print('python', sys.version)
print('platform', platform.platform())
try:
    import torch
    report['torch'] = torch.__version__
    report['cuda_available'] = torch.cuda.is_available()
    if torch.cuda.is_available():
        report['gpu'] = torch.cuda.get_device_name(0)
        report['cuda'] = torch.version.cuda
    print('torch', torch.__version__, 'cuda', torch.version.cuda,
          'gpu', report.get('gpu'))
except Exception as exc:
    report['torch_error'] = repr(exc)
smi = subprocess.run(['nvidia-smi', '--query-gpu=driver_version,name',
                      '--format=csv,noheader'], capture_output=True, text=True)
report['nvidia_smi'] = (smi.stdout or smi.stderr).strip()
print('nvidia-smi:', report['nvidia_smi'])
"""

CELL_INSTALL = """
# Estágio 1 — instalação via pip (a pergunta central do spike).
t0 = time.time()
proc = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                       'nvidia-curobo'], capture_output=True, text=True)
report['stages']['install_seconds'] = round(time.time() - t0, 1)
report['stages']['pip_returncode'] = proc.returncode
tail = (proc.stdout + proc.stderr).strip().splitlines()
report['stages']['pip_tail'] = tail[-15:]
for line in tail[-15:]:
    print(line)
print('CUROBO_INSTALL_OK' if proc.returncode == 0 else 'CUROBO_INSTALL_FAIL')
"""

CELL_IK = """
# Estágio 2 — import + tentativa de IK em lote na GPU.
import importlib.metadata as md_
try:
    report['curobo_version'] = md_.version('nvidia-curobo')
except Exception:
    report['curobo_version'] = None
print('nvidia-curobo version:', report['curobo_version'])
try:
    import curobo
    print('CUROBO_IMPORT_OK', getattr(curobo, '__version__', '?'))
    report['stages']['import'] = 'ok'
except Exception as exc:
    print('CUROBO_IMPORT_FAIL', repr(exc))
    report['stages']['import'] = repr(exc)
    curobo = None

if curobo is not None:
    # Mapa do pacote: ajuda a localizar a API da V2 se a clássica mudou.
    import pkgutil
    mods = sorted(m.name for m in pkgutil.walk_packages(curobo.__path__,
                                                        prefix='curobo.'))
    report['ik_like_modules'] = [m for m in mods
                                 if 'ik' in m.lower() or 'solver' in m.lower()][:60]
    print('modulos ik/solver:', report['ik_like_modules'])
    try:
        import torch
        from curobo.types.base import TensorDeviceType
        from curobo.types.math import Pose
        from curobo.util_file import (get_robot_configs_path, join_path,
                                      load_yaml)
        from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

        tensor_args = TensorDeviceType(device=torch.device('cuda'),
                                       dtype=torch.float32)
        cfg_name = None
        for cand in ('ur10e.yml', 'franka.yml', 'ur5e.yml'):
            try:
                load_yaml(join_path(get_robot_configs_path(), cand))
                cfg_name = cand
                break
            except Exception:
                continue
        report['robot_config'] = cfg_name
        print('robot config:', cfg_name)
        robot_cfg = load_yaml(join_path(get_robot_configs_path(),
                                        cfg_name))['robot_cfg']
        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg, None, rotation_threshold=0.05,
            position_threshold=0.005, num_seeds=20,
            self_collision_check=False, self_collision_opt=False,
            tensor_args=tensor_args, use_cuda_graph=True)
        solver = IKSolver(ik_config)
        n = 64
        g = torch.Generator(device='cuda').manual_seed(0)
        pos = torch.rand((n, 3), device='cuda', generator=g)
        pos = pos * torch.tensor([0.3, 0.4, 0.3], device='cuda') \\
            + torch.tensor([0.35, -0.2, 0.25], device='cuda')
        quat = torch.zeros((n, 4), device='cuda')
        quat[:, 0] = 1.0
        goal = Pose(pos, quat)
        t0 = time.time()
        result = solver.solve_batch(goal)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        succ = float(result.success.float().mean().item())
        report['stages']['ik'] = {'batch': n, 'success_rate': succ,
                                  'seconds': round(elapsed, 3)}
        print(f'CUROBO_IK_OK batch={n} success={succ:.3f} '
              f'seconds={elapsed:.3f}')
    except Exception as exc:
        import traceback
        traceback.print_exc()
        report['stages']['ik_error'] = repr(exc)
        print('CUROBO_IK_FAIL', repr(exc))
"""

CELL_REPORT = """
import json
with open('/kaggle/working/curobo-spike-report.json', 'w') as fh:
    json.dump(report, fh, indent=1, default=str)
print(json.dumps(report, indent=1, default=str))
print('CUROBO_SPIKE_DONE')
"""

cells = [
    md("# FANUC — spike cuRobo V2: instalação pip + IK em lote (Kaggle T4)\n"
       "\n"
       "Pergunta: `pip install nvidia-curobo` funciona num kernel Kaggle "
       "padrão e o solver resolve IK em lote na GPU? Veredito graduado no log "
       "(CUROBO_INSTALL_*, CUROBO_IMPORT_*, CUROBO_IK_*) e relatório JSON em "
       "`/kaggle/working/curobo-spike-report.json`. Spike de viabilidade da "
       "integração (auditoria de integrações v2, Rank 3) — nenhum código do "
       "projeto é tocado."),
    code(CELL_ENV),
    code(CELL_INSTALL),
    code(CELL_IK),
    code(CELL_REPORT),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
        "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}
(OUT / "curobo-spike.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")

meta = {"id": "pascalcoimbra/fanuc-curobo-spike",
        "title": "FANUC cuRobo V2 spike install IK",
        "code_file": "curobo-spike.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
        "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
        "category_ids": []}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
print("notebook + metadata escritos em", OUT)
