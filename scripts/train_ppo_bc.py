"""Entrada de treino PPO com warm-start BC do ator (issue #3).

Substitui o ``isaaclab train`` SOMENTE no kernel BC (flag
``--bc-warmstart`` do builder; o caminho padrão não muda). Cria o
runner RSL-RL padrão com as configs da própria tarefa, injeta os
pesos do ator pré-treinados por BC (``scripts/pretrain_bc.py``) por
ORDEM de camadas lineares do módulo ``actor`` — com conferência de
shapes e marcador ``BC_WARMSTART_APPLIED`` no log — e roda o
``learn()`` normal: checkpoints, logs e métricas são os do runner
oficial, então playback/benchmark do pipeline funcionam iguais.

Defensivo por desenho: se a arquitetura encontrada não casar com os
pesos (nº de lineares/shape), o script levanta erro ANTES de treinar
— o portão de fumaça do protocolo BC (SPEC) valida esta entrada no
início do kernel, nunca no meio do treino.

Modo ANCORADO (issue #23, receita DAPG/RFT anti-esquecimento),
atrás das flags ``--anchor-*``/``--gate-warmup``/``--freeze-std``
e DESLIGADO por default: sobre o modo protegido do Experimento 1,
o ``update()`` do PPO é embrulhado para (i) atenuar o passo do
ator pelo gating ``tanh(t/T)`` no espaço de parâmetros, (ii) rodar
passos extras de âncora (KL para o professor BC congelado ou MSE
para as ações das demos) com peso decaindo, e (iii) manter o std
da distribuição congelado. Os cronogramas e as contas são do
núcleo puro ``fanuc_cell.anchored_ppo``; aqui mora só a cola com
o torch/rsl_rl, defensiva como o resto do script (API não
reconhecida = erro alto ANTES do ``learn()``).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Fanuc-Conveyor-PickPlace-M10id12-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--max_iterations", type=int, default=1500)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--bc-weights", type=Path, required=True,
                    help=".npz do ator treinado por BC (pretrain_bc.py).")
parser.add_argument("--physics", default="ovphysx")
parser.add_argument("--protected", action="store_true",
                    help="Estágio B do Experimento 1 (issue #8): higiene "
                         "de fine-tuning — entropia 0 e crítico aquecido "
                         "com o ator congelado por --warmup-iters "
                         "iterações antes do treino conjunto.")
parser.add_argument("--warmup-iters", type=int, default=50)
parser.add_argument("--entropy-coef", type=float, default=None,
                    help="Sobrescreve o entropy_coef do cfg do agente. "
                         "Sem a flag: 0 no modo protegido/ancorado "
                         "(higiene da receita), o do cfg nos demais.")
parser.add_argument("--anchor-weight", type=float, default=0.0,
                    help="Issue #23: peso inicial (beta0) da âncora "
                         "BC/KL durante o PPO. 0 = desligada (default; "
                         "o caminho é o de sempre). A referência da "
                         "receita é 0,02 no modo kl.")
parser.add_argument("--anchor-mode", choices=("kl", "mse"),
                    default="kl",
                    help="kl = KL fechada para o professor BC "
                         "congelado; mse = MSE da média do ator para "
                         "as ações das demos (forma DAPG).")
parser.add_argument("--anchor-final-frac", type=float, default=0.0,
                    help="Fração de beta0 ao fim do decaimento linear "
                         "(0 = a âncora some na última iteração).")
parser.add_argument("--anchor-steps", type=int, default=4,
                    help="Passos de gradiente da âncora por iteração "
                         "da fase conjunta.")
parser.add_argument("--anchor-batch", type=int, default=1024,
                    help="Tamanho do minibatch de demos da âncora.")
parser.add_argument("--anchor-demos", type=Path, default=None,
                    help=".npz de demos (formato DemoSet) de onde a "
                         "âncora amostra. Obrigatório com "
                         "--anchor-weight > 0.")
parser.add_argument("--anchor-teacher-std", type=float, default=0.1,
                    help="Desvio da Gaussiana do professor na KL "
                         "(o BC entrega só a média; default = "
                         "init_noise_std da tarefa).")
parser.add_argument("--gate-warmup", type=int, default=0,
                    help="Issue #23: T do gating tanh(t/T) do passo "
                         "do ator na fase conjunta. 0 = sem gating.")
parser.add_argument("--freeze-std", action="store_true",
                    help="Issue #23: congela o parâmetro de desvio "
                         "da distribuição durante todo o treino "
                         "(o ruído de exploração não apaga o clone).")
add_launcher_args(parser)
args = parser.parse_args()
args.visualizer = "none"

ANCHORED = args.anchor_weight > 0 or args.gate_warmup > 0 \
    or args.freeze_std
if args.anchor_weight < 0:
    parser.error("--anchor-weight deve ser >= 0")
if args.anchor_weight > 0 and args.anchor_demos is None:
    parser.error("--anchor-weight > 0 exige --anchor-demos "
                 "(as demos de onde a âncora amostra)")
if not 0.0 <= args.anchor_final_frac <= 1.0:
    parser.error("--anchor-final-frac deve estar em [0, 1]")
if args.anchor_steps < 1 or args.anchor_batch < 1:
    parser.error("--anchor-steps e --anchor-batch devem ser >= 1")
if args.anchor_teacher_std <= 0:
    parser.error("--anchor-teacher-std deve ser > 0")
if args.gate_warmup < 0:
    parser.error("--gate-warmup deve ser >= 0")
if args.entropy_coef is not None and args.entropy_coef < 0:
    parser.error("--entropy-coef deve ser >= 0")
if ANCHORED and args.entropy_coef is not None \
        and args.entropy_coef > 0.0005:
    parser.error("--entropy-coef no modo ancorado deve ficar na "
                 "faixa da receita (0 a 0,0005) — issue #23")

import gymnasium as gym
import numpy as np
import torch

from fanuc_cell.anchored_ppo import AnchorSchedule, DemoBuffer
from fanuc_cell.bc import ActorMLP
from fanuc_cell.demos import DemoSet
from fanuc_cell_lab.tasks.conveyor_pick_place.config.m10id12.env_cfg import M10id12EnvCfg
from fanuc_cell_lab.tasks.conveyor_pick_place.config.m10id12.agents.rsl_rl_ppo_cfg import M10id12PPORunnerCfg
import fanuc_cell_lab.tasks  # noqa: F401


def _actor_linears(alg) -> list:
    """Lineares do ator, na ordem, nas APIs nova e antiga do rsl_rl.

    rsl-rl >= 4: o PPO guarda ``alg.actor`` (MLPModel, pesos em
    ``actor.mlp``). rsl-rl antigo: ``alg.actor_critic.actor`` (Sequential).
    Restringir ao ``mlp`` quando ele existe evita capturar Lineares de
    outros submodulos (distribuicao, normalizador).
    """
    actor = getattr(alg, "actor", None)
    if actor is None:
        actor_critic = getattr(alg, "actor_critic", None)
        if actor_critic is None:
            raise RuntimeError(
                "BC warm-start: runner.alg sem 'actor' nem 'actor_critic' "
                "— API do rsl_rl nao reconhecida.")
        actor = getattr(actor_critic, "actor", actor_critic)
    root = getattr(actor, "mlp", None) or actor
    return [m for _, m in root.named_modules()
            if isinstance(m, torch.nn.Linear)]


def inject_bc_weights(runner, weights_path: Path) -> dict:
    """Copia os pesos BC para as camadas lineares do ator, em ordem."""
    mlp, meta = ActorMLP.load_npz(weights_path)
    linears = _actor_linears(runner.alg)
    if len(linears) != len(mlp.weights):
        raise RuntimeError(
            f"BC warm-start: ator tem {len(linears)} camadas lineares, "
            f"pesos BC têm {len(mlp.weights)} — arquiteturas divergem.")
    with torch.no_grad():
        for layer, w, b in zip(linears, mlp.weights, mlp.biases):
            if tuple(layer.weight.shape) != tuple(w.shape) \
                    or tuple(layer.bias.shape) != tuple(b.shape):
                raise RuntimeError(
                    f"BC warm-start: shape diverge "
                    f"({tuple(layer.weight.shape)} vs {tuple(w.shape)})")
            layer.weight.copy_(
                torch.as_tensor(w, device=layer.weight.device))
            layer.bias.copy_(
                torch.as_tensor(b, device=layer.bias.device))
    info = {"layers": len(linears), "meta": {
        k: meta.get(k) for k in ("obs_dim", "act_dim", "hidden_dims",
                                 "bc_final_loss")}}
    print("BC_WARMSTART_APPLIED", info, flush=True)
    return info


def _actor_module(alg):
    """O módulo do ator, nas APIs nova e antiga do rsl_rl."""
    actor = getattr(alg, "actor", None)
    if actor is not None:
        return actor
    actor_critic = getattr(alg, "actor_critic", None)
    if actor_critic is None:
        raise RuntimeError(
            "Âncora BC: runner.alg sem 'actor' nem 'actor_critic' "
            "— API do rsl_rl nao reconhecida.")
    return actor_critic


def _actor_mean(alg, obs: torch.Tensor) -> torch.Tensor:
    """Média da Gaussiana do ator para ``obs`` (sem amostrar).

    rsl_rl >= 4 (MLPModel): normalizador de obs (Identity quando
    desligado) + ``actor.mlp`` — a distribuição é separada e o MLP
    devolve a média. API antiga (ActorCritic): o Sequential
    ``actor`` interno. Qualquer outra forma é erro alto: a âncora
    não pode operar sobre uma média que não sabe extrair.
    """
    actor = _actor_module(alg)
    x = obs
    normalizer = getattr(actor, "obs_normalizer", None)
    if normalizer is not None:
        x = normalizer(x)
    mlp = getattr(actor, "mlp", None)
    if mlp is not None:
        return mlp(x)
    inner = getattr(actor, "actor", None)
    if inner is not None and inner is not actor:
        return inner(x)
    return actor(x)


def _find_std_params(actor) -> list:
    """Parâmetros de desvio da distribuição: [(nome, param, is_log)].

    Procura por nome nos parâmetros do ator (ex.:
    ``distribution.std_param`` no GaussianDistribution do rsl_rl).
    Lista vazia = a distribuição não expõe o desvio como parâmetro
    — quem chamou decide se isso é fatal (é, para KL e freeze).
    """
    found = []
    for name, param in actor.named_parameters():
        leaf = name.rsplit(".", 1)[-1]
        if "std" in leaf:
            found.append((name, param, "log" in leaf))
    return found


def _build_teacher(weights_path: Path, device) -> torch.nn.Module:
    """Professor BC congelado em torch, dos mesmos pesos do warm-start."""
    mlp, _meta = ActorMLP.load_npz(weights_path)
    layers: list[torch.nn.Module] = []
    for i, (w, b) in enumerate(zip(mlp.weights, mlp.biases)):
        linear = torch.nn.Linear(w.shape[1], w.shape[0])
        with torch.no_grad():
            linear.weight.copy_(torch.as_tensor(w))
            linear.bias.copy_(torch.as_tensor(b))
        layers.append(linear)
        if i < len(mlp.weights) - 1:
            layers.append(torch.nn.ELU())
    teacher = torch.nn.Sequential(*layers).to(device)
    for param in teacher.parameters():
        param.requires_grad_(False)
    teacher.eval()
    return teacher


class _AnchorController:
    """Embrulha o ``update()`` do PPO com gating + âncora (issue #23).

    Fases: durante o aquecimento do crítico (``warmup``) o embrulho
    é transparente — o ator está congelado e os cronogramas não
    correm. Na fase ``joint``, cada ``update()`` do PPO é seguido
    de (i) gating no espaço de parâmetros — os parâmetros do ator
    são interpolados de volta para o valor pré-update por
    ``beta = tanh(t/T)``, o que escala o PASSO do ator sem tocar
    crítico, otimizador nem o schedule de LR — e (ii) os passos de
    âncora sobre minibatches das demos. O contador de iteração é
    próprio (um por chamada de update na fase joint), então os
    cronogramas do núcleo puro são indexados na fase conjunta,
    nunca no relógio global do runner.
    """

    def __init__(self, alg, schedule: AnchorSchedule,
                 buffer: DemoBuffer | None, teacher,
                 std_params: list, starts_in_warmup: bool) -> None:
        self.alg = alg
        self.schedule = schedule
        self.buffer = buffer
        self.teacher = teacher
        self.std_params = std_params
        self.phase = "warmup" if starts_in_warmup else "joint"
        self.joint_iter = 0
        self.losses: list[float] = []
        actor = _actor_module(alg)
        self.actor_params = list(actor.parameters())
        self.device = self.actor_params[0].device
        self.max_grad_norm = float(getattr(alg, "max_grad_norm", 1.0))

    def _student_log_std(self, act_dim: int) -> torch.Tensor:
        _name, param, is_log = self.std_params[0]
        log_std = param if is_log else torch.log(param.clamp_min(1e-8))
        return log_std.reshape(-1).expand(act_dim)

    def _anchor_loss(self, obs: torch.Tensor,
                     actions: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        mean_student = _actor_mean(self.alg, obs)
        if self.schedule.mode == "mse":
            return F.mse_loss(mean_student, actions)
        with torch.no_grad():
            mean_teacher = self.teacher(obs)
        log_std_student = self._student_log_std(mean_student.shape[-1])
        log_std_teacher = float(np.log(self.schedule.teacher_std))
        var_s = torch.exp(2.0 * log_std_student)
        var_t = self.schedule.teacher_std ** 2
        kl = (log_std_teacher - log_std_student) \
            + (var_s + (mean_student - mean_teacher) ** 2) \
            / (2.0 * var_t) - 0.5
        return kl.sum(dim=-1).mean()

    def _anchor_steps(self, weight: float) -> float:
        optimizer = self.alg.optimizer
        total_loss = 0.0
        for step in range(self.schedule.steps_per_iteration):
            obs_np, act_np = self.buffer.batch(
                self.joint_iter * self.schedule.steps_per_iteration
                + step,
                self.schedule.batch_size)
            obs = torch.as_tensor(obs_np, device=self.device)
            actions = torch.as_tensor(act_np, device=self.device)
            loss = self._anchor_loss(obs, actions)
            optimizer.zero_grad()
            (loss * weight).backward()
            torch.nn.utils.clip_grad_norm_(
                self.actor_params, self.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += float(loss.detach())
        return total_loss / self.schedule.steps_per_iteration

    def install(self) -> None:
        orig_update = self.alg.update

        def anchored_update():
            if self.phase != "joint":
                return orig_update()
            state = self.schedule.state(self.joint_iter)
            beta = state["gate_beta"]
            snapshot = None
            if beta < 1.0:
                snapshot = [p.detach().clone()
                            for p in self.actor_params]
            losses = orig_update()
            if snapshot is not None:
                with torch.no_grad():
                    for param, pre in zip(self.actor_params, snapshot):
                        param.mul_(beta).add_(pre, alpha=1.0 - beta)
            weight = state["anchor_weight"]
            if weight > 0.0:
                loss = self._anchor_steps(weight)
                self.losses.append(loss)
                if self.joint_iter % 50 == 0:
                    print("BC_ANCHOR_STEP",
                          {"iter": self.joint_iter, "weight": weight,
                           "gate_beta": beta, "loss": loss},
                          flush=True)
            self.joint_iter += 1
            return losses

        self.alg.update = anchored_update

    def summary(self) -> dict:
        tail = self.losses[-50:]
        return {
            "joint_iterations": self.joint_iter,
            "anchor_steps_total": len(self.losses),
            "anchor_loss_mean_tail": (
                float(np.mean(tail)) if tail else None),
            "schedule": self.schedule.summary(),
        }


def install_anchor(runner, joint_iterations: int,
                   starts_in_warmup: bool):
    """Monta a receita ancorada sobre o runner já warm-started.

    Tudo é validado ANTES de qualquer ``learn()``: buffer de demos
    com as dimensões do ator BC, professor congelado, forward de
    fumaça da média do ator e — para KL e ``--freeze-std`` — o
    parâmetro de desvio localizado. Devolve (controlador, info
    para o registro durável).
    """
    alg = runner.alg
    actor = _actor_module(alg)
    _mlp, bc_meta = ActorMLP.load_npz(args.bc_weights)
    obs_dim = int(bc_meta["obs_dim"])
    act_dim = int(bc_meta["act_dim"])

    std_params = _find_std_params(actor)
    frozen_std: list[str] = []
    if args.freeze_std:
        if not std_params:
            raise RuntimeError(
                "Âncora BC: --freeze-std pedido mas nenhum parâmetro "
                "de desvio foi encontrado no ator — API do rsl_rl "
                "não reconhecida.")
        for name, param, _is_log in std_params:
            param.requires_grad_(False)
            frozen_std.append(name)

    schedule = AnchorSchedule(
        beta0=args.anchor_weight, mode=args.anchor_mode,
        final_fraction=args.anchor_final_frac,
        gate_warmup=args.gate_warmup,
        total_iterations=max(1, joint_iterations),
        teacher_std=args.anchor_teacher_std,
        steps_per_iteration=args.anchor_steps,
        batch_size=args.anchor_batch)

    buffer = None
    teacher = None
    device = next(actor.parameters()).device
    if schedule.enabled:
        demos = DemoSet.load_npz(args.anchor_demos)
        if demos.obs_dim != obs_dim or demos.act_dim != act_dim:
            raise RuntimeError(
                f"Âncora BC: demos {demos.obs_dim}x{demos.act_dim} "
                f"divergem do ator BC {obs_dim}x{act_dim}.")
        buffer = DemoBuffer(demos, seed=args.seed)
        teacher = _build_teacher(args.bc_weights, device)
        if args.anchor_mode == "kl" and not std_params:
            raise RuntimeError(
                "Âncora BC (modo kl): nenhum parâmetro de desvio "
                "encontrado no ator para a KL do estudante — API "
                "do rsl_rl não reconhecida.")
        # Forward de fumaça: a média do ator precisa sair com o
        # shape do batch de demos ANTES do treino (defensivo como
        # o resto do script).
        obs_np, _act_np = buffer.batch(0, 2)
        with torch.no_grad():
            mean = _actor_mean(
                alg, torch.as_tensor(obs_np, device=device))
        if tuple(mean.shape) != (2, act_dim):
            raise RuntimeError(
                f"Âncora BC: média do ator saiu {tuple(mean.shape)}, "
                f"esperado (2, {act_dim}).")

    controller = _AnchorController(
        alg, schedule, buffer, teacher, std_params, starts_in_warmup)
    controller.install()
    info = {
        "schedule": schedule.summary(),
        "buffer": buffer.summary() if buffer is not None else None,
        "frozen_std_params": frozen_std,
        "entropy_coef": float(args.entropy_coef)
        if args.entropy_coef is not None else 0.0,
    }
    print("BC_ANCHOR_APPLIED", info, flush=True)
    return controller, info


def main() -> int:
    env_cfg = M10id12EnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    agent_cfg = M10id12PPORunnerCfg()
    agent_cfg.seed = args.seed
    agent_cfg.max_iterations = args.max_iterations
    agent_cfg.device = args.device

    with launch_simulation(env_cfg, args) as physics_cfg:
        env_cfg.sim.physics = physics_cfg
        env = gym.make(args.task, cfg=env_cfg)
        try:
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
            try:
                env = RslRlVecEnvWrapper(
                    env, clip_actions=agent_cfg.clip_actions)
            except TypeError:
                env = RslRlVecEnvWrapper(env)
            # Mesmo passo do treino oficial (isaaclab train): sem ele, o
            # to_dict() cru mantem a cfg no formato antigo (policy) e o
            # rsl-rl >= 4 morre com KeyError 'class_name' em
            # resolve_class(cfg["actor"]) — foi o que matou a BC v4
            # depois do portao das demos aprovado.
            if args.entropy_coef is not None:
                agent_cfg.algorithm.entropy_coef = args.entropy_coef
            elif args.protected or ANCHORED:
                # Higiene de fine-tuning (issue #8) e da receita
                # ancorada (issue #23): o bônus de entropia empurra
                # o ator para longe da imitação nos primeiros
                # passos — nestes modos ele é zero.
                agent_cfg.algorithm.entropy_coef = 0.0
            import importlib.metadata as metadata
            from isaaclab_rl.rsl_rl.utils import \
                handle_deprecated_rsl_rl_cfg
            agent_cfg = handle_deprecated_rsl_rl_cfg(
                agent_cfg, metadata.version("rsl-rl-lib"))
            from rsl_rl.runners import OnPolicyRunner
            log_dir = Path("logs") / "rsl_rl" / agent_cfg.experiment_name \
                / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                   + "_bc_warmstart")
            runner = OnPolicyRunner(
                env, agent_cfg.to_dict(), log_dir=str(log_dir),
                device=agent_cfg.device)
            warmstart_info = inject_bc_weights(runner, args.bc_weights)
            if args.protected:
                warmstart_info["protected"] = {
                    "entropy_coef": 0.0,
                    "warmup_iters": args.warmup_iters,
                    "obs_normalization": bool(
                        agent_cfg.to_dict().get("actor", {})
                        .get("obs_normalization", False)),
                }
            total_iters = agent_cfg.max_iterations
            warmup_iters = args.warmup_iters \
                if (args.protected or ANCHORED) else 0
            anchor_ctl = None
            if ANCHORED:
                # Receita ancorada (issue #23): instalada e
                # validada ANTES do learn() — qualquer divergência
                # de API/dimensão é erro alto aqui, nunca no meio
                # do treino. O congelamento do std acontece dentro
                # da instalação, ANTES do freeze do warmup: assim o
                # std não entra na lista de descongelamento do
                # warmup e fica congelado a corrida inteira.
                anchor_ctl, anchor_info = install_anchor(
                    runner, total_iters - warmup_iters,
                    starts_in_warmup=warmup_iters > 0)
                warmstart_info["anchored"] = anchor_info
            # Registro duravel do warm-start: o runtime-report do
            # notebook guarda so a cauda do stdout e o marcador
            # impresso na injecao ficou fora dela na BC v6.
            Path("bc-warmstart.json").write_text(
                json.dumps(warmstart_info, indent=1))
            if warmup_iters > 0:
                # Estágio B (issue #8): aquece o crítico com o ator
                # CONGELADO. O otimizador do rsl_rl 5.x é único para
                # ator+crítico (chain dos parâmetros), então o freeze
                # é por requires_grad nos parâmetros do ator — o que
                # inclui o std da distribuição. learn() continua a
                # contagem de iterações entre chamadas (fonte 5.5.1:
                # start_it = current_learning_iteration).
                frozen = [p for p in runner.alg.actor.parameters()
                          if p.requires_grad]
                for p in frozen:
                    p.requires_grad_(False)
                try:
                    runner.learn(
                        num_learning_iterations=args.warmup_iters,
                        init_at_random_ep_len=True)
                finally:
                    for p in frozen:
                        p.requires_grad_(True)
                # Off-by-one do rsl_rl 5.x (mapeado na fonte 5.5.1):
                # current_learning_iteration guarda o ÍNDICE da última
                # iteração concluída — sem o ajuste, a 2ª chamada a
                # learn() reexecuta a última iteração do warmup já com
                # o ator descongelado e a corrida termina um índice
                # atrás (model_1498 em vez de model_1499).
                runner.current_learning_iteration += 1
                print("BC_PROTECTED_WARMUP_DONE",
                      {"warmup_iters": args.warmup_iters,
                       "frozen_params": len(frozen)}, flush=True)
                if anchor_ctl is not None:
                    # Os cronogramas da âncora/gating começam aqui:
                    # a fase conjunta é indexada do zero.
                    anchor_ctl.phase = "joint"
                runner.learn(
                    num_learning_iterations=total_iters - args.warmup_iters)
            else:
                runner.learn(
                    num_learning_iterations=total_iters,
                    init_at_random_ep_len=True)
            if anchor_ctl is not None:
                print("BC_ANCHOR_DONE", anchor_ctl.summary(),
                      flush=True)
            print("BC_WARMSTART_APPLIED", warmstart_info, flush=True)
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
