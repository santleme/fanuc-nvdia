"""PPO ancorado em BC — receita DAPG/RFT anti-esquecimento (issue #23).

Núcleo PURO (só numpy): cronogramas e contas da âncora que protege o
warm-start BC durante o PPO. A cola com o torch/rsl_rl mora em
``scripts/train_ppo_bc.py`` (roda só no kernel); aqui ficam as
decisões numéricas testáveis sem Isaac:

- **Gating do PPO**: ``beta(t) = tanh(t / T_warmup)`` — enquanto o
  crítico é ruim, o passo do ator é atenuado por beta (a cola aplica
  o gating no espaço de parâmetros: ``theta <- theta_pre +
  beta * (theta_pos - theta_pre)``, equivalente a escalar o passo
  do ator sem tocar o crítico nem o otimizador).
- **Âncora BC/KL**: peso ``w(t)`` com decaimento LINEAR de ``beta0``
  até ``beta0 * final_fraction`` ao longo da fase conjunta. A loss é
  KL fechada entre as Gaussianas diagonais do estudante e do
  professor congelado (modo ``kl``, referência β≈0,02 do ExT/IBRL-RFT
  via auditoria da issue) ou MSE entre a média do estudante e a ação
  do professor nas demos (modo ``mse``, a forma DAPG clássica).
- **DemoBuffer**: o conjunto de demos (``DemoSet``) do qual a âncora
  amostra minibatches determinísticos. É também o ponto de extensão
  do DAgger desenhado na SPEC: rodadas futuras ACRESCENTAM pares
  (obs visitada pelo estudante, ação do professor) ao buffer via
  ``append`` — mesmo formato, sem mudar a âncora.

Nada aqui importa torch nem Isaac Lab: treinar na VM é proibido
pela regra do projeto; estes testes são de código.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .demos import DemoSet

# Referência da auditoria da issue #23: KL β≈0,02 (ExT/IBRL-RFT).
DEFAULT_ANCHOR_WEIGHT = 0.02
# Desvio do professor congelado na KL: o BC entrega só a média; a
# Gaussiana do professor usa o init_noise_std da tarefa (0,1).
DEFAULT_TEACHER_STD = 0.1


def gate_beta(iteration: int, t_warmup: int) -> float:
    """Fator de gating do passo do ator: ``tanh(t / T_warmup)``.

    ``iteration`` é o índice (base 0) da iteração da FASE CONJUNTA
    (depois do aquecimento do crítico). Com ``t_warmup <= 0`` o
    gating está desligado e o fator é 1. Em t=0 o fator é 0 (o
    primeiro passo do PPO não mexe no ator); em t=T vale
    tanh(1) ≈ 0,7616; tende a 1.
    """
    if iteration < 0:
        raise ValueError("iteration deve ser >= 0")
    if t_warmup <= 0:
        return 1.0
    return float(np.tanh(iteration / t_warmup))


def anchor_weight_at(iteration: int, total_iterations: int,
                     beta0: float, final_fraction: float) -> float:
    """Peso da âncora na iteração: decaimento linear beta0 → beta0*frac.

    ``iteration`` base 0 na fase conjunta; a última iteração
    (``total_iterations - 1``) recebe exatamente
    ``beta0 * final_fraction``. Com ``total_iterations <= 1`` o peso
    é ``beta0``.
    """
    if beta0 < 0:
        raise ValueError("beta0 deve ser >= 0")
    if not 0.0 <= final_fraction <= 1.0:
        raise ValueError("final_fraction deve estar em [0, 1]")
    if iteration < 0:
        raise ValueError("iteration deve ser >= 0")
    if total_iterations <= 1:
        return float(beta0)
    frac = min(iteration, total_iterations - 1) / (total_iterations - 1)
    return float(beta0 * (1.0 - frac * (1.0 - final_fraction)))


def gaussian_kl_diag(mean_student: np.ndarray, log_std_student,
                     mean_teacher: np.ndarray, log_std_teacher) -> float:
    """KL(estudante || professor) de Gaussianas diagonais, média no batch.

    Forma fechada por dimensão:
    ``log(σt/σs) + (σs² + (μs−μt)²) / (2σt²) − 1/2``, somada nas
    dimensões de ação e mediada no batch. É a fórmula que a cola
    torch de ``train_ppo_bc.py`` implementa para o modo ``kl`` —
    este gêmeo numpy existe para piná-la em teste.
    """
    ms = np.asarray(mean_student, dtype=np.float64)
    mt = np.asarray(mean_teacher, dtype=np.float64)
    if ms.shape != mt.shape or ms.ndim != 2:
        raise ValueError("médias devem ser matrizes (batch, act_dim) iguais")
    ls = np.broadcast_to(np.asarray(log_std_student, dtype=np.float64),
                         (ms.shape[1],))
    lt = np.broadcast_to(np.asarray(log_std_teacher, dtype=np.float64),
                         (ms.shape[1],))
    var_s = np.exp(2.0 * ls)
    var_t = np.exp(2.0 * lt)
    kl = (lt - ls) + (var_s + (ms - mt) ** 2) / (2.0 * var_t) - 0.5
    return float(kl.sum(axis=1).mean())


@dataclass
class AnchorSchedule:
    """Configuração validada da receita ancorada (issue #23).

    ``total_iterations`` é o tamanho da FASE CONJUNTA (total do
    treino menos o aquecimento do crítico): os dois cronogramas são
    indexados nela, nunca no relógio global do runner — o warmup
    congelado não consome decaimento nem gating.
    """

    beta0: float = DEFAULT_ANCHOR_WEIGHT
    mode: str = "kl"
    final_fraction: float = 0.0
    gate_warmup: int = 0
    total_iterations: int = 1
    teacher_std: float = DEFAULT_TEACHER_STD
    steps_per_iteration: int = 4
    batch_size: int = 1024

    def __post_init__(self) -> None:
        if self.beta0 < 0:
            raise ValueError("beta0 (peso inicial da âncora) deve ser >= 0")
        if self.mode not in ("kl", "mse"):
            raise ValueError("mode deve ser 'kl' ou 'mse'")
        if not 0.0 <= self.final_fraction <= 1.0:
            raise ValueError("final_fraction deve estar em [0, 1]")
        if self.gate_warmup < 0:
            raise ValueError("gate_warmup deve ser >= 0")
        if self.total_iterations < 1:
            raise ValueError("total_iterations deve ser >= 1")
        if self.teacher_std <= 0:
            raise ValueError("teacher_std deve ser > 0")
        if self.steps_per_iteration < 1:
            raise ValueError("steps_per_iteration deve ser >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size deve ser >= 1")

    @property
    def enabled(self) -> bool:
        """Âncora ativa = peso inicial positivo (gating é à parte)."""
        return self.beta0 > 0

    def state(self, iteration: int) -> dict:
        """Cronogramas na iteração da fase conjunta (base 0)."""
        return {
            "iteration": int(iteration),
            "anchor_weight": anchor_weight_at(
                iteration, self.total_iterations, self.beta0,
                self.final_fraction),
            "gate_beta": gate_beta(iteration, self.gate_warmup),
        }

    def summary(self) -> dict:
        return {
            "beta0": self.beta0,
            "mode": self.mode,
            "final_fraction": self.final_fraction,
            "gate_warmup": self.gate_warmup,
            "total_iterations": self.total_iterations,
            "teacher_std": self.teacher_std,
            "steps_per_iteration": self.steps_per_iteration,
            "batch_size": self.batch_size,
        }


class DemoBuffer:
    """Buffer de demos da âncora, com amostragem determinística.

    Envolve um ``DemoSet`` (formato da coleta, issue #3). A âncora
    pede ``batch(iteration)``: os índices dependem só de
    (seed, iteration) — a mesma corrida repete os mesmos minibatches.
    ``append`` é o gancho do DAgger desenhado na SPEC (issue #23):
    pares re-rotulados (obs do estudante, ação do professor) entram
    como um ``DemoSet`` a mais, sem mudar a interface da âncora.
    """

    def __init__(self, demos: DemoSet, seed: int = 0) -> None:
        if demos.num_steps == 0:
            raise ValueError("DemoBuffer exige demos não vazias")
        self._sets: list[DemoSet] = [demos]
        self._seed = int(seed)
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._observations = np.concatenate(
            [s.observations for s in self._sets]).astype(np.float32)
        self._actions = np.concatenate(
            [s.actions for s in self._sets]).astype(np.float32)

    @property
    def num_steps(self) -> int:
        return len(self._observations)

    @property
    def obs_dim(self) -> int:
        return int(self._observations.shape[1])

    @property
    def act_dim(self) -> int:
        return int(self._actions.shape[1])

    @property
    def num_sets(self) -> int:
        return len(self._sets)

    def append(self, demos: DemoSet) -> None:
        """Acrescenta um DemoSet (gancho DAgger); dims devem casar."""
        if demos.obs_dim != self.obs_dim or demos.act_dim != self.act_dim:
            raise ValueError(
                "DemoSet anexado diverge em dimensão "
                f"({demos.obs_dim}x{demos.act_dim} vs "
                f"{self.obs_dim}x{self.act_dim})")
        if demos.num_steps == 0:
            raise ValueError("DemoSet anexado está vazio")
        self._sets.append(demos)
        self._rebuild_index()

    def batch(self, iteration: int,
              batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Minibatch determinístico da iteração (obs, ações)."""
        if iteration < 0:
            raise ValueError("iteration deve ser >= 0")
        if batch_size < 1:
            raise ValueError("batch_size deve ser >= 1")
        rng = np.random.default_rng(self._seed * 1_000_003 + iteration)
        n = self.num_steps
        idx = rng.choice(n, size=batch_size, replace=batch_size > n)
        return self._observations[idx], self._actions[idx]

    def summary(self) -> dict:
        return {
            "num_steps": self.num_steps,
            "num_sets": self.num_sets,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "seed": self._seed,
        }
