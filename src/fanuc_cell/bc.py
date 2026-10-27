"""Behavior cloning do ator da política (issue #3, BC warm-start).

MLP em numpy puro com a MESMA arquitetura do ator RSL-RL da tarefa
(``rsl_rl_ppo_cfg.py``: hidden [256, 128, 64], ativação ELU, saída
linear = média da Gaussiana; o desvio é parâmetro separado do runner
e NÃO é tocado pelo BC). Treinar aqui na VM com dados sintéticos é
teste de código; o treino de verdade roda no kernel, sobre as demos
coletadas no Isaac (``scripts/pretrain_bc.py``).

Exportação: ``.npz`` com os pesos por camada (``layer{i}_weight`` /
``layer{i}_bias``) e também no endereçamento de um ``nn.Sequential``
Linear/ELU (``actor.0``, ``actor.2``, ``actor.4``, ``actor.6``), para a
injeção no Kaggle (``scripts/train_ppo_bc.py``) mapear por ORDEM de
camadas lineares — robusto a renomeações internas do rsl_rl.
"""

from __future__ import annotations

import json

import numpy as np

ACTOR_HIDDEN_DIMS = (256, 128, 64)


def _elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))


def _elu_grad(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1.0, np.exp(np.minimum(x, 0.0)))


class ActorMLP:
    """MLP [obs, *hidden, act] com ELU — espelho numpy do ator RSL-RL."""

    def __init__(self, obs_dim: int, act_dim: int,
                 hidden_dims=ACTOR_HIDDEN_DIMS, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        dims = [obs_dim, *hidden_dims, act_dim]
        self.dims = dims
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for fan_in, fan_out in zip(dims[:-1], dims[1:]):
            # Inicialização ortogonal (ganho raiz de 2 nas ocultas,
            # 0.01 na saída) — a mesma receita do rsl_rl.
            gain = np.sqrt(2.0) if fan_out != act_dim else 0.01
            if fan_out <= fan_in:
                q, _ = np.linalg.qr(
                    rng.standard_normal((fan_in, fan_out)))
                w = (gain * q.T).astype(np.float32)
            else:
                q, _ = np.linalg.qr(
                    rng.standard_normal((fan_out, fan_in)))
                w = (gain * q).astype(np.float32)
            self.weights.append(w)
            self.biases.append(np.zeros(fan_out, dtype=np.float32))

    def forward(self, x: np.ndarray, cache: bool = False):
        h = x.astype(np.float32)
        preactivations = []
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ w.T + b
            preactivations.append(z)
            h = z if i == len(self.weights) - 1 else _elu(z)
        if cache:
            return h, preactivations
        return h

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(np.asarray(x, dtype=np.float32))

    def state_arrays(self) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            arrays[f"layer{i}_weight"] = w
            arrays[f"layer{i}_bias"] = b
            # Endereçamento nn.Sequential (Linear, ELU, Linear, ...).
            arrays[f"actor.{2 * i}.weight"] = w
            arrays[f"actor.{2 * i}.bias"] = b
        return arrays

    def save_npz(self, path, metadata: dict | None = None) -> None:
        meta = {
            "obs_dim": self.dims[0],
            "act_dim": self.dims[-1],
            "hidden_dims": list(self.dims[1:-1]),
            "activation": "elu",
        }
        meta.update(metadata or {})
        np.savez_compressed(path, **self.state_arrays(),
                            metadata_json=np.asarray(json.dumps(meta)))

    @classmethod
    def load_npz(cls, path) -> tuple["ActorMLP", dict]:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata_json"]))
            mlp = cls(meta["obs_dim"], meta["act_dim"],
                      hidden_dims=tuple(meta["hidden_dims"]))
            for i in range(len(mlp.weights)):
                mlp.weights[i] = np.asarray(data[f"layer{i}_weight"],
                                            dtype=np.float32)
                mlp.biases[i] = np.asarray(data[f"layer{i}_bias"],
                                           dtype=np.float32)
        return mlp, meta


def train_bc(observations: np.ndarray, actions: np.ndarray,
             hidden_dims=ACTOR_HIDDEN_DIMS, epochs: int = 300,
             lr: float = 1e-3, batch_size: int = 256, seed: int = 0,
             weight_decay: float = 0.0) -> tuple[ActorMLP, list[float]]:
    """Treina o ator por BC (MSE nas ações do professor), Adam full/mini-batch.

    Devolve (modelo, histórico de perda por época). Determinístico
    dado o seed.
    """
    obs = np.asarray(observations, dtype=np.float32)
    act = np.asarray(actions, dtype=np.float32)
    if obs.ndim != 2 or act.ndim != 2 or len(obs) != len(act) or len(obs) == 0:
        raise ValueError("demos inválidas para BC")
    mlp = ActorMLP(obs.shape[1], act.shape[1], hidden_dims, seed=seed)
    rng = np.random.default_rng(seed)
    mw = [np.zeros_like(w) for w in mlp.weights]
    vw = [np.zeros_like(w) for w in mlp.weights]
    mb = [np.zeros_like(b) for b in mlp.biases]
    vb = [np.zeros_like(b) for b in mlp.biases]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    history: list[float] = []
    step = 0
    n = len(obs)
    for _ in range(epochs):
        order = rng.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            xb, yb = obs[idx], act[idx]
            # forward com cache
            hs = [xb]
            zs = []
            h = xb
            for i, (w, b) in enumerate(zip(mlp.weights, mlp.biases)):
                z = h @ w.T + b
                zs.append(z)
                h = z if i == len(mlp.weights) - 1 else _elu(z)
                hs.append(h)
            err = h - yb
            epoch_loss += float(np.mean(err ** 2)) * len(idx)
            grad = (2.0 / err.size) * err  # d(MSE)/d(saída)
            for i in reversed(range(len(mlp.weights))):
                gw = grad.T @ hs[i] + weight_decay * mlp.weights[i]
                gb = grad.sum(axis=0)
                if i > 0:
                    grad = (grad @ mlp.weights[i]) * _elu_grad(zs[i - 1])
                step_i = step + 1
                # Adam por parâmetro (peso e bias da camada i)
                for arr_list, g, m_list, v_list in (
                        (mlp.weights, gw, mw, vw), (mlp.biases, gb, mb, vb)):
                    m_list[i] = beta1 * m_list[i] + (1 - beta1) * g
                    v_list[i] = beta2 * v_list[i] + (1 - beta2) * (g * g)
                    m_hat = m_list[i] / (1 - beta1 ** step_i)
                    v_hat = v_list[i] / (1 - beta2 ** step_i)
                    arr_list[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)
            step += 1
        history.append(epoch_loss / n)
    return mlp, history
