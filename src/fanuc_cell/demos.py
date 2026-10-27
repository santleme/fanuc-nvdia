"""Gravação de demonstrações do professor scriptado (issue #3, BC warm-start).

Núcleo puro (só numpy): quem roda o rollout é o script
``scripts/collect_demos.py`` dentro do env Isaac; aqui mora só o
registro dos pares (obs, ação) no espaço exato da política, com
metadados de qualidade (taxa de grasp/success por episódio) — o
portão do experimento BC lê esses números antes de qualquer treino.

Formato em disco: ``.npz`` com ``observations`` (N, obs_dim),
``actions`` (N, act_dim), ``episode_index`` (N,), ``episode_success``
(E,), ``episode_grasped`` (E,) e ``metadata_json`` (string JSON).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DemoSet:
    """Conjunto de demonstrações + metadados de coleta."""

    observations: np.ndarray
    actions: np.ndarray
    episode_index: np.ndarray
    episode_success: np.ndarray
    episode_grasped: np.ndarray
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.observations = np.asarray(self.observations, dtype=np.float32)
        self.actions = np.asarray(self.actions, dtype=np.float32)
        self.episode_index = np.asarray(self.episode_index, dtype=np.int64)
        self.episode_success = np.asarray(self.episode_success, dtype=bool)
        self.episode_grasped = np.asarray(self.episode_grasped, dtype=bool)
        if self.observations.ndim != 2 or self.actions.ndim != 2:
            raise ValueError("observations/actions devem ser matrizes (N, dim)")
        if len(self.observations) != len(self.actions) \
                or len(self.observations) != len(self.episode_index):
            raise ValueError("observations, actions e episode_index desalinhados")
        if len(self.episode_success) != len(self.episode_grasped):
            raise ValueError("episode_success/episode_grasped desalinhados")

    @property
    def num_steps(self) -> int:
        return int(len(self.observations))

    @property
    def num_episodes(self) -> int:
        return int(len(self.episode_success))

    @property
    def obs_dim(self) -> int:
        return int(self.observations.shape[1])

    @property
    def act_dim(self) -> int:
        return int(self.actions.shape[1])

    @property
    def grasp_rate(self) -> float:
        return float(self.episode_grasped.mean()) if self.num_episodes else 0.0

    @property
    def success_rate(self) -> float:
        return float(self.episode_success.mean()) if self.num_episodes else 0.0

    def summary(self) -> dict:
        return {
            "num_steps": self.num_steps,
            "num_episodes": self.num_episodes,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "grasp_rate": self.grasp_rate,
            "success_rate": self.success_rate,
        }

    def save_npz(self, path) -> None:
        meta = dict(self.metadata)
        meta.setdefault("summary", self.summary())
        np.savez_compressed(
            path,
            observations=self.observations,
            actions=self.actions,
            episode_index=self.episode_index,
            episode_success=self.episode_success,
            episode_grasped=self.episode_grasped,
            metadata_json=np.asarray(json.dumps(meta)),
        )

    @classmethod
    def load_npz(cls, path) -> "DemoSet":
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            return cls(
                observations=data["observations"],
                actions=data["actions"],
                episode_index=data["episode_index"],
                episode_success=data["episode_success"],
                episode_grasped=data["episode_grasped"],
                metadata=metadata,
            )


class DemoRecorder:
    """Acumula passos de rollouts do professor, episódio a episódio."""

    def __init__(self, metadata: dict | None = None) -> None:
        self._obs: list[np.ndarray] = []
        self._act: list[np.ndarray] = []
        self._ep_index: list[int] = []
        self._ep_success: list[bool] = []
        self._ep_grasped: list[bool] = []
        self._current: int | None = None
        self.metadata = dict(metadata or {})

    def begin_episode(self) -> int:
        if self._current is not None:
            raise RuntimeError("begin_episode com episódio ainda aberto")
        self._current = len(self._ep_success)
        return self._current

    def record(self, observation, action) -> None:
        if self._current is None:
            raise RuntimeError("record fora de episódio; chame begin_episode")
        self._obs.append(np.asarray(observation, dtype=np.float32).ravel())
        self._act.append(np.asarray(action, dtype=np.float32).ravel())
        self._ep_index.append(self._current)

    def discard_current_episode(self) -> None:
        """Descarta os pares do episódio aberto e o registra como falha.

        Usado quando o rollout do professor aborta no meio (ex.: IK sem
        convergência): os pares parciais NÃO entram no DemoSet (o BC
        treina em todos os pares gravados), mas o episódio conta no
        denominador do success_rate — o portão de qualidade mede o
        professor inteiro, incluindo as falhas.
        """
        if self._current is None:
            raise RuntimeError("discard sem episódio aberto")
        idx = self._current
        keep = [i for i, ep in enumerate(self._ep_index) if ep != idx]
        self._obs = [self._obs[i] for i in keep]
        self._act = [self._act[i] for i in keep]
        self._ep_index = [self._ep_index[i] for i in keep]
        self._ep_success.append(False)
        self._ep_grasped.append(False)
        self._current = None

    def register_failed_episode(self) -> None:
        """Registra um episódio abortado como falha do professor.

        Com episódio aberto, descarta os pares parciais; sem episódio
        aberto (falha antes do rollout), abre e fecha um episódio
        vazio falho. Nos dois casos o episódio entra no denominador
        do success_rate, sem pares para o BC.
        """
        if self._current is not None:
            self.discard_current_episode()
        else:
            self.begin_episode()
            self.end_episode(success=False, grasped=False)

    def end_episode(self, *, success: bool, grasped: bool) -> None:
        if self._current is None:
            raise RuntimeError("end_episode sem episódio aberto")
        self._ep_success.append(bool(success))
        self._ep_grasped.append(bool(grasped))
        self._current = None

    def to_demoset(self) -> DemoSet:
        if self._current is not None:
            raise RuntimeError("episódio aberto ao fechar o DemoSet")
        if not self._ep_success:
            raise ValueError("nenhum episódio gravado")
        return DemoSet(
            observations=np.stack(self._obs),
            actions=np.stack(self._act),
            episode_index=np.asarray(self._ep_index, dtype=np.int64),
            episode_success=np.asarray(self._ep_success, dtype=bool),
            episode_grasped=np.asarray(self._ep_grasped, dtype=bool),
            metadata=dict(self.metadata),
        )
