"""Logger Rerun de rollouts para diagnóstico visual do grasp (issue #6).

Ferramenta de diagnóstico OPCIONAL: grava um ``.rrd`` com as séries
sincronizadas por passo que a análise da coleta BC v3 precisou
extrair de JSON/log na mão — pose do TCP × pose da peça, forças de
contato dos dois dedos, juntas da garra, fase do professor e eventos
(grasp/captura/sucesso). Abrir o ``.rrd`` no viewer do Rerun permite
fazer scrub do episódio e ver o desalinhamento na descida.

Proteções (aceitação da issue):
- import do ``rerun-sdk`` é protegido: sem o pacote instalado, o
  logger vira no-op silencioso — nenhum script, kernel ou treino
  quebra por causa desta ferramenta;
- default DESLIGADO: os scripts só criam o logger com ``--rerun``;
  sem a flag, nenhum método é chamado e o comportamento é o atual;
- o SDK é dependência opcional (extra ``rerun`` no pyproject), nunca
  requisito duro do projeto.

O módulo é puro (só numpy) e aceita um SDK injetado — os testes usam
um stub que grava as chamadas, sem rede nem display.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:  # import protegido: ferramenta opcional, nunca um ponto de falha
    import rerun as _rerun_sdk
except ImportError:  # pragma: no cover - depende do ambiente
    _rerun_sdk = None


def rerun_available() -> bool:
    """True se o rerun-sdk está importável neste ambiente."""
    return _rerun_sdk is not None


class RolloutLogger:
    """Grava um rollout (1+ episódios) em ``.rrd`` via rerun-sdk.

    Inativo (``active == False``) quando desabilitado ou sem SDK:
    todos os métodos viram no-op imediato, então os scripts podem
    chamar sem branch quando o logger existe.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        app_id: str = "fanuc_cell_rollout",
        enabled: bool = True,
        part_size_m: float = 0.08,
        sdk=None,
    ) -> None:
        self._path = Path(path) if path is not None else None
        self._part_size = float(part_size_m)
        self._sdk = sdk if sdk is not None else _rerun_sdk
        self._active = bool(enabled) and self._sdk is not None
        self._episode: int | None = None
        self._last_phase: str | None = None
        self._finished = False
        if self._active:
            self._sdk.init(app_id)
            if self._path is not None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._sdk.save(str(self._path))

    @property
    def active(self) -> bool:
        return self._active

    @property
    def path(self) -> Path | None:
        return self._path

    # -- episódios e eventos -------------------------------------------------

    def start_episode(self, episode_index: int) -> None:
        if not self._active:
            return
        self._episode = int(episode_index)
        self._last_phase = None
        self.log_event("episode_start", episode=self._episode)

    def log_event(self, name: str, **details) -> None:
        if not self._active:
            return
        text = name
        if details:
            text += " " + " ".join(f"{k}={v}" for k, v in details.items())
        self._sdk.log("events", self._sdk.TextLog(text))

    def finish_episode(self, *, success: bool | None = None,
                       grasped: bool | None = None) -> None:
        if not self._active:
            return
        self.log_event("episode_end", episode=self._episode,
                       success=success, grasped=grasped)
        self._episode = None

    # -- passos ---------------------------------------------------------------

    def log_step(
        self,
        *,
        step: int,
        time_s: float | None = None,
        phase: str | None = None,
        tcp_pos,
        part_pos,
        left_force_n: float | None = None,
        right_force_n: float | None = None,
        gripper_joints=None,
    ) -> None:
        """Um passo do rollout: poses, contato, garra e fase.

        ``tcp_pos``/``part_pos`` em metros, no mesmo referencial
        (os scripts passam posições relativas à origem do env).
        """
        if not self._active:
            return
        rr = self._sdk
        rr.set_time_sequence("step", int(step))
        if time_s is not None:
            rr.set_time_seconds("sim_time", float(time_s))
        tcp = np.asarray(tcp_pos, dtype=np.float64).ravel()
        part = np.asarray(part_pos, dtype=np.float64).ravel()
        rr.log("world/tcp", rr.Transform3D(translation=tcp, axis_length=0.08))
        rr.log("world/part", rr.Boxes3D(
            centers=[part],
            sizes=[[self._part_size, self._part_size, self._part_size]],
        ))
        distance = float(np.linalg.norm(part - tcp))
        rr.log("tcp_part/distance_m", rr.Scalars([distance]))
        if left_force_n is not None:
            rr.log("contact/left_force_n", rr.Scalars([float(left_force_n)]))
        if right_force_n is not None:
            rr.log("contact/right_force_n", rr.Scalars([float(right_force_n)]))
        if gripper_joints is not None:
            joints = np.asarray(gripper_joints, dtype=np.float64).ravel()
            if len(joints) >= 2:
                rr.log("gripper/left_joint_rad", rr.Scalars([float(joints[0])]))
                rr.log("gripper/right_joint_rad", rr.Scalars([float(joints[1])]))
        if phase is not None and phase != self._last_phase:
            rr.log("phase", rr.TextLog(str(phase)))
            self._last_phase = str(phase)

    # -- encerramento -----------------------------------------------------------

    def finish(self) -> Path | None:
        """Fecha a gravação; devolve o caminho do ``.rrd`` (ou None)."""
        if not self._active or self._finished:
            return self._path if self._active else None
        self._finished = True
        disconnect = getattr(self._sdk, "disconnect", None)
        if callable(disconnect):
            disconnect()
        return self._path


def make_rollout_logger(enabled: bool, path: str | Path | None = None,
                        **kwargs) -> RolloutLogger | None:
    """Fábrica usada pelos scripts: None sem a flag ``--rerun``.

    Sem a flag, os scripts nem criam o objeto — garantia estrutural
    de que o comportamento sem Rerun é idêntico ao atual. Com a flag
    mas sem o SDK instalado, devolve um logger inativo (no-op) e o
    script avisa no log; nada quebra.
    """
    if not enabled:
        return None
    return RolloutLogger(path, enabled=True, **kwargs)
