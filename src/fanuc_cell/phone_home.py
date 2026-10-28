"""Phone-home de telemetria para kernels Kaggle (regra do fundador, 28/10/2026).

O Kaggle NÃO publica log nem saída de um kernel enquanto ele roda — pela
API, uma corrida longa é uma caixa-preta até o terminal (foi assim que uma
sessão do cache builder ficou ~11 h pendurada sem ninguém enxergar). Este
módulo inverte o fluxo: o próprio kernel manda sinais curtos para fora, em
tempo real, e o monitor lê de fora.

Canal: um tópico público num servidor ntfy (instância pública
ntfy.envs.net por padrão; a base é configurável porque o ntfy.sh é
inacessível do ambiente do monitor). O tópico é gerado por lançamento e
funciona como segredo fraco — só trafegam nome de estágio, tempos e
contadores (nunca credencial, nunca dado do usuário). Sem tópico
configurado, tudo aqui é no-op silencioso.

Uso dentro do notebook (o builder injeta quando --phone-home-topic é
passado; o fonte deste módulo é embutido na célula principal):
    configure("fanuc-<slug>-<sufixo>"); start_heartbeat(120)
    stage_done("coleta", 123.4) a cada fim de estágio.

Leitura pelo monitor (de fora):
    curl -s "https://ntfy.envs.net/<topico>/json?poll=1&since=30m"

Stdlib apenas (urllib/threading). Toda falha de rede é engolida: telemetria
nunca pode derrubar o experimento.
"""

from __future__ import annotations

import os
import threading
import time
import urllib.request

_DEFAULT_BASE = "https://ntfy.envs.net"
_state: dict = {"topic": None, "base": _DEFAULT_BASE,
                "started_at": None, "last_stage": None}
_heartbeat_stop: threading.Event | None = None


def configure(topic: str | None = None, base: str | None = None) -> str | None:
    """Define o tópico (ou lê FANUC_PHONE_HOME_TOPIC do ambiente)."""
    chosen = topic or os.environ.get("FANUC_PHONE_HOME_TOPIC") or None
    _state["topic"] = chosen
    _state["base"] = (base or os.environ.get("FANUC_PHONE_HOME_BASE")
                      or _DEFAULT_BASE).rstrip("/")
    if _state["started_at"] is None:
        _state["started_at"] = time.monotonic()
    return chosen


def _post(message: str) -> bool:
    topic = _state.get("topic")
    if not topic:
        return False
    try:
        request = urllib.request.Request(
            f"{_state.get('base', _DEFAULT_BASE)}/{topic}",
            data=message[:600].encode("utf-8"),
            method="POST",
            headers={"Title": "FANUC kernel"},
        )
        with urllib.request.urlopen(request, timeout=10):
            pass
        return True
    except Exception:
        return False


def note(message: str) -> bool:
    """Manda uma linha livre (boot, marco, aviso)."""
    return _post(str(message))


def stage_done(name: str, seconds: float | None = None) -> bool:
    """Registra o fim de um estágio e avisa com o tempo dele."""
    _state["last_stage"] = name
    if seconds is None:
        return _post(f"stage={name} done")
    return _post(f"stage={name} done seconds={seconds}")


def _beat_once() -> bool:
    started = _state.get("started_at")
    elapsed = 0.0 if started is None else time.monotonic() - started
    last = _state.get("last_stage") or "-"
    return _post(f"heartbeat elapsed_s={elapsed:.0f} last_stage={last}")


def start_heartbeat(interval_seconds: float = 120.0) -> threading.Thread | None:
    """Sobe a thread de heartbeat (daemon). Sem tópico, não sobe nada."""
    global _heartbeat_stop
    if not _state.get("topic"):
        return None
    if _heartbeat_stop is not None and not _heartbeat_stop.is_set():
        return None
    _heartbeat_stop = threading.Event()
    stop = _heartbeat_stop

    def _loop() -> None:
        while not stop.wait(interval_seconds):
            _beat_once()

    thread = threading.Thread(target=_loop, name="phone-home", daemon=True)
    thread.start()
    return thread
