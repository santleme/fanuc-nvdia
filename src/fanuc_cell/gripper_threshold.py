"""Medição do limiar de fechamento da garra (issue #13, backlog B6).

A ambiguidade: o MDP declara a garra FECHADA quando as duas juntas
estão ``>= 0,004 m`` (``object_held``/``object_grasped`` em
``mdp/cell.py``, parâmetro ``closed_joint_position``; o baseline usa o
mesmo literal), mas declara a garra ABERTA quando as duas juntas estão
``< 0,005 m`` (``object_at_goal``, literal solto, sem parâmetro). Na
banda [0,004; 0,005) os dois predicados são verdadeiros AO MESMO
TEMPO: a mesma pose conta como grasp para um termo e como soltura
para o termo de sucesso. Com o grasp sendo o gargalo central do
projeto, o limiar que define sucesso não pode viver nessa zona
cinzenta — e a correção não é chutar um dos dois valores: é MEDIR
onde a física põe o pinch de verdade.

Este módulo é a análise pura dessa medição (numpy, sem Isaac/torch):
recebe as amostras por passo gravadas pelo probe
(``scripts/probe_gripper_threshold.py``, que roda no Kaggle) e devolve
a varredura do limiar contra a força de contato medida + a
recomendação pela regra pré-registrada na SPEC (protocolo
``fanuc-gripper-threshold-b6``):

- verdade física do pinch = contato bilateral (as duas forças normais
  dos dedos acima de ``CONTACT_FORCE_N``, o mesmo limiar do MDP);
- ``pinch_p05`` = percentil 5 da junta mínima nos passos de contato
  bilateral SUSTENTADO (>= 5 passos) das fases de segurar — o teto que
  o limiar de fechamento não pode ultrapassar sem perder pinches
  reais;
- ``contact_loss_p05`` = percentil 5 da junta máxima no passo de
  perda de contato durante a abertura — o teto que o limiar de
  abertura não pode ultrapassar sem declarar "aberta" com a peça
  ainda pinçada;
- valor único admissível: ``T* <= min(pinch_p05, contact_loss_p05)``;
  adota-se o maior ponto da grade de candidatos sob esse teto.
"""

from __future__ import annotations

import numpy as np

# Valores atuais em disputa (a banda é o intervalo entre eles).
CURRENT_CLOSED_THRESHOLD_M = 0.004  # mdp (x2) + baseline scriptado
CURRENT_OPEN_THRESHOLD_M = 0.005  # mdp object_at_goal (literal solto)

# Verdade física: o mesmo limiar de contato bilateral do MDP.
CONTACT_FORCE_N = 0.01
# Fases em que segurar a peça é o comportamento esperado (pinch real).
PINCH_PHASES = ("grasp", "hold", "lift", "transfer", "place")
RELEASE_PHASES = ("release", "retreat")
# Contato só conta como pinch sustentado em corridas >= este tamanho.
MIN_SUSTAINED_CONTACT_STEPS = 5
# A perda de contato só é confirmada com esta sequência sem contato.
CONTACT_LOSS_CONFIRM_STEPS = 10
# Grade de candidatos varridos contra a força medida (m).
CANDIDATE_GRID_M = tuple(
    round(float(value), 5) for value in np.arange(0.0020, 0.0080001, 0.00025)
)
# Fronteiras da regra pré-registrada (SPEC, protocolo B6).
PROVA_MIN_M = 0.0045

_SAMPLE_FIELDS = (
    "scenario",
    "phase",
    "joint_left",
    "joint_right",
    "force_left_n",
    "force_right_n",
)


def bilateral_contact(sample: dict) -> bool:
    """Verdade física do pinch: os dois dedos tocando a peça."""
    return (
        float(sample["force_left_n"]) > CONTACT_FORCE_N
        and float(sample["force_right_n"]) > CONTACT_FORCE_N
    )


def _min_joint(sample: dict) -> float:
    return min(float(sample["joint_left"]), float(sample["joint_right"]))


def _max_joint(sample: dict) -> float:
    return max(float(sample["joint_left"]), float(sample["joint_right"]))


def _segments(samples: list[dict]) -> list[list[dict]]:
    """Agrupa amostras em sequências contíguas (scenario, rep)."""
    segments: list[list[dict]] = []
    current: list[dict] = []
    current_key: tuple | None = None
    for sample in samples:
        key = (sample["scenario"], sample.get("rep", 0))
        if key != current_key:
            if current:
                segments.append(current)
            current = []
            current_key = key
        current.append(sample)
    if current:
        segments.append(current)
    return segments


def sustained_contact_mask(samples: list[dict]) -> list[bool]:
    """Marca passos em corridas de contato bilateral >= o mínimo.

    Um toque isolado (bounce do sensor, raspão na descida) não é
    pinch: só corridas sustentadas dentro do mesmo (scenario, rep)
    contam para medir onde a garra segura de verdade.
    """
    mask = [False] * len(samples)
    offset = 0
    for segment in _segments(samples):
        contacts = [bilateral_contact(sample) for sample in segment]
        index = 0
        while index < len(contacts):
            if not contacts[index]:
                index += 1
                continue
            end = index
            while end < len(contacts) and contacts[end]:
                end += 1
            if end - index >= MIN_SUSTAINED_CONTACT_STEPS:
                for position in range(offset + index, offset + end):
                    mask[position] = True
            index = end
        offset += len(segment)
    return mask


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _confirmed_loss(contacts: list[bool], index: int) -> bool:
    """Perda confirmada: contato em ``index`` e a janela seguinte limpa."""
    window = contacts[index + 1: index + 1 + CONTACT_LOSS_CONFIRM_STEPS]
    return len(window) == CONTACT_LOSS_CONFIRM_STEPS and not any(window)


def _contact_loss_max_joints(samples: list[dict]) -> list[float]:
    """Junta máxima no passo de perda de contato, por rep do pinch.

    Procura, nas fases de abertura, o último passo com contato
    bilateral seguido de uma sequência confirmada sem contato; se a
    abertura não produzir um, cai para o último contato do episódio
    com a mesma confirmação. É o ponto em que os dedos fisicamente
    soltam a peça: declarar "aberta" acima dele é declarar soltura
    com a peça ainda pinçada.
    """
    losses: list[float] = []
    for segment in _segments(samples):
        if segment[0]["scenario"] != "pinch":
            continue
        contacts = [bilateral_contact(sample) for sample in segment]

        loss_index: int | None = None
        for index in range(len(segment) - 1, -1, -1):
            if (
                contacts[index]
                and segment[index]["phase"] in RELEASE_PHASES
                and _confirmed_loss(contacts, index)
            ):
                loss_index = index
                break
        if loss_index is None:
            for index in range(len(segment) - 1, -1, -1):
                if contacts[index] and _confirmed_loss(contacts, index):
                    loss_index = index
                    break
        if loss_index is not None:
            losses.append(_max_joint(segment[loss_index]))
    return losses


def _empty_stall_median(samples: list[dict]) -> float | None:
    """Platô da junta mínima no fechamento vazio (garra sem peça)."""
    closing = [
        sample
        for sample in samples
        if sample["scenario"] == "empty" and sample["phase"] == "empty_close"
    ]
    if not closing:
        return None
    tail = closing[len(closing) - max(1, int(np.ceil(len(closing) * 0.2))):]
    return float(np.median([_min_joint(sample) for sample in tail]))


def sweep_thresholds(samples: list[dict]) -> list[dict]:
    """Varredura: cada candidato contra a verdade do contato medido.

    ``closed(T)`` replica a semântica do MDP (as DUAS juntas >= T).
    FP = declarar fechada sem contato (fechamento vazio contando como
    grasp); FN = não declarar fechada com contato (pinch real
    perdido). Taxas sobre o total de passos sem/com contato.
    """
    contacts = [bilateral_contact(sample) for sample in samples]
    min_joints = np.asarray([_min_joint(sample) for sample in samples])
    contact_count = int(sum(contacts))
    no_contact_count = len(samples) - contact_count
    rows = []
    for candidate in CANDIDATE_GRID_M:
        closed = min_joints >= candidate
        false_positive = int(sum(closed & ~np.asarray(contacts)))
        false_negative = int(sum(~closed & np.asarray(contacts)))
        rows.append(
            {
                "threshold_m": candidate,
                "false_positive_rate": (
                    false_positive / no_contact_count if no_contact_count else None
                ),
                "false_negative_rate": (
                    false_negative / contact_count if contact_count else None
                ),
                "false_positives": false_positive,
                "false_negatives": false_negative,
            }
        )
    return rows


def recommend_threshold(
    pinch_p05: float | None,
    contact_loss_p05: float | None,
) -> dict:
    """Regra pré-registrada do protocolo B6 (SPEC) — não ajustar aqui.

    PROVA: T* >= 0,0045 — um valor único acima da banda separa os
    dois lados; adotar T* (fechada := min >= T*; aberta := max < T*).
    PARCIAL: 0,004 <= T* < 0,0045 — manter 0,004 como valor único e
    alinhar só o literal de abertura; a banda some por construção.
    MATA: T* < 0,004 (ou grade vazia) — nem o valor do código se
    sustenta na medição; escalar o desenho da detecção de grasp.
    SEM_SINAL: falta pinch sustentado ou perda de contato medida —
    a medição não aconteceu; nada se decide.
    """
    if pinch_p05 is None or contact_loss_p05 is None:
        return {
            "verdict": "SEM_SINAL",
            "threshold_m": None,
            "admissible_max_m": None,
            "pinch_p05_m": pinch_p05,
            "contact_loss_p05_m": contact_loss_p05,
            "current_closed_ok": None,
            "current_open_ok": None,
        }
    admissible_max = min(pinch_p05, contact_loss_p05)
    candidates = [c for c in CANDIDATE_GRID_M if c <= admissible_max + 1e-9]
    threshold = candidates[-1] if candidates else None
    if threshold is None or threshold < CURRENT_CLOSED_THRESHOLD_M:
        verdict = "MATA"
    elif threshold < PROVA_MIN_M:
        verdict = "PARCIAL"
        threshold = CURRENT_CLOSED_THRESHOLD_M
    else:
        verdict = "PROVA"
    return {
        "verdict": verdict,
        "threshold_m": threshold,
        "admissible_max_m": admissible_max,
        "pinch_p05_m": pinch_p05,
        "contact_loss_p05_m": contact_loss_p05,
        "current_closed_ok": CURRENT_CLOSED_THRESHOLD_M <= admissible_max,
        "current_open_ok": CURRENT_OPEN_THRESHOLD_M <= contact_loss_p05,
    }


def analyze_threshold_samples(samples: list[dict]) -> dict:
    """Análise completa das amostras do probe -> relatório + veredito."""
    if not samples:
        raise ValueError("probe nao gravou nenhuma amostra")
    for sample in samples:
        missing = [field for field in _SAMPLE_FIELDS if field not in sample]
        if missing:
            raise ValueError(f"amostra sem campos {missing}: {sample!r}")

    sustained = sustained_contact_mask(samples)
    pinch_min_joints = [
        _min_joint(sample)
        for sample, is_sustained in zip(samples, sustained)
        if is_sustained
        and sample["scenario"] == "pinch"
        and sample["phase"] in PINCH_PHASES
    ]
    loss_max_joints = _contact_loss_max_joints(samples)
    pinch_p05 = _percentile(pinch_min_joints, 5.0)
    contact_loss_p05 = _percentile(loss_max_joints, 5.0)

    contacts = [bilateral_contact(sample) for sample in samples]
    in_band = [
        _min_joint(sample) >= CURRENT_CLOSED_THRESHOLD_M
        and _max_joint(sample) < CURRENT_OPEN_THRESHOLD_M
        for sample in samples
    ]
    contact_count = int(sum(contacts))

    recommendation = recommend_threshold(pinch_p05, contact_loss_p05)
    return {
        "protocol": "fanuc-gripper-threshold-b6",
        "sample_count": len(samples),
        "contact_step_count": contact_count,
        "pinch_sustained_step_count": len(pinch_min_joints),
        "contact_loss_measurement_count": len(loss_max_joints),
        "pinch_min_joint_m": {
            "p05": pinch_p05,
            "median": _percentile(pinch_min_joints, 50.0),
            "min": float(min(pinch_min_joints)) if pinch_min_joints else None,
        },
        "contact_loss_max_joint_m": {
            "p05": contact_loss_p05,
            "values": loss_max_joints,
        },
        "empty_close_stall_min_joint_m": _empty_stall_median(samples),
        # Ocupação da banda ambígua atual: passos em que o MDP diz
        # "fechada" (min >= 0,004) e "aberta" (max < 0,005) ao mesmo
        # tempo — a medida do tamanho real da zona cinzenta.
        "current_band_occupancy": {
            "all_steps": float(np.mean(in_band)),
            "contact_steps": (
                float(np.mean([b for b, c in zip(in_band, contacts) if c]))
                if contact_count
                else None
            ),
            "no_contact_steps": (
                float(np.mean([b for b, c in zip(in_band, contacts) if not c]))
                if len(samples) - contact_count
                else None
            ),
        },
        "sweep": sweep_thresholds(samples),
        "recommendation": recommendation,
        "verdict": recommendation["verdict"],
    }
