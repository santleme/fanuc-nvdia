"""Agregação multi-seed do benchmark oficial (B3, issue #10).

Módulo puro (só ``math`` da stdlib): sem numpy, sem Isaac — pode ser
importado em qualquer contexto, inclusive dentro da célula de
benchmark do notebook Kaggle, que é quem grava o veredito no
runtime-report. Fonte única da matemática do protocolo: média,
desvio amostral e IC95 da média das taxas por seed.

Seed única não separa sinal de ruído — todos os vereditos até a
BC v7 foram lidos num benchmark de seed única (101). O protocolo
B3 roda o benchmark oficial nas seeds fixas de
``DEFAULT_BENCHMARK_SEEDS`` e reporta média ± desvio (e o IC95).
"""

from __future__ import annotations

import math
from collections.abc import Iterable

DEFAULT_BENCHMARK_SEEDS: tuple[int, ...] = (101, 202, 303)

# z do IC95 (aproximação normal). Com n=3 o intervalo é largo por
# construção — é o preço honesto de separar sinal de ruído com
# poucas seeds; ele existe para o veredito não fingir precisão.
_Z_95 = 1.96


def summarize_rates(rates: Iterable[float]) -> dict[str, float | int]:
    """Resume taxas por seed: ``n``, ``mean``, ``std``, ``ci95_half``.

    ``std`` é o desvio AMOSTRAL (ddof=1) — com 1 seed, ``std`` e
    ``ci95_half`` são 0.0 (não há dispersão observável, não uma
    alegação de precisão). ``ci95_half`` é a meia-largura do
    intervalo de confiança de 95% da média (z × std / √n).
    """
    values = [float(rate) for rate in rates]
    if not values:
        raise ValueError("summarize_rates exige ao menos uma taxa (uma seed).")
    n = len(values)
    mean = sum(values) / n
    if n > 1:
        variance = sum((value - mean) ** 2 for value in values) / (n - 1)
        std = math.sqrt(variance)
        ci95_half = _Z_95 * std / math.sqrt(n)
    else:
        std = 0.0
        ci95_half = 0.0
    return {"n": n, "mean": mean, "std": std, "ci95_half": ci95_half}


def format_rate_summary(summary: dict[str, float | int]) -> str:
    """Linha de veredito: ``0.500 ± 0.100 (IC95 ± 0.113, n=3)``."""
    return (
        f"{summary['mean']:.3f} ± {summary['std']:.3f} "
        f"(IC95 ± {summary['ci95_half']:.3f}, n={summary['n']})"
    )
