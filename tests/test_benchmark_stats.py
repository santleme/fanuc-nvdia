"""B3 (issue #10): agregação multi-seed do benchmark oficial.

A matemática do protocolo mora em ``fanuc_cell.benchmark_stats``
(fonte única, usada pela célula de benchmark do notebook) —
média, desvio AMOSTRAL (ddof=1) e meia-largura do IC95 da média.
"""

from __future__ import annotations

import math

import pytest

from fanuc_cell.benchmark_stats import (
    DEFAULT_BENCHMARK_SEEDS,
    format_rate_summary,
    summarize_rates,
)


def test_default_seeds_are_the_registered_protocol() -> None:
    assert DEFAULT_BENCHMARK_SEEDS == (101, 202, 303)


def test_summarize_three_seeds_known_values() -> None:
    summary = summarize_rates([0.0, 0.5, 1.0])
    assert summary["n"] == 3
    assert summary["mean"] == pytest.approx(0.5)
    # desvio amostral (ddof=1): sqrt((0.25 + 0 + 0.25) / 2) = 0.5
    assert summary["std"] == pytest.approx(0.5)
    # IC95 da média: 1.96 * std / sqrt(n)
    assert summary["ci95_half"] == pytest.approx(1.96 * 0.5 / math.sqrt(3))


def test_summarize_identical_rates_has_zero_dispersion() -> None:
    summary = summarize_rates([0.3, 0.3, 0.3])
    assert summary["mean"] == pytest.approx(0.3)
    assert summary["std"] == 0.0
    assert summary["ci95_half"] == 0.0


def test_summarize_single_seed_is_legacy_shaped() -> None:
    # Seed única (leitura legada pré-B3): a taxa é a média e não
    # há dispersão observável — std/IC95 = 0, não precisão alegada.
    summary = summarize_rates([0.42])
    assert summary["n"] == 1
    assert summary["mean"] == pytest.approx(0.42)
    assert summary["std"] == 0.0
    assert summary["ci95_half"] == 0.0


def test_summarize_rejects_empty() -> None:
    with pytest.raises(ValueError):
        summarize_rates([])


def test_format_rate_summary_reports_mean_std_ci_and_n() -> None:
    text = format_rate_summary(summarize_rates([0.0, 0.5, 1.0]))
    assert text.startswith("0.500 ± 0.500")
    assert "IC95 ± 0.566" in text
    assert "n=3" in text
