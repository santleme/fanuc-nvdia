"""Phone-home dos kernels (regra do fundador, 28/10/2026): testes estruturais.

O módulo é stdlib e fail-silent; os testes trocam o urlopen por um coletor
local — nenhuma rede de verdade é tocada. Os testes do builder seguem o
padrão da casa: o builder roda de verdade (saída em tmp) e as afirmações
são sobre o fonte do notebook gerado.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fanuc_cell import phone_home  # noqa: E402


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def sent(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append((request.full_url, request.data.decode("utf-8")))
        return _FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    phone_home._state.update(
        {"topic": None, "base": phone_home._DEFAULT_BASE,
         "started_at": None, "last_stage": None})
    return calls


def test_no_topic_is_silent_noop(sent):
    phone_home.configure(None)
    assert phone_home.note("oi") is False
    assert phone_home.stage_done("coleta", 1.0) is False
    assert phone_home.start_heartbeat(0.01) is None
    assert sent == []


def test_note_posts_to_topic(sent):
    phone_home.configure("fanuc-teste-123")
    assert phone_home.note("boot: kernel iniciado") is True
    assert sent == [("https://ntfy.envs.net/fanuc-teste-123",
                     "boot: kernel iniciado")]


def test_configure_base_override(sent):
    phone_home.configure("fanuc-teste-123", base="https://exemplo.local/")
    assert phone_home.note("base propria") is True
    assert sent[-1][0] == "https://exemplo.local/fanuc-teste-123"


def test_stage_done_records_and_posts(sent):
    phone_home.configure("fanuc-teste-123")
    assert phone_home.stage_done("coleta", 12.5) is True
    assert phone_home._state["last_stage"] == "coleta"
    assert "stage=coleta done seconds=12.5" in sent[-1][1]


def test_network_failure_is_swallowed(monkeypatch):
    def boom(request, timeout=None):
        raise OSError("rede fora")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    phone_home._state.update(
        {"topic": "fanuc-teste-123", "base": phone_home._DEFAULT_BASE,
         "started_at": None, "last_stage": None})
    assert phone_home.note("nao pode levantar") is False


def test_heartbeat_beat_once(sent):
    phone_home.configure("fanuc-teste-123")
    phone_home.stage_done("treino", 3.0)
    sent.clear()
    assert phone_home._beat_once() is True
    assert "heartbeat" in sent[-1][1]
    assert "last_stage=treino" in sent[-1][1]


def test_env_var_configures(monkeypatch, sent):
    monkeypatch.setenv("FANUC_PHONE_HOME_TOPIC", "fanuc-do-env-1")
    assert phone_home.configure() == "fanuc-do-env-1"
    assert phone_home.note("via env") is True


# ---------------------------------------------------------------- builder


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "builder_phone_home", ROOT / "scripts/build_kaggle_notebook.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    return builder


def _build(tmp_path, monkeypatch, extra_argv):
    builder = _load_builder()
    monkeypatch.setattr(builder, "KAGGLE_DIR", tmp_path)
    monkeypatch.setattr(
        sys, "argv",
        ["build_kaggle_notebook.py", "--kaggle-user", "tester",
         *extra_argv])
    builder.main()
    notebook = json.loads((tmp_path / builder.NOTEBOOK_NAME).read_text())
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


def test_builder_flag_and_manifest_wiring() -> None:
    builder = (ROOT / "scripts/build_kaggle_notebook.py").read_text()
    assert '"--phone-home-topic"' in builder
    assert '"src/fanuc_cell/phone_home.py"' in builder
    assert '"tests/test_phone_home.py"' in builder


def test_builder_default_notebook_has_no_phone_home(tmp_path, monkeypatch):
    source = _build(tmp_path, monkeypatch, [])
    assert "PHONE_HOME_TOPIC" not in source
    assert "ntfy.envs.net" not in source


def test_builder_phone_home_notebook_is_wired(tmp_path, monkeypatch):
    source = _build(tmp_path, monkeypatch,
                    ["--phone-home-topic", "fanuc-teste-123"])
    assert "PHONE_HOME_TOPIC = 'fanuc-teste-123'" in source
    assert "ntfy.envs.net" in source
    assert "_ph.configure(PHONE_HOME_TOPIC, base='https://ntfy.envs.net')" \
        in source
    assert "_ph.start_heartbeat(120)" in source
    assert "_ph_stage(name, elapsed)" in source
    # o gancho entra no stage_end, depois do print [PERF] do estágio
    assert source.index("_ph_stage(name, elapsed)") > source.index(
        "[PERF] stage={name}")


def test_builder_rejects_bad_topic(tmp_path, monkeypatch):
    builder = _load_builder()
    monkeypatch.setattr(builder, "KAGGLE_DIR", tmp_path)
    monkeypatch.setattr(
        sys, "argv",
        ["build_kaggle_notebook.py", "--kaggle-user", "tester",
         "--phone-home-topic", "topico com espaco!"])
    with pytest.raises(SystemExit):
        builder.main()
