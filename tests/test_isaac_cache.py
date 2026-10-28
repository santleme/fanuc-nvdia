"""Cache da instalação do Isaac Lab (issue #12, B4): contratos estruturais.

Sem Isaac, sem Kaggle, sem rede: manifesto/descoberta/plano são
funções puras sobre diretórios temporários; o prepare roda sobre uma
árvore sintética; e a montagem dos notebooks é conferida gerando-os
de verdade com os builders — default intocado, flag ligada com
fallback literal, e as células geradas compilam.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fanuc_cell import isaac_cache
from fanuc_cell.isaac_cache import (
    ISAACLAB_PART,
    MANIFEST_NAME,
    UV_CACHE_PART,
    build_cache_manifest,
    cache_restore_plan,
    cache_snapshot_plan,
    find_cache_mount,
    snapshot_size_problems,
    validate_cache_manifest,
)


def _manifest(**overrides) -> dict:
    manifest = build_cache_manifest(
        parts={ISAACLAB_PART: 100, UV_CACHE_PART: 50},
        python_version="3.12.3",
        platform="Linux-test",
        created_by="test",
    )
    manifest.update(overrides)
    return manifest


def _write_mount(mount: Path, manifest: dict, *, with_part: bool = True) -> Path:
    mount.mkdir(parents=True, exist_ok=True)
    (mount / MANIFEST_NAME).write_text(json.dumps(manifest))
    if with_part:
        (mount / ISAACLAB_PART).mkdir(exist_ok=True)
    return mount


def test_manifest_roundtrip_is_valid() -> None:
    manifest = _manifest()
    assert validate_cache_manifest(manifest) == []
    assert manifest["isaaclab_ref"] == isaac_cache.ISAACLAB_REF == "v3.0.0"
    assert manifest["format"] == isaac_cache.CACHE_FORMAT_VERSION
    assert set(manifest["warmed_extras"]) == set(isaac_cache.WARMED_EXTRAS)


def test_manifest_rejects_wrong_version_format_and_missing_part() -> None:
    assert validate_cache_manifest(_manifest(isaaclab_ref="v2.1.0"))
    assert validate_cache_manifest(_manifest(format=99))
    assert validate_cache_manifest(_manifest(parts={UV_CACHE_PART: 1}))
    assert validate_cache_manifest(["not", "a", "dict"])
    assert validate_cache_manifest(None)


def test_find_cache_mount_skips_absent_invalid_and_picks_valid(tmp_path) -> None:
    valid = _write_mount(tmp_path / "ok", _manifest())
    stale = _write_mount(tmp_path / "stale", _manifest(isaaclab_ref="v2.0.0"))
    empty = tmp_path / "empty"
    empty.mkdir()

    assert find_cache_mount([empty]) is None
    assert find_cache_mount([stale]) is None
    assert find_cache_mount([empty, stale, valid]) == valid
    assert find_cache_mount([None, valid]) == valid


def test_find_cache_mount_requires_the_tree_not_just_the_manifest(tmp_path) -> None:
    mount = _write_mount(tmp_path, _manifest(), with_part=False)
    assert find_cache_mount([mount]) is None


def test_restore_plan_targets_the_install_paths(tmp_path) -> None:
    mount = _write_mount(tmp_path, _manifest())
    (mount / UV_CACHE_PART).mkdir()
    plan = cache_restore_plan(mount, working_dir="/kaggle/working",
                              home_dir="/root")
    by_dest = {step["dest"]: step for step in plan}
    isaaclab = by_dest["/kaggle/working/IsaacLab"]
    assert isaaclab["source"] == str(mount / ISAACLAB_PART)
    assert isaaclab["merge"] is False
    uv_cache = by_dest["/root/.cache/uv"]
    assert uv_cache["source"] == str(mount / UV_CACHE_PART)
    assert uv_cache["merge"] is True


def test_restore_plan_omits_uv_cache_when_not_in_dataset(tmp_path) -> None:
    mount = _write_mount(tmp_path, _manifest())
    plan = cache_restore_plan(mount, working_dir="/w", home_dir="/h")
    assert [step["dest"] for step in plan] == ["/w/IsaacLab"]


def test_prepare_snapshot_layout_manifest_and_verify(tmp_path) -> None:
    spec = importlib.util.spec_from_file_location(
        "prepare_isaac_cache", ROOT / "scripts/prepare_isaac_cache.py")
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)

    source = tmp_path / "IsaacLab"
    (source / ".venv" / "bin").mkdir(parents=True)
    (source / ".venv" / "bin" / "python").write_text("#!python\n")
    (source / "pyproject.toml").write_text("[project]\n")
    (source / ".git").mkdir()
    (source / ".git" / "HEAD").write_text("ref\n")
    uv_cache = tmp_path / "uvcache"
    uv_cache.mkdir()
    (uv_cache / "wheel.bin").write_bytes(b"x" * 10)

    out = tmp_path / "isaaclab-cache"
    manifest = prep.snapshot_cache(
        source_dir=source, uv_cache_dir=uv_cache, out_dir=out,
        created_by="test-builder")

    assert (out / MANIFEST_NAME).is_file()
    assert (out / ISAACLAB_PART / ".venv" / "bin" / "python").is_file()
    assert not (out / ISAACLAB_PART / ".git").exists()
    assert (out / UV_CACHE_PART / "wheel.bin").is_file()
    assert manifest["parts"][UV_CACHE_PART] == 10
    assert prep.verify_cache(out) == []
    # Snapshot é MOVIMENTO (correção da tentativa v2, Errno 28 por
    # cópia): as origens não existem mais — nada foi duplicado.
    assert not source.exists()
    assert not uv_cache.exists()


def test_prepare_verify_rejects_tree_without_venv(tmp_path) -> None:
    spec = importlib.util.spec_from_file_location(
        "prepare_isaac_cache", ROOT / "scripts/prepare_isaac_cache.py")
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)

    broken = _write_mount(tmp_path, _manifest())
    assert any(".venv" in p for p in prep.verify_cache(broken))
    assert prep.verify_cache(tmp_path / "nada") != []


def _load_prepare():
    spec = importlib.util.spec_from_file_location(
        "prepare_isaac_cache", ROOT / "scripts/prepare_isaac_cache.py")
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    return prep


def test_snapshot_plan_is_prune_then_move_as_data() -> None:
    # O plano do snapshot (espelho do plano de restauração): podar
    # primeiro, MOVER depois — nenhuma ação de cópia existe no plano.
    plan = cache_snapshot_plan(
        source_dir="/kaggle/working/IsaacLab",
        uv_cache_dir="/kaggle/working/uv-cache",
        out_dir="/kaggle/working/isaaclab-cache")
    assert [step["action"] for step in plan] == [
        "prune", "prune", "move", "move"]
    prune_lab, prune_uv = plan[0], plan[1]
    assert prune_lab["root"] == "/kaggle/working/IsaacLab"
    assert prune_lab["dir_names"] == ["__pycache__"]
    assert prune_lab["drop_root_names"] == [".git", "logs"]
    assert prune_uv["root"] == "/kaggle/working/uv-cache"
    assert prune_uv["dir_names"] == ["__pycache__"]
    assert prune_uv["drop_root_names"] == []
    moves = {step["part"]: step for step in plan if step["action"] == "move"}
    assert moves[ISAACLAB_PART]["source"] == "/kaggle/working/IsaacLab"
    assert moves[ISAACLAB_PART]["dest"] == "/kaggle/working/isaaclab-cache/isaaclab"
    assert moves[UV_CACHE_PART]["source"] == "/kaggle/working/uv-cache"
    assert moves[UV_CACHE_PART]["dest"] == "/kaggle/working/isaaclab-cache/uv-cache"


def test_snapshot_plan_omits_uv_cache_when_absent() -> None:
    plan = cache_snapshot_plan(
        source_dir="/s", uv_cache_dir="/u", out_dir="/o",
        uv_cache_exists=False)
    assert [step["action"] for step in plan] == ["prune", "move"]
    assert plan[-1]["dest"] == "/o/isaaclab"


def test_snapshot_size_problems_gate() -> None:
    assert snapshot_size_problems(
        {ISAACLAB_PART: 10, UV_CACHE_PART: 5}, max_total_bytes=15) == []
    problems = snapshot_size_problems(
        {ISAACLAB_PART: 16}, max_total_bytes=15)
    assert len(problems) == 1
    assert "16" in problems[0] and "15" in problems[0]
    # Sem a parte obrigatória, não há snapshot que preste.
    assert snapshot_size_problems({}, max_total_bytes=15)
    # O limiar declarado é o default e a fronteira exata passa.
    assert isaac_cache.SNAPSHOT_MAX_TOTAL_BYTES == 32 * 1024**3
    assert snapshot_size_problems(
        {ISAACLAB_PART: isaac_cache.SNAPSHOT_MAX_TOTAL_BYTES}) == []


def test_prepare_snapshot_moves_prunes_and_verifies(tmp_path) -> None:
    prep = _load_prepare()
    source = tmp_path / "IsaacLab"
    (source / ".venv" / "bin").mkdir(parents=True)
    (source / ".venv" / "bin" / "python").write_bytes(b"p" * 9)
    pycache = source / ".venv" / "lib" / "__pycache__"
    pycache.mkdir(parents=True)
    (pycache / "mod.cpython-312.pyc").write_bytes(b"c" * 5)
    (source / "pyproject.toml").write_bytes(b"t" * 10)
    (source / ".git").mkdir()
    (source / ".git" / "HEAD").write_bytes(b"g" * 4)
    (source / "logs").mkdir()
    (source / "logs" / "run.log").write_bytes(b"l" * 3)
    uv_cache = tmp_path / "uv-cache"
    (uv_cache / "archive" / "__pycache__").mkdir(parents=True)
    (uv_cache / "archive" / "__pycache__" / "x.pyc").write_bytes(b"c" * 4)
    (uv_cache / "wheel.bin").write_bytes(b"w" * 10)

    out = tmp_path / "isaaclab-cache"
    manifest = prep.snapshot_cache(
        source_dir=source, uv_cache_dir=uv_cache, out_dir=out,
        created_by="test-builder")

    # Movimento, não cópia: origens somem, destino tem o conteúdo.
    assert not source.exists()
    assert not uv_cache.exists()
    assert (out / ISAACLAB_PART / ".venv" / "bin" / "python").is_file()
    # Poda: .git, logs e TODO __pycache__ ficam fora do snapshot.
    assert not (out / ISAACLAB_PART / ".git").exists()
    assert not (out / ISAACLAB_PART / "logs").exists()
    assert list(out.rglob("__pycache__")) == []
    # Manifesto mede o destino JÁ podado/movido.
    assert manifest["parts"] == {ISAACLAB_PART: 19, UV_CACHE_PART: 10}
    assert prep.verify_cache(out) == []


def test_prepare_snapshot_refuses_oversize_before_touching_trees(
        tmp_path) -> None:
    prep = _load_prepare()
    source = tmp_path / "IsaacLab"
    (source / ".venv" / "__pycache__").mkdir(parents=True)
    (source / ".venv" / "bin").mkdir(parents=True)
    (source / ".venv" / "bin" / "python").write_bytes(b"p" * 9)
    (source / ".venv" / "__pycache__" / "m.pyc").write_bytes(b"c" * 5)
    uv_cache = tmp_path / "uv-cache"
    uv_cache.mkdir()
    (uv_cache / "wheel.bin").write_bytes(b"w" * 10)
    out = tmp_path / "isaaclab-cache"

    with pytest.raises(RuntimeError, match="ANTES de mover"):
        prep.snapshot_cache(
            source_dir=source, uv_cache_dir=uv_cache, out_dir=out,
            created_by="test-builder", max_total_bytes=1)

    # A recusa é ANTES de qualquer mutação: nem poda, nem movimento.
    assert (source / ".venv" / "bin" / "python").is_file()
    assert (source / ".venv" / "__pycache__" / "m.pyc").is_file()
    assert (uv_cache / "wheel.bin").is_file()
    assert not (out / ISAACLAB_PART).exists()


def _load_builder(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_main_notebook(tmp_path, monkeypatch, extra_argv):
    builder = _load_builder("builder_cache_test", "build_kaggle_notebook.py")
    monkeypatch.setattr(builder, "KAGGLE_DIR", tmp_path)
    monkeypatch.setattr(
        sys, "argv",
        ["build_kaggle_notebook.py", "--kaggle-user", "tester", *extra_argv])
    builder.main()
    notebook = json.loads((tmp_path / builder.NOTEBOOK_NAME).read_text())
    metadata = json.loads((tmp_path / "kernel-metadata.json").read_text())
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    return source, metadata, notebook


def test_builder_default_notebook_has_no_cache_traces(tmp_path, monkeypatch) -> None:
    source, metadata, _ = _build_main_notebook(tmp_path, monkeypatch, [])
    assert "isaac_cache" not in source
    assert "find_cache_mount" not in source
    assert "dataset_sources" not in metadata
    # A instalação de sempre, no nível base de indentação da célula.
    assert "    clone = run(['git', 'clone'" in source
    assert "    if not _cache_restored:" not in source


def test_builder_cache_flag_wires_mount_fallback_and_metadata(
        tmp_path, monkeypatch) -> None:
    argv = ["--isaac-cache-dataset", "tester/fanuc-isaaclab-cache"]
    source, metadata, notebook = _build_main_notebook(
        tmp_path, monkeypatch, argv)

    assert metadata["dataset_sources"] == ["tester/fanuc-isaaclab-cache"]
    # Descoberta do mount + override + núcleo puro decidindo.
    assert "find_cache_mount" in source
    assert "'/kaggle/input/fanuc-isaaclab-cache'" in source
    assert "FANUC_ISAAC_CACHE_DIR" in source
    assert "cache_restore_plan" in source
    # Sonda de import como portão final.
    assert "'import isaaclab, torch'" in source
    assert "'--no-sync'" in source
    # Fallback = o trecho original, reindentado sob o `if not`.
    assert "    if not _cache_restored:\n" in source
    assert "        clone = run(['git', 'clone'" in source
    assert "        sync = run(['uv', 'sync'], cwd=source_dir)" in source
    # Medição da aceitação registrada no relatório.
    assert "runtime_report['isaac_cache'] = _cache_info" in source
    assert "stage_start('isaac_setup')" in source
    # A célula principal gerada é Python válido.
    main_cell = next(
        "".join(cell["source"]) for cell in notebook["cells"]
        if "find_cache_mount" in "".join(cell["source"]))
    compile(main_cell, "<isaac-install-cell>", "exec")


def test_builder_cache_composes_with_newton_physics(tmp_path, monkeypatch) -> None:
    argv = ["--physics", "newton_mjwarp",
            "--isaac-cache-dataset", "tester/fanuc-isaaclab-cache"]
    source, _, notebook = _build_main_notebook(tmp_path, monkeypatch, argv)
    assert "find_cache_mount" in source
    main_cell = next(
        "".join(cell["source"]) for cell in notebook["cells"]
        if "find_cache_mount" in "".join(cell["source"]))
    compile(main_cell, "<isaac-install-cell-newton>", "exec")


def test_builder_cache_composes_with_benchmark_seeds(
        tmp_path, monkeypatch) -> None:
    # B4 (issue #12) x B3 (issue #10): as duas flags juntas — o merge
    # das duas frentes no builder — geram um notebook com a fiação do
    # cache E a lista de seeds trocada, células geradas válidas.
    argv = ["--isaac-cache-dataset", "tester/fanuc-isaaclab-cache",
            "--benchmark-seeds", "7,8"]
    source, metadata, notebook = _build_main_notebook(
        tmp_path, monkeypatch, argv)
    assert metadata["dataset_sources"] == ["tester/fanuc-isaaclab-cache"]
    assert "find_cache_mount" in source
    assert "benchmark_seeds = (7, 8)" in source
    assert "benchmark_seeds = (101, 202, 303)" not in source
    install_cell = next(
        "".join(cell["source"]) for cell in notebook["cells"]
        if "find_cache_mount" in "".join(cell["source"]))
    compile(install_cell, "<isaac-install-cell-seeds>", "exec")
    benchmark_cell = next(
        "".join(cell["source"]) for cell in notebook["cells"]
        if "benchmark_seeds = (7, 8)" in "".join(cell["source"]))
    compile(benchmark_cell, "<benchmark-cell-cache>", "exec")


def test_builder_cache_composes_with_eval_only_dataset_sources(
        tmp_path, monkeypatch) -> None:
    # B4 (issue #12) x B2 (issue #11): as duas flags anexam dataset ao
    # kernel — cache E checkpoint coexistem em dataset_sources (a
    # resolução do merge acumula; nenhum dos dois sobrescreve).
    argv = ["--isaac-cache-dataset", "tester/fanuc-isaaclab-cache",
            "--eval-only",
            "--eval-checkpoint", "fanuc-checkpoint-teste/model_1499.pt"]
    source, metadata, _ = _build_main_notebook(tmp_path, monkeypatch, argv)
    assert metadata["dataset_sources"] == [
        "tester/fanuc-isaaclab-cache", "tester/fanuc-checkpoint-teste"]
    assert "find_cache_mount" in source
    assert "'--checkpoint', eval_checkpoint," in source


def test_builder_cache_dataset_requires_full_slug(tmp_path, monkeypatch) -> None:
    builder = _load_builder("builder_cache_slug", "build_kaggle_notebook.py")
    monkeypatch.setattr(builder, "KAGGLE_DIR", tmp_path)
    monkeypatch.setattr(
        sys, "argv",
        ["build_kaggle_notebook.py", "--isaac-cache-dataset", "sem-usuario"])
    with pytest.raises(SystemExit):
        builder.main()


def test_cache_builder_kernel_installs_warms_and_snapshots(
        tmp_path, monkeypatch) -> None:
    builder = _load_builder("builder_cache_kernel", "build_cache_notebook.py")
    monkeypatch.setattr(builder, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["build_cache_notebook.py", "--kaggle-user", "tester"])
    builder.main()

    notebook = json.loads((tmp_path / builder.NOTEBOOK_NAME).read_text())
    metadata = json.loads((tmp_path / "kernel-metadata.json").read_text())
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert metadata["id"] == "tester/fanuc-isaaclab-cache-builder"
    assert metadata["enable_gpu"] is False
    assert "UV_LINK_MODE'] = 'copy'" in source
    assert "'--branch', 'v3.0.0'" in source
    for extra in ("ovphysx", "video", "importers", "ovrtx"):
        assert f"'--extra', '{extra}'" in source
    assert "prepare_isaac_cache.py" in source
    assert "isaac-cache-manifest.json" in source


def test_cache_builder_notebook_measures_and_born_uv_cache(
        tmp_path, monkeypatch) -> None:
    """Correção da tentativa v2 (issue #12): o construtor mede (du)
    árvore e cache do uv ANTES do snapshot e o cache do uv nasce sob
    /kaggle/working (UV_CACHE_DIR) — mesma partição da saída, para o
    snapshot mover em vez de copiar (a cópia deu Errno 28)."""
    builder = _load_builder("builder_cache_measure", "build_cache_notebook.py")
    monkeypatch.setattr(builder, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["build_cache_notebook.py", "--kaggle-user", "tester"])
    builder.main()

    notebook = json.loads((tmp_path / builder.NOTEBOOK_NAME).read_text())
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "UV_CACHE_DIR'] = '/kaggle/working/uv-cache'" in source
    assert "'du', '-sh'" in source
    assert "'--uv-cache-dir', '/kaggle/working/uv-cache'" in source
    # A medição entra no log ANTES da chamada do prepare.
    assert source.index("'du', '-sh'") < source.index("prepare_isaac_cache.py',")


def test_cache_builder_notebook_cells_all_compile(
        tmp_path, monkeypatch) -> None:
    """Regressão (issue #12): TODAS as células de código do notebook do
    CONSTRUTOR compilam. Os testes do consumidor compilavam as células
    dele, mas as do construtor nunca eram compiladas — a aspa de abertura
    faltante no print final do builder passou e só explodiu como
    SyntaxError (U+2014) na célula final do kernel, depois de a
    instalação inteira rodar, sem produzir o snapshot isaaclab-cache/.
    Linhas de magic/shell (``%cd``, ``!...``) são sintaxe do IPython,
    não do CPython: saem antes do compile, o resto é Python puro."""
    builder = _load_builder("builder_cache_compile", "build_cache_notebook.py")
    monkeypatch.setattr(builder, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["build_cache_notebook.py", "--kaggle-user", "tester"])
    builder.main()

    notebook = json.loads((tmp_path / builder.NOTEBOOK_NAME).read_text())
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert code_cells, "o notebook do construtor precisa de células de código"
    for idx, cell in enumerate(code_cells):
        source = "".join(cell["source"])
        python_source = "\n".join(
            line for line in source.splitlines()
            if not line.lstrip().startswith(("%", "!")))
        compile(python_source, f"<cache-builder-cell-{idx}>", "exec")


def test_cache_module_is_bundled_and_exported() -> None:
    builder = (ROOT / "scripts/build_kaggle_notebook.py").read_text()
    assert '"src/fanuc_cell/isaac_cache.py"' in builder
    assert '"scripts/prepare_isaac_cache.py"' in builder
    init = (ROOT / "src/fanuc_cell/__init__.py").read_text()
    assert "from .isaac_cache import" in init
    docs = (ROOT / "docs/isaac-cache.md").read_text()
    assert "fanuc-isaaclab-cache" in docs
    assert "--isaac-cache-dataset" in docs
