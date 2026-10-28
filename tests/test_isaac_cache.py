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
    find_cache_mount,
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


def test_prepare_verify_rejects_tree_without_venv(tmp_path) -> None:
    spec = importlib.util.spec_from_file_location(
        "prepare_isaac_cache", ROOT / "scripts/prepare_isaac_cache.py")
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)

    broken = _write_mount(tmp_path, _manifest())
    assert any(".venv" in p for p in prep.verify_cache(broken))
    assert prep.verify_cache(tmp_path / "nada") != []


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


def test_cache_module_is_bundled_and_exported() -> None:
    builder = (ROOT / "scripts/build_kaggle_notebook.py").read_text()
    assert '"src/fanuc_cell/isaac_cache.py"' in builder
    assert '"scripts/prepare_isaac_cache.py"' in builder
    init = (ROOT / "src/fanuc_cell/__init__.py").read_text()
    assert "from .isaac_cache import" in init
    docs = (ROOT / "docs/isaac-cache.md").read_text()
    assert "fanuc-isaaclab-cache" in docs
    assert "--isaac-cache-dataset" in docs
