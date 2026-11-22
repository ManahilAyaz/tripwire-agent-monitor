"""Tests for the CLI path-validation helpers (addresses SonarCloud pythonsecurity:S8707)."""

from __future__ import annotations

import pytest

from tripwire.paths import resolve_config_path, resolve_output_dir


def test_resolve_config_path_returns_resolved_existing_file(tmp_path):
    config = tmp_path / "agent.yaml"
    config.write_text("model: qwen/qwen3.8-27b\n")

    resolved = resolve_config_path(str(config))

    assert resolved == config.resolve()
    assert resolved.is_file()


def test_resolve_config_path_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_config_path(str(tmp_path / "does_not_exist.yaml"))


def test_resolve_config_path_collapses_traversal_components(tmp_path):
    config = tmp_path / "agent.yaml"
    config.write_text("model: qwen/qwen3.8-27b\n")
    traversal_path = tmp_path / "sub" / ".." / "agent.yaml"

    resolved = resolve_config_path(str(traversal_path))

    assert resolved == config.resolve()


def test_resolve_output_dir_creates_directory_if_missing(tmp_path):
    out_dir = tmp_path / "pilot"

    resolved = resolve_output_dir(str(out_dir))

    assert resolved == out_dir.resolve()
    assert resolved.is_dir()


def test_resolve_output_dir_accepts_existing_directory(tmp_path):
    resolved = resolve_output_dir(str(tmp_path))
    assert resolved == tmp_path.resolve()
