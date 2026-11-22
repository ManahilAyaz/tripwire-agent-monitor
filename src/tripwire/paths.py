"""Path validation for CLI-supplied file/directory arguments (a system boundary).

`argparse` strings (config paths, output directories) are resolved and checked here before
any script or module uses them in a filesystem call, rather than passing the raw string
straight into `open()`/`Path.unlink()` — resolving collapses any ".." traversal components and
follows symlinks, so what the rest of the code acts on is always a concrete, validated path.
"""

from __future__ import annotations

from pathlib import Path


def resolve_config_path(raw_path: str) -> Path:
    """Resolve a CLI-supplied config file path and confirm it's a real file."""
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    return path


def resolve_output_dir(raw_path: str) -> Path:
    """Resolve a CLI-supplied output directory path, creating it if it doesn't exist yet."""
    path = Path(raw_path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path
