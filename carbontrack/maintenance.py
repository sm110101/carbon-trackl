from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any
import shutil
import os
import glob

from .utils import DEFAULT_CACHE_ENV, DEFAULT_CACHE_DIRNAME


def _resolve_cache_dir_without_creating(custom_dir: Optional[str]) -> Path:
    if custom_dir:
        return Path(custom_dir).expanduser().resolve()
    env_dir = os.environ.get(DEFAULT_CACHE_ENV)
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return Path.home() / DEFAULT_CACHE_DIRNAME


def _safe_rmtree(path: Path) -> bool:
    try:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
            return True
    except Exception:
        return False
    return False


def _safe_remove_file(path: Path) -> bool:
    try:
        if path.exists() and path.is_file():
            path.unlink(missing_ok=True)
            return True
    except Exception:
        return False
    return False


def clean(
    cache_dir: Optional[str] = None,
    remove_cache: bool = True,
    remove_build: bool = True,
    remove_local_outputs: bool = True,
    venv_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Perform cleanup tasks. Returns a summary dict.
    - remove_cache: removes the entire carbontrack cache dir
    - remove_build: removes dist/, build/, and *.egg-info in CWD
    - remove_local_outputs: removes *_emissions.png in CWD
    - venv_path: if provided, recursively deletes that directory
    """
    summary: Dict[str, Any] = {
        "cache_removed": False,
        "build_removed": [],
        "outputs_removed": [],
        "venv_removed": False,
    }

    if remove_cache:
        cache_path = _resolve_cache_dir_without_creating(cache_dir)
        summary["cache_removed"] = _safe_rmtree(cache_path)

    if remove_build:
        cwd = Path.cwd()
        for name in ("dist", "build"):
            p = cwd / name
            if _safe_rmtree(p):
                summary["build_removed"].append(str(p))
        for egg_dir in glob.glob(str(cwd / "*.egg-info")):
            if _safe_rmtree(Path(egg_dir)):
                summary["build_removed"].append(egg_dir)

    if remove_local_outputs:
        cwd = Path.cwd()
        for pattern in ("*_emissions.png",):
            for file in cwd.glob(pattern):
                if _safe_remove_file(file):
                    summary["outputs_removed"].append(str(file))

    if venv_path:
        vp = Path(venv_path).expanduser().resolve()
        summary["venv_removed"] = _safe_rmtree(vp)

    return summary