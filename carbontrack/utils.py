import os
from pathlib import Path
from typing import Optional


DEFAULT_CACHE_ENV = "CARBONTRACK_CACHE_DIR"
DEFAULT_CACHE_DIRNAME = ".carbontrack"


def get_cache_dir(custom_dir: Optional[str] = None) -> Path:
    """
    Resolve the cache directory, creating it if needed.
    Precedence:
      1) explicit custom_dir
      2) env CARBONTRACK_CACHE_DIR
      3) ~/.carbontrack
    """
    if custom_dir:
        path = Path(custom_dir).expanduser().resolve()
    else:
        env_dir = os.environ.get(DEFAULT_CACHE_ENV)
        path = Path(env_dir).expanduser().resolve() if env_dir else Path.home() / DEFAULT_CACHE_DIRNAME
    # subdirs
    raw_dir = path / "raw"
    processed_dir = path / "processed"
    forecast_dir = path / "forecasts"
    for sub in (path, raw_dir, processed_dir, forecast_dir):
        sub.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)


