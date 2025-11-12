from pathlib import Path
from typing import Optional, Tuple
import requests

from .utils import get_cache_dir, ensure_parent


OWID_CO2_URL = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
OWID_CO2_CACHE_NAME = "owid_co2.csv"


def fetch_owid_co2(cache_dir: Optional[str] = None, timeout: int = 60) -> Path:
    """
    Download the OWID CO2 dataset and cache it under raw/.
    """
    base = get_cache_dir(cache_dir)
    target = base / "raw" / OWID_CO2_CACHE_NAME
    response = requests.get(OWID_CO2_URL, timeout=timeout)
    response.raise_for_status()
    ensure_parent(target)
    target.write_bytes(response.content)
    return target


def resolve_raw_paths(cache_dir: Optional[str] = None) -> Tuple[Path]:
    """
    Return tuple of available raw file paths (present or expected).
    """
    base = get_cache_dir(cache_dir)
    return (base / "raw" / OWID_CO2_CACHE_NAME,)


