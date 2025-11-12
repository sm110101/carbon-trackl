from pathlib import Path
from typing import Optional
import pandas as pd

from .utils import get_cache_dir
from .data import OWID_CO2_CACHE_NAME


STANDARDIZED_NAME = "standardized.csv"


def _load_owid(raw_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    # Minimal subset needed for MVP
    cols = ["country", "year", "co2", "population", "gdp"]
    present = [c for c in cols if c in df.columns]
    df = df[present].copy()
    # Fill missing numeric with 0 for MVP simplicity
    for c in ("co2", "population", "gdp"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    # set sector='all' for OWID aggregate
    df["sector"] = "all"
    # rename co2 to emissions_co2e (MtCO2 as provided by OWID)
    df = df.rename(columns={"co2": "emissions_co2e"})
    # Reorder/ensure columns
    final_cols = ["country", "year", "sector", "emissions_co2e", "population", "gdp"]
    for c in final_cols:
        if c not in df.columns:
            df[c] = 0 if c in ("emissions_co2e", "population", "gdp") else ""
    return df[final_cols]


def standardize(cache_dir: Optional[str] = None) -> Path:
    """
    Create standardized CSV combining sources into schema:
    country, year, sector, emissions_co2e, population, gdp
    For MVP, only OWID aggregate is included with sector='all'.
    """
    base = get_cache_dir(cache_dir)
    raw_owid = base / "raw" / OWID_CO2_CACHE_NAME
    if not raw_owid.exists():
        raise FileNotFoundError(f"OWID raw file not found at {raw_owid}. Run fetch first.")
    df = _load_owid(raw_owid)
    out = base / "processed" / STANDARDIZED_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


