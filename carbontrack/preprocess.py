from pathlib import Path
from typing import Optional, Dict, List, Tuple
import pandas as pd

from .utils import get_cache_dir
from .data import OWID_CO2_CACHE_NAME


STANDARDIZED_NAME = "standardized.csv"


def _load_owid(raw_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    # Core identifiers
    id_cols = ["country", "year", "population", "gdp"]
    present_ids = [c for c in id_cols if c in df.columns]
    df_ids = df[present_ids + [c for c in df.columns if c not in present_ids]].copy()
    # Numeric cleaning for id-level fields
    for c in ("population", "gdp"):
        if c in df_ids.columns:
            df_ids[c] = pd.to_numeric(df_ids[c], errors="coerce")

    # Map OWID per-source columns into "sector" rows
    sector_map: Dict[str, str] = {
        "co2": "all",
        "cement_co2": "cement",
        "coal_co2": "coal",
        "oil_co2": "oil",
        "gas_co2": "gas",
        "flaring_co2": "flaring",
        "other_industry_co2": "other_industry",
        "land_use_change_co2": "land_use_change",
    }

    records: List[Tuple[str, int, str, float, float, float]] = []
    for _, row in df_ids.iterrows():
        country = row.get("country")
        year = row.get("year")
        population = row.get("population", float("nan"))
        gdp = row.get("gdp", float("nan"))
        try:
            year = int(year)
        except Exception:
            continue
        for col, sector in sector_map.items():
            if col in df_ids.columns:
                val = row.get(col)
                try:
                    val = float(val)
                except Exception:
                    val = float("nan")
                # Keep rows where we have a non-null value
                if pd.notna(val):
                    records.append((country, year, sector, val, population, gdp))

    if not records:
        # Fallback: return empty standardized schema
        return pd.DataFrame(columns=["country", "year", "sector", "emissions_co2e", "population", "gdp"])

    out_df = pd.DataFrame.from_records(
        records,
        columns=["country", "year", "sector", "emissions_co2e", "population", "gdp"],
    )
    # Fill missing numeric with 0 for MVP simplicity (keep NaNs would be okay too)
    for c in ("emissions_co2e", "population", "gdp"):
        if c in out_df.columns:
            out_df[c] = pd.to_numeric(out_df[c], errors="coerce").fillna(0.0)
    # Ensure dtypes
    out_df["year"] = out_df["year"].astype(int)
    return out_df[["country", "year", "sector", "emissions_co2e", "population", "gdp"]]


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


