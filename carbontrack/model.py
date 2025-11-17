from pathlib import Path
from typing import Optional, Tuple, Iterable, List
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

from .utils import get_cache_dir
from .preprocess import STANDARDIZED_NAME


FORECAST_NAME = "forecasts.csv"


def _prepare_country_timeseries(df: pd.DataFrame, country: str, sector: Optional[str] = None) -> pd.DataFrame:
    sub = df[df["country"] == country].copy()
    if sector is not None and "sector" in sub.columns:
        sub = sub[sub["sector"] == sector]
    sub = sub.sort_values("year")
    # Simple features: year, population, gdp, lag1 emissions
    sub["lag1"] = sub["emissions_co2e"].shift(1).fillna(0.0)
    features = ["year", "population", "gdp", "lag1"]
    target = "emissions_co2e"
    # Drop rows with all-zero features to avoid degenerate fit
    sub = sub.fillna(0.0)
    return sub, features, target


def train_and_forecast(country: str, horizon: int = 5, cache_dir: Optional[str] = None, sector: Optional[str] = "all") -> pd.DataFrame:
    """
    Train a simple per-country regression and forecast next N years.
    Returns a DataFrame with year, country, sector, yhat.
    """
    base = get_cache_dir(cache_dir)
    std_path = base / "processed" / STANDARDIZED_NAME
    if not std_path.exists():
        raise FileNotFoundError(f"Standardized data not found at {std_path}. Run prepare first.")
    df = pd.read_csv(std_path)
    sub, features, target = _prepare_country_timeseries(df, country, sector)
    if len(sub) < 3:
        raise ValueError(f"Not enough data to model {country} (sector={sector}). Need at least 3 rows.")
    X = sub[features].values
    y = sub[target].values
    model = LinearRegression()
    model.fit(X, y)
    last_year = int(sub["year"].max())
    # Forecast iteratively, updating lag1 each step with last prediction
    forecasts = []
    last_lag = float(sub.iloc[-1]["emissions_co2e"])
    last_pop = float(sub.iloc[-1]["population"])
    last_gdp = float(sub.iloc[-1]["gdp"])
    for i in range(1, horizon + 1):
        year = last_year + i
        x_row = np.array([[year, last_pop, last_gdp, last_lag]])
        yhat = float(model.predict(x_row)[0])
        forecasts.append({"country": country, "sector": sector if sector is not None else "all", "year": year, "yhat_emissions_co2e": yhat})
        # keep lag rolling with predicted value
        last_lag = yhat
    pred_df = pd.DataFrame(forecasts)
    out = base / "forecasts" / FORECAST_NAME
    if out.exists():
        existing = pd.read_csv(out)
        if "sector" not in existing.columns:
            existing["sector"] = "all"
        # remove existing rows for this country and overlapping years
        mask = ~((existing["country"] == country) & (existing.get("sector") == (sector if sector is not None else "all")) & (existing["year"].isin(pred_df["year"])))
        existing = existing[mask]
        combined = pd.concat([existing, pred_df], ignore_index=True)
    else:
        combined = pred_df
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)
    return pred_df


def available_countries(cache_dir: Optional[str] = None) -> Tuple[str, ...]:
    base = get_cache_dir(cache_dir)
    std_path = base / "processed" / STANDARDIZED_NAME
    if not std_path.exists():
        return tuple()
    df = pd.read_csv(std_path, usecols=["country"])
    return tuple(sorted(df["country"].unique()))


def train_and_forecast_many(
    countries: Iterable[str],
    horizon: int = 5,
    cache_dir: Optional[str] = None,
    sector: Optional[str] = "all",
) -> pd.DataFrame:
    base = get_cache_dir(cache_dir)
    results: List[pd.DataFrame] = []
    for c in countries:
        try:
            pred = train_and_forecast(country=c, horizon=horizon, cache_dir=str(base), sector=sector)
            results.append(pred.assign(country=c, sector=sector if sector is not None else "all"))
        except Exception:
            continue
    if not results:
        return pd.DataFrame(columns=["country", "sector", "year", "yhat_emissions_co2e"])
    return pd.concat(results, ignore_index=True)

