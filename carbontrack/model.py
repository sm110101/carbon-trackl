from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

from .utils import get_cache_dir
from .preprocess import STANDARDIZED_NAME


FORECAST_NAME = "forecasts.csv"


def _prepare_country_timeseries(df: pd.DataFrame, country: str) -> pd.DataFrame:
    sub = df[df["country"] == country].copy()
    sub = sub.sort_values("year")
    # Simple features: year, population, gdp, lag1 emissions
    sub["lag1"] = sub["emissions_co2e"].shift(1).fillna(0.0)
    features = ["year", "population", "gdp", "lag1"]
    target = "emissions_co2e"
    # Drop rows with all-zero features to avoid degenerate fit
    sub = sub.fillna(0.0)
    return sub, features, target


def train_and_forecast(country: str, horizon: int = 5, cache_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Train a simple per-country regression and forecast next N years.
    Returns a DataFrame with year, country, yhat.
    """
    base = get_cache_dir(cache_dir)
    std_path = base / "processed" / STANDARDIZED_NAME
    if not std_path.exists():
        raise FileNotFoundError(f"Standardized data not found at {std_path}. Run prepare first.")
    df = pd.read_csv(std_path)
    sub, features, target = _prepare_country_timeseries(df, country)
    if len(sub) < 3:
        raise ValueError(f"Not enough data to model {country}. Need at least 3 rows.")
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
        forecasts.append({"country": country, "year": year, "yhat_emissions_co2e": yhat})
        # keep lag rolling with predicted value
        last_lag = yhat
    pred_df = pd.DataFrame(forecasts)
    out = base / "forecasts" / FORECAST_NAME
    if out.exists():
        existing = pd.read_csv(out)
        # remove existing rows for this country and overlapping years
        mask = ~((existing["country"] == country) & (existing["year"].isin(pred_df["year"])))
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


