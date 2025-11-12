from pathlib import Path
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt

from .utils import get_cache_dir
from .preprocess import STANDARDIZED_NAME
from .model import FORECAST_NAME


def plot_country(country: str, save_path: Optional[str] = None, cache_dir: Optional[str] = None) -> Path:
    """
    Plot historical emissions and latest forecast for a country.
    Saves to PNG and returns path.
    """
    base = get_cache_dir(cache_dir)
    std_path = base / "processed" / STANDARDIZED_NAME
    hist = pd.read_csv(std_path)
    hist = hist[hist["country"] == country].copy()
    hist = hist.sort_values("year")

    forecast_path = base / "forecasts" / FORECAST_NAME
    fc = None
    if forecast_path.exists():
        fc = pd.read_csv(forecast_path)
        fc = fc[fc["country"] == country].copy()
        if len(fc) == 0:
            fc = None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hist["year"], hist["emissions_co2e"], label="Historical", color="#1f77b4", linewidth=2)
    if fc is not None:
        ax.plot(fc["year"], fc["yhat_emissions_co2e"], label="Forecast", color="#ff7f0e", linestyle="--", linewidth=2)
    ax.set_title(f"{country} CO2 emissions (MtCO2)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Emissions (MtCO2)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        out_path = Path(save_path).expanduser().resolve()
    else:
        out_path = base / "forecasts" / f"{country.replace(' ', '_').lower()}_emissions.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


