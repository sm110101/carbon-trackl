from pathlib import Path
from typing import Optional, Iterable, List
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from .utils import get_cache_dir
from .preprocess import STANDARDIZED_NAME
from .model import FORECAST_NAME


def plot_country(country: str, save_path: Optional[str] = None, cache_dir: Optional[str] = None, sector: Optional[str] = "all") -> Path:
    """
    Plot historical emissions and latest forecast for a country.
    Saves to PNG and returns path.
    """
    base = get_cache_dir(cache_dir)
    std_path = base / "processed" / STANDARDIZED_NAME
    hist = pd.read_csv(std_path)
    hist = hist[hist["country"] == country].copy()
    if sector is not None and "sector" in hist.columns:
        hist = hist[hist["sector"] == sector]
    hist = hist.sort_values("year")

    forecast_path = base / "forecasts" / FORECAST_NAME
    fc = None
    if forecast_path.exists():
        fc = pd.read_csv(forecast_path)
        if "sector" not in fc.columns:
            fc["sector"] = "all"
        fc = fc[(fc["country"] == country) & (fc["sector"] == (sector if sector is not None else "all"))].copy()
        if len(fc) == 0:
            fc = None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hist["year"], hist["emissions_co2e"], label="Historical", color="#1f77b4", linewidth=2)
    if fc is not None:
        ax.plot(fc["year"], fc["yhat_emissions_co2e"], label="Forecast", color="#ff7f0e", linestyle="--", linewidth=2)
    title = f"{country} CO2 emissions (MtCO2)"
    if sector and sector != "all":
        title += f" – {sector}"
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Emissions (MtCO2)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        out_path = Path(save_path).expanduser().resolve()
    else:
        suffix = f"_{sector}" if sector and sector != "all" else ""
        out_path = base / "forecasts" / f"{country.replace(' ', '_').lower()}{suffix}_emissions.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_grid(
    countries: Iterable[str],
    save_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    sector: Optional[str] = "all",
    cols: int = 2,
) -> Path:
    base = get_cache_dir(cache_dir)
    std_path = base / "processed" / STANDARDIZED_NAME
    hist_df = pd.read_csv(std_path)
    forecast_path = base / "forecasts" / FORECAST_NAME
    fc_df = None
    if forecast_path.exists():
        fc_df = pd.read_csv(forecast_path)
        if "sector" not in fc_df.columns:
            fc_df["sector"] = "all"

    countries_list: List[str] = list(countries)
    n = len(countries_list)
    rows = (n + cols - 1) // cols if n else 1
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = axes if isinstance(axes, (list, tuple, np.ndarray)) else [axes]  # type: ignore
    axes = [ax for sub in (axes if isinstance(axes, np.ndarray) else [axes]) for ax in (sub if isinstance(sub, (list, np.ndarray)) else [sub])]  # flatten

    for idx, country in enumerate(countries_list):
        ax = axes[idx]
        hist = hist_df[hist_df["country"] == country].copy()
        if sector is not None and "sector" in hist.columns:
            hist = hist[hist["sector"] == sector]
        hist = hist.sort_values("year")
        ax.plot(hist["year"], hist["emissions_co2e"], label="Historical", color="#1f77b4", linewidth=2)
        if fc_df is not None:
            fc = fc_df[(fc_df["country"] == country) & (fc_df["sector"] == (sector if sector is not None else "all"))].copy()
            if len(fc) > 0:
                ax.plot(fc["year"], fc["yhat_emissions_co2e"], label="Forecast", color="#ff7f0e", linestyle="--", linewidth=2)
        t = country
        if sector and sector != "all":
            t += f" – {sector}"
        ax.set_title(t)
        ax.set_xlabel("Year")
        ax.set_ylabel("Emissions (MtCO2)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    # hide any unused axes
    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()

    if save_path:
        out_path = Path(save_path).expanduser().resolve()
    else:
        out_path = base / "forecasts" / f"compare_grid_{len(countries_list)}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

