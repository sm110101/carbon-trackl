CarbonTrackCLI – Track, preprocess, forecast, and visualize CO₂ emissions from your terminal.

Quickstart

1) Install (dev mode)

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
pip install -U pip
pip install -e .
```

2) Fetch data (OWID)

```bash
carbontrack fetch --source owid
```

3) Prepare standardized dataset

```bash
carbontrack prepare
```

4) Forecast next 5 years for a country

```bash
carbontrack forecast --country "United States" --horizon 5
```

5) Plot historical + forecast

```bash
carbontrack plot --country "United States" -o us_emissions.png
```

Reset / Clean (one command)

```bash
# remove cache + build artifacts + local *_emissions.png
carbontrack clean --all
```

Notes
- Cache location defaults to `~/.carbontrack`. Override with `--cache-dir PATH` or env `CARBONTRACK_CACHE_DIR`.
- Standardized schema: `country, year, sector, emissions_co2e, population, gdp`. For MVP, `sector="all"` from OWID aggregates.
- The forecasting model is a simple Linear Regression baseline. It can be swapped later for XGBoost.

Uninstall

```bash
pip uninstall carbontrack
```


