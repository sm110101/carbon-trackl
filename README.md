CarbonTrackCLI – Track, preprocess, forecast, and visualize CO₂ emissions from your terminal.

Quickstart

1) Install (dev mode)

Clone this repository:
```bash 
git clone https://github.com/sm110101/carbon-trackl.git
```


```bash
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
source .venv/bin/activate # Mac
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
- CodeCarbon is enabled by default. Each command run is tracked and a CSV log is written to `<cache_dir>/emissions/`. A brief summary (kg CO2e) is printed after the command completes.

Display emissions

```bash
# Show last 5 runs (default)
carbontrack emissions

# Show last 10 runs in a table
carbontrack emissions --last 10

# Show total kg CO2e across all runs
carbontrack emissions --total

# Emit JSON for last run (for scripts)
carbontrack emissions --last 1 --json

# Print the path to the CSV log
carbontrack emissions --path
```

Uninstall

```bash
pip uninstall carbontrack
```


