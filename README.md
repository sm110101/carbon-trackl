## CarbonTrackCLI – Track, preprocess, forecast, and visualize CO₂ emissions from your terminal.

## Installation

```bash 
git clone https://github.com/sm110101/carbon-trackl.git
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
source .venv/bin/activate # Mac
pip install -U pip
pip install -e .
```

## Quickstart

```bash
# 1) Fetch OWID data
carbontrack fetch --source owid

# 2) Prepare standardized dataset
carbontrack prepare

# 3) Forecast next 5 years for a country
carbontrack forecast --country "United States" --horizon 5

# 4) Plot historical + forecast
carbontrack plot --country "United States" -o us_emissions.png

# 5) Batch forecast multiple countries
carbontrack forecast-batch --countries "United States,China,India" --horizon 5

# 6) Plot multiple countries side-by-side
carbontrack plot-grid --countries "United States,China,India" --cols 3 -o compare.png
```

## Data model and sectors
- Standardized long-form schema: `country, year, sector, emissions_co2e, population, gdp`
- Sectors mapped from OWID per-source columns:
  - all (`co2`)
  - cement (`cement_co2`)
  - coal (`coal_co2`)
  - oil (`oil_co2`)
  - gas (`gas_co2`)
  - flaring (`flaring_co2`)
  - other_industry (`other_industry_co2`)
  - land_use_change (`land_use_change_co2`)

## Core commands

```bash
# Fetch and cache raw datasets
carbontrack fetch --source owid

# Create standardized dataset CSV
carbontrack prepare

# Train simple model and forecast next N years
carbontrack forecast --country "United States" --horizon 5 --sector all

# Plot historical and forecast for a country
carbontrack plot --country "United States" --sector all -o us_emissions.png

# Forecast multiple countries
carbontrack forecast-batch --countries "United States,China,India" --horizon 5 --sector all

# Plot multiple countries in a grid
carbontrack plot-grid --countries "United States,China,India" --sector all --cols 3 -o compare.png
```

## Explore OWID data (built-in)

```bash
# Summary of standardized dataset
carbontrack info

# List countries (first 20)
carbontrack countries --limit 20

# List sectors (first 20)
carbontrack sectors --limit 20

# Search countries by substring
carbontrack countries --search "United"

# Search sectors by substring
carbontrack sectors --search "energy"

# Year coverage for a country
carbontrack years --country "United States"

# Top 10 emitters for a given year
carbontrack top --year 2019 --limit 10

# Show column names (schema)
carbontrack schema

# Preview first 5 rows
carbontrack head --rows 5
```

## Emissions tracking
CodeCarbon is enabled by default. Each command run is tracked and logs are written to `<cache_dir>/emissions/`. A brief summary in scaled units (kg/g/mg) prints after each command.

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

## Clean / reset

```bash
# remove cache + build artifacts + local *_emissions.png
carbontrack clean --all
```

## Configuration notes
- Cache location defaults to `~/.carbontrack`. Override with `--cache-dir PATH` or env `CARBONTRACK_CACHE_DIR`.
- Standardized schema: `country, year, sector, emissions_co2e, population, gdp` (OWID aggregate uses `sector="all"`).
- Forecasting model: simple Linear Regression baseline (swappable later).

## Uninstall

```bash
pip uninstall carbontrack
```

