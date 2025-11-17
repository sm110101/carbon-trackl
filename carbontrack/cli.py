import argparse
from typing import Optional
from pathlib import Path
import sys
import os
import logging

from .utils import get_cache_dir
from . import data as data_mod
from . import preprocess as prep_mod
from . import model as model_mod
from . import viz as viz_mod
from . import maintenance as maint_mod
from .preprocess import STANDARDIZED_NAME

try:
    # Optional import; dependency is declared, but handle gracefully if unavailable
    from codecarbon import EmissionsTracker  # type: ignore
except Exception:  # pragma: no cover
    EmissionsTracker = None  # type: ignore


def _configure_codecarbon_logging() -> None:
    try:
        os.environ.setdefault("CODECARBON_LOG_LEVEL", "ERROR")
        logger = logging.getLogger("codecarbon")
        logger.setLevel(logging.ERROR)
        logger.propagate = False
        for handler in list(logger.handlers):
            try:
                handler.setLevel(logging.ERROR)
            except Exception:
                pass
    except Exception:
        # Best-effort only
        pass


def _format_emissions(emissions_kg: float) -> str:
    if emissions_kg is None:
        return "n/a"
    if emissions_kg >= 1.0:
        return f"{emissions_kg:.3f} kg CO2e"
    grams = emissions_kg * 1_000.0
    if grams >= 1.0:
        return f"{grams:.1f} g CO2e"
    mg = grams * 1_000.0
    return f"{mg:.0f} mg CO2e"


def cmd_emissions(args: argparse.Namespace) -> None:
    from pandas import read_csv, read_json  # local import to keep CLI light

    base = get_cache_dir(args.cache_dir)
    emissions_dir = base / "emissions"

    # Try common CodeCarbon outputs in priority order
    candidates = []
    candidates.append(emissions_dir / "emissions.csv")
    candidates.extend(sorted(emissions_dir.glob("*.csv")))
    jsonl_candidates = [emissions_dir / "emissions.jsonl"]
    jsonl_candidates.extend(sorted(emissions_dir.glob("*.jsonl")))

    df = None
    last_error = None
    for csv_path in candidates:
        if csv_path.exists():
            try:
                df = read_csv(csv_path)
                break
            except Exception as e:
                last_error = e
                continue
    if df is None:
        # Try jsonl as a fallback
        for jl_path in jsonl_candidates:
            if jl_path.exists():
                try:
                    df = read_json(jl_path, lines=True)
                    break
                except Exception as e:
                    last_error = e
                    continue

    if df is None:
        hint = "Install CodeCarbon in your environment: pip install codecarbon"
        if last_error:
            print(f"No emissions logs found under {emissions_dir}.\nHint: {hint}\n(Last read error: {last_error})")
        else:
            print(f"No emissions logs found under {emissions_dir}.\nHint: {hint}")
        return

    # Sort by timestamp if present; otherwise by index
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    if args.path:
        print(str(csv_path))
        return

    if args.total:
        col = "emissions" if "emissions" in df.columns else None
        if not col:
            print("Emissions column not found in log.")
            return
        total = float(df[col].sum())
        count = int(len(df))
        print(f"Total tracked emissions: {total:.6f} kg CO2e across {count} runs")
        return

    # Select a compact set of columns if available
    preferred = [
        "timestamp",
        "duration",
        "emissions",  # kg
        "run_id",
        "project_name",
    ]
    present_cols = [c for c in preferred if c in df.columns]
    if not present_cols:
        present_cols = list(df.columns)

    out_df = df.tail(args.last)[present_cols]
    if args.json:
        # Minimal JSON array of records
        import json

        print(json.dumps(out_df.to_dict(orient="records")))
    else:
        # Nicely formatted table
        # Rename a couple columns for readability if present
        rename_map = {}
        if "duration" in out_df.columns:
            rename_map["duration"] = "duration_s"
        if "emissions" in out_df.columns:
            rename_map["emissions"] = "emissions_kg"
        if rename_map:
            out_df = out_df.rename(columns=rename_map)
        # Add human-readable emissions column
        if "emissions_kg" in out_df.columns:
            out_df = out_df.copy()
            out_df["co2e"] = out_df["emissions_kg"].apply(_format_emissions)
            # Prefer a concise default column order if possible
            order = [c for c in ["timestamp", "duration_s", "co2e", "run_id", "project_name"] if c in out_df.columns]
            if order:
                out_df = out_df[order]
        print(out_df.to_string(index=False))


def _load_standardized_df(cache_dir: Optional[str]):
    import pandas as pd

    base = get_cache_dir(cache_dir)
    std_path = base / "processed" / STANDARDIZED_NAME
    if not std_path.exists():
        raise FileNotFoundError(f"Standardized data not found at {std_path}. Run 'carbontrack prepare' first.")
    return pd.read_csv(std_path)


def cmd_info(args: argparse.Namespace) -> None:
    df = _load_standardized_df(args.cache_dir)
    num_rows = len(df)
    num_countries = df["country"].nunique() if "country" in df.columns else 0
    min_year = int(df["year"].min()) if "year" in df.columns and len(df) else 0
    max_year = int(df["year"].max()) if "year" in df.columns and len(df) else 0
    print(f"Rows: {num_rows}")
    print(f"Countries: {num_countries}")
    print(f"Year range: {min_year}–{max_year}")


def cmd_countries(args: argparse.Namespace) -> None:
    df = _load_standardized_df(args.cache_dir)
    series = df["country"].drop_duplicates().sort_values()
    if args.search:
        needle = args.search.lower()
        series = series[series.str.lower().str.contains(needle)]
    if args.limit:
        series = series.head(args.limit)
    for c in series:
        print(c)


def cmd_sectors(args: argparse.Namespace) -> None:
    import pandas as pd
    df = _load_standardized_df(args.cache_dir)
    if "sector" in df.columns:
        series = df["sector"].drop_duplicates().sort_values()
    else:
        # Fallback for datasets without explicit sector column
        series = pd.Series(["all"])
    if args.search:
        needle = args.search.lower()
        series = series[series.str.lower().str.contains(needle)]
    if args.limit:
        series = series.head(args.limit)
    for s in series:
        print(s)


def cmd_years(args: argparse.Namespace) -> None:
    if not args.country:
        print("Please specify --country")
        sys.exit(2)
    df = _load_standardized_df(args.cache_dir)
    sub = df[df["country"] == args.country]
    if len(sub) == 0:
        print(f"No rows for country '{args.country}'.")
        return
    years = sub["year"].astype(int)
    print(f"{args.country}: {int(years.min())}–{int(years.max())}  ({len(sub)} rows)")


def cmd_top(args: argparse.Namespace) -> None:
    import pandas as pd

    df = _load_standardized_df(args.cache_dir)
    if "year" not in df.columns or "emissions_co2e" not in df.columns:
        print("Required columns missing in standardized data.")
        return
    sub = df[df["year"] == args.year][["country", "emissions_co2e"]].copy()
    if len(sub) == 0:
        print(f"No rows for year {args.year}.")
        return
    sub = sub.sort_values("emissions_co2e", ascending=False).head(args.limit)
    print(sub.to_string(index=False))


def cmd_schema(args: argparse.Namespace) -> None:
    df = _load_standardized_df(args.cache_dir)
    for c in df.columns:
        print(c)


def cmd_head(args: argparse.Namespace) -> None:
    df = _load_standardized_df(args.cache_dir)
    print(df.head(args.rows).to_string(index=False))


def cmd_fetch(args: argparse.Namespace) -> None:
    if args.source.lower() in ("owid", "all"):
        path = data_mod.fetch_owid_co2(cache_dir=args.cache_dir)
        print(f"Fetched OWID CO2 data -> {path}")
    else:
        print("Only 'owid' is supported in MVP. Skipping.")


def cmd_prepare(args: argparse.Namespace) -> None:
    out = prep_mod.standardize(cache_dir=args.cache_dir)
    print(f"Standardized dataset -> {out}")


def cmd_forecast(args: argparse.Namespace) -> None:
    if not args.country:
        print("Please specify --country, e.g., --country 'United States'")
        sys.exit(2)
    preds = model_mod.train_and_forecast(country=args.country, horizon=args.horizon, cache_dir=args.cache_dir, sector=args.sector)
    print(preds.to_string(index=False))


def cmd_plot(args: argparse.Namespace) -> None:
    if not args.country:
        print("Please specify --country, e.g., --country 'United States'")
        sys.exit(2)
    out = viz_mod.plot_country(country=args.country, save_path=args.output, cache_dir=args.cache_dir, sector=args.sector)
    print(f"Saved plot -> {out}")


def _parse_countries_arg(countries: Optional[str], countries_file: Optional[str]) -> list[str]:
    result: list[str] = []
    if countries:
        parts = [p.strip() for p in countries.split(",") if p.strip()]
        result.extend(parts)
    if countries_file:
        p = Path(countries_file).expanduser().resolve()
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    result.append(line)
    return sorted(set(result))


def cmd_forecast_batch(args: argparse.Namespace) -> None:
    countries = _parse_countries_arg(args.countries, args.countries_file)
    if not countries:
        print("Please provide --countries CSV or --countries-file path")
        sys.exit(2)
    preds = model_mod.train_and_forecast_many(countries=countries, horizon=args.horizon, cache_dir=args.cache_dir, sector=args.sector)
    if len(preds) == 0:
        print("No forecasts generated.")
        return
    print(preds.to_string(index=False))


def cmd_plot_grid(args: argparse.Namespace) -> None:
    countries = _parse_countries_arg(args.countries, args.countries_file)
    if not countries:
        print("Please provide --countries CSV or --countries-file path")
        sys.exit(2)
    out = viz_mod.plot_grid(countries=countries, save_path=args.output, cache_dir=args.cache_dir, sector=args.sector, cols=args.cols)
    print(f"Saved grid -> {out}")


def cmd_clean(args: argparse.Namespace) -> None:
    remove_cache = args.all or args.cache
    remove_build = args.all or args.build
    remove_outputs = args.all or args.outputs
    summary = maint_mod.clean(
        cache_dir=args.cache_dir,
        remove_cache=remove_cache,
        remove_build=remove_build,
        remove_local_outputs=remove_outputs,
        venv_path=args.venv_path,
    )
    print("Cleanup summary:")
    print(f"- cache_removed: {summary['cache_removed']}")
    print(f"- build_removed: {len(summary['build_removed'])} items")
    for p in summary["build_removed"]:
        print(f"  - {p}")
    print(f"- outputs_removed: {len(summary['outputs_removed'])} files")
    for p in summary["outputs_removed"]:
        print(f"  - {p}")
    print(f"- venv_removed: {summary['venv_removed']}")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="carbontrack", description="CarbonTrackCLI – Track, forecast, and visualize CO2 emissions.")
    p.add_argument("--cache-dir", help="Custom cache directory. Defaults to ~/.carbontrack", default=None)
    sub = p.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="Fetch and cache raw datasets")
    p_fetch.add_argument("--source", default="owid", help="Data source: owid (MVP)")
    p_fetch.set_defaults(func=cmd_fetch)

    p_prep = sub.add_parser("prepare", help="Create standardized dataset CSV")
    p_prep.set_defaults(func=cmd_prepare)

    p_fc = sub.add_parser("forecast", help="Train simple model and forecast next N years")
    p_fc.add_argument("--country", required=True, help="Country name as in OWID dataset, e.g., 'United States'")
    p_fc.add_argument("--horizon", type=int, default=5, help="Forecast horizon in years (default 5)")
    p_fc.add_argument("--sector", default="all", help="Sector filter (default 'all')")
    p_fc.set_defaults(func=cmd_forecast)

    p_plot = sub.add_parser("plot", help="Plot historical and forecast for a country")
    p_plot.add_argument("--country", required=True, help="Country name as in OWID dataset")
    p_plot.add_argument("-o", "--output", help="Optional output file path (PNG). Defaults to cache dir.")
    p_plot.add_argument("--sector", default="all", help="Sector filter (default 'all')")
    p_plot.set_defaults(func=cmd_plot)

    p_clean = sub.add_parser("clean", help="Remove cache, build artifacts, local outputs, and optional venv")
    p_clean.add_argument("--cache-dir", help="If set, deletes this cache dir instead of default/env", default=None)
    p_clean.add_argument("--all", action="store_true", help="Remove cache, build artifacts, and local outputs")
    p_clean.add_argument("--cache", action="store_true", help="Remove cache directory only")
    p_clean.add_argument("--build", action="store_true", help="Remove dist/, build/, *.egg-info in CWD")
    p_clean.add_argument("--outputs", action="store_true", help="Remove local *_emissions.png files in CWD")
    p_clean.add_argument("--venv-path", help="Optional path to virtual environment to delete", default=None)
    p_clean.set_defaults(func=cmd_clean)

    p_em = sub.add_parser("emissions", help="Display CodeCarbon emissions logs")
    p_em.add_argument("--last", type=int, default=5, help="Show last N runs (default 5)")
    p_em.add_argument("--total", action="store_true", help="Show total emissions across all runs")
    p_em.add_argument("--json", action="store_true", help="Output JSON instead of table for the selected rows")
    p_em.add_argument("--path", action="store_true", help="Print the path to the emissions CSV and exit")
    p_em.set_defaults(func=cmd_emissions)

    p_fcb = sub.add_parser("forecast-batch", help="Forecast next N years for multiple countries")
    p_fcb.add_argument("--countries", help="Comma-separated list of countries")
    p_fcb.add_argument("--countries-file", help="Path to file with one country per line")
    p_fcb.add_argument("--horizon", type=int, default=5, help="Forecast horizon in years (default 5)")
    p_fcb.add_argument("--sector", default="all", help="Sector filter (default 'all')")
    p_fcb.set_defaults(func=cmd_forecast_batch)

    p_grid = sub.add_parser("plot-grid", help="Plot multiple countries side-by-side")
    p_grid.add_argument("--countries", help="Comma-separated list of countries")
    p_grid.add_argument("--countries-file", help="Path to file with one country per line")
    p_grid.add_argument("--sector", default="all", help="Sector filter (default 'all')")
    p_grid.add_argument("--cols", type=int, default=2, help="Number of columns in the subplot grid (default 2)")
    p_grid.add_argument("-o", "--output", help="Optional output file path (PNG). Defaults to cache dir.")
    p_grid.set_defaults(func=cmd_plot_grid)

    p_info = sub.add_parser("info", help="Show standardized dataset summary")
    p_info.set_defaults(func=cmd_info)

    p_countries = sub.add_parser("countries", help="List countries found in standardized dataset")
    p_countries.add_argument("--search", help="Substring to filter country names (case-insensitive)", default=None)
    p_countries.add_argument("--limit", type=int, help="Limit number of countries shown", default=None)
    p_countries.set_defaults(func=cmd_countries)

    p_sectors = sub.add_parser("sectors", help="List sectors found in standardized dataset")
    p_sectors.add_argument("--search", help="Substring to filter sector names (case-insensitive)", default=None)
    p_sectors.add_argument("--limit", type=int, help="Limit number of sectors shown", default=None)
    p_sectors.set_defaults(func=cmd_sectors)

    p_years = sub.add_parser("years", help="Show available year range for a country")
    p_years.add_argument("--country", required=True, help="Country name as in dataset")
    p_years.set_defaults(func=cmd_years)

    p_top = sub.add_parser("top", help="Top emitters for a given year")
    p_top.add_argument("--year", type=int, required=True, help="Year to rank by emissions")
    p_top.add_argument("--limit", type=int, default=10, help="Number of countries to show (default 10)")
    p_top.set_defaults(func=cmd_top)

    p_schema = sub.add_parser("schema", help="Print standardized dataset column names")
    p_schema.set_defaults(func=cmd_schema)

    p_head = sub.add_parser("head", help="Preview first N rows of standardized dataset")
    p_head.add_argument("--rows", type=int, default=5, help="Number of rows to show (default 5)")
    p_head.set_defaults(func=cmd_head)

    return p


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    # initialize cache dir early to ensure subdirs exist
    base = get_cache_dir(args.cache_dir)

    tracker = None
    emissions_dir = base / "emissions"
    emissions_kg = None
    # Only track for workload commands; skip for exploration/listing
    should_track = getattr(args, "command", None) in ("fetch", "prepare", "forecast", "plot", "forecast-batch", "plot-grid")
    if EmissionsTracker is not None and should_track:
        _configure_codecarbon_logging()
        emissions_dir.mkdir(parents=True, exist_ok=True)
        try:
            tracker = EmissionsTracker(
                project_name="CarbonTrackCLI",
                output_dir=str(emissions_dir),
                log_level="error",
                save_to_file=True,
                tracking_mode="process",
            )
            tracker.start()
        except Exception:
            tracker = None

    try:
        args.func(args)
    finally:
        if tracker is not None:
            try:
                emissions_kg = tracker.stop()
            except Exception:
                emissions_kg = None
        if emissions_kg is not None:
            print(f"CO2e: {_format_emissions(float(emissions_kg))}  |  log: {emissions_dir}")


if __name__ == "__main__":
    main()


