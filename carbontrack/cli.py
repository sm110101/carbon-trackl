import argparse
from typing import Optional
from pathlib import Path
import sys

from .utils import get_cache_dir
from . import data as data_mod
from . import preprocess as prep_mod
from . import model as model_mod
from . import viz as viz_mod
from . import maintenance as maint_mod

try:
    # Optional import; dependency is declared, but handle gracefully if unavailable
    from codecarbon import EmissionsTracker  # type: ignore
except Exception:  # pragma: no cover
    EmissionsTracker = None  # type: ignore


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
        print(out_df.to_string(index=False))


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
    preds = model_mod.train_and_forecast(country=args.country, horizon=args.horizon, cache_dir=args.cache_dir)
    print(preds.to_string(index=False))


def cmd_plot(args: argparse.Namespace) -> None:
    if not args.country:
        print("Please specify --country, e.g., --country 'United States'")
        sys.exit(2)
    out = viz_mod.plot_country(country=args.country, save_path=args.output, cache_dir=args.cache_dir)
    print(f"Saved plot -> {out}")


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
    p_fc.set_defaults(func=cmd_forecast)

    p_plot = sub.add_parser("plot", help="Plot historical and forecast for a country")
    p_plot.add_argument("--country", required=True, help="Country name as in OWID dataset")
    p_plot.add_argument("-o", "--output", help="Optional output file path (PNG). Defaults to cache dir.")
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

    return p


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    # initialize cache dir early to ensure subdirs exist
    base = get_cache_dir(args.cache_dir)

    tracker = None
    emissions_dir = base / "emissions"
    emissions_kg = None
    if EmissionsTracker is not None:
        emissions_dir.mkdir(parents=True, exist_ok=True)
        try:
            tracker = EmissionsTracker(
                project_name="CarbonTrackCLI",
                output_dir=str(emissions_dir),
                log_level="warning",
                save_to_file=True,
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
            print(f"CodeCarbon: {emissions_kg:.6f} kg CO2e logged -> {emissions_dir}")


if __name__ == "__main__":
    main()


