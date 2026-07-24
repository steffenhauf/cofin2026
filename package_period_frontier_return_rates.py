#!/usr/bin/env python3
"""Package period-frontier return-rate CSVs from a portfolio sweep."""

from __future__ import annotations

import argparse
import csv
from io import StringIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


LEGACY_README_TEMPLATE = """# Period-frontier return-rate data

This archive contains the `period_frontier_return_rates` output for every
organisation profile found in the source sweep.

## Folder layout

```text
<organisation profile>/
└── period_frontier_return_rates/
    └── users_<N>/
        └── max_hardware_invest_usd_<budget>/
            └── confidential_document_fraction_<fraction>/
                └── access_plan_<plan>/
                    └── <configuration>_access_plan_<plan>.csv
```

The organisation profile is the company work-mix profile, for example
`mixed` or `software_engineering_heavy`.

The nested configuration directories identify:

- `users_<N>`: organisation size in employees;
- `max_hardware_invest_usd_<budget>`: maximum upfront hardware investment;
- `confidential_document_fraction_<fraction>`: confidential-document share,
  with decimal points encoded as `p` in paths (for example, `0p25`).
- `access_plan_<plan>`: access-plan-specific frontier output, one of
  `pay_per_use`, `flatrate_limited`, or `flatrate_limited_topup`.

The filename additionally records the service budget and capability-plateau
setting so that every simulation configuration remains uniquely identifiable.

## CSV layout

Each CSV represents one organisation configuration. It has one row for every
simulation period and contains:

- `period`: zero-based simulation period index;
- `month`: simulation month corresponding to that period;
- `period_<P>_low_risk`: return rate of the asset selected at period `P` in
  the low-risk band;
- `period_<P>_medium_risk`: return rate of the asset selected at period `P`
  in the medium-risk band;
- `period_<P>_high_risk`: return rate of the asset selected at period `P`
  in the high-risk band.

There are three time-series columns for each selection period `P`. Each such
column contains the selected asset's full simulation history, including
periods before and after the period at which that asset was selected.

The risk bands are based on the final-period Pareto frontier available to the
sweep selection logic: low risk is at or below median risk minus one standard
deviation; medium risk is above that lower boundary and at or below median
risk plus one standard deviation; high risk is above the upper boundary.
Within each band, the asset with the highest return/risk is selected. If a
band contains no frontier point, the sweep's documented fallback keeps the
three-column layout complete.

The return values are median annualized return rates across paired Monte Carlo
futures. The CSVs are intended for time-series analysis and do not contain
the individual future paths or the full asset metadata manifest.
"""

FINAL_SHOCK_README_TEMPLATE = """# Final-horizon shock-split frontier data

This archive contains the `final_shock_frontier_return_rates` output for every
organisation profile found in the source sweep.

Each CSV has one row per simulation period and 48 selected-asset rate columns:
the low-, medium-, and high-risk Pareto selections for each combination of
realized plateau, gradual price, abrupt price, and embargo shocks. The
selection occurs at the final horizon; each selected asset then contributes
its complete monthly return-rate history. Column names record the four yes/no
shock states, allowing posterior comparison by the shock outcomes accumulated
in each Monte Carlo future.
"""

PORTABLE_LAYOUT_NOTE = """
## Archive layout

CSV files use short, stable archive names to remain extractable on Windows:

```text
<organisation profile>/<sequence number>.csv
```

`manifest.csv` maps each archive path to its original relative source path.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_dir", type=Path, help="Root directory of a completed portfolio sweep.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Zip path (default: <sweep_dir>/final_shock_frontier_return_rates.zip).",
    )
    parser.add_argument(
        "--legacy-period-frontier-return-rates",
        action="store_true",
        help="Package the previous full-horizon period-frontier CSVs instead.",
    )
    parser.add_argument(
        "--legacy-layout",
        action="store_true",
        help="Preserve the original nested paths in the archive.",
    )
    return parser.parse_args()


def find_csvs(sweep_dir: Path, legacy: bool = False) -> list[Path]:
    folder_name = "period_frontier_return_rates" if legacy else "final_shock_frontier_return_rates"
    csvs = sorted(
        path
        for profile_dir in sweep_dir.iterdir()
        if profile_dir.is_dir()
        for path in (profile_dir / folder_name).rglob("*.csv")
        if path.is_file()
    )
    if not csvs:
        raise SystemExit(f"No {folder_name} CSVs found below {sweep_dir}.")
    return csvs


def package(sweep_dir: Path, output: Path, legacy: bool = False, legacy_layout: bool = False) -> int:
    if not sweep_dir.is_dir():
        raise SystemExit(f"Sweep directory does not exist: {sweep_dir}")
    csvs = find_csvs(sweep_dir, legacy)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        readme = LEGACY_README_TEMPLATE if legacy else FINAL_SHOCK_README_TEMPLATE
        if not legacy_layout:
            readme += PORTABLE_LAYOUT_NOTE
        archive.writestr("README.md", readme)
        manifest = StringIO(newline="")
        writer = csv.writer(manifest)
        if not legacy_layout:
            writer.writerow(("archive_path", "source_path"))
        for index, path in enumerate(csvs, start=1):
            source_path = path.relative_to(sweep_dir).as_posix()
            archive_path = source_path if legacy_layout else f"{path.relative_to(sweep_dir).parts[0]}/{index:06d}.csv"
            archive.write(path, archive_path)
            if not legacy_layout:
                writer.writerow((archive_path, source_path))
        if not legacy_layout:
            archive.writestr("manifest.csv", manifest.getvalue())
    return len(csvs)


def main() -> None:
    args = parse_args()
    output = args.output or args.sweep_dir / (
        "period_frontier_return_rates.zip" if args.legacy_period_frontier_return_rates else "final_shock_frontier_return_rates.zip"
    )
    count = package(
        args.sweep_dir,
        output,
        args.legacy_period_frontier_return_rates,
        args.legacy_layout,
    )
    print(f"Wrote {output} with {count} frontier CSVs.")


if __name__ == "__main__":
    main()
