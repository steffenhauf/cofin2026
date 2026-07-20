#!/usr/bin/env python3
"""Package period-frontier return-rate CSVs from a portfolio sweep."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


README_TEMPLATE = """# Period-frontier return-rate data

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_dir", type=Path, help="Root directory of a completed portfolio sweep.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Zip path (default: <sweep_dir>/period_frontier_return_rates.zip).",
    )
    return parser.parse_args()


def find_csvs(sweep_dir: Path) -> list[Path]:
    csvs = sorted(
        path
        for profile_dir in sweep_dir.iterdir()
        if profile_dir.is_dir()
        for path in (profile_dir / "period_frontier_return_rates").rglob("*.csv")
        if path.is_file()
    )
    if not csvs:
        raise SystemExit(f"No period_frontier_return_rates CSVs found below {sweep_dir}.")
    return csvs


def package(sweep_dir: Path, output: Path) -> int:
    if not sweep_dir.is_dir():
        raise SystemExit(f"Sweep directory does not exist: {sweep_dir}")
    csvs = find_csvs(sweep_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("README.md", README_TEMPLATE)
        for path in csvs:
            archive.write(path, path.relative_to(sweep_dir).as_posix())
    return len(csvs)


def main() -> None:
    args = parse_args()
    output = args.output or args.sweep_dir / "period_frontier_return_rates.zip"
    count = package(args.sweep_dir, output)
    print(f"Wrote {output} with {count} period-frontier CSVs.")


if __name__ == "__main__":
    main()
