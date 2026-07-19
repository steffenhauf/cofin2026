#!/usr/bin/env python3
"""Create an interactive Excel mean-variance workbook from portfolio sweeps.

The workbook stores one compact mean/covariance record per organisation scenario.
Excel users select their company profile and constraints, then edit asset weights.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path


MANIFEST_NAME = "portfolio_paired_future_manifest.csv"
PATHS_NAME = "portfolio_paired_future_return_rates.csv"
STATISTICS_NAME = "portfolio_service_sleeve_statistics.csv"
SLEEVE_MANIFEST_NAME = "portfolio_service_sleeve_manifest.csv"
ASSETS = ("global_cloud", "eu_cloud", "onprem_oss", "cloud_oss")


@dataclass
class Moments:
    count: int = 0
    mean: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    m2: list[list[float]] = field(default_factory=lambda: [[0.0] * 3 for _ in range(3)])

    def add(self, values: list[float]) -> None:
        self.count += 1
        delta = [value - self.mean[index] for index, value in enumerate(values)]
        self.mean = [mean + change / self.count for mean, change in zip(self.mean, delta)]
        for row in range(3):
            for column in range(3):
                self.m2[row][column] += delta[row] * (values[column] - self.mean[column])

    def covariance(self) -> list[list[float]]:
        if self.count < 2:
            raise ValueError("At least two paired futures are required for covariance.")
        return [[value / (self.count - 1) for value in row] for row in self.m2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_folder", type=Path, help="Sweep root or one company-profile output folder.")
    parser.add_argument("--month", type=float, help="Simulation month to use (default: final month for every scenario).")
    parser.add_argument("--output", type=Path, help="Workbook path (default: sweep_folder/welch_ai_portfolio_analysis.xlsx).")
    return parser.parse_args()


def discover_outputs(sweep_folder: Path) -> list[Path]:
    if (sweep_folder / MANIFEST_NAME).exists() and (sweep_folder / PATHS_NAME).exists():
        return [sweep_folder]
    outputs = sorted(path.parent for path in sweep_folder.rglob(MANIFEST_NAME) if (path.parent / PATHS_NAME).exists())
    if not outputs:
        raise SystemExit(f"No paired-future CSVs found below {sweep_folder}.")
    return outputs


def parse_scenario_id(scenario_id: str) -> dict[str, str]:
    profile, *parts = scenario_id.split("|")
    values = {"company_profile": profile}
    for part in parts:
        key, value = part.split("=", 1)
        values[key] = value
    return values


def selected_assets(folder: Path) -> dict[str, dict[str, dict[str, str]]]:
    selected: dict[str, dict[str, dict[str, str]]] = {}
    with (folder / MANIFEST_NAME).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            selected.setdefault(row["organization_scenario_id"], {})[row["portfolio"]] = row
    return selected


def final_months(path: Path) -> dict[str, float]:
    months: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            scenario_id = row["organization_scenario_id"]
            month = float(row["month"])
            months[scenario_id] = max(month, months.get(scenario_id, month))
    return months


def scenario_statistics(folder: Path, requested_month: float | None) -> dict[str, tuple[float, Moments]]:
    paths = folder / PATHS_NAME
    selected_months = final_months(paths) if requested_month is None else {}
    moments: dict[str, Moments] = {}
    with paths.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            scenario_id = row["organization_scenario_id"]
            month = float(row["month"])
            target_month = selected_months.get(scenario_id, requested_month)
            if month != target_month:
                continue
            if any(not row[asset] for asset in ASSETS):
                continue
            moments.setdefault(scenario_id, Moments()).add([float(row[asset]) for asset in ASSETS])
    return {scenario_id: ((selected_months.get(scenario_id, requested_month)), value) for scenario_id, value in moments.items()}


def collect_data(output_folders: list[Path], requested_month: float | None) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    scenario_rows: list[dict[str, object]] = []
    asset_rows: list[dict[str, str]] = []
    for folder in output_folders:
        statistics_path = folder / STATISTICS_NAME
        manifest_path = folder / SLEEVE_MANIFEST_NAME
        if statistics_path.exists() and manifest_path.exists():
            with statistics_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if requested_month is not None and float(row["month"]) != requested_month:
                        continue
                    scenario_rows.append(
                        {
                            key: (float(value) if key.startswith(("mean_", "cov_", "service_", "hardware_", "confidential")) else int(value) if key in {"users", "month", "futures"} else value)
                            for key, value in row.items()
                        }
                    )
            with manifest_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    row["asset_description"] = " | ".join(
                        f"{key}={row[key]}" for key in ("model", "access_plan", "hardware", "service_provider", "shock_combination")
                    )
                    asset_rows.append(row)
            continue
        raise SystemExit(
            f"{folder} lacks {STATISTICS_NAME}. Rerun the portfolio sweep with the current code first."
        )
        asset_map = selected_assets(folder)
        for scenario_id, (month, moments) in scenario_statistics(folder, requested_month).items():
            if scenario_id not in asset_map or set(asset_map[scenario_id]) != set(ASSETS):
                continue
            values = parse_scenario_id(scenario_id)
            covariance = moments.covariance()
            scenario_rows.append(
                {
                    "organization_scenario_id": scenario_id,
                    "company_profile": values["company_profile"],
                    "users": int(values["users"]),
                    "service_usd": float(values["service_usd"]),
                    "hardware_usd": float(values["hardware_usd"]),
                    "confidential": float(values["confidential"]),
                    "plateau": str(values["plateau"]).capitalize(),
                    "month": month,
                    "futures": moments.count,
                    "mean_low_risk": moments.mean[0],
                    "mean_optimum": moments.mean[1],
                    "mean_high_gain": moments.mean[2],
                    "cov_ll": covariance[0][0], "cov_lo": covariance[0][1], "cov_lh": covariance[0][2],
                    "cov_oo": covariance[1][1], "cov_oh": covariance[1][2], "cov_hh": covariance[2][2],
                }
            )
            for asset in ASSETS:
                asset_rows.append({"organization_scenario_id": scenario_id, "service_sleeve": asset, **asset_map[scenario_id][asset]})
    if not scenario_rows:
        raise SystemExit("No complete scenario records were found.")
    scenario_rows = sorted(scenario_rows, key=lambda row: str(row["organization_scenario_id"]))
    selector_keys: dict[str, int] = {}
    for row in scenario_rows:
        selector_keys.setdefault(row["organization_scenario_id"], len(selector_keys) + 1)
        row["selector_key"] = selector_keys[row["organization_scenario_id"]]
    for row in asset_rows:
        row["selector_key"] = selector_keys[row["organization_scenario_id"]]
    return scenario_rows, asset_rows


def create_workbook(output_path: Path, scenarios: list[dict[str, object]], assets: list[dict[str, str]]) -> None:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.worksheet.datavalidation import DataValidation
    except ImportError as exc:
        raise SystemExit("Install the workbook dependency first: python -m pip install openpyxl") from exc

    from openpyxl.utils import get_column_letter

    workbook = Workbook()
    portfolio = workbook.active
    portfolio.title = "Portfolio"
    data = workbook.create_sheet("Scenario data")
    asset_sheet = workbook.create_sheet("Selected assets")
    lists = workbook.create_sheet("Lists")
    lists.sheet_state = "hidden"
    title_fill = PatternFill("solid", fgColor="1F4E78")
    header_fill = PatternFill("solid", fgColor="D9EAF7")
    input_fill = PatternFill("solid", fgColor="FFF2CC")

    headers = list(scenarios[0])
    data.append(headers)
    for row in scenarios:
        data.append([row[header] for header in headers])
    for cell in data[1]:
        cell.font = Font(bold=True)
        cell.fill = header_fill
    data.freeze_panes = "A2"
    data.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{len(scenarios) + 1}"
    for index in range(1, len(headers) + 1):
        data.column_dimensions[get_column_letter(index)].width = 18

    asset_headers = ["scenario_asset_key", "selector_key", "organization_scenario_id", "service_sleeve", "model", "token_cost", "access_plan", "hardware", "hardware_refresh", "local_fallback", "service_provider", "shock_combination", "asset_description"]
    asset_sheet.append(asset_headers)
    for row in assets:
        asset_sheet.append([str(row["selector_key"]) + row["service_sleeve"], row["selector_key"], *(row.get(header, "") for header in asset_headers[2:])])
    for cell in asset_sheet[1]:
        cell.font = Font(bold=True)
        cell.fill = header_fill
    asset_sheet.freeze_panes = "A2"
    asset_sheet.auto_filter.ref = f"A1:M{len(assets) + 1}"
    for index in range(1, 14):
        asset_sheet.column_dimensions[get_column_letter(index)].width = 22
    asset_sheet.column_dimensions["M"].width = 110

    selectors = {
        "company_profile": sorted({str(row["company_profile"]) for row in scenarios}),
        "users": sorted({int(row["users"]) for row in scenarios}),
        "service_usd": sorted({float(row["service_usd"]) for row in scenarios}),
        "hardware_usd": sorted({float(row["hardware_usd"]) for row in scenarios}),
        "confidential": sorted({float(row["confidential"]) for row in scenarios}),
        "plateau": ["False", "True"],
        "month": sorted({float(row["month"]) for row in scenarios}),
    }
    for column, (name, values) in enumerate(selectors.items(), start=1):
        lists.cell(1, column, name)
        for row, value in enumerate(values, start=2):
            lists.cell(row, column, value)

    portfolio["A1"] = "Welch-like AI service portfolio analysis"
    portfolio["A1"].font = Font(size=16, bold=True, color="FFFFFF")
    portfolio["A1"].fill = title_fill
    portfolio.merge_cells("A1:D1")
    fields = [("Company profile", "company_profile"), ("Employees", "users"), ("Service budget USD/person/month", "service_usd"), ("Hardware budget USD", "hardware_usd"), ("Confidential-work share", "confidential"), ("Capability plateau", "plateau"), ("Evaluation month", "month")]
    first = scenarios[0]
    for row, (label, key) in enumerate(fields, start=3):
        portfolio.cell(row, 1, label)
        portfolio.cell(row, 2, first[key])
        portfolio.cell(row, 2).fill = input_fill
        validation = DataValidation(type="list", formula1=f"=Lists!${get_column_letter(row - 2)}$2:${get_column_letter(row - 2)}${len(selectors[key]) + 1}")
        portfolio.add_data_validation(validation)
        validation.add(portfolio.cell(row, 2))
    selector_column = get_column_letter(headers.index("selector_key") + 1)
    portfolio["A10"] = "Scenario selector key"
    data_last = len(scenarios) + 1
    criteria = (
        f"('Scenario data'!$B$2:$B${data_last}=B3)*('Scenario data'!$C$2:$C${data_last}=B4)*"
        f"('Scenario data'!$D$2:$D${data_last}=B5)*('Scenario data'!$E$2:$E${data_last}=B6)*"
        f"('Scenario data'!$F$2:$F${data_last}=B7)*('Scenario data'!$G$2:$G${data_last}=B8)"
    )
    portfolio["B10"] = f"=IFERROR(SUMPRODUCT({criteria}*'Scenario data'!${selector_column}$2:${selector_column}${data_last})/SUMPRODUCT({criteria}),0)"
    portfolio["A11"] = "Matching scenario-month rows"
    portfolio["B11"] = f"=SUMPRODUCT({criteria}*('Scenario data'!$H$2:$H${data_last}=B9))"
    portfolio["A12"] = "Selected month"
    portfolio["B12"] = "=B9"
    portfolio["A13"] = "Asset"
    portfolio["B13"] = "Weight"
    portfolio["C13"] = "Expected annualized return"
    portfolio["D13"] = "Selected asset description"
    for cell in portfolio[13]:
        cell.font = Font(bold=True)
        cell.fill = header_fill
    mean_columns = {asset: get_column_letter(headers.index(f"mean_{asset}") + 1) for asset in ASSETS}
    for row, asset in enumerate(ASSETS, start=14):
        portfolio.cell(row, 1, asset.replace("_", " ").title())
        portfolio.cell(row, 2, 1 / 3).fill = input_fill
        portfolio.cell(row, 3, f'=IFERROR(SUMIFS(\'Scenario data\'!${mean_columns[asset]}$2:${mean_columns[asset]}${data_last},\'Scenario data\'!${selector_column}$2:${selector_column}${data_last},$B$10,\'Scenario data\'!$H$2:$H${data_last},$B$9),"")')
        portfolio.cell(row, 4, f'=IFERROR(INDEX(\'Selected assets\'!$M$2:$M${len(assets) + 1},MATCH($B$10&"{asset}",\'Selected assets\'!$A$2:$A${len(assets) + 1},0)),"")')
        portfolio.cell(row, 4).alignment = Alignment(wrap_text=True, vertical="top")
    weight_validation = DataValidation(type="decimal", operator="between", formula1="0", formula2="1")
    portfolio.add_data_validation(weight_validation)
    weight_validation.add("B14:B17")
    portfolio["A19"] = "Weight total"
    portfolio["B19"] = "=SUM(B14:B17)"
    portfolio["A21"] = "Portfolio expected annualized return"
    portfolio["B21"] = "=SUMPRODUCT(B14:B17,C14:C17)"
    portfolio["A22"] = "Portfolio risk (standard deviation)"
    portfolio["B22"] = "=SQRT(SUMPRODUCT(B14:B17,MMULT(B32:E35,B14:B17)))"
    portfolio["A23"] = "Return per unit of risk"
    portfolio["B23"] = "=IFERROR(B21/B22,\"\")"
    portfolio["A25"] = "Target annualized return"
    portfolio["B25"] = "=B21"
    portfolio["B25"].fill = input_fill
    portfolio["A26"] = "Target risk"
    portfolio["B26"] = "=B22"
    portfolio["B26"].fill = input_fill
    portfolio["A27"] = "Solver objective (minimize)"
    portfolio["A24"] = "Optimize for"
    portfolio["B24"] = "Return"
    portfolio["B24"].fill = input_fill
    optimization_validation = DataValidation(type="list", formula1='="Return,Risk"')
    portfolio.add_data_validation(optimization_validation)
    optimization_validation.add("B24")
    portfolio["B27"] = '=IF(B24="Return",(B21-B25)^2,(B22-B26)^2)'
    portfolio["A30"] = "Covariance matrix (paired Monte Carlo futures)"
    for index, asset in enumerate(ASSETS, start=2):
        portfolio.cell(31, index, asset)
        portfolio.cell(index + 30, 1, asset)
    for row, left in enumerate(ASSETS):
        for column, right in enumerate(ASSETS):
            source = get_column_letter(headers.index(f"cov_{left}_{right}") + 1)
            portfolio.cell(row + 32, column + 2, f'=IFERROR(SUMIFS(\'Scenario data\'!${source}$2:${source}${data_last},\'Scenario data\'!${selector_column}$2:${selector_column}${data_last},$B$10,\'Scenario data\'!$H$2:$H${data_last},$B$9),"")')
    portfolio["A37"] = "Use the yellow cells to select the organisation, month, optimization target, and weights. In Excel Solver, minimize B27 by changing B14:B17 subject to B19=1 and B14:B17>=0. Only the selected Return or Risk target is used."
    portfolio.merge_cells("A37:D38")
    portfolio["A37"].alignment = Alignment(wrap_text=True, vertical="top")
    portfolio.column_dimensions["A"].width = 34
    portfolio.column_dimensions["B"].width = 24
    portfolio.column_dimensions["C"].width = 28
    portfolio.column_dimensions["D"].width = 110
    portfolio.freeze_panes = "A13"

    for sheet in (portfolio, data):
        for row in sheet.iter_rows():
            for cell in row:
                if isinstance(cell.value, (float, int)) or (isinstance(cell.value, str) and cell.value.startswith("=")):
                    cell.number_format = "0.0000"
    workbook.calculation.fullCalcOnLoad = True
    workbook.calculation.forceFullCalc = True
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)


def main() -> None:
    args = parse_args()
    scenarios, assets = collect_data(discover_outputs(args.sweep_folder), args.month)
    output = args.output or args.sweep_folder / "welch_ai_portfolio_analysis.xlsx"
    create_workbook(output, scenarios, assets)
    print(f"Wrote {output} with {len(scenarios)} organisation scenarios.")


if __name__ == "__main__":
    main()
