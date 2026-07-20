#!/usr/bin/env python3
"""Render the current simulation flow and a compact linear risk/reward frontier."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from simulate_llm_efficiency import SimulationConfig, _resolve_backend
from sweep_portfolio_rate_changes import (
    COMPANY_PROFILES,
    ASSET_IDENTITY_COLUMNS,
    SweepScenario,
    return_change_summary,
    service_sleeve_statistics,
    simulate_portfolio_scenario,
)


RISK = "annualized_return_rate_risk_stddev"
REWARD = "mean_annualized_return_rate"
IDENTITY = (
    "model", "token_cost", "access_plan", "hardware", "hardware_refresh",
    "local_fallback", "service_provider", "shock_combination",
)


def efficient_frontier(points: pd.DataFrame) -> pd.DataFrame:
    ordered = points.sort_values([RISK, REWARD], ascending=[True, False])
    best = -np.inf
    keep = []
    for index, row in ordered.iterrows():
        if float(row[REWARD]) > best:
            keep.append(index)
            best = float(row[REWARD])
    return ordered.loc[keep]


def select_scenarios(points: pd.DataFrame) -> dict[str, pd.Series]:
    candidates = points.dropna(subset=[RISK, REWARD])
    if candidates.empty:
        raise SystemExit("The selected slice has no finite risk/reward points.")
    selected: dict[str, pd.Series] = {}

    def add(name: str, row: pd.Series) -> bool:
        identity = tuple(row[column] for column in IDENTITY)
        if all(identity != tuple(old[column] for column in IDENTITY) for old in selected.values()):
            selected[name] = row
            return True
        return False

    add("Low risk", candidates.sort_values([RISK, REWARD], ascending=[True, False]).iloc[0])
    add("High reward", candidates.sort_values([REWARD, RISK], ascending=[False, True]).iloc[0])
    positive = candidates[candidates[RISK] > 0].copy()
    if not positive.empty:
        positive["reward_risk"] = positive[REWARD] / positive[RISK]
        if not add("Best reward / risk", positive.sort_values("reward_risk", ascending=False).iloc[0]):
            for _, row in positive.sort_values("reward_risk", ascending=False).iterrows():
                if add("Balanced", row):
                    break
    for _, row in candidates.iterrows():
        if len(selected) == 3:
            break
        add("Balanced", row)
    return selected


SERVICE_MARKERS = {
    "cloud": "o",
    "on-prem": "^",
    "oss-cloud": "D",
    "European": "s",
}


def _service_type(row: pd.Series) -> str:
    provider = str(row["service_provider"])
    if provider == "global_service":
        return "cloud"
    if provider == "local_hardware":
        return "on-prem"
    if provider == "eu_cloud_oss":
        return "oss-cloud"
    if provider == "european_service":
        return "European"
    return "cloud"


def _illustrative(ax, service_paths: pd.DataFrame, individual_paths: int) -> None:
    service_columns = {
        "global_cloud": ("Cloud", "#2563eb"),
        "onprem_oss": ("On-prem", "#dc2626"),
        "cloud_oss": ("OSS cloud", "#059669"),
        "eu_cloud": ("European", "#7c3aed"),
    }
    for sleeve, (name, color) in service_columns.items():
        values = service_paths[["run", "period", "month", sleeve]].dropna()
        bands = values.groupby(["period", "month"], sort=True)[sleeve].agg(
            p10=lambda series: series.quantile(0.10),
            p50="median",
            p90=lambda series: series.quantile(0.90),
        ).reset_index()
        paths = list(values.groupby("run", sort=False))
        if individual_paths:
            path_indices = [round(index * (len(paths) - 1) / max(1, individual_paths - 1)) for index in range(individual_paths)]
            paths = [paths[index] for index in path_indices]
        for _, path in paths:
            path = path.sort_values("month")
            ax.plot(path["month"], path[sleeve], color=color, alpha=0.16, linewidth=0.55, zorder=1)
        risk = float(np.nanstd(values.loc[values["month"] == values["month"].max(), sleeve], ddof=1))
        ax.fill_between(bands["month"], bands["p10"], bands["p90"], color=color, alpha=0.16)
        ax.plot(bands["month"], bands["p50"], color=color, linewidth=1.4, marker="o",
                markersize=2.5, label=f"{name} (risk {risk:.3f})", zorder=3)

    ax.set_xlabel("Month", fontsize=7)
    ax.set_ylabel("Annualized return rate", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.legend(loc="best", fontsize=5.8)


def _frontier(
    ax, points: pd.DataFrame, scenarios: dict[str, pd.Series], x_max: float,
    workbook_assets: pd.DataFrame | None = None,
) -> None:
    ax.scatter(points[RISK], points[REWARD], s=9, alpha=0.32, color="#6b7280", linewidths=0)
    frontier = efficient_frontier(points)
    ax.plot(frontier[RISK], frontier[REWARD], color="#2563eb", linewidth=1.3, label="Pareto frontier")
    frontier = frontier.copy()
    frontier["service_type"] = frontier.apply(_service_type, axis=1)
    for service_type, service_points in frontier.groupby("service_type", sort=False):
        ax.scatter(
            service_points[RISK], service_points[REWARD], s=24,
            marker=SERVICE_MARKERS[service_type], color="#2563eb",
            edgecolor="white", linewidth=0.5, zorder=3,
        )
    colors = {"Low risk": "#f59e0b", "High reward": "#dc2626", "Best reward / risk": "#059669"}
    for name, row in scenarios.items():
        ax.scatter([row[RISK]], [row[REWARD]], s=30, marker=SERVICE_MARKERS[_service_type(row)],
                   color=colors.get(name, "#7c3aed"),
                   edgecolor="white", linewidth=0.7, zorder=4, label=name)
    sleeve_labels = {
        "global_cloud": ("Cloud", "cloud"),
        "onprem_oss": ("On-prem", "on-prem"),
        "cloud_oss": ("OSS cloud", "oss-cloud"),
        "eu_cloud": ("European", "European"),
    }
    if workbook_assets is not None:
        for _, row in workbook_assets.iterrows():
            label, service_type = sleeve_labels[row["service_sleeve"]]
            ax.scatter(
                [row[RISK]], [row[REWARD]], s=95,
                marker=SERVICE_MARKERS[service_type],
                color="#f59e0b", edgecolor="#111827", linewidth=1.5,
                zorder=6, label=f"Workbook {label}",
            )
    ax.set_xscale("linear")
    ax.set_xlim(0.0, x_max)
    values = points[REWARD].to_numpy(float)
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    margin = max(1e-6, (high - low) * 0.06)
    ax.set_ylim(low - margin, high + margin)
    ax.set_xlabel("Annualized return-rate risk (SD)", fontsize=7)
    ax.set_ylabel("Mean annualized return rate", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.grid(alpha=0.18, linewidth=0.5)
    handles, labels = ax.get_legend_handles_labels()
    handles.extend(
        Line2D([], [], marker=marker, linestyle="", color="#2563eb", markeredgecolor="white",
               markeredgewidth=0.5, label=service_type)
        for service_type, marker in SERVICE_MARKERS.items()
    )
    ax.legend(handles, labels + list(SERVICE_MARKERS), fontsize=5.6, loc="best", framealpha=0.9)


def run_service_sleeve_sweep(
    users: int, profile: str, individual_paths: int, max_hardware_budget: float,
    aggregate_over_time: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if profile not in COMPANY_PROFILES:
        raise SystemExit(f"Unknown company profile for the service-sleeve sweep: {profile}")
    runs = max(20, individual_paths * 4)
    scenario = SweepScenario(users, 25.0, max_hardware_budget, 0.25, False)
    config = SimulationConfig(
        years=3.0,
        resolution_months=1,
        runs=runs,
        seed=20260711,
        plateau_quarter=24,
        employee_mix=COMPANY_PROFILES[profile],
        backend=_resolve_backend("auto"),
    )
    results, _ = simulate_portfolio_scenario(scenario, config, 0)
    summary = return_change_summary(results)
    _, manifest = service_sleeve_statistics(results, summary, scenario, profile, 0.0)
    if manifest.empty:
        raise SystemExit("The minimal service-sleeve sweep produced no usable service assets.")
    selected = manifest[["service_sleeve", *ASSET_IDENTITY_COLUMNS]]
    paired = results.merge(selected, on=ASSET_IDENTITY_COLUMNS, how="inner")
    service_paths = paired.pivot_table(
        index=["run", "period", "month"],
        columns="service_sleeve",
        values="annualized_return_rate",
    ).reset_index()
    if aggregate_over_time:
        per_run = (
            results.groupby([*ASSET_IDENTITY_COLUMNS, "run"], as_index=False)["annualized_return_rate"]
            .mean()
        )
    else:
        final_period = results["period"].max()
        per_run = results[results["period"] == final_period]
    points = per_run.groupby(list(ASSET_IDENTITY_COLUMNS), as_index=False).agg(
        **{
            REWARD: ("annualized_return_rate", "mean"),
            RISK: ("annualized_return_rate", "std"),
        }
    ).fillna({RISK: 0.0})
    selected_points = points.merge(selected, on=ASSET_IDENTITY_COLUMNS, how="inner")
    return service_paths, points, selected_points


def render(
    output: Path, users: int, profile: str, x_max: float, individual_paths: int,
    max_hardware_budget: float, aggregate_over_time: bool,
) -> None:
    service_paths, points, selected_points = run_service_sleeve_sweep(
        users, profile, individual_paths, max_hardware_budget, aggregate_over_time
    )
    scenarios = select_scenarios(points)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2),
                                gridspec_kw={"width_ratios": (1.08, 1.25), "wspace": 0.20})
    _illustrative(axes[0], service_paths, individual_paths)
    _frontier(axes[1], points, scenarios, x_max, selected_points)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("simulation_dir", type=Path,
                        help="Output directory for the rendered figure.")
    parser.add_argument("--users", type=int, default=25)
    parser.add_argument("--company-profile", default="software_engineering_heavy")
    parser.add_argument("--individual-paths", type=int, default=5,
                        help="Number of individual Monte Carlo paths shown per left-panel category (default: 5).")
    parser.add_argument("--max-hardware-budget", type=float, default=1_000_000.0,
                        help="Maximum upfront hardware budget for the left-panel minimal sweep (default: 1000000).")
    parser.add_argument("--aggregate-over-time", action="store_true",
                        help="Use each run's mean annualized return over the full simulation horizon.")
    parser.add_argument("--x-max", type=float, default=5.0,
                        help="Maximum linear risk displayed on x (default: 5).")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.users <= 0:
        raise SystemExit("--users must be positive.")
    if args.x_max <= 0:
        raise SystemExit("--x-max must be positive.")
    if args.individual_paths < 0:
        raise SystemExit("--individual-paths must be non-negative.")
    if args.max_hardware_budget < 0:
        raise SystemExit("--max-hardware-budget must be non-negative.")
    if args.output is None:
        args.output = args.simulation_dir / f"simulation_and_frontier_{args.users}_{args.company_profile}.png"
    return args


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    render(
        args.output, args.users, args.company_profile, args.x_max,
        args.individual_paths, args.max_hardware_budget, args.aggregate_over_time,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
