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
    efficient_frontier,
    period_frontier_selections,
    return_change_summary,
    service_sleeve_statistics,
    simulate_portfolio_scenario,
)


RISK = "annualized_return_rate_risk_stddev"
REWARD = "median_annualized_return_rate"


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


def _frontier(ax, final_points: pd.DataFrame, intermediate_points: pd.DataFrame,
              selections: pd.DataFrame, x_max: float) -> None:
    if not intermediate_points.empty:
        ax.scatter(
            intermediate_points[RISK], intermediate_points[REWARD], s=9, alpha=0.07,
            color="#9ca3af", linewidths=0, label="Intermediate-period candidates", zorder=1,
        )
    ax.scatter(
        final_points[RISK], final_points[REWARD], s=22, alpha=0.48, color="#4b5563",
        linewidths=0, label="Final-period candidates", zorder=2,
    )
    global_frontier = efficient_frontier(
        pd.concat([intermediate_points, final_points], ignore_index=True), RISK, REWARD
    )
    ax.plot(
        global_frontier[RISK], global_frontier[REWARD], color="#7c3aed", linewidth=1.0,
        linestyle="--", label="All-horizon Pareto frontier", zorder=3,
    )
    last_month: float | None = None
    for index, (_, row) in enumerate(global_frontier.sort_values(RISK).iterrows()):
        month = float(row["month"])
        if month == last_month:
            continue
        ax.annotate(
            f"{month:g}m", (row[RISK], row[REWARD]), xytext=(2, 4 if index % 2 else -8),
            textcoords="offset points", color="#6d28d9", fontsize=4.8, zorder=4,
        )
        last_month = month
    frontier = efficient_frontier(final_points, RISK, REWARD)
    ax.plot(frontier[RISK], frontier[REWARD], color="#2563eb", linewidth=1.3, label="Final-period Pareto frontier", zorder=4)
    frontier = frontier.copy()
    frontier["service_type"] = frontier.apply(_service_type, axis=1)
    for service_type, service_points in frontier.groupby("service_type", sort=False):
        ax.scatter(
            service_points[RISK], service_points[REWARD], s=30,
            marker=SERVICE_MARKERS[service_type], color="#2563eb",
            edgecolor="white", linewidth=0.5, zorder=5,
        )
    colors = {"low_risk": "#f59e0b", "medium_risk": "#059669", "high_risk": "#dc2626"}
    labels = {"low_risk": "Low risk", "medium_risk": "Medium risk", "high_risk": "High risk"}
    for _, row in selections.iterrows():
        name = row["portfolio"]
        ax.scatter([row[RISK]], [row[REWARD]], s=64, marker=SERVICE_MARKERS[_service_type(row)],
                   color=colors.get(name, "#7c3aed"),
                   edgecolor="#111827", linewidth=0.9, zorder=6)
    ax.set_xscale("linear")
    selected_x_max = float(selections[RISK].max()) if not selections.empty else 0.0
    display_x_max = max(x_max, selected_x_max * 1.05)
    ax.set_xlim(0.0, display_x_max)
    visible_final = final_points[final_points[RISK].between(0.0, display_x_max)]
    values = visible_final[REWARD].to_numpy(float)
    low, high = np.nanpercentile(values, [1.0, 99.0])
    if not selections.empty:
        low = min(low, float(selections[REWARD].min()))
        high = max(high, float(selections[REWARD].max()))
    margin = max(1e-6, (high - low) * 0.10)
    ax.set_ylim(low - margin, high + margin)
    ax.set_xlabel("Annualized return-rate risk (SD)", fontsize=7)
    ax.set_ylabel("Median annualized return rate", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.grid(alpha=0.18, linewidth=0.5)
    context_legend = ax.legend(
        *ax.get_legend_handles_labels(), fontsize=5.6, loc="upper left", framealpha=0.9
    )
    ax.add_artist(context_legend)
    service_legend = ax.legend(
        [
            Line2D([], [], marker=marker, linestyle="", color="#2563eb", markeredgecolor="white",
                   markeredgewidth=0.5)
            for marker in SERVICE_MARKERS.values()
        ],
        list(SERVICE_MARKERS), title="Final service type", fontsize=5.2, title_fontsize=5.5,
        loc="center left", framealpha=0.9,
    )
    ax.add_artist(service_legend)
    ax.legend(
        [Line2D([], [], marker="o", linestyle="", color=color, markeredgecolor="#111827") for color in colors.values()],
        list(labels.values()), title="Risk-band selection", fontsize=5.2, title_fontsize=5.5,
        loc="lower right", framealpha=0.9,
    )


def run_service_sleeve_sweep(
    users: int, profile: str, individual_paths: int, max_hardware_budget: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return service_paths, summary


def render(
    output: Path, users: int, profile: str, x_max: float, individual_paths: int,
    max_hardware_budget: float,
) -> None:
    service_paths, summary = run_service_sleeve_sweep(
        users, profile, individual_paths, max_hardware_budget
    )
    final_period = summary["period"].max()
    final_points = summary[summary["period"] == final_period]
    intermediate_points = summary[summary["period"] < final_period]
    selections = period_frontier_selections(summary)
    selections = selections[selections["selection_period"] == final_period]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2),
                                gridspec_kw={"width_ratios": (1.08, 1.25), "wspace": 0.20})
    _illustrative(axes[0], service_paths, individual_paths)
    _frontier(axes[1], final_points, intermediate_points, selections, x_max)
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
        args.individual_paths, args.max_hardware_budget,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
