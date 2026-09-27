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
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from simulate_llm_efficiency import SimulationConfig, _resolve_backend
from sweep_portfolio_rate_changes import (
    ASSET_IDENTITY_COLUMNS, COMPANY_PROFILES, SweepScenario,
    efficient_frontier, period_frontier_selections, return_change_summary,
    service_sleeve_statistics, simulate_portfolio_scenario)

RISK = "annualized_return_rate_risk_stddev"
REWARD = "median_annualized_return_rate"

SERVICE_MARKERS = {
    "global_service": "o",
    "local_hardware": "^",
    "eu_cloud_oss": "D",
    "european_service": "s",
}

SERVICE_LABELS = {
    "global_service": "Cloud",
    "local_hardware": "On-prem",
    "eu_cloud_oss": "OSS cloud",
    "european_service": "European",
}


def _illustrative(
    ax, service_paths: pd.DataFrame, individual_paths: int
) -> None:
    service_columns = {
        "global_cloud": ("Cloud", "#2563eb"),
        "onprem_oss": ("On-prem", "#dc2626"),
        "cloud_oss": ("OSS cloud", "#059669"),
        "eu_cloud": ("European", "#7c3aed"),
    }
    for sleeve, (name, color) in service_columns.items():
        values = service_paths[["run", "period", "month", sleeve]].dropna()
        bands = (
            values.groupby(["period", "month"], sort=True)[sleeve]
            .agg(
                p10=lambda series: series.quantile(0.10),
                p50="median",
                p90=lambda series: series.quantile(0.90),
            )
            .reset_index()
        )
        paths = list(values.groupby("run", sort=False))
        if individual_paths:
            path_indices = [
                round(index * (len(paths) - 1) / max(1, individual_paths - 1))
                for index in range(individual_paths)
            ]
            paths = [paths[index] for index in path_indices]
        for _, path in paths:
            path = path.sort_values("month")
            ax.plot(
                path["month"],
                path[sleeve],
                color=color,
                alpha=0.16,
                linewidth=0.55,
                zorder=1,
            )
        risk = float(
            np.nanstd(
                values.loc[values["month"] == values["month"].max(), sleeve],
                ddof=1,
            )
        )
        ax.fill_between(
            bands["month"], bands["p10"], bands["p90"], color=color, alpha=0.16
        )
        ax.plot(
            bands["month"],
            bands["p50"],
            color=color,
            linewidth=1.4,
            marker="o",
            markersize=2.5,
            label=f"{name} (risk {risk:.3f})",
            zorder=3,
        )

    ax.set_xlabel("Month", fontsize=7)
    ax.set_ylabel("Annualized return rate", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.legend(loc="best", fontsize=5.8)


def _shock_colors(results: pd.DataFrame) -> pd.DataFrame:
    """Return RGB colours from shocks actually triggered in each asset's runs."""
    final_flags = (
        results.groupby([*ASSET_IDENTITY_COLUMNS, "run"], sort=False)[
            [
                "embargo_shock",
                "sudden_break_even_shock",
                "gradual_break_even_shock",
                "capability_plateau_shock",
            ]
        ]
        .max()
        .reset_index()
    )
    final_flags["price_shock"] = final_flags[
        ["sudden_break_even_shock", "gradual_break_even_shock"]
    ].max(axis=1)
    return (
        final_flags.groupby(ASSET_IDENTITY_COLUMNS, sort=False)[
            ["price_shock", "embargo_shock", "capability_plateau_shock"]
        ]
        .mean()
        .rename(
            columns={
                "price_shock": "shock_red",
                "embargo_shock": "shock_blue",
                "capability_plateau_shock": "shock_green",
            }
        )
        .reset_index()
    )


def _portfolio_mixes(
    results: pd.DataFrame, sleeves: pd.DataFrame, shock_colours: pd.DataFrame
) -> pd.DataFrame:
    """Sample fully allocated mixes of the four paired service sleeves at the final horizon."""
    final_period = results["period"].max()
    selected = (
        sleeves[["service_sleeve", *ASSET_IDENTITY_COLUMNS]]
        .merge(shock_colours, on=ASSET_IDENTITY_COLUMNS, how="left")
        .fillna(0.0)
    )
    paired = results[results["period"] == final_period].merge(
        selected, on=ASSET_IDENTITY_COLUMNS, how="inner"
    )
    paths = paired.pivot(
        index="run", columns="service_sleeve", values="annualized_return_rate"
    ).dropna()
    weights = np.random.default_rng(20260711).dirichlet(
        np.ones(paths.shape[1]), size=200
    )
    returns = paths.to_numpy() @ weights.T
    sleeve_colours = (
        selected.set_index("service_sleeve")
        .reindex(paths.columns)[["shock_red", "shock_green", "shock_blue"]]
        .to_numpy()
    )
    mix_colours = weights @ sleeve_colours
    return pd.DataFrame(
        {
            "sample": np.arange(len(weights)),
            "period": final_period,
            RISK: returns.std(axis=0, ddof=1),
            REWARD: np.median(returns, axis=0),
            "shock_red": mix_colours[:, 0],
            "shock_green": mix_colours[:, 1],
            "shock_blue": mix_colours[:, 2],
        }
    )


def _display_colours(
    values: pd.DataFrame | np.ndarray | tuple[float, float, float],
    floor: float,
) -> np.ndarray:
    return floor + (1.0 - floor) * np.asarray(values, dtype=float)


def _top_shock_colours(
    *frames: pd.DataFrame, floor: float
) -> tuple[list[Line2D], list[str]]:
    """Return the nine most common displayed shock colours, excluding pure primaries."""
    colours = pd.concat(
        [
            frame[["shock_red", "shock_green", "shock_blue"]]
            for frame in frames
        ],
        ignore_index=True,
    )
    rounded = np.rint(colours.to_numpy() * 4.0) / 4.0
    counts = pd.DataFrame(
        rounded, columns=["shock_red", "shock_green", "shock_blue"]
    ).value_counts()
    handles = []
    labels = []
    names = ("Price", "Plateau", "Embargo")
    primary_colours = {(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)}
    for colour, _ in counts.items():
        if colour in primary_colours:
            continue
        active = [
            f"{name} {value:.0%}"
            for name, value in zip(names, colour)
            if value >= 0.125
        ]
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color=_display_colours(colour, floor),
            )
        )
        labels.append(" + ".join(active) if active else "No shocks")
        if len(handles) == 9:
            break
    return handles, labels


def _frontier_marker(
    ax, row: pd.Series, marker: str, size: float, zorder: int, floor: float
) -> None:
    colour = _display_colours(
        [row["shock_red"], row["shock_green"], row["shock_blue"]], floor
    )
    ax.scatter(
        [row[RISK]],
        [row[REWARD]],
        s=size,
        marker=marker,
        color=[colour],
        linewidths=0,
        zorder=zorder,
    )


def _frontier(
    ax,
    final_points: pd.DataFrame,
    results: pd.DataFrame,
    sleeves: pd.DataFrame,
    x_max: float,
    shock_color_floor: float,
) -> None:
    shock_colours = _shock_colors(results)
    final_points = final_points.merge(
        shock_colours, on=ASSET_IDENTITY_COLUMNS, how="left"
    ).fillna(0.0)
    for provider, values in final_points.groupby(
        "service_provider", sort=False
    ):
        ax.scatter(
            values[RISK],
            values[REWARD],
            s=22,
            marker=SERVICE_MARKERS[provider],
            color=_display_colours(
                values[["shock_red", "shock_green", "shock_blue"]],
                shock_color_floor,
            ),
            linewidths=0,
            zorder=2,
        )
    service_pareto = efficient_frontier(final_points, RISK, REWARD)
    ax.plot(
        service_pareto[RISK],
        service_pareto[REWARD],
        color="#a855f7",
        linewidth=1.0,
        linestyle="--",
        label="Single-service Pareto frontier",
        zorder=10,
    )
    for _, row in service_pareto.iterrows():
        _frontier_marker(
            ax,
            row,
            SERVICE_MARKERS[row["service_provider"]],
            36,
            4,
            shock_color_floor,
        )
    mixes = _portfolio_mixes(results, sleeves, shock_colours)
    ax.scatter(
        mixes[RISK],
        mixes[REWARD],
        s=20,
        marker="X",
        color=_display_colours(
            mixes[["shock_red", "shock_green", "shock_blue"]],
            shock_color_floor,
        ),
        linewidths=0,
        zorder=3,
    )
    pareto = efficient_frontier(mixes, RISK, REWARD)
    ax.plot(
        pareto[RISK],
        pareto[REWARD],
        color="#c026d3",
        linewidth=1.1,
        label="Portfolio Pareto frontier",
        zorder=11,
    )
    for _, row in pareto.iterrows():
        _frontier_marker(ax, row, "X", 36, 5, shock_color_floor)
    mix_selections = period_frontier_selections(mixes)
    mix_selections = mix_selections[
        mix_selections["selection_period"]
        == mix_selections["selection_period"].max()
    ]
    for _, row in mix_selections.iterrows():
        _frontier_marker(ax, row, "X", 70, 7, shock_color_floor)
    service_selections = period_frontier_selections(final_points)
    service_selections = service_selections[
        service_selections["selection_period"]
        == service_selections["selection_period"].max()
    ]
    for _, row in service_selections.iterrows():
        _frontier_marker(
            ax,
            row,
            SERVICE_MARKERS[row["service_provider"]],
            70,
            9,
            shock_color_floor,
        )
    ax.set_xscale("linear")
    selected_x_max = max(
        float(mix_selections[RISK].max()) if not mix_selections.empty else 0.0,
        (
            float(service_selections[RISK].max())
            if not service_selections.empty
            else 0.0
        ),
    )
    display_x_max = max(x_max, selected_x_max * 1.05)
    ax.set_xlim(0.0, display_x_max)
    visible_final = final_points[
        final_points[RISK].between(0.0, display_x_max)
    ]
    values = visible_final[REWARD].to_numpy(float)
    low, high = np.nanpercentile(values, [1.0, 99.0])
    selected_rewards = pd.concat(
        [mix_selections[REWARD], service_selections[REWARD]], ignore_index=True
    )
    if not selected_rewards.empty:
        low = min(low, float(selected_rewards.min()))
        high = max(high, float(selected_rewards.max()))
    margin = max(1e-6, (high - low) * 0.10)
    ax.set_ylim(low - margin, high + margin)
    ax.set_xlabel("Annualized return-rate risk (SD)", fontsize=7)
    ax.set_ylabel("Median annualized return rate", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.grid(alpha=0.18, linewidth=0.5)
    shock_handles, shock_labels = _top_shock_colours(
        final_points, mixes, floor=shock_color_floor
    )
    shock_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=_display_colours(colour, shock_color_floor),
        )
        for colour in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    ] + shock_handles
    shock_labels = [
        "R: Price only",
        "G: Plateau only",
        "B: Embargo only",
    ] + shock_labels
    global_legend = ax.legend(
        [
            Line2D([], [], color="#c026d3", linewidth=1.1),
            Line2D([], [], color="#a855f7", linewidth=1.0, linestyle="--"),
            *[
                Line2D(
                    [],
                    [],
                    marker=SERVICE_MARKERS[provider],
                    linestyle="",
                    color="#111827",
                )
                for provider in SERVICE_MARKERS
            ],
            Line2D([], [], marker="X", linestyle="", color="#111827"),
        ],
        [
            "Mixed portfolio frontier",
            "Single-service frontier",
            *SERVICE_LABELS.values(),
            "Mix",
        ],
        title="Frontiers and service type",
        fontsize=4.8,
        title_fontsize=5.5,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=2,
        framealpha=0.9,
    )
    ax.add_artist(global_legend)
    ax.legend(
        shock_handles,
        shock_labels,
        title="Shocks — intensity = share of simulations",
        fontsize=4.8,
        title_fontsize=5.5,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.80),
        ncol=2,
        framealpha=0.9,
    )


def run_service_sleeve_sweep(
    users: int,
    profile: str,
    individual_paths: int,
    max_hardware_budget: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if profile not in COMPANY_PROFILES:
        raise SystemExit(
            f"Unknown company profile for the service-sleeve sweep: {profile}"
        )
    runs = max(20, individual_paths * 4)
    scenario = SweepScenario(users, 25.0, max_hardware_budget, 0.25, False)
    config = SimulationConfig(
        years=3.0,
        resolution_months=1,
        runs=runs,
        seed=20260711,
        plateau_quarter=24,
        stochastic_shocks=True,
        employee_mix=COMPANY_PROFILES[profile],
        backend=_resolve_backend("auto"),
    )
    results, _ = simulate_portfolio_scenario(scenario, config, 0)
    summary = return_change_summary(results)
    _, manifest = service_sleeve_statistics(
        results, summary, scenario, profile, 0.0
    )
    if manifest.empty:
        raise SystemExit(
            "The minimal service-sleeve sweep produced no usable service assets."
        )
    selected = manifest[["service_sleeve", *ASSET_IDENTITY_COLUMNS]]
    paired = results.merge(selected, on=ASSET_IDENTITY_COLUMNS, how="inner")
    service_paths = paired.pivot_table(
        index=["run", "period", "month"],
        columns="service_sleeve",
        values="annualized_return_rate",
    ).reset_index()
    return service_paths, summary, results, manifest


def render(
    output: Path,
    users: int,
    profile: str,
    x_max: float,
    individual_paths: int,
    max_hardware_budget: float,
    shock_color_floor: float,
) -> None:
    service_paths, summary, results, sleeves = run_service_sleeve_sweep(
        users, profile, individual_paths, max_hardware_budget
    )
    final_period = summary["period"].max()
    final_points = summary[summary["period"] == final_period]
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.5, 4.2),
        gridspec_kw={"width_ratios": (1.08, 1.25), "wspace": 0.20},
    )
    _illustrative(axes[0], service_paths, individual_paths)
    _frontier(
        axes[1], final_points, results, sleeves, x_max, shock_color_floor
    )
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "simulation_dir",
        type=Path,
        help="Output directory for the rendered figure.",
    )
    parser.add_argument("--users", type=int, default=25)
    parser.add_argument(
        "--company-profile", default="software_engineering_heavy"
    )
    parser.add_argument(
        "--individual-paths",
        type=int,
        default=5,
        help="Number of individual Monte Carlo paths shown per left-panel category (default: 5).",
    )
    parser.add_argument(
        "--max-hardware-budget",
        type=float,
        default=1_000_000.0,
        help="Maximum upfront hardware budget for the left-panel minimal sweep (default: 1000000).",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=5.0,
        help="Maximum linear risk displayed on x (default: 5).",
    )
    parser.add_argument(
        "--shock-color-floor",
        type=float,
        default=0.30,
        help="RGB baseline for no-shock points; 0 is black and 1 is white (default: 0.30).",
    )
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
    if not 0.0 <= args.shock_color_floor <= 1.0:
        raise SystemExit("--shock-color-floor must be between 0 and 1.")
    if args.output is None:
        args.output = (
            args.simulation_dir
            / f"simulation_and_frontier_{args.users}_{args.company_profile}.png"
        )
    return args


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    render(
        args.output,
        args.users,
        args.company_profile,
        args.x_max,
        args.individual_paths,
        args.max_hardware_budget,
        args.shock_color_floor,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
