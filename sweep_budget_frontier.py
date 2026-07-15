#!/usr/bin/env python3
"""Sweep budget pairs, fit final-period efficient frontiers, and plot selected portfolios."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.tri as mtri
from tqdm.auto import tqdm

from simulate_llm_efficiency import (
    HARDWARE_CALIBRATIONS,
    HARDWARE_SCENARIOS,
    SimulationConfig,
    _hardware_calibration_profile,
    _resolve_backend,
    run_simulation,
    summarize,
)


def parse_employee_mix(mix_arg: str | None) -> dict[str, float] | None:
    if mix_arg is None:
        return None

    mix: dict[str, float] = {}
    for item in mix_arg.split(","):
        key, value = item.split("=", 1)
        mix[key.strip()] = float(value)
    return mix


def budget_pairs(args: argparse.Namespace) -> list[tuple[float, float]]:
    service_values = np.linspace(args.service_budget_min_usd, args.service_budget_max_usd, args.service_budget_points)
    hardware_values = np.linspace(args.hardware_budget_min_usd, args.hardware_budget_max_usd, args.hardware_budget_points)
    pairs = []
    for service_budget in service_values:
        for hardware_budget in hardware_values:
            lifetime_budget = service_budget * 12.0 * args.years + hardware_budget
            if lifetime_budget <= args.total_budget_usd + 1e-9:
                pairs.append((float(service_budget), float(hardware_budget)))
    return pairs


def efficient_frontier(final_summary: pd.DataFrame) -> pd.DataFrame:
    points = final_summary.sort_values(["risk_mean", "return_rate_mean"], ascending=[True, False]).copy()
    frontier_rows = []
    best_return = -np.inf
    for _, row in points.iterrows():
        if row["return_rate_mean"] > best_return + 1e-12:
            frontier_rows.append(row.to_dict())
            best_return = row["return_rate_mean"]
    return pd.DataFrame(frontier_rows)


def frontier_line(frontier: pd.DataFrame) -> pd.DataFrame:
    frontier = frontier.sort_values("risk_mean").drop_duplicates(subset=["risk_mean"], keep="last")
    if len(frontier) <= 1:
        return frontier[["risk_mean", "return_rate_mean"]].copy()

    dense_risk = np.linspace(float(frontier["risk_mean"].min()), float(frontier["risk_mean"].max()), 80)
    dense_return = np.interp(dense_risk, frontier["risk_mean"], frontier["return_rate_mean"])
    return pd.DataFrame({"risk_mean": dense_risk, "return_rate_mean": dense_return})


def _finite_panel(panel: pd.DataFrame, metric: str) -> pd.DataFrame:
    mask = np.isfinite(panel["service_budget_usd"]) & np.isfinite(panel["hardware_budget_usd"]) & np.isfinite(panel[metric])
    return panel.loc[mask].copy()


def _load_selection_summary(output_dir: Path) -> pd.DataFrame:
    selections_path = output_dir / "budget_frontier_selections.csv"
    if not selections_path.exists():
        raise SystemExit(f"Missing selection summary for replot: {selections_path}")
    return _prepare_selection_summary(pd.read_csv(selections_path))


def _shared_budget_limits(selection_summary: pd.DataFrame) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    x = pd.to_numeric(selection_summary["service_budget_usd"], errors="coerce")
    y = pd.to_numeric(selection_summary["hardware_budget_usd"], errors="coerce")
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.empty or y.empty:
        return None, None

    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    if xmin == xmax:
        xmin -= 0.5
        xmax += 0.5
    if ymin == ymax:
        ymin -= 0.5
        ymax += 0.5
    return (xmin, xmax), (ymin, ymax)


def _infer_budget_constraint(selection_summary: pd.DataFrame) -> tuple[float | None, float | None]:
    if "total_lifetime_budget_usd" not in selection_summary:
        return None, None

    finite = selection_summary[
        np.isfinite(selection_summary["service_budget_usd"])
        & np.isfinite(selection_summary["hardware_budget_usd"])
        & np.isfinite(selection_summary["total_lifetime_budget_usd"])
    ][["service_budget_usd", "hardware_budget_usd", "total_lifetime_budget_usd"]].drop_duplicates()
    if finite.empty:
        return None, None

    total_budget_usd = float(finite["total_lifetime_budget_usd"].max())
    years: float | None = None
    for _, group in finite.groupby("hardware_budget_usd"):
        group = group.sort_values("service_budget_usd")
        if len(group) < 2:
            continue
        first = group.iloc[0]
        last = group.iloc[-1]
        delta_service = float(last["service_budget_usd"] - first["service_budget_usd"])
        if abs(delta_service) <= 1e-12:
            continue
        delta_total = float(last["total_lifetime_budget_usd"] - first["total_lifetime_budget_usd"])
        years = delta_total / (12.0 * delta_service)
        break
    return total_budget_usd, years


def _prepare_selection_summary(selection_summary: pd.DataFrame) -> pd.DataFrame:
    selection_summary = selection_summary.copy()
    if "delivered_usage_share_mean" not in selection_summary:
        selection_summary["delivered_usage_share_mean"] = np.nan
    if "unserved_work_share_mean" not in selection_summary:
        selection_summary["unserved_work_share_mean"] = 1.0 - selection_summary["delivered_usage_share_mean"]
    if "return_rate_adjusted_mean" not in selection_summary:
        selection_summary["return_rate_adjusted_mean"] = (
            selection_summary["return_rate_mean"] * selection_summary["delivered_usage_share_mean"]
        )
    return selection_summary


def _plot_panel_specs() -> list[tuple[str, str]]:
    return [
        ("return_rate_adjusted_mean", "Return Rate Adjusted For Unserved Work"),
        ("risk_mean", "Risk"),
        ("mean_efficiency_gain_mean", "Mean Efficiency Gain"),
        ("delivered_usage_share_mean", "Delivered Usage Share"),
        ("unserved_work_share_mean", "Unserved Work Share"),
    ]


def _overlay_budget_exceeded_region(
    ax: plt.Axes,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    total_budget_usd: float | None,
    years: float | None,
) -> None:
    if xlim is None or ylim is None or total_budget_usd is None or years is None or years <= 0.0:
        return

    x = np.linspace(xlim[0], xlim[1], 512)
    boundary = total_budget_usd - (12.0 * years * x)
    visible = boundary < ylim[1]
    if not np.any(visible):
        return

    lower = np.clip(boundary, ylim[0], ylim[1])
    ax.fill_between(
        x,
        lower,
        ylim[1],
        where=visible,
        facecolor="#d9d9d9",
        edgecolor="#8c8c8c",
        hatch="///",
        alpha=0.35,
        linewidth=0.0,
        zorder=4,
    )
    ax.plot(x, boundary, color="#666666", linestyle="--", linewidth=1.0, zorder=5)

    visible_x = x[visible]
    visible_y = np.clip(boundary[visible], ylim[0], ylim[1])
    if len(visible_x) == 0:
        return
    ax.text(
        float(np.median(visible_x)),
        float(np.median((visible_y + ylim[1]) * 0.5)),
        "total budget exceeded",
        ha="center",
        va="center",
        fontsize=9,
        color="#444444",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2.0},
        zorder=6,
    )


def _prepare_frontier_timeseries(frontier_timeseries: pd.DataFrame) -> pd.DataFrame:
    frontier_timeseries = frontier_timeseries.copy()
    if "delivered_usage_share_mean" not in frontier_timeseries:
        frontier_timeseries["delivered_usage_share_mean"] = np.nan
    if "unserved_work_share_mean" not in frontier_timeseries:
        frontier_timeseries["unserved_work_share_mean"] = 1.0 - frontier_timeseries["delivered_usage_share_mean"]
    if "return_rate_adjusted_mean" not in frontier_timeseries:
        frontier_timeseries["return_rate_adjusted_mean"] = (
            frontier_timeseries["return_rate_mean"] * frontier_timeseries["delivered_usage_share_mean"]
        )
    return frontier_timeseries


def _load_frontier_timeseries(output_dir: Path) -> pd.DataFrame:
    timeseries_path = output_dir / "budget_frontier_timeseries.csv"
    if not timeseries_path.exists():
        raise FileNotFoundError(timeseries_path)
    return _prepare_frontier_timeseries(pd.read_csv(timeseries_path))


def evaluate_budget_pair(
    service_budget_usd: float,
    hardware_budget_usd: float,
    base_config: SimulationConfig,
    pair_index: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = SimulationConfig(
        users=base_config.users,
        years=base_config.years,
        resolution_months=base_config.resolution_months,
        runs=base_config.runs,
        concurrency=base_config.concurrency,
        seed=base_config.seed + pair_index * 10007,
        plateau_quarter=base_config.plateau_quarter,
        base_cost_index_per_active_user_quarter=base_config.base_cost_index_per_active_user_quarter,
        max_monthly_service_budget_usd=service_budget_usd,
        max_upfront_hardware_budget_usd=hardware_budget_usd,
        confidential_document_fraction=base_config.confidential_document_fraction,
        usd_per_service_cost_index_quarter=base_config.usd_per_service_cost_index_quarter,
        usd_per_hardware_capex_index=base_config.usd_per_hardware_capex_index,
        hardware_calibration=base_config.hardware_calibration,
        onprem_capability_multiplier=base_config.onprem_capability_multiplier,
        employee_mix=base_config.employee_mix,
        engineering_context=base_config.engineering_context,
        backend=base_config.backend,
        show_progress=False,
        output_dir=base_config.output_dir,
    )

    results = run_simulation(config)
    summary = summarize(results)
    final_summary = summary[summary["period"] == summary["period"].max()].copy()
    frontier = efficient_frontier(final_summary)
    frontier_fit = frontier_line(frontier)
    frontier_scenarios = frontier["scenario"].unique().tolist()
    frontier_timeseries = summary[summary["scenario"].isin(frontier_scenarios)].copy()

    low_risk = frontier.sort_values(["risk_mean", "return_rate_mean"], ascending=[True, False]).iloc[0]
    high_return = frontier.sort_values(["return_rate_mean", "risk_mean"], ascending=[False, True]).iloc[0]

    selection_rows = pd.DataFrame(
        [
            {
                "selection": "low_risk",
                "service_budget_usd": service_budget_usd,
                "hardware_budget_usd": hardware_budget_usd,
                "total_lifetime_budget_usd": service_budget_usd * 12.0 * config.years + hardware_budget_usd,
                "scenario": low_risk["scenario"],
                "local_fallback": low_risk["local_fallback"],
                "risk_mean": float(low_risk["risk_mean"]),
                "return_rate_mean": float(low_risk["return_rate_mean"]),
                "return_rate_adjusted_mean": float(low_risk["return_rate_mean"] * low_risk["delivered_usage_share_mean"]),
                "mean_efficiency_gain_mean": float(low_risk["mean_efficiency_gain_mean"]),
                "delivered_usage_share_mean": float(low_risk["delivered_usage_share_mean"]),
                "unserved_work_share_mean": float(1.0 - low_risk["delivered_usage_share_mean"]),
                "confidential_usage_share_mean": float(low_risk["confidential_usage_share_mean"]),
                "hardware_budget_feasible_mean": float(low_risk["hardware_budget_feasible_mean"]),
                "hardware_budget_scale_mean": float(low_risk["hardware_budget_scale_mean"]),
                "service_budget_scale_mean": float(low_risk["service_budget_scale_mean"]),
            },
            {
                "selection": "high_return",
                "service_budget_usd": service_budget_usd,
                "hardware_budget_usd": hardware_budget_usd,
                "total_lifetime_budget_usd": service_budget_usd * 12.0 * config.years + hardware_budget_usd,
                "scenario": high_return["scenario"],
                "local_fallback": high_return["local_fallback"],
                "risk_mean": float(high_return["risk_mean"]),
                "return_rate_mean": float(high_return["return_rate_mean"]),
                "return_rate_adjusted_mean": float(high_return["return_rate_mean"] * high_return["delivered_usage_share_mean"]),
                "mean_efficiency_gain_mean": float(high_return["mean_efficiency_gain_mean"]),
                "delivered_usage_share_mean": float(high_return["delivered_usage_share_mean"]),
                "unserved_work_share_mean": float(1.0 - high_return["delivered_usage_share_mean"]),
                "confidential_usage_share_mean": float(high_return["confidential_usage_share_mean"]),
                "hardware_budget_feasible_mean": float(high_return["hardware_budget_feasible_mean"]),
                "hardware_budget_scale_mean": float(high_return["hardware_budget_scale_mean"]),
                "service_budget_scale_mean": float(high_return["service_budget_scale_mean"]),
            },
        ]
    )

    budget_meta = {
        "service_budget_usd": service_budget_usd,
        "hardware_budget_usd": hardware_budget_usd,
        "total_lifetime_budget_usd": service_budget_usd * 12.0 * config.years + hardware_budget_usd,
    }
    frontier = frontier.assign(
        line_type="frontier_point",
        **budget_meta,
    )
    frontier_fit = frontier_fit.assign(
        line_type="frontier_fit",
        service_budget_usd=service_budget_usd,
        hardware_budget_usd=hardware_budget_usd,
        total_lifetime_budget_usd=budget_meta["total_lifetime_budget_usd"],
    )
    frontier_output = pd.concat([frontier, frontier_fit], ignore_index=True, sort=False)
    frontier_timeseries = frontier_timeseries.assign(
        service_budget_usd=service_budget_usd,
        hardware_budget_usd=hardware_budget_usd,
        total_lifetime_budget_usd=budget_meta["total_lifetime_budget_usd"],
    )
    return _prepare_selection_summary(selection_rows), frontier_output, _prepare_frontier_timeseries(frontier_timeseries)


def _plot_budget_matrix(
    selection_summary: pd.DataFrame,
    output_dir: Path,
    transpose: bool,
    contours: bool,
    total_budget_usd: float | None = None,
    years: float | None = None,
) -> None:
    selection_summary = _prepare_selection_summary(selection_summary)
    if total_budget_usd is None or years is None:
        inferred_total_budget_usd, inferred_years = _infer_budget_constraint(selection_summary)
        if total_budget_usd is None:
            total_budget_usd = inferred_total_budget_usd
        if years is None:
            years = inferred_years
    selections = [("low_risk", "Low-Risk Frontier"), ("high_return", "High-Return Frontier")]
    metrics = _plot_panel_specs()
    if transpose:
        rows = len(metrics)
        cols = len(selections)
        figsize = (5.2 * cols, 3.6 * rows)
    else:
        rows = len(selections)
        cols = len(metrics)
        figsize = (4.6 * cols, 4.4 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes_array = np.atleast_2d(axes)
    xlim, ylim = _shared_budget_limits(selection_summary)

    for row_idx in range(rows):
        for col_idx in range(cols):
            ax = axes_array[row_idx, col_idx]
            if transpose:
                metric, metric_title = metrics[row_idx]
                selection, selection_title = selections[col_idx]
            else:
                selection, selection_title = selections[row_idx]
                metric, metric_title = metrics[col_idx]

            title_suffix = "Contours" if contours else "Grid"
            panel = _finite_panel(selection_summary[selection_summary["selection"] == selection], metric)
            ax.set_title(f"{selection_title}: {metric_title} {title_suffix}")
            ax.set_xlabel("Monthly service budget (USD)")
            ax.set_ylabel("Upfront hardware budget (USD)")
            if xlim is not None and ylim is not None:
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)

            if contours:
                unique_points = panel[["service_budget_usd", "hardware_budget_usd"]].drop_duplicates() if not panel.empty else panel
                if len(panel) < 3 or len(unique_points) < 3:
                    ax.text(0.5, 0.5, "Need at least 3 finite budget points", ha="center", va="center", transform=ax.transAxes)
                    continue
                triangulation = mtri.Triangulation(panel["service_budget_usd"], panel["hardware_budget_usd"])
                value_min = float(panel[metric].min())
                value_max = float(panel[metric].max())
                if np.isclose(value_min, value_max):
                    fill_levels = np.array([value_min - 1e-12, value_max + 1e-12], dtype=float)
                    line_levels = np.array([value_min], dtype=float)
                else:
                    top = np.nextafter(value_max, np.inf)
                    fill_levels = np.linspace(value_min, top, 12)
                    line_levels = np.linspace(value_min, value_max, 8)
                try:
                    contour = ax.tricontourf(
                        triangulation,
                        panel[metric],
                        levels=fill_levels,
                        cmap="viridis",
                        extend="max",
                    )
                    ax.tricontour(
                        triangulation,
                        panel[metric],
                        levels=line_levels,
                        colors="white",
                        linewidths=0.4,
                        alpha=0.6,
                    )
                except ValueError:
                    ax.text(0.5, 0.5, "Degenerate budget grid", ha="center", va="center", transform=ax.transAxes)
                    continue
                ax.scatter(
                    panel["service_budget_usd"],
                    panel["hardware_budget_usd"],
                    s=16,
                    color="black",
                    alpha=0.6,
                )
                color_artist = contour
            else:
                if panel.empty:
                    ax.text(0.5, 0.5, "No finite values", ha="center", va="center", transform=ax.transAxes)
                    continue
                color_artist = ax.scatter(
                    panel["service_budget_usd"],
                    panel["hardware_budget_usd"],
                    c=panel[metric],
                    cmap="viridis",
                    s=90,
                    edgecolors="black",
                    linewidths=0.3,
                )

            _overlay_budget_exceeded_region(ax, xlim, ylim, total_budget_usd, years)
            colorbar = fig.colorbar(color_artist, ax=ax, shrink=0.84)
            colorbar.set_label(metric)

    fig.tight_layout()
    if contours:
        filename = "budget_frontier_contours_transposed.png" if transpose else "budget_frontier_contours.png"
    else:
        filename = "budget_frontier_grid_transposed.png" if transpose else "budget_frontier_grid.png"
    fig.savefig(output_dir / filename, dpi=180)
    plt.close(fig)


def plot_budget_grid(
    selection_summary: pd.DataFrame,
    output_dir: Path,
    total_budget_usd: float | None = None,
    years: float | None = None,
) -> None:
    _plot_budget_matrix(
        selection_summary,
        output_dir,
        transpose=False,
        contours=False,
        total_budget_usd=total_budget_usd,
        years=years,
    )
    _plot_budget_matrix(
        selection_summary,
        output_dir,
        transpose=True,
        contours=False,
        total_budget_usd=total_budget_usd,
        years=years,
    )


def plot_budget_contours(
    selection_summary: pd.DataFrame,
    output_dir: Path,
    total_budget_usd: float | None = None,
    years: float | None = None,
) -> None:
    _plot_budget_matrix(
        selection_summary,
        output_dir,
        transpose=False,
        contours=True,
        total_budget_usd=total_budget_usd,
        years=years,
    )
    _plot_budget_matrix(
        selection_summary,
        output_dir,
        transpose=True,
        contours=True,
        total_budget_usd=total_budget_usd,
        years=years,
    )


def _plot_frontier_timeseries(frontier_timeseries: pd.DataFrame, output_dir: Path, transpose: bool) -> None:
    frontier_timeseries = _prepare_frontier_timeseries(frontier_timeseries)
    metrics = _plot_panel_specs()
    if transpose:
        rows = len(metrics)
        cols = 1
        figsize = (8.0, 3.3 * rows)
    else:
        rows = 1
        cols = len(metrics)
        figsize = (4.8 * cols, 4.4)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes_array = axes if transpose else axes

    for idx, (metric, metric_title) in enumerate(metrics):
        ax = axes[idx, 0] if transpose else axes[0, idx]
        panel = frontier_timeseries[["month", metric]].copy()
        panel = panel[np.isfinite(panel["month"]) & np.isfinite(panel[metric])]
        ax.set_title(f"Frontier Spread Over Time: {metric_title}")
        ax.set_xlabel("Month")
        ax.set_ylabel(metric)
        if panel.empty:
            ax.text(0.5, 0.5, "No finite values", ha="center", va="center", transform=ax.transAxes)
            continue

        grouped = panel.groupby("month", sort=True)[metric]
        month_index = sorted(grouped.groups)
        months = np.array(month_index, dtype=float)
        mins = grouped.min().reindex(month_index).to_numpy(float)
        p10 = grouped.quantile(0.10).reindex(month_index).to_numpy(float)
        medians = grouped.median().reindex(month_index).to_numpy(float)
        p90 = grouped.quantile(0.90).reindex(month_index).to_numpy(float)
        maxs = grouped.max().reindex(month_index).to_numpy(float)

        ax.fill_between(months, mins, maxs, color="#93c5fd", alpha=0.25, label="Min-Max")
        ax.fill_between(months, p10, p90, color="#2563eb", alpha=0.28, label="P10-P90")
        ax.plot(months, medians, color="#111827", linewidth=2.0, label="Median")
        ax.scatter(months, medians, color="#111827", s=14, zorder=3)
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    filename = "budget_frontier_timeseries_transposed.png" if transpose else "budget_frontier_timeseries.png"
    fig.savefig(output_dir / filename, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replot-only", action="store_true", help="Reuse budget_frontier_selections.csv in --output-dir and regenerate plots only.")
    parser.add_argument("--users", type=int, default=500)
    parser.add_argument("--years", type=float, default=3.0)
    parser.add_argument("--resolution-months", type=int, default=3)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--simulation-concurrency", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--plateau-quarter", type=int, default=5)
    parser.add_argument("--employee-mix", type=str, default=None)
    parser.add_argument(
        "--engineering-context",
        choices=["bounded", "maintenance", "mixed"],
        default="mixed",
    )
    parser.add_argument("--backend", choices=["numpy", "numba", "torch-mps", "auto"], default="auto")
    parser.add_argument("--total-budget-usd", type=float, default=None)
    parser.add_argument("--service-budget-min-usd", type=float, default=0.0)
    parser.add_argument("--service-budget-max-usd", type=float, default=None)
    parser.add_argument("--service-budget-points", type=int, default=5)
    parser.add_argument("--hardware-budget-min-usd", type=float, default=0.0)
    parser.add_argument("--hardware-budget-max-usd", type=float, default=None)
    parser.add_argument("--hardware-budget-points", type=int, default=5)
    parser.add_argument("--confidential-document-fraction", type=float, default=0.0)
    parser.add_argument("--usd-per-service-cost-index-quarter", type=float, default=60.0)
    parser.add_argument("--usd-per-hardware-capex-index", type=float, default=None)
    parser.add_argument("--hardware-calibration", choices=sorted(HARDWARE_CALIBRATIONS), default="h100-gemma4-31b")
    parser.add_argument("--output-dir", type=Path, default=Path("budget_sweep_outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.replot_only:
        selection_summary = _load_selection_summary(args.output_dir)
        plot_budget_grid(selection_summary, args.output_dir)
        plot_budget_contours(selection_summary, args.output_dir)
        try:
            frontier_timeseries = _load_frontier_timeseries(args.output_dir)
        except FileNotFoundError:
            frontier_timeseries = None
        if frontier_timeseries is not None:
            _plot_frontier_timeseries(frontier_timeseries, args.output_dir, transpose=False)
            _plot_frontier_timeseries(frontier_timeseries, args.output_dir, transpose=True)
        print(f"Replotted outputs from {args.output_dir / 'budget_frontier_selections.csv'}")
        print(f"Wrote plots to {args.output_dir}")
        return

    missing_budget_args = [
        name
        for name, value in (
            ("--total-budget-usd", args.total_budget_usd),
            ("--service-budget-max-usd", args.service_budget_max_usd),
            ("--hardware-budget-max-usd", args.hardware_budget_max_usd),
        )
        if value is None
    ]
    if missing_budget_args:
        raise SystemExit(f"Missing required arguments unless --replot-only is set: {', '.join(missing_budget_args)}")
    if not 0.0 <= args.confidential_document_fraction <= 1.0:
        raise SystemExit("--confidential-document-fraction must be between 0 and 1.")

    backend = _resolve_backend(args.backend)
    employee_mix = parse_employee_mix(args.employee_mix)

    calibration = _hardware_calibration_profile(args.hardware_calibration)
    hardware_index_usd = (
        args.usd_per_hardware_capex_index
        if args.usd_per_hardware_capex_index is not None
        else float(calibration["target_unit_usd"]) / HARDWARE_SCENARIOS["onprem_10pct_capacity"]["capex_units"]
    )
    base_config = SimulationConfig(
        users=args.users,
        years=args.years,
        resolution_months=args.resolution_months,
        runs=args.runs,
        concurrency=args.simulation_concurrency,
        seed=args.seed,
        plateau_quarter=args.plateau_quarter,
        confidential_document_fraction=args.confidential_document_fraction,
        usd_per_service_cost_index_quarter=args.usd_per_service_cost_index_quarter,
        usd_per_hardware_capex_index=hardware_index_usd,
        hardware_calibration=args.hardware_calibration,
        onprem_capability_multiplier=float(calibration["onprem_capability_multiplier"]),
        employee_mix=employee_mix,
        engineering_context=args.engineering_context,
        backend=backend,
        show_progress=True,
        output_dir=args.output_dir,
    )

    pairs = budget_pairs(args)
    if not pairs:
        raise SystemExit("No budget pairs fit within the total budget constraint.")

    selections: list[pd.DataFrame] = []
    frontier_frames: list[pd.DataFrame] = []
    frontier_timeseries_frames: list[pd.DataFrame] = []

    executor_cls = ThreadPoolExecutor if backend == "torch-mps" else ProcessPoolExecutor
    with executor_cls(max_workers=min(args.concurrency, len(pairs))) as executor:
        futures = {
            executor.submit(evaluate_budget_pair, service_budget, hardware_budget, base_config, index): (service_budget, hardware_budget)
            for index, (service_budget, hardware_budget) in enumerate(pairs)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Sweeping budgets", unit="budget-pair"):
            selection_rows, frontier, frontier_timeseries = future.result()
            selections.append(selection_rows)
            frontier_frames.append(frontier)
            frontier_timeseries_frames.append(frontier_timeseries)

    selection_summary = pd.concat(selections, ignore_index=True)
    frontier_points = pd.concat(frontier_frames, ignore_index=True)
    frontier_timeseries = pd.concat(frontier_timeseries_frames, ignore_index=True)

    selection_summary.to_csv(args.output_dir / "budget_frontier_selections.csv", index=False)
    frontier_points.to_csv(args.output_dir / "budget_frontier_points.csv", index=False)
    frontier_timeseries.to_csv(args.output_dir / "budget_frontier_timeseries.csv", index=False)
    plot_budget_grid(selection_summary, args.output_dir, total_budget_usd=args.total_budget_usd, years=args.years)
    plot_budget_contours(selection_summary, args.output_dir, total_budget_usd=args.total_budget_usd, years=args.years)
    _plot_frontier_timeseries(frontier_timeseries, args.output_dir, transpose=False)
    _plot_frontier_timeseries(frontier_timeseries, args.output_dir, transpose=True)

    print(f"Wrote outputs to {args.output_dir}")
    print(selection_summary.head(6).to_string(index=False))


if __name__ == "__main__":
    main()
