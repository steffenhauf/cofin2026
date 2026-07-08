#!/usr/bin/env python3
"""Sweep budget pairs, fit final-period efficient frontiers, and plot selected portfolios."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    SimulationConfig,
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


def evaluate_budget_pair(
    service_budget_usd: float,
    hardware_budget_usd: float,
    base_config: SimulationConfig,
    pair_index: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = SimulationConfig(
        users=base_config.users,
        years=base_config.years,
        resolution_months=base_config.resolution_months,
        runs=base_config.runs,
        concurrency=1,
        seed=base_config.seed + pair_index * 10007,
        plateau_quarter=base_config.plateau_quarter,
        base_cost_index_per_active_user_quarter=base_config.base_cost_index_per_active_user_quarter,
        max_monthly_service_budget_usd=service_budget_usd,
        max_upfront_hardware_budget_usd=hardware_budget_usd,
        usd_per_service_cost_index_quarter=base_config.usd_per_service_cost_index_quarter,
        usd_per_hardware_capex_index=base_config.usd_per_hardware_capex_index,
        employee_mix=base_config.employee_mix,
        engineering_context=base_config.engineering_context,
        backend=base_config.backend,
        output_dir=base_config.output_dir,
    )

    results = run_simulation(config)
    summary = summarize(results)
    final_summary = summary[summary["period"] == summary["period"].max()].copy()
    frontier = efficient_frontier(final_summary)
    frontier_fit = frontier_line(frontier)

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
                "risk_mean": float(low_risk["risk_mean"]),
                "return_rate_mean": float(low_risk["return_rate_mean"]),
                "mean_efficiency_gain_mean": float(low_risk["mean_efficiency_gain_mean"]),
                "hardware_budget_feasible_mean": float(low_risk["hardware_budget_feasible_mean"]),
                "service_budget_scale_mean": float(low_risk["service_budget_scale_mean"]),
            },
            {
                "selection": "high_return",
                "service_budget_usd": service_budget_usd,
                "hardware_budget_usd": hardware_budget_usd,
                "total_lifetime_budget_usd": service_budget_usd * 12.0 * config.years + hardware_budget_usd,
                "scenario": high_return["scenario"],
                "risk_mean": float(high_return["risk_mean"]),
                "return_rate_mean": float(high_return["return_rate_mean"]),
                "mean_efficiency_gain_mean": float(high_return["mean_efficiency_gain_mean"]),
                "hardware_budget_feasible_mean": float(high_return["hardware_budget_feasible_mean"]),
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
    return selection_rows, frontier_output


def plot_budget_grid(selection_summary: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    panels = [
        ("low_risk", "return_rate_mean", "Low-Risk Frontier Return"),
        ("low_risk", "risk_mean", "Low-Risk Frontier Risk"),
        ("high_return", "return_rate_mean", "High-Return Frontier Return"),
        ("high_return", "risk_mean", "High-Return Frontier Risk"),
    ]

    for ax, (selection, metric, title) in zip(axes.ravel(), panels, strict=True):
        panel = selection_summary[selection_summary["selection"] == selection]
        scatter = ax.scatter(
            panel["service_budget_usd"],
            panel["hardware_budget_usd"],
            c=panel[metric],
            cmap="viridis",
            s=90,
            edgecolors="black",
            linewidths=0.3,
        )
        ax.set_title(title)
        ax.set_xlabel("Monthly service budget (USD)")
        ax.set_ylabel("Upfront hardware budget (USD)")
        colorbar = fig.colorbar(scatter, ax=ax, shrink=0.84)
        colorbar.set_label(metric)

    fig.tight_layout()
    fig.savefig(output_dir / "budget_frontier_grid.png", dpi=180)
    plt.close(fig)


def plot_budget_contours(selection_summary: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    panels = [
        ("low_risk", "return_rate_mean", "Low-Risk Frontier Return Contours"),
        ("low_risk", "risk_mean", "Low-Risk Frontier Risk Contours"),
        ("high_return", "return_rate_mean", "High-Return Frontier Return Contours"),
        ("high_return", "risk_mean", "High-Return Frontier Risk Contours"),
    ]

    for ax, (selection, metric, title) in zip(axes.ravel(), panels, strict=True):
        panel = selection_summary[selection_summary["selection"] == selection].copy()
        if len(panel) < 3:
            ax.text(0.5, 0.5, "Need at least 3 budget points", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            ax.set_xlabel("Monthly service budget (USD)")
            ax.set_ylabel("Upfront hardware budget (USD)")
            continue

        triangulation = mtri.Triangulation(panel["service_budget_usd"], panel["hardware_budget_usd"])
        contour = ax.tricontourf(
            triangulation,
            panel[metric],
            levels=12,
            cmap="viridis",
        )
        ax.tricontour(
            triangulation,
            panel[metric],
            levels=8,
            colors="white",
            linewidths=0.4,
            alpha=0.6,
        )
        ax.scatter(
            panel["service_budget_usd"],
            panel["hardware_budget_usd"],
            s=16,
            color="black",
            alpha=0.6,
        )
        ax.set_title(title)
        ax.set_xlabel("Monthly service budget (USD)")
        ax.set_ylabel("Upfront hardware budget (USD)")
        colorbar = fig.colorbar(contour, ax=ax, shrink=0.84)
        colorbar.set_label(metric)

    fig.tight_layout()
    fig.savefig(output_dir / "budget_frontier_contours.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=500)
    parser.add_argument("--years", type=float, default=3.0)
    parser.add_argument("--resolution-months", type=int, default=3)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--plateau-quarter", type=int, default=5)
    parser.add_argument("--employee-mix", type=str, default=None)
    parser.add_argument(
        "--engineering-context",
        choices=["bounded", "maintenance", "mixed"],
        default="mixed",
    )
    parser.add_argument("--backend", choices=["numpy", "torch-mps", "auto"], default="numpy")
    parser.add_argument("--total-budget-usd", type=float, required=True)
    parser.add_argument("--service-budget-min-usd", type=float, default=0.0)
    parser.add_argument("--service-budget-max-usd", type=float, required=True)
    parser.add_argument("--service-budget-points", type=int, default=5)
    parser.add_argument("--hardware-budget-min-usd", type=float, default=0.0)
    parser.add_argument("--hardware-budget-max-usd", type=float, required=True)
    parser.add_argument("--hardware-budget-points", type=int, default=5)
    parser.add_argument("--usd-per-service-cost-index-quarter", type=float, default=60.0)
    parser.add_argument("--usd-per-hardware-capex-index", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("budget_sweep_outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend = _resolve_backend(args.backend)
    employee_mix = parse_employee_mix(args.employee_mix)

    import simulate_llm_efficiency as sim

    hardware_index_usd = (
        args.usd_per_hardware_capex_index
        if args.usd_per_hardware_capex_index is not None
        else sim.USD_PER_HARDWARE_CAPEX_INDEX_DEFAULT
    )
    base_config = SimulationConfig(
        users=args.users,
        years=args.years,
        resolution_months=args.resolution_months,
        runs=args.runs,
        concurrency=1,
        seed=args.seed,
        plateau_quarter=args.plateau_quarter,
        usd_per_service_cost_index_quarter=args.usd_per_service_cost_index_quarter,
        usd_per_hardware_capex_index=hardware_index_usd,
        employee_mix=employee_mix,
        engineering_context=args.engineering_context,
        backend=backend,
        output_dir=args.output_dir,
    )

    pairs = budget_pairs(args)
    if not pairs:
        raise SystemExit("No budget pairs fit within the total budget constraint.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selections: list[pd.DataFrame] = []
    frontier_frames: list[pd.DataFrame] = []

    with ThreadPoolExecutor(max_workers=min(args.concurrency, len(pairs))) as executor:
        futures = {
            executor.submit(evaluate_budget_pair, service_budget, hardware_budget, base_config, index): (service_budget, hardware_budget)
            for index, (service_budget, hardware_budget) in enumerate(pairs)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Sweeping budgets", unit="budget-pair"):
            selection_rows, frontier = future.result()
            selections.append(selection_rows)
            frontier_frames.append(frontier)

    selection_summary = pd.concat(selections, ignore_index=True)
    frontier_points = pd.concat(frontier_frames, ignore_index=True)

    selection_summary.to_csv(args.output_dir / "budget_frontier_selections.csv", index=False)
    frontier_points.to_csv(args.output_dir / "budget_frontier_points.csv", index=False)
    plot_budget_grid(selection_summary, args.output_dir)
    plot_budget_contours(selection_summary, args.output_dir)

    print(f"Wrote outputs to {args.output_dir}")
    print(selection_summary.head(6).to_string(index=False))


if __name__ == "__main__":
    main()
