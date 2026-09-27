#!/usr/bin/env python3
"""Plot illustrative full-service, full-hardware, and mixed optimum portfolios."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from simulate_llm_efficiency import (
    HARDWARE_CALIBRATIONS, HARDWARE_SCENARIOS, SimulationConfig,
    _hardware_calibration_profile, _resolve_backend)
from sweep_portfolio_rate_changes import (
    SweepScenario, return_change_summary, select_portfolios,
    simulate_portfolio_scenario)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=500)
    parser.add_argument("--years", type=float, default=3.0)
    parser.add_argument("--resolution-months", type=int, default=3)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument(
        "--individual-paths",
        type=int,
        default=5,
        help="Number of individual Monte Carlo return-rate paths to overlay for each illustrative category.",
    )
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument(
        "--backend",
        choices=["numpy", "numba", "torch-mps", "auto"],
        default="auto",
    )
    parser.add_argument(
        "--software-group",
        action="store_true",
        help="Use a workforce mix of 90% software engineering and 10% administration.",
    )
    parser.add_argument(
        "--frontier",
        choices=["low-risk", "optimum", "high-return"],
        default="optimum",
        help="Frontier selection to plot. high-return uses the sweep's high-gain selection rule.",
    )
    parser.add_argument(
        "--token-cost",
        choices=["auto", "flat", "gradual_break_even", "sudden_break_even"],
        default="auto",
        help="Restrict the frontier selection to one token-cost path; auto lets all paths compete.",
    )
    parser.add_argument(
        "--highest-fluctuation",
        action="store_true",
        help="Plot the fixed scenario with the greatest within-run return-rate-change volatility for each budget category.",
    )
    parser.add_argument(
        "--confidential-document-share", type=float, default=0.25
    )
    parser.add_argument("--plateau-months", type=int, default=24)
    parser.add_argument(
        "--full-service-usd-per-person-month", type=float, default=50.0
    )
    parser.add_argument("--full-service-hardware-usd", type=float, default=0.0)
    parser.add_argument(
        "--full-hardware-usd-per-person-month", type=float, default=0.0
    )
    parser.add_argument(
        "--full-hardware-hardware-usd", type=float, default=1_000_000.0
    )
    parser.add_argument(
        "--mixed-usd-per-person-month", type=float, default=25.0
    )
    parser.add_argument("--mixed-hardware-usd", type=float, default=500_000.0)
    parser.add_argument(
        "--usd-per-service-cost-index-quarter", type=float, default=60.0
    )
    parser.add_argument(
        "--usd-per-hardware-capex-index", type=float, default=None
    )
    parser.add_argument(
        "--hardware-calibration",
        choices=sorted(HARDWARE_CALIBRATIONS),
        default="h100-gemma4-31b",
    )
    parser.add_argument(
        "--figure-width",
        choices=["default", "one-column"],
        default="default",
        help="Use one-column for a compact 3.5-inch-wide figure suitable for a two-column paper.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("illustrative_portfolio_outputs"),
    )
    return parser.parse_args()


def return_rate_quantiles(results: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["scenario", "period", "month"]
    return (
        results.groupby(group_cols, sort=False)["return_rate"]
        .agg(
            return_rate_p10=lambda values: values.quantile(0.10),
            return_rate_p50="median",
            return_rate_p90=lambda values: values.quantile(0.90),
        )
        .reset_index()
    )


def scenario_fluctuation(results: pd.DataFrame) -> pd.DataFrame:
    ordered = results.sort_values(["scenario", "run", "period"]).copy()
    ordered["return_rate_change"] = ordered.groupby(
        ["scenario", "run"], sort=False
    )["return_rate"].diff()
    per_run = (
        ordered.groupby(["scenario", "run"], sort=False)["return_rate_change"]
        .std()
        .rename("return_rate_change_volatility")
        .reset_index()
    )
    return (
        per_run.groupby("scenario", sort=False)[
            "return_rate_change_volatility"
        ]
        .median()
        .reset_index()
    )


def main() -> None:
    args = parse_args()
    if args.runs < 2:
        raise SystemExit(
            "--runs must be at least 2 to select an optimum risk frontier."
        )
    if args.individual_paths < 0:
        raise SystemExit("--individual-paths must be non-negative.")
    if (
        args.resolution_months <= 0
        or args.plateau_months <= 0
        or args.plateau_months % args.resolution_months
    ):
        raise SystemExit(
            "--plateau-months must be a positive multiple of --resolution-months."
        )
    if not 0.0 <= args.confidential_document_share <= 1.0:
        raise SystemExit(
            "--confidential-document-share must be between 0 and 1."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibration = _hardware_calibration_profile(args.hardware_calibration)
    hardware_index_usd = (
        args.usd_per_hardware_capex_index
        if args.usd_per_hardware_capex_index is not None
        else float(calibration["target_unit_usd"])
        / HARDWARE_SCENARIOS["onprem_10pct_capacity"]["capex_units"]
    )
    base_config = SimulationConfig(
        users=args.users,
        years=args.years,
        resolution_months=args.resolution_months,
        runs=args.runs,
        seed=args.seed,
        plateau_quarter=args.plateau_months // args.resolution_months,
        usd_per_service_cost_index_quarter=args.usd_per_service_cost_index_quarter,
        usd_per_hardware_capex_index=hardware_index_usd,
        hardware_calibration=args.hardware_calibration,
        onprem_capability_multiplier=float(
            calibration["onprem_capability_multiplier"]
        ),
        employee_mix=(
            {"software_engineering": 0.9, "administration": 0.1}
            if args.software_group
            else None
        ),
        backend=_resolve_backend(args.backend),
        output_dir=args.output_dir,
    )
    scenarios = [
        (
            "Full service",
            SweepScenario(
                args.users,
                args.full_service_usd_per_person_month,
                args.full_service_hardware_usd,
                args.confidential_document_share,
                True,
            ),
        ),
        (
            "Full hardware",
            SweepScenario(
                args.users,
                args.full_hardware_usd_per_person_month,
                args.full_hardware_hardware_usd,
                args.confidential_document_share,
                True,
            ),
        ),
        (
            "Mixed",
            SweepScenario(
                args.users,
                args.mixed_usd_per_person_month,
                args.mixed_hardware_usd,
                args.confidential_document_share,
                True,
            ),
        ),
    ]
    portfolio_by_frontier = {
        "low-risk": "low_risk",
        "optimum": "optimum",
        "high-return": "high_gain",
    }
    selected_portfolio = portfolio_by_frontier[args.frontier]
    selected_frames = []
    path_frames = []
    for index, (name, scenario) in enumerate(scenarios):
        results, metadata = simulate_portfolio_scenario(
            scenario, base_config, index
        )
        quantiles = return_rate_quantiles(results)
        changes = return_change_summary(results)
        if args.token_cost != "auto":
            changes = changes[changes["token_cost"] == args.token_cost]
            results = results[results["token_cost"] == args.token_cost]
            quantiles = quantiles[
                quantiles["scenario"].isin(results["scenario"].unique())
            ]
        if args.highest_fluctuation:
            final_selection = (
                scenario_fluctuation(results)
                .sort_values(
                    ["return_rate_change_volatility", "scenario"],
                    ascending=[False, True],
                )
                .iloc[0]
            )
            selection_rule = (
                "maximum median within-run return-rate-change volatility"
            )
        else:
            frontier = select_portfolios(changes, scenario, metadata)
            final_selection = frontier[
                (frontier["portfolio"] == selected_portfolio)
                & (frontier["period"] == frontier["period"].max())
            ].iloc[0]
            selection_rule = final_selection["selection_rule"]
        timeseries = changes[
            changes["scenario"] == final_selection["scenario"]
        ].merge(quantiles, on=["scenario", "period", "month"], how="left")
        timeseries["portfolio"] = (
            "highest_fluctuation"
            if args.highest_fluctuation
            else selected_portfolio
        )
        timeseries["selection_rule"] = selection_rule
        timeseries["selected_at_final_period"] = True
        timeseries["final_selection_month"] = (
            results["month"].max()
            if args.highest_fluctuation
            else final_selection["month"]
        )
        timeseries["forced_token_cost"] = args.token_cost
        timeseries["highest_fluctuation"] = args.highest_fluctuation
        if args.highest_fluctuation:
            timeseries["selected_return_rate_change_volatility"] = (
                final_selection["return_rate_change_volatility"]
            )
        timeseries["workforce_mix"] = (
            "90% software engineering; 10% administration"
            if args.software_group
            else "default mixed workforce"
        )
        timeseries["illustrative_scenario"] = name
        selected_frames.append(timeseries)
        run_ids = sorted(
            results.loc[
                results["scenario"] == final_selection["scenario"], "run"
            ].unique()
        )
        path_count = min(args.individual_paths, len(run_ids))
        if path_count:
            path_indices = [
                round(index * (len(run_ids) - 1) / max(1, path_count - 1))
                for index in range(path_count)
            ]
            paths = results[
                (results["scenario"] == final_selection["scenario"])
                & (
                    results["run"].isin(
                        [run_ids[index] for index in path_indices]
                    )
                )
            ][["run", "month", "return_rate"]].copy()
            paths["illustrative_scenario"] = name
            path_frames.append(paths)

    selected = pd.concat(selected_frames, ignore_index=True)
    paths = (
        pd.concat(path_frames, ignore_index=True)
        if path_frames
        else pd.DataFrame()
    )
    selected.to_csv(
        args.output_dir / "illustrative_portfolio_selections.csv", index=False
    )
    one_column = args.figure_width == "one-column"
    fig, ax = plt.subplots(figsize=(3.5, 2.8) if one_column else (10, 5.5))
    colors = {
        "Full service": "#2563eb",
        "Full hardware": "#dc2626",
        "Mixed": "#059669",
    }
    for name, values in selected.groupby("illustrative_scenario", sort=False):
        values = values.sort_values("month")
        color = colors[name]
        if not paths.empty:
            for _, path in paths[
                paths["illustrative_scenario"] == name
            ].groupby("run", sort=False):
                path = path.sort_values("month")
                ax.plot(
                    path["month"],
                    path["return_rate"],
                    color=color,
                    alpha=0.18,
                    linewidth=0.55 if one_column else 0.8,
                    zorder=1,
                )
        ax.fill_between(
            values["month"],
            values["return_rate_p10"],
            values["return_rate_p90"],
            color=color,
            alpha=0.16,
        )
        ax.plot(
            values["month"],
            values["return_rate_p50"],
            color=color,
            linewidth=1.4 if one_column else 2.4,
            marker="o",
            markersize=3.5 if one_column else 6,
            label=name,
            zorder=3,
        )
    ax.axvline(
        args.plateau_months,
        color="#4b5563",
        linestyle="--",
        linewidth=1.2,
        label="Capability plateau",
    )
    ax.set_xlabel("Month", fontsize=8 if one_column else None)
    ax.set_ylabel(
        f"Efficiency-index gain per AI cost unit\n"
        f"(mean gain/employee ÷ total AI cost/employee; {args.resolution_months}-month period)",
        fontsize=7 if one_column else None,
    )
    ax.tick_params(labelsize=7 if one_column else None)
    ax.legend(loc="best", fontsize=6 if one_column else None)
    fig.tight_layout()
    suffix = "_one_column" if one_column else ""
    fig.savefig(
        args.output_dir
        / f"illustrative_{args.frontier}_portfolio_return_rates{suffix}.png",
        dpi=300 if one_column else 180,
    )
    plt.close(fig)
    print(f"Wrote illustrative plot and selections to {args.output_dir}")


if __name__ == "__main__":
    main()
