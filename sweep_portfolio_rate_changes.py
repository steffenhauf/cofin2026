#!/usr/bin/env python3
"""Sweep company portfolios and export median return-rate changes and their risk."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from tqdm.auto import tqdm

from simulate_llm_efficiency import (
    ACCESS_PLANS,
    HARDWARE_CALIBRATIONS,
    HARDWARE_SCENARIOS,
    LOCAL_FALLBACK_POLICIES,
    MODEL_SCENARIOS,
    SHOCK_COMBINATIONS,
    TOKEN_SCENARIOS,
    SimulationConfig,
    _hardware_calibration_profile,
    hardware_purchase_cost_usd,
    _resolve_backend,
    _simulate_one_scenario,
)


HEADCOUNTS = (5, 25, 50, 100, 500, 2500)
SERVICE_BUDGETS_USD = (5.0, 10.0, 25.0, 50.0)
HARDWARE_BUDGETS_USD = (0.0, 10_000.0, 50_000.0, 100_000.0, 500_000.0, 1_000_000.0)
CONFIDENTIAL_SHARES = (0.0, 0.1, 0.25, 0.5, 0.9)
PORTFOLIOS = ("low_risk", "optimum", "high_gain")
COMPANY_PROFILES = {
    "administration_heavy": {
        "software_engineering": 0.05,
        "administration": 0.70,
        "manual_labor": 0.10,
        "knowledge_work": 0.15,
        "creative_work": 0.00,
    },
    "mixed": {
        "software_engineering": 0.25,
        "administration": 0.35,
        "manual_labor": 0.20,
        "knowledge_work": 0.20,
        "creative_work": 0.10,
    },
    "manual_labor_heavy": {
        "software_engineering": 0.05,
        "administration": 0.15,
        "manual_labor": 0.70,
        "knowledge_work": 0.10,
        "creative_work": 0.00,
    },
    "software_engineering_heavy": {
        "software_engineering": 0.75,
        "administration": 0.10,
        "manual_labor": 0.05,
        "knowledge_work": 0.10,
        "creative_work": 0.00,
    },
    "knowledge_worker_heavy": {
        "software_engineering": 0.10,
        "administration": 0.20,
        "manual_labor": 0.05,
        "knowledge_work": 0.65,
        "creative_work": 0.00,
    },
    "creative_worker_heavy": {
        "software_engineering": 0.05,
        "administration": 0.10,
        "manual_labor": 0.05,
        "knowledge_work": 0.10,
        "creative_work": 0.70,
    },
}


@dataclass(frozen=True)
class SweepScenario:
    users: int
    service_budget_per_person_usd: float
    hardware_budget_usd: float
    confidential_share: float
    plateau: bool


def hardware_for_budget(budget_usd: float, hardware_calibration: str) -> str:
    """Return the highest-capacity maxed-out tier affordable with the full upfront budget."""
    affordable = ["cloud_api_only"]
    for name, hardware in HARDWARE_SCENARIOS.items():
        if name == "cloud_oss":
            continue
        cost_usd = hardware_purchase_cost_usd(hardware_calibration, name)
        if cost_usd <= budget_usd + 1e-9:
            affordable.append(name)
    return max(affordable, key=lambda name: HARDWARE_SCENARIOS[name]["capacity"])


def hardware_selection_for_budget(budget_usd: float) -> tuple[str, str]:
    """Choose the highest system tier in the supplied hardware-cost table that fits the CAPEX budget."""
    if budget_usd < 8_500.0:
        return "rtx6000-ada-gemma4-12b", "cloud_api_only"
    if budget_usd < 14_500.0:
        return "rtx6000-ada-gemma4-12b", "onprem_full_capacity"
    if budget_usd < 380_000.0:
        return "rtx6000-pro-gemma4-26b-moe", "onprem_full_capacity"
    if budget_usd < 750_000.0:
        return "h100-gemma4-31b", "onprem_full_capacity"
    return "b200-gemma4-31b", "onprem_full_capacity"

def friendly_label(scenario: SweepScenario, portfolio: str) -> str:
    plateau = "capability plateaus after 18 months" if scenario.plateau else "continuous capability improvement"
    names = {
        "low_risk": "low-risk portfolio",
        "optimum": "balanced optimum portfolio",
        "high_gain": "high-gain portfolio",
    }
    return (
        f"{scenario.users} employees | ${scenario.service_budget_per_person_usd:g}/person/month service | "
        f"${scenario.hardware_budget_usd:,.0f} upfront hardware | "
        f"{scenario.confidential_share:.0%} confidential documents | {plateau} | {names[portfolio]}"
    )


def scenario_grid(stochastic_shocks: bool = False) -> list[SweepScenario]:
    return [
        SweepScenario(users, service, hardware, confidential, plateau)
        for users in HEADCOUNTS
        for service in SERVICE_BUDGETS_USD
        for hardware in HARDWARE_BUDGETS_USD
        for confidential in CONFIDENTIAL_SHARES
        for plateau in ((False,) if stochastic_shocks else (True, False))
    ]


def simulation_scenarios(
    hardware_name: str,
    plateau: bool,
    hardware_calibration: str,
    stochastic_shocks: bool = False,
) -> list[tuple]:
    rows = []
    model_scenarios = (
        ((model, specification) for model, specification in MODEL_SCENARIOS.items() if specification["growth"])
        if stochastic_shocks
        else MODEL_SCENARIOS.items()
    )
    token_scenarios = ("flat",) if stochastic_shocks else tuple(TOKEN_SCENARIOS)
    shock_combinations = SHOCK_COMBINATIONS if stochastic_shocks else ("none",)
    for model, model_specification in model_scenarios:
        if plateau and model_specification["growth"]:
            continue
        if not plateau and not model_specification["growth"]:
            continue
        if hardware_name in ("cloud_oss",) and model_specification["kind"] != "oss":
            continue
        if hardware_name not in ("cloud_api_only", "cloud_oss") and model_specification["kind"] != "oss":
            continue
        if hardware_name == "cloud_oss":
            for shock_combination in shock_combinations:
                rows.append((model, "flat", hardware_name, "maxed_out", "pay_per_use", "persona_choice", "eu_cloud_oss", shock_combination))
            continue
        for token_cost in token_scenarios:
            for access_plan in ACCESS_PLANS:
                fallbacks = ("persona_choice",) if hardware_name == "cloud_api_only" else LOCAL_FALLBACK_POLICIES
                providers = ("global_service", "european_service") if hardware_name == "cloud_api_only" else ("local_hardware",)
                for fallback in fallbacks:
                    for service_provider in providers:
                        for shock_combination in shock_combinations:
                            rows.append((model, token_cost, hardware_name, "maxed_out", access_plan, fallback, service_provider, shock_combination))
    return rows

def simulate_portfolio_scenario(
    scenario: SweepScenario,
    base_config: SimulationConfig,
    scenario_index: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    hardware_calibration, hardware_name = hardware_selection_for_budget(scenario.hardware_budget_usd)
    calibration = _hardware_calibration_profile(hardware_calibration)
    config = SimulationConfig(
        users=scenario.users,
        years=base_config.years,
        resolution_months=base_config.resolution_months,
        runs=base_config.runs,
        concurrency=1,
        seed=base_config.seed + scenario_index * 100_003,
        plateau_quarter=base_config.plateau_quarter if scenario.plateau else 10**6,
        max_monthly_service_budget_usd=scenario.service_budget_per_person_usd * scenario.users,
        max_upfront_hardware_budget_usd=scenario.hardware_budget_usd,
        confidential_document_fraction=min(scenario.confidential_share, 0.9),
        zero_risk_confidential_work_share=base_config.zero_risk_confidential_work_share,
        counterfactual_skill_growth_enabled=base_config.counterfactual_skill_growth_enabled,
        counterfactual_skill_growth_rate_per_year=base_config.counterfactual_skill_growth_rate_per_year,
        usd_per_service_cost_index_quarter=base_config.usd_per_service_cost_index_quarter,
        usd_per_hardware_capex_index=base_config.usd_per_hardware_capex_index,
        hardware_calibration=hardware_calibration,
        onprem_capability_multiplier=float(calibration["onprem_capability_multiplier"]),
        employee_mix=base_config.employee_mix,
        engineering_context=base_config.engineering_context,
        adoption_propensity_concentration=base_config.adoption_propensity_concentration,
        adoption_capability_elasticity=base_config.adoption_capability_elasticity,
        improvement_base_probability=base_config.improvement_base_probability,
        improvement_capability_weight=base_config.improvement_capability_weight,
        improvement_probability_min=base_config.improvement_probability_min,
        improvement_probability_max=base_config.improvement_probability_max,
        ai_gain_min=base_config.ai_gain_min,
        ai_gain_max=base_config.ai_gain_max,
        zero_risk_feature_gain=base_config.zero_risk_feature_gain,
        embargo_shock_probability=base_config.embargo_shock_probability,
        embargo_rollback_months=base_config.embargo_rollback_months,
        stochastic_shocks=base_config.stochastic_shocks,
        sudden_break_even_probability_per_month=base_config.sudden_break_even_probability_per_month,
        gradual_break_even_probability_per_month=base_config.gradual_break_even_probability_per_month,
        capability_plateau_probability_per_month=base_config.capability_plateau_probability_per_month,
        confidential_mixing_ratio=base_config.confidential_mixing_ratio,
        confidential_mixing_penalty=base_config.confidential_mixing_penalty,
        cloud_oss_hours_per_usage_unit_period=base_config.cloud_oss_hours_per_usage_unit_period,
        it_support_enabled=base_config.it_support_enabled,
        it_support_max_users=base_config.it_support_max_users,
        it_support_monthly_cost_eur=base_config.it_support_monthly_cost_eur,
        backend=base_config.backend,
        show_progress=False,
        output_dir=base_config.output_dir,
    )
    frames = []
    asset_names = (hardware_name, "cloud_oss")
    for offset, asset_name in enumerate(asset_names):
        for model_scenario in simulation_scenarios(asset_name, scenario.plateau, config.hardware_calibration, config.stochastic_shocks):
            frames.append(_simulate_one_scenario((*model_scenario[:7], config, config.seed + offset * 1009, model_scenario[7])))
    unit_cost_usd = hardware_purchase_cost_usd(hardware_calibration, hardware_name)
    replicas = int(scenario.hardware_budget_usd // unit_cost_usd) if unit_cost_usd else 0
    metadata = {
        "hardware_scenario": hardware_name,
        "hardware_calibration": hardware_calibration,
        "hardware_replicas": replicas,
        "hardware_cost_usd": unit_cost_usd * replicas,
    }
    return pd.concat(frames, ignore_index=True), metadata


def return_change_summary(results: pd.DataFrame) -> pd.DataFrame:
    ordered = results.sort_values(["scenario", "run", "period"]).copy()
    ordered["return_rate_change"] = ordered.groupby(["scenario", "run"], sort=False)["return_rate"].diff().fillna(0.0)
    group_cols = ["scenario", "model", "token_cost", "access_plan", "hardware", "hardware_refresh", "local_fallback", "service_provider", "shock_combination", "period", "month"]
    return (
        ordered.groupby(group_cols, sort=False)
        .agg(
            median_return_rate_change=("return_rate_change", "median"),
            risk_stddev=("return_rate_change", "std"),
            median_annualized_return_rate=("annualized_return_rate", "median"),
            annualized_return_rate_risk_stddev=("annualized_return_rate", "std"),
            median_efficiency_gain=("mean_efficiency_gain", "median"),
            efficiency_gain_risk_stddev=("mean_efficiency_gain", "std"),
            median_zero_risk_efficiency_gain=("zero_risk_efficiency_gain", "median"),
        )
        .reset_index()
        .fillna(
            {
                "risk_stddev": 0.0,
                "annualized_return_rate_risk_stddev": 0.0,
                "efficiency_gain_risk_stddev": 0.0,
            }
        )
    )


def efficient_frontier(
    points: pd.DataFrame,
    risk_column: str = "risk_stddev",
    reward_column: str = "median_return_rate_change",
) -> pd.DataFrame:
    finite = np.isfinite(points[risk_column]) & np.isfinite(points[reward_column])
    ordered = points.loc[finite].sort_values([risk_column, reward_column], ascending=[True, False])
    best_gain = -np.inf
    selected = []
    for _, row in ordered.iterrows():
        if row[reward_column] > best_gain + 1e-12:
            selected.append(row)
            best_gain = row[reward_column]
    return pd.DataFrame(selected)


def _choose(
    points: pd.DataFrame,
    portfolio: str,
    risk_column: str = "risk_stddev",
    reward_column: str = "median_return_rate_change",
) -> tuple[pd.Series, str]:
    risk_mean, risk_std = points[risk_column].mean(), points[risk_column].std(ddof=0)
    gain_mean, gain_std = points[reward_column].mean(), points[reward_column].std(ddof=0)
    if portfolio == "low_risk":
        eligible = points[points[risk_column] <= risk_mean - risk_std]
        if eligible.empty:
            return points.sort_values([risk_column, reward_column], ascending=[True, False]).iloc[0], "minimum-risk fallback"
        return eligible.sort_values([reward_column, risk_column], ascending=[False, True]).iloc[0], "risk <= mean - 1 sigma"
    if portfolio == "optimum":
        eligible = points[points[risk_column].between(risk_mean - risk_std, risk_mean + risk_std)]
        if eligible.empty:
            eligible = points
            reason = "all-frontier fallback"
        else:
            reason = "risk within mean +/- 1 sigma"
        return eligible.sort_values([reward_column, risk_column], ascending=[False, True]).iloc[0], reason
    eligible = points[points[reward_column] >= gain_mean + 3.0 * gain_std]
    if eligible.empty:
        return points.sort_values([reward_column, risk_column], ascending=[False, True]).iloc[0], "maximum-gain fallback"
    return eligible.sort_values([risk_column, reward_column], ascending=[True, False]).iloc[0], "gain >= mean + 3 sigma"


def select_portfolios(summary: pd.DataFrame, scenario: SweepScenario, metadata: dict[str, object]) -> pd.DataFrame:
    rows = []
    for (period, month), points in summary.groupby(["period", "month"], sort=True):
        frontier = efficient_frontier(
            points,
            "annualized_return_rate_risk_stddev",
            "median_annualized_return_rate",
        )
        if frontier.empty:
            continue
        for portfolio in PORTFOLIOS:
            selected, rule = _choose(
                frontier,
                portfolio,
                "annualized_return_rate_risk_stddev",
                "median_annualized_return_rate",
            )
            row = selected.to_dict()
            provider_labels = {
                "global_service": "global service",
                "european_service": "European service",
                "local_hardware": "local hardware",
            }
            row.update(
                portfolio=portfolio,
                selection_rule=rule,
                column_label=friendly_label(scenario, portfolio) + f" | {provider_labels.get(selected.get('service_provider'), selected.get('service_provider', 'service'))} | shocks: {selected.get('shock_combination', 'none')}",
                users=scenario.users,
                service_budget_per_person_usd=scenario.service_budget_per_person_usd,
                hardware_budget_usd=scenario.hardware_budget_usd,
                confidential_document_share=scenario.confidential_share,
                capability_plateau_after_18_months=scenario.plateau,
                **metadata,
            )
            rows.append(row)
    return pd.DataFrame(rows)


def write_wide_csvs(selected: pd.DataFrame, output_dir: Path) -> None:
    for metric, filename in (("median_return_rate_change", "portfolio_rate_changes.csv"), ("risk_stddev", "portfolio_rate_change_risk.csv")):
        wide = selected.pivot(index="month", columns="column_label", values=metric).reset_index()
        wide = wide.reindex(columns=["month", *sorted(selected["column_label"].unique())])
        wide.to_csv(output_dir / filename, index=False)


SERVICE_MARKERS = {
    "cloud": "o",
    "on-prem": "^",
    "oss-cloud": "D",
    "European": "s",
}


def _service_type(row: pd.Series) -> str:
    provider = str(row.get("service_provider", ""))
    if provider == "global_service":
        return "cloud"
    if provider == "local_hardware":
        return "on-prem"
    if provider == "eu_cloud_oss":
        return "oss-cloud"
    if provider == "european_service":
        return "European"
    return "cloud"


def _selected_median_rows(selected: pd.DataFrame, month: float | None = None) -> pd.DataFrame:
    if selected.empty:
        return selected
    values = selected.copy()
    if month is None and "month" in values:
        month = values["month"].max()
    if month is not None:
        values = values[values["month"] == month]
    if values.empty:
        return values
    values["service_type"] = values.apply(_service_type, axis=1)
    group_cols = ["portfolio", "service_type"]
    if "users" in values:
        group_cols.append("users")
    numeric = [
        column
        for column in (
            "risk_stddev",
            "median_return_rate_change",
            "annualized_return_rate_risk_stddev",
            "median_annualized_return_rate",
            "efficiency_gain_risk_stddev",
            "median_efficiency_gain",
        )
        if column in values
    ]
    return values.groupby(group_cols, as_index=False)[numeric].median()


def _employee_colors(values: pd.DataFrame) -> dict[int, tuple[float, float, float, float]]:
    users = sorted(int(value) for value in values["users"].dropna().unique()) if "users" in values else []
    palette = plt.get_cmap("tab10").colors
    return {user: palette[index % len(palette)] for index, user in enumerate(users)}


def _plot_selected_service_medians(ax, selected: pd.DataFrame, x: str, y: str, colors: dict, month: float | None = None) -> None:
    medians = _selected_median_rows(selected, month)
    for _, row in medians.iterrows():
        if x not in row or y not in row or not np.isfinite(row[x]) or not np.isfinite(row[y]):
            continue
        portfolio = row["portfolio"]
        user_color = colors.get(int(row["users"])) if "users" in row and int(row["users"]) in colors else None
        marker_color = user_color if user_color is not None else colors.get(portfolio, "#111827")
        ax.scatter(
            [row[x]], [row[y]], marker=SERVICE_MARKERS[row["service_type"]],
            s=52, color=marker_color, edgecolor="black", linewidth=0.7,
            zorder=6,
        )


def _add_employee_legend(ax, values: pd.DataFrame, loc="center right") -> None:
    colors = _employee_colors(values)
    handles = [Line2D([], [], marker="o", linestyle="", color=color, label=f"{users} employees") for users, color in colors.items()]
    legend = ax.legend(handles=handles, title="Employee bracket", fontsize=6, title_fontsize=7, loc=loc)
    ax.add_artist(legend)


def _add_service_legend(ax, loc="lower right") -> None:
    handles = [Line2D([], [], marker=marker, linestyle="", color="#374151", markeredgecolor="black", label=label) for label, marker in SERVICE_MARKERS.items()]
    legend = ax.legend(handles=handles, title="Selected CSV median", fontsize=6, title_fontsize=7, loc=loc)
    ax.add_artist(legend)



def _symlog_threshold(values: pd.Series | np.ndarray) -> float:
    finite = np.abs(np.asarray(values, dtype=float))
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    if finite.size == 0:
        return 1e-6
    return max(1e-6, float(np.percentile(finite, 10)))


def _apply_frontier_scale(ax, risk_values, reward_values, symlog: bool) -> None:
    if not symlog:
        return
    ax.set_xscale("symlog", linthresh=_symlog_threshold(risk_values))
    ax.set_yscale("symlog", linthresh=_symlog_threshold(reward_values))


def _combined_legend(ax, values: pd.DataFrame, extra_handles: list[Line2D] | None = None) -> None:
    colors = _employee_colors(values)
    handles = [Line2D([], [], marker="o", linestyle="", color=color, label=f"Employees: {users}") for users, color in colors.items()]
    handles.extend(
        Line2D([], [], marker=marker, linestyle="", color="#374151", markeredgecolor="black", label=f"Service: {label}")
        for label, marker in SERVICE_MARKERS.items()
    )
    handles.append(Line2D([], [], marker="o", linestyle="", color="#9ca3af", markeredgecolor="#111827", markeredgewidth=2.0, label="Per-bracket Pareto asset"))
    if extra_handles:
        handles.extend(extra_handles)
    ax.legend(
        handles=handles,
        title="Legend",
        fontsize=6,
        title_fontsize=7,
        loc="lower right",
        bbox_to_anchor=(1.0, 0.0),
        borderaxespad=0.0,
        framealpha=0.92,
    )


def per_bracket_pareto_assets(
    points: pd.DataFrame,
    grouping_columns: list[str] | None = None,
    risk_column: str = "efficiency_gain_risk_stddev",
    reward_column: str = "median_efficiency_gain",
) -> pd.DataFrame:
    """Select low-risk, optimum, and high-gain assets independently per employee bracket."""
    if points.empty:
        return pd.DataFrame()
    grouping_columns = grouping_columns or []
    group_columns = [column for column in grouping_columns if column in points.columns and column != "users"] + ["users"]
    rows = []
    for group_key, group in points.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_metadata = dict(zip(group_columns, group_key, strict=True))
        users = group_metadata["users"]

        frontier = efficient_frontier(group, risk_column, reward_column)
        if frontier.empty:
            continue
        scored = frontier.copy()
        scored["risk_stddev"] = scored[risk_column]
        scored["median_return_rate_change"] = scored[reward_column]
        for portfolio in PORTFOLIOS:
            chosen, rule = _choose(scored, portfolio)
            row = chosen.drop(labels=["risk_stddev", "median_return_rate_change"], errors="ignore").to_dict()
            row["portfolio"] = portfolio
            row["selection_rule"] = rule
            row.update(group_metadata)
            row["employee_bracket"] = int(users)
            row["service_type"] = _service_type(chosen)
            rows.append(row)
    return pd.DataFrame(rows)


def write_per_bracket_pareto_csvs(assets: pd.DataFrame, output_dir: Path) -> None:
    assets.to_csv(output_dir / "portfolio_per_bracket_pareto_assets.csv", index=False)
    for portfolio in PORTFOLIOS:
        assets[assets["portfolio"] == portfolio].to_csv(
            output_dir / f"portfolio_per_bracket_{portfolio}_pareto_assets.csv", index=False
        )


def write_pareto_markdown_table(points: pd.DataFrame, output_path: Path) -> None:
    """Write a Markdown matrix grouped by shock assumptions and work mix."""
    stochastic_shocks = (
        "shock_combination" in points
        and points["shock_combination"].fillna("none").ne("none").any()
    )
    grouping = (
        ["company_profile", "shock_combination", "users"]
        if stochastic_shocks
        else ["company_profile", "capability_plateau_after_18_months", "token_cost", "shock_combination", "users"]
    )
    annualized_return_columns = {
        "median_annualized_return_rate",
        "annualized_return_rate_risk_stddev",
    }
    show_annualized_returns = annualized_return_columns.issubset(points.columns)
    assets = per_bracket_pareto_assets(
        points,
        grouping,
        "annualized_return_rate_risk_stddev" if show_annualized_returns else "efficiency_gain_risk_stddev",
        "median_annualized_return_rate" if show_annualized_returns else "median_efficiency_gain",
    )
    if assets.empty:
        return
    if stochastic_shocks:
        assets["shock_assumption"] = assets["shock_combination"].map(
            lambda combination: f"shocks: {combination}"
        )
    else:
        assets["shock_assumption"] = assets.apply(
            lambda row: f"{row['token_cost']}; {'plateau after 18 months' if bool(row['capability_plateau_after_18_months']) else 'continuous capability growth'}; shocks: {row.get('shock_combination', 'none')}",
            axis=1,
        )
    assets["work_mix"] = assets["company_profile"]
    asset_labels = {
        "global_service": "Global API",
        "european_service": "EU API",
        "local_hardware": "On-prem",
        "eu_cloud_oss": "EU OSS cloud",
    }
    assets["asset_label"] = assets["service_provider"].map(asset_labels).fillna(assets["hardware"])
    brackets = sorted(int(value) for value in assets["employee_bracket"].unique())
    portfolio_labels = {"low_risk": "Low risk", "optimum": "Optimum", "high_gain": "High return"}
    lines = [
        "# Per-bracket Pareto assets",
        "",
        (
            "Each cell shows `asset (annualized return rate, annualized return-rate SD)` for the selected per-bracket Pareto asset."
            if show_annualized_returns
            else "Each cell shows `asset (return, risk)` for the selected per-bracket Pareto asset. Return is median per-employee efficiency gain; risk is its standard deviation."
        ),
        "",
        "| Shock assumptions | Work mix | " + " | ".join(f"{users} employees" for users in brackets for _ in PORTFOLIOS) + " |",
        "|---|---|" + "|".join("---" for _ in range(len(brackets) * len(PORTFOLIOS))) + "|",
        "|  |  | " + " | ".join(portfolio_labels[portfolio] for _ in brackets for portfolio in PORTFOLIOS) + " |",
    ]
    for (shock, work_mix), group in assets.groupby(["shock_assumption", "work_mix"], sort=True):
        cells = []
        for users in brackets:
            for portfolio in PORTFOLIOS:
                match = group[(group["employee_bracket"] == users) & (group["portfolio"] == portfolio)]
                if match.empty:
                    cells.append("—")
                    continue
                row = match.iloc[0]
                if show_annualized_returns:
                    cells.append(
                        f"{row['asset_label']} ({row['median_annualized_return_rate']:.3f}, {row['annualized_return_rate_risk_stddev']:.3f})"
                    )
                else:
                    cells.append(f"{row['asset_label']} ({row['median_efficiency_gain']:.3f}, {row['efficiency_gain_risk_stddev']:.3f})")
        lines.append(f"| {shock} | {work_mix} | " + " | ".join(cells) + " |")
    output_path.write_text("\n".join(lines) + "\n")


def _plot_pareto_assets(ax, assets: pd.DataFrame, points: pd.DataFrame, x: str, y: str) -> None:
    if assets is None or assets.empty or x not in assets or y not in assets:
        return
    colors = _employee_colors(points)
    for _, row in assets.iterrows():
        if not np.isfinite(row[x]) or not np.isfinite(row[y]):
            continue
        ax.scatter(
            [row[x]], [row[y]], marker=SERVICE_MARKERS[row["service_type"]],
            s=78, color=colors.get(int(row["employee_bracket"]), "#111827"),
            edgecolor="#111827", linewidth=2.2, zorder=8,
        )


def _add_pareto_legend(ax) -> None:
    handle = Line2D([], [], marker="o", linestyle="", color="#9ca3af", markeredgecolor="#111827", markeredgewidth=2.0, label="Per-bracket Pareto asset")
    legend = ax.legend(handles=[handle], fontsize=6, loc="lower center")
    ax.add_artist(legend)


def plot_month_grid(selected: pd.DataFrame, output_dir: Path) -> None:
    months = sorted(selected["month"].unique())
    columns = min(4, len(months))
    rows = int(np.ceil(len(months) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.5 * columns, 3.6 * rows), squeeze=False)
    colors = {"low_risk": "#2563eb", "optimum": "#059669", "high_gain": "#dc2626"}
    bracket_colors = _employee_colors(selected)
    for ax, month in zip(axes.flat, months, strict=False):
        panel = selected[selected["month"] == month]
        for portfolio in PORTFOLIOS:
            values = panel[panel["portfolio"] == portfolio]
            ax.scatter(values["annualized_return_rate_risk_stddev"], values["median_annualized_return_rate"], s=8, alpha=0.20, color=colors[portfolio], label=portfolio.replace("_", " "))
        _plot_selected_service_medians(ax, panel, "annualized_return_rate_risk_stddev", "median_annualized_return_rate", bracket_colors)
        ax.set_title(f"Month {month}")
        ax.set_xlabel("Annualized return-rate risk (SD)")
        ax.set_ylabel("Median annualized return rate")
    for ax in axes.flat[len(months):]:
        ax.set_visible(False)
    _combined_legend(
        axes.flat[0],
        selected,
        [Line2D([], [], marker="o", linestyle="", color=color, label=label.replace("_", " ")) for label, color in {"low_risk": "#2563eb", "optimum": "#059669", "high_gain": "#dc2626"}.items()],
    )
    fig.tight_layout()
    fig.savefig(output_dir / "portfolio_month_by_month_grid.png", dpi=180)
    plt.close(fig)


def plot_animation(selected: pd.DataFrame, output_dir: Path) -> None:
    try:
        import plotly.express as px
    except ImportError:
        return
    values = selected.copy()
    values["portfolio"] = values["portfolio"].str.replace("_", " ")
    figure = px.scatter(
        values,
        x="annualized_return_rate_risk_stddev",
        y="median_annualized_return_rate",
        color="portfolio",
        animation_frame="month",
        hover_name="column_label",
        title="Portfolio annualized return rates by simulated month",
        labels={
            "annualized_return_rate_risk_stddev": "Annualized return-rate risk (SD)",
            "median_annualized_return_rate": "Median annualized return rate",
        },
    )
    figure.write_html(output_dir / "portfolio_risk_return_animation.html", include_plotlyjs="cdn")


def plot_efficiency_risk_frontier(points: pd.DataFrame, output_dir: Path, one_column: bool = False, selected: pd.DataFrame | None = None, pareto_assets: pd.DataFrame | None = None, symlog: bool = False) -> None:
    """Plot the final-period annualized return-rate frontier."""
    if points.empty:
        return
    risk = "annualized_return_rate_risk_stddev"
    reward = "median_annualized_return_rate"
    frontier = efficient_frontier(points, risk, reward)
    positive_risk = frontier[frontier[risk] > 0.0].copy()
    if positive_risk.empty:
        tangency = None
    else:
        tangency = positive_risk.loc[(positive_risk[reward] / positive_risk[risk]).idxmax()]

    fig, ax = plt.subplots(figsize=(3.5, 3.0) if one_column else (8.2, 5.8))
    ax.scatter(points[risk], points[reward], s=5 if one_column else 8, color="#111827", alpha=0.18, linewidths=0, label="Individual simulated portfolios")
    if selected is not None:
        _plot_selected_service_medians(ax, selected, risk, reward, _employee_colors(points))
    ax.plot(frontier[risk], frontier[reward], color="#2563eb", linewidth=1.5 if one_column else 2.2, marker="o", markersize=2.5 if one_column else 3.5, label="Efficient frontier")
    if selected is not None:
        _plot_pareto_assets(ax, pareto_assets, points, risk, reward)
    if tangency is not None:
        end_risk = max(float(points[risk].max()), float(tangency[risk])) * 1.05
        slope = float(tangency[reward]) / float(tangency[risk])
        ax.plot([0.0, end_risk], [0.0, slope * end_risk], "--", color="#7c3aed", linewidth=1.4, label="Best risk-adjusted allocation")
        ax.scatter([tangency[risk]], [tangency[reward]], s=42, color="#7c3aed", zorder=4)
    _apply_frontier_scale(ax, points[risk], points[reward], symlog)
    reward_min, reward_max = float(points[reward].min()), float(points[reward].max())
    reward_margin = max(1e-6, (reward_max - reward_min) * 0.06)
    ax.set_ylim(reward_min - reward_margin, reward_max + reward_margin)
    ax.axvline(0.0, color="#6b7280", linewidth=0.8)
    ax.set_xlabel("Annualized return-rate risk (SD)", fontsize=8 if one_column else None)
    ax.set_ylabel("Median annualized return rate", fontsize=8 if one_column else None)
    if not one_column:
        ax.set_title("Final-period annualized return-rate frontier" + (" (symlog axes)" if symlog else ""))
    ax.tick_params(labelsize=7 if one_column else None)
    extra_handles = [
        Line2D([], [], color="#111827", linewidth=1.8, label="Efficient frontier"),
    ]
    if tangency is not None:
        extra_handles.append(Line2D([], [], color="#7c3aed", linestyle="--", label="Best risk-adjusted allocation"))
    _combined_legend(ax, points, extra_handles)
    fig.tight_layout()
    suffix = ("_symlog" if symlog else "") + ("_one_column" if one_column else "")
    fig.savefig(output_dir / f"portfolio_efficiency_risk_frontier{suffix}.png", dpi=300 if one_column else 180)
    plt.close(fig)


def plot_efficiency_risk_frontier_by_asset_mix(
    points: pd.DataFrame,
    output_dir: Path,
    simulation_months: float,
    one_column: bool = False,
    selected: pd.DataFrame | None = None,
    pareto_assets: pd.DataFrame | None = None,
    symlog: bool = False,
) -> None:
    """Show the final frontier while encoding company size and spending mix."""
    if points.empty:
        return
    risk = "annualized_return_rate_risk_stddev"
    reward = "median_annualized_return_rate"
    values = points.copy()
    service_commitment = values["service_budget_per_person_usd"] * values["users"] * simulation_months
    hardware_share = values["hardware_budget_usd"] / (values["hardware_budget_usd"] + service_commitment)
    values["asset_mix"] = np.select(
        [hardware_share >= 2.0 / 3.0, hardware_share <= 1.0 / 3.0],
        ["Hardware-heavy", "Service-heavy"],
        default="Mixed",
    )
    colors = dict(zip(sorted(values["users"].unique()), plt.get_cmap("tab10").colors, strict=False))
    markers = {"Hardware-heavy": "^", "Service-heavy": "o", "Mixed": "s"}
    frontier = efficient_frontier(values, risk, reward)

    fig, ax = plt.subplots(figsize=(3.5, 3.0) if one_column else (8.2, 5.8))
    for (users, asset_mix), group in values.groupby(["users", "asset_mix"], sort=True):
        ax.scatter(
            group[risk],
            group[reward],
            s=7 if one_column else 13,
            marker=markers[asset_mix],
            color=colors[users],
            alpha=0.18,
            linewidths=0,
        )
    if selected is not None:
        _plot_selected_service_medians(ax, selected, risk, reward, _employee_colors(values if "values" in locals() else points))
    ax.plot(frontier[risk], frontier[reward], color="#111827", linewidth=1.4 if one_column else 2.0, zorder=3)
    if selected is not None:
        _plot_pareto_assets(ax, pareto_assets, values, risk, reward)
    _apply_frontier_scale(ax, values[risk], values[reward], symlog)
    reward_min, reward_max = float(values[reward].min()), float(values[reward].max())
    ax.set_ylim(reward_min - max(1e-6, (reward_max - reward_min) * 0.06), reward_max + max(1e-6, (reward_max - reward_min) * 0.06))
    ax.axvline(0.0, color="#6b7280", linewidth=0.8)
    ax.set_xlabel("Annualized return-rate risk (SD)", fontsize=8 if one_column else None)
    ax.set_ylabel("Median annualized return rate", fontsize=8 if one_column else None)
    if not one_column:
        ax.set_title("Annualized return-rate frontier by company size and asset mix" + (" (symlog axes)" if symlog else ""))
    mix_handles = [Line2D([], [], marker=marker, linestyle="", color="#374151", label=f"Asset mix: {asset_mix}") for asset_mix, marker in markers.items()]
    _combined_legend(ax, values, mix_handles)
    ax.tick_params(labelsize=7 if one_column else None)
    fig.tight_layout()
    suffix = ("_symlog" if symlog else "") + ("_one_column" if one_column else "")
    fig.savefig(output_dir / f"portfolio_efficiency_risk_frontier_by_asset_mix{suffix}.png", dpi=300 if one_column else 180)
    plt.close(fig)


def _hyperbolic_frontier_curve(
    frontier: pd.DataFrame,
    risk: str,
    reward: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the left-opening branch x(y) of a size-specific hyperbola."""
    values = frontier.sort_values(reward)
    risks = values[risk].to_numpy(float)
    rewards = values[reward].to_numpy(float)
    if len(values) < 3 or np.ptp(rewards) <= 1e-12:
        return risks, rewards

    def hyperbola(y: np.ndarray, vertex: float, curvature: float, center: float, spread: float) -> np.ndarray:
        return vertex - curvature * (np.sqrt(1.0 + ((y - center) / spread) ** 2) - 1.0)

    risk_span = max(np.ptp(risks), 1e-9)
    reward_span = max(np.ptp(rewards), 1e-9)
    try:
        parameters, _ = curve_fit(
            hyperbola,
            rewards,
            risks,
            p0=(float(risks.max()), risk_span, float(np.median(rewards)), reward_span / 2.0),
            bounds=(
                [-np.inf, 1e-9, rewards.min() - reward_span, 1e-9],
                [np.inf, np.inf, rewards.max() + reward_span, np.inf],
            ),
            maxfev=20_000,
        )
    except (RuntimeError, ValueError):
        return risks, rewards

    curve_rewards = np.linspace(float(rewards.min()), float(rewards.max()), 160)
    curve_risks = hyperbola(curve_rewards, *parameters)
    return curve_risks, curve_rewards


def plot_efficiency_risk_frontier_by_size(points: pd.DataFrame, output_dir: Path, one_column: bool = False, selected: pd.DataFrame | None = None, pareto_assets: pd.DataFrame | None = None, symlog: bool = False) -> None:
    """Show one hyperbola-style annualized return-rate frontier per company size."""
    if points.empty:
        return
    risk = "annualized_return_rate_risk_stddev"
    reward = "median_annualized_return_rate"
    colors = dict(zip(sorted(points["users"].unique()), plt.get_cmap("tab10").colors, strict=False))
    fig, ax = plt.subplots(figsize=(3.5, 3.0) if one_column else (8.2, 5.8))
    for users, group in points.groupby("users", sort=True):
        color = colors[users]
        ax.scatter(group[risk], group[reward], s=6 if one_column else 11, color=color, alpha=0.14, linewidths=0)
        frontier = efficient_frontier(group, risk, reward)
        curve_risk, curve_reward = _hyperbolic_frontier_curve(frontier, risk, reward)
        curve_risk = np.array(curve_risk, copy=True)
        fitted_risks = np.interp(group[reward].to_numpy(float), curve_reward, curve_risk)
        curve_risk -= max(0.0, float((fitted_risks - group[risk].to_numpy(float)).max())) + 1e-12
        ax.plot(curve_risk, curve_reward, color=color, linewidth=1.5 if one_column else 2.2)
    if selected is not None:
        _plot_selected_service_medians(ax, selected, risk, reward, _employee_colors(values if "values" in locals() else points))
    reward_min, reward_max = float(points[reward].min()), float(points[reward].max())
    reward_margin = max(1e-6, (reward_max - reward_min) * 0.06)
    ax.set_ylim(reward_min - reward_margin, reward_max + reward_margin)
    ax.axvline(0.0, color="#6b7280", linewidth=0.8)
    ax.set_xlabel("Annualized return-rate risk (SD)", fontsize=8 if one_column else None)
    ax.set_ylabel("Median annualized return rate", fontsize=8 if one_column else None)
    if selected is not None:
        _plot_pareto_assets(ax, pareto_assets, points, risk, reward)
    _apply_frontier_scale(ax, points[risk], points[reward], symlog)
    if not one_column:
        ax.set_title("Hyperbolic annualized return-rate frontiers by company size" + (" (symlog axes)" if symlog else ""))
    ax.tick_params(labelsize=7 if one_column else None)
    _combined_legend(ax, points, [Line2D([], [], color="#111827", linewidth=1.8, label="Hyperbolic frontier")])
    fig.tight_layout()
    suffix = ("_symlog" if symlog else "") + ("_one_column" if one_column else "")
    fig.savefig(output_dir / f"portfolio_efficiency_risk_frontier_by_size{suffix}.png", dpi=300 if one_column else 180)
    plt.close(fig)


def plot_efficiency_risk_frontier_by_size_empirical(points: pd.DataFrame, output_dir: Path, one_column: bool = False, selected: pd.DataFrame | None = None, pareto_assets: pd.DataFrame | None = None, symlog: bool = False) -> None:
    """Show the observed annualized return-rate frontier points for each company size."""
    if points.empty:
        return
    risk = "annualized_return_rate_risk_stddev"
    reward = "median_annualized_return_rate"
    colors = dict(zip(sorted(points["users"].unique()), plt.get_cmap("tab10").colors, strict=False))
    fig, ax = plt.subplots(figsize=(3.5, 3.0) if one_column else (8.2, 5.8))
    for users, group in points.groupby("users", sort=True):
        color = colors[users]
        ax.scatter(group[risk], group[reward], s=6 if one_column else 11, color=color, alpha=0.14, linewidths=0)
        frontier = efficient_frontier(group, risk, reward).sort_values(risk)
        ax.plot(
            frontier[risk],
            frontier[reward],
            color=color,
            linewidth=1.5 if one_column else 2.2,
            marker="o",
            markersize=2.5 if one_column else 3.5,
        )
    if selected is not None:
        _plot_selected_service_medians(ax, selected, risk, reward, _employee_colors(values if "values" in locals() else points))
    reward_min, reward_max = float(points[reward].min()), float(points[reward].max())
    reward_margin = max(1e-6, (reward_max - reward_min) * 0.06)
    ax.set_ylim(reward_min - reward_margin, reward_max + reward_margin)
    ax.axvline(0.0, color="#6b7280", linewidth=0.8)
    ax.set_xlabel("Annualized return-rate risk (SD)", fontsize=8 if one_column else None)
    ax.set_ylabel("Median annualized return rate", fontsize=8 if one_column else None)
    if selected is not None:
        _plot_pareto_assets(ax, pareto_assets, points, risk, reward)
    _apply_frontier_scale(ax, points[risk], points[reward], symlog)
    if not one_column:
        ax.set_title("Observed annualized return-rate frontiers by company size" + (" (symlog axes)" if symlog else ""))
    ax.tick_params(labelsize=7 if one_column else None)
    _combined_legend(ax, points, [Line2D([], [], color="#111827", linewidth=1.8, marker="o", markersize=3, label="Observed Pareto frontier")])
    fig.tight_layout()
    suffix = ("_symlog" if symlog else "") + ("_one_column" if one_column else "")
    fig.savefig(output_dir / f"portfolio_efficiency_risk_frontier_by_size_empirical{suffix}.png", dpi=300 if one_column else 180)
    plt.close(fig)


def _bands(values: pd.DataFrame, group_cols: list[str], metric: str) -> pd.DataFrame:
    return (
        values.groupby(group_cols, sort=True)[metric]
        .agg(median="median", p10=lambda series: series.quantile(0.10), p90=lambda series: series.quantile(0.90))
        .reset_index()
    )


def plot_faceted_timeseries(selected: pd.DataFrame, output_dir: Path, metric: str, filename: str, ylabel: str, title: str) -> None:
    """Show the distribution over budgets and confidentiality settings for each size."""
    headcounts = sorted(selected["users"].unique())
    fig, axes = plt.subplots(len(PORTFOLIOS), len(headcounts), figsize=(3.5 * len(headcounts), 2.8 * len(PORTFOLIOS)), sharex=True)
    colors = {False: "#2563eb", True: "#dc2626"}
    labels = {False: "Continuous improvement", True: "Plateau after 18 months"}
    bands = _bands(selected, ["portfolio", "users", "capability_plateau_after_18_months", "month"], metric)
    for row, portfolio in enumerate(PORTFOLIOS):
        for column, users in enumerate(headcounts):
            ax = axes[row, column]
            panel = bands[(bands["portfolio"] == portfolio) & (bands["users"] == users)]
            for plateau, values in panel.groupby("capability_plateau_after_18_months", sort=False):
                values = values.sort_values("month")
                ax.fill_between(values["month"], values["p10"], values["p90"], color=colors[plateau], alpha=0.14)
                ax.plot(values["month"], values["median"], color=colors[plateau], linewidth=1.7, label=labels[plateau])
            if row == 0:
                ax.set_title(f"{users} employees")
            if column == 0:
                ax.set_ylabel(f"{portfolio.replace('_', ' ')}\n{ylabel}")
            if row == len(PORTFOLIOS) - 1:
                ax.set_xlabel("Month")
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle(f"{title} across service, hardware, and confidentiality settings", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_heatmaps(selected: pd.DataFrame, output_dir: Path, metric: str, filename: str, title: str) -> None:
    final_period = selected[selected["period"] == selected["period"].max()]
    values = (
        final_period.groupby(["portfolio", "users", "service_budget_per_person_usd", "hardware_budget_usd"], sort=True)[metric]
        .median()
        .reset_index()
    )
    headcounts = sorted(values["users"].unique())
    fig, axes = plt.subplots(
        len(PORTFOLIOS),
        len(headcounts),
        figsize=(3.8 * len(headcounts), 2.8 * len(PORTFOLIOS)),
        squeeze=False,
        layout="constrained",
    )
    images: list[object] = []
    for row, portfolio in enumerate(PORTFOLIOS):
        for column, users in enumerate(headcounts):
            ax = axes[row, column]
            panel = values[(values["portfolio"] == portfolio) & (values["users"] == users)]
            pivot = panel.pivot(index="hardware_budget_usd", columns="service_budget_per_person_usd", values=metric).sort_index().sort_index(axis=1)
            bracket_values = values[values["users"] == users][metric]
            image = ax.imshow(
                pivot.to_numpy(),
                origin="lower",
                aspect="auto",
                vmin=bracket_values.min(),
                vmax=bracket_values.max(),
                cmap="viridis",
            )
            if row == 0:
                images.append(image)
            ax.set_xticks(range(len(pivot.columns)), [f"${value:g}" for value in pivot.columns], rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(pivot.index)), [f"${value / 1000:g}k" for value in pivot.index], fontsize=7)
            if row == 0:
                ax.set_title(f"{users} employees")
            if column == 0:
                ax.set_ylabel(f"{portfolio.replace('_', ' ')}\nupfront hardware")
            if row == len(PORTFOLIOS) - 1:
                ax.set_xlabel("Service/person/month")
    for column, image in enumerate(images):
        fig.colorbar(image, ax=axes[:, column], location="right", shrink=0.82, label=title)
    fig.suptitle(f"Final-period {title}: median across confidentiality and capability paths")
    fig.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_plateau_comparison(selected: pd.DataFrame, output_dir: Path) -> None:
    bands = _bands(selected, ["portfolio", "capability_plateau_after_18_months", "month"], "median_annualized_return_rate")
    fig, axes = plt.subplots(1, len(PORTFOLIOS), figsize=(5 * len(PORTFOLIOS), 4), sharey=True)
    colors = {False: "#2563eb", True: "#dc2626"}
    labels = {False: "Continuous improvement", True: "Plateau after 18 months"}
    for ax, portfolio in zip(axes, PORTFOLIOS, strict=True):
        panel = bands[bands["portfolio"] == portfolio]
        for plateau, values in panel.groupby("capability_plateau_after_18_months", sort=False):
            values = values.sort_values("month")
            ax.fill_between(values["month"], values["p10"], values["p90"], color=colors[plateau], alpha=0.15)
            ax.plot(values["month"], values["median"], color=colors[plateau], linewidth=2, label=labels[plateau])
        ax.axvline(18, color="#6b7280", linewidth=1, linestyle="--")
        ax.set_title(portfolio.replace("_", " "))
        ax.set_xlabel("Month")
    axes[0].set_ylabel("Median annualized return rate")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "portfolio_plateau_comparison.png", dpi=180)
    plt.close(fig)


def plot_all(
    selected: pd.DataFrame,
    output_dir: Path,
    frontier_points: pd.DataFrame | None = None,
    one_column: bool = False,
    simulation_months: float = 36.0,
    pareto_assets: pd.DataFrame | None = None,
) -> None:
    plot_month_grid(selected, output_dir)
    plot_animation(selected, output_dir)
    plot_faceted_timeseries(
        selected,
        output_dir,
        "median_annualized_return_rate",
        "portfolio_faceted_return_rate_timeseries.png",
        "median annualized return rate",
        "Annualized return-rate distributions",
    )
    plot_faceted_timeseries(
        selected,
        output_dir,
        "annualized_return_rate_risk_stddev",
        "portfolio_faceted_risk_timeseries.png",
        "annualized return-rate risk (SD)",
        "Annualized return-rate risk distributions",
    )
    plot_sensitivity_heatmaps(selected, output_dir, "median_annualized_return_rate", "portfolio_final_return_sensitivity.png", "median annualized return rate")
    plot_sensitivity_heatmaps(selected, output_dir, "annualized_return_rate_risk_stddev", "portfolio_final_risk_sensitivity.png", "annualized return-rate risk (SD)")
    plot_plateau_comparison(selected, output_dir)
    if frontier_points is not None:
        pareto_assets = (
            per_bracket_pareto_assets(
                frontier_points,
                risk_column="annualized_return_rate_risk_stddev",
                reward_column="median_annualized_return_rate",
            )
            if pareto_assets is None
            else pareto_assets
        )
        plot_efficiency_risk_frontier(frontier_points, output_dir, one_column, selected, pareto_assets)
        plot_efficiency_risk_frontier(frontier_points, output_dir, one_column, selected, pareto_assets, symlog=True)
        plot_efficiency_risk_frontier_by_asset_mix(frontier_points, output_dir, simulation_months, one_column, selected, pareto_assets)
        plot_efficiency_risk_frontier_by_asset_mix(frontier_points, output_dir, simulation_months, one_column, selected, pareto_assets, symlog=True)
        plot_efficiency_risk_frontier_by_size(frontier_points, output_dir, one_column, selected, pareto_assets)
        plot_efficiency_risk_frontier_by_size(frontier_points, output_dir, one_column, selected, pareto_assets, symlog=True)
        plot_efficiency_risk_frontier_by_size_empirical(frontier_points, output_dir, one_column, selected, pareto_assets)
        plot_efficiency_risk_frontier_by_size_empirical(frontier_points, output_dir, one_column, selected, pareto_assets, symlog=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replot-only", action="store_true", help="Regenerate all plots from portfolio_scenario_manifest.csv without resimulating.")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write sweep CSVs without generating plots. Use this for parallel array workers.",
    )
    parser.add_argument(
        "--skip-combined-summary",
        action="store_true",
        help="Do not write the cross-profile summary in the parent output directory.",
    )
    parser.add_argument("--years", type=float, default=3.0)
    parser.add_argument("--resolution-months", type=int, default=3)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=4, help="Number of independent portfolio scenarios to simulate locally.")
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--employee-mix", type=str, default=None)
    parser.add_argument(
        "--company-profiles",
        type=str,
        default=",".join(COMPANY_PROFILES),
        help="Comma-separated work-mix profiles to run. Defaults to all built-in profiles.",
    )
    parser.add_argument("--engineering-context", choices=["bounded", "maintenance", "mixed"], default="mixed")
    parser.add_argument("--backend", choices=["numpy", "numba", "torch-mps", "auto"], default="auto")
    parser.add_argument("--usd-per-service-cost-index-quarter", type=float, default=60.0)
    parser.add_argument("--usd-per-hardware-capex-index", type=float, default=None)
    parser.add_argument("--hardware-calibration", choices=sorted(HARDWARE_CALIBRATIONS), default="h100-gemma4-31b")
    parser.add_argument("--embargo-shock-probability", type=float, default=0.05)
    parser.add_argument("--embargo-rollback-months", type=int, default=6)
    parser.add_argument(
        "--stochastic-shocks",
        action="store_true",
        help="Replace deterministic token-cost and plateau scenarios with persistent stochastic shock combinations.",
    )
    parser.add_argument("--sudden-break-even-probability-per-month", type=float, default=0.01)
    parser.add_argument("--gradual-break-even-probability-per-month", type=float, default=0.01)
    parser.add_argument("--capability-plateau-probability-per-month", type=float, default=0.01)
    parser.add_argument("--confidential-mixing-ratio", type=float, default=0.5)
    parser.add_argument("--confidential-mixing-penalty", type=float, default=0.5)
    parser.add_argument("--zero-risk-confidential-work-share", type=float, default=0.20)
    parser.add_argument("--disable-counterfactual-skill-growth", action="store_true")
    parser.add_argument("--counterfactual-skill-growth-rate-per-year", type=float, default=0.015)
    parser.add_argument("--cloud-oss-hours-per-usage-unit-period", type=float, default=None)
    parser.add_argument("--disable-it-support", action="store_true", help="Disable the default 0.5-FTE IT-support cost for small on-premise and OSS-cloud companies.")
    parser.add_argument("--it-support-max-users", type=int, default=50, help="Apply the IT-support cost at or below this employee count.")
    parser.add_argument(
        "--figure-width",
        choices=["default", "one-column"],
        default="default",
        help="Use one-column for the compact 3.5-inch-wide efficiency-risk frontier figure for a two-column paper.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("portfolio_sweep_outputs"))
    return parser.parse_args()


def parse_employee_mix(value: str | None) -> dict[str, float] | None:
    if value is None:
        return None
    return {item.split("=", 1)[0].strip(): float(item.split("=", 1)[1]) for item in value.split(",")}


def selected_company_profiles(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    if args.employee_mix is not None:
        return {"custom": parse_employee_mix(args.employee_mix) or {}}
    names = [name.strip() for name in args.company_profiles.split(",") if name.strip()]
    unknown = set(names) - set(COMPANY_PROFILES)
    if unknown:
        raise SystemExit(f"Unknown company profiles: {', '.join(sorted(unknown))}")
    return {name: COMPANY_PROFILES[name] for name in names}


def company_profile_index(name: str) -> int:
    """Return the stable seed index used by a named built-in profile."""
    try:
        return list(COMPANY_PROFILES).index(name)
    except ValueError:
        return 0


def run_company_profile(
    name: str,
    employee_mix: dict[str, float],
    base_config: SimulationConfig,
    args: argparse.Namespace,
    profile_index: int,
) -> None:
    output_dir = args.output_dir / name
    output_dir.mkdir(parents=True, exist_ok=True)
    config = replace(
        base_config,
        employee_mix=employee_mix,
        output_dir=output_dir,
        seed=base_config.seed + profile_index * 10_000_019,
    )
    scenarios = scenario_grid(config.stochastic_shocks)
    executor_cls = ThreadPoolExecutor if config.backend == "torch-mps" else ProcessPoolExecutor
    selected_frames = []
    frontier_point_frames = []
    with executor_cls(max_workers=min(args.concurrency, len(scenarios))) as executor:
        futures = {executor.submit(simulate_portfolio_scenario, scenario, config, index): scenario for index, scenario in enumerate(scenarios)}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Sweeping {name}", unit="scenario"):
            scenario = futures[future]
            results, metadata = future.result()
            summary = return_change_summary(results)
            selected = select_portfolios(summary, scenario, metadata)
            selected["company_profile"] = name
            selected_frames.append(selected)
            final_points = summary[summary["period"] == summary["period"].max()].copy()
            final_points["users"] = scenario.users
            final_points["service_budget_per_person_usd"] = scenario.service_budget_per_person_usd
            final_points["hardware_budget_usd"] = scenario.hardware_budget_usd
            final_points["confidential_document_share"] = scenario.confidential_share
            final_points["capability_plateau_after_18_months"] = scenario.plateau
            final_points["company_profile"] = name
            frontier_point_frames.append(final_points)
    selected = pd.concat(selected_frames, ignore_index=True).sort_values(
        ["users", "service_budget_per_person_usd", "hardware_budget_usd", "confidential_document_share", "capability_plateau_after_18_months", "month", "portfolio"]
    )
    selected.to_csv(output_dir / "portfolio_scenario_manifest.csv", index=False)
    frontier_points = pd.concat(frontier_point_frames, ignore_index=True)
    frontier_points.to_csv(output_dir / "portfolio_efficiency_risk_points.csv", index=False)
    pareto_assets = per_bracket_pareto_assets(
        frontier_points,
        risk_column="annualized_return_rate_risk_stddev",
        reward_column="median_annualized_return_rate",
    )
    write_per_bracket_pareto_csvs(pareto_assets, output_dir)
    write_wide_csvs(selected, output_dir)
    if not args.no_plots:
        plot_all(selected, output_dir, frontier_points, args.figure_width == "one-column", config.periods * config.resolution_months, pareto_assets)
    print(f"Wrote {name} portfolio sweep outputs to {output_dir}")


def main() -> None:
    args = parse_args()
    profiles = selected_company_profiles(args)
    if args.replot_only:
        output_dirs = [args.output_dir] if (args.output_dir / "portfolio_scenario_manifest.csv").exists() else [args.output_dir / name for name in profiles]
        missing = [output_dir for output_dir in output_dirs if not (output_dir / "portfolio_scenario_manifest.csv").exists()]
        if missing:
            raise SystemExit(f"Missing sweep manifest: {missing[0] / 'portfolio_scenario_manifest.csv'}")
        for output_dir in output_dirs:
            manifest = output_dir / "portfolio_scenario_manifest.csv"
            points_path = output_dir / "portfolio_efficiency_risk_points.csv"
            frontier_points = pd.read_csv(points_path) if points_path.exists() else None
            required_annualized_columns = {
                "median_annualized_return_rate",
                "annualized_return_rate_risk_stddev",
            }
            if frontier_points is not None and not required_annualized_columns.issubset(frontier_points.columns):
                raise SystemExit(
                    "This sweep predates annualized return-rate frontier data; rerun the simulation before replotting."
                )
            pareto_assets = (
                per_bracket_pareto_assets(
                    frontier_points,
                    risk_column="annualized_return_rate_risk_stddev",
                    reward_column="median_annualized_return_rate",
                )
                if frontier_points is not None
                else pd.DataFrame()
            )
            if frontier_points is not None:
                write_per_bracket_pareto_csvs(pareto_assets, output_dir)
            if not args.no_plots:
                plot_all(pd.read_csv(manifest), output_dir, frontier_points, args.figure_width == "one-column", args.years * 12, pareto_assets)
            print(f"Replotted outputs from {manifest}")
        if not args.skip_combined_summary:
            combined_points = pd.concat([pd.read_csv(output_dir / "portfolio_efficiency_risk_points.csv") for output_dir in output_dirs], ignore_index=True)
            write_pareto_markdown_table(combined_points, args.output_dir / "portfolio_per_bracket_pareto_table.md")
        return
    if args.resolution_months <= 0 or 18 % args.resolution_months != 0:
        raise SystemExit("--resolution-months must be a positive divisor of 18 so the plateau occurs exactly after 18 months.")
    if args.runs < 2:
        raise SystemExit("--runs must be at least 2 to calculate standard deviations.")
    if not 0.0 <= args.embargo_shock_probability <= 1.0:
        raise SystemExit("--embargo-shock-probability must be between 0 and 1.")
    if args.embargo_rollback_months < 0:
        raise SystemExit("--embargo-rollback-months must be non-negative.")
    for option_name in (
        "sudden_break_even_probability_per_month",
        "gradual_break_even_probability_per_month",
        "capability_plateau_probability_per_month",
    ):
        if not 0.0 <= getattr(args, option_name) <= 1.0:
            raise SystemExit(f"--{option_name.replace('_', '-')} must be between 0 and 1.")
    if not 0.0 <= args.confidential_mixing_ratio <= 1.0:
        raise SystemExit("--confidential-mixing-ratio must be between 0 and 1.")
    if not 0.0 <= args.confidential_mixing_penalty <= 1.0:
        raise SystemExit("--confidential-mixing-penalty must be between 0 and 1.")
    if not 0.0 <= args.zero_risk_confidential_work_share <= 1.0:
        raise SystemExit("--zero-risk-confidential-work-share must be between 0 and 1.")
    if args.counterfactual_skill_growth_rate_per_year < 0:
        raise SystemExit("--counterfactual-skill-growth-rate-per-year must be non-negative.")
    if args.cloud_oss_hours_per_usage_unit_period is not None and args.cloud_oss_hours_per_usage_unit_period <= 0:
        raise SystemExit("--cloud-oss-hours-per-usage-unit-period must be positive.")
    if args.it_support_max_users < 0:
        raise SystemExit("--it-support-max-users must be non-negative.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibration = _hardware_calibration_profile(args.hardware_calibration)
    if args.usd_per_hardware_capex_index is not None and args.usd_per_hardware_capex_index <= 0:
        raise SystemExit("--usd-per-hardware-capex-index must be positive.")
    hardware_index_usd = (
        args.usd_per_hardware_capex_index
        if args.usd_per_hardware_capex_index is not None
        else float(calibration["target_unit_usd"]) / HARDWARE_SCENARIOS["onprem_10pct_capacity"]["capex_units"]
    )
    base_config = SimulationConfig(
        years=args.years, resolution_months=args.resolution_months, runs=args.runs, seed=args.seed,
        plateau_quarter=18 // args.resolution_months, usd_per_service_cost_index_quarter=args.usd_per_service_cost_index_quarter,
        usd_per_hardware_capex_index=hardware_index_usd, hardware_calibration=args.hardware_calibration,
        onprem_capability_multiplier=float(calibration["onprem_capability_multiplier"]),
        engineering_context=args.engineering_context, backend=_resolve_backend(args.backend), output_dir=args.output_dir,
        embargo_shock_probability=args.embargo_shock_probability,
        embargo_rollback_months=args.embargo_rollback_months,
        stochastic_shocks=args.stochastic_shocks,
        sudden_break_even_probability_per_month=args.sudden_break_even_probability_per_month,
        gradual_break_even_probability_per_month=args.gradual_break_even_probability_per_month,
        capability_plateau_probability_per_month=args.capability_plateau_probability_per_month,
        confidential_mixing_ratio=args.confidential_mixing_ratio,
        confidential_mixing_penalty=args.confidential_mixing_penalty,
        zero_risk_confidential_work_share=args.zero_risk_confidential_work_share,
        counterfactual_skill_growth_enabled=not args.disable_counterfactual_skill_growth,
        counterfactual_skill_growth_rate_per_year=args.counterfactual_skill_growth_rate_per_year,
        cloud_oss_hours_per_usage_unit_period=args.cloud_oss_hours_per_usage_unit_period,
        it_support_enabled=not args.disable_it_support,
        it_support_max_users=args.it_support_max_users,
    )
    for name, employee_mix in profiles.items():
        run_company_profile(name, employee_mix, base_config, args, company_profile_index(name))
    if not args.skip_combined_summary:
        combined_points = pd.concat(
            [pd.read_csv(args.output_dir / name / "portfolio_efficiency_risk_points.csv") for name in profiles],
            ignore_index=True,
        )
        write_pareto_markdown_table(combined_points, args.output_dir / "portfolio_per_bracket_pareto_table.md")


if __name__ == "__main__":
    main()
