#!/usr/bin/env python3
"""Sweep company portfolios and export median return-rate changes and their risk."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import (
    ProcessPoolExecutor, ThreadPoolExecutor, as_completed)
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
from simulate_llm_efficiency import (
    ACCESS_PLANS, HARDWARE_CALIBRATIONS, HARDWARE_SCENARIOS,
    LOCAL_FALLBACK_POLICIES, MODEL_SCENARIOS, SHOCK_COMBINATIONS,
    SIMULATION_DEFAULTS, SWEEP_DEFAULTS, TOKEN_SCENARIOS, SimulationConfig,
    _hardware_calibration_profile, _resolve_backend, _simulate_one_scenario,
    configuration_path_from_argv, hardware_purchase_cost_usd,
    load_simulation_configuration)
from tqdm.auto import tqdm

HEADCOUNTS = (5, 25, 50, 100, 500, 2500)
SERVICE_BUDGETS_USD = (5.0, 10.0, 25.0, 50.0)
HARDWARE_BUDGETS_USD = (
    0.0,
    10_000.0,
    50_000.0,
    100_000.0,
    500_000.0,
    1_000_000.0,
)
CONFIDENTIAL_SHARES = (0.0, 0.1, 0.25, 0.5, 0.9)
PORTFOLIOS = ("low_risk", "optimum", "high_gain")
PERIOD_FRONTIER_PORTFOLIOS = ("low_risk", "medium_risk", "high_risk")
FINAL_SHOCK_COLUMNS = (
    ("plateau", "capability_plateau_shock"),
    ("price_gradual", "gradual_break_even_shock"),
    ("price_abrupt", "sudden_break_even_shock"),
    ("embargo", "embargo_shock"),
)
SERVICE_SLEEVES = {
    "global_service": "global_cloud",
    "european_service": "eu_cloud",
    "local_hardware": "onprem_oss",
    "eu_cloud_oss": "cloud_oss",
}
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
        "administration": 0.25,
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
    return max(
        affordable, key=lambda name: HARDWARE_SCENARIOS[name]["capacity"]
    )


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
    plateau = (
        "capability plateaus after 18 months"
        if scenario.plateau
        else "continuous capability improvement"
    )
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
        (
            (model, specification)
            for model, specification in MODEL_SCENARIOS.items()
            if specification["growth"]
        )
        if stochastic_shocks
        else MODEL_SCENARIOS.items()
    )
    token_scenarios = (
        ("flat",) if stochastic_shocks else tuple(TOKEN_SCENARIOS)
    )
    shock_combinations = SHOCK_COMBINATIONS if stochastic_shocks else ("none",)
    for model, model_specification in model_scenarios:
        if plateau and model_specification["growth"]:
            continue
        if not plateau and not model_specification["growth"]:
            continue
        if (
            hardware_name in ("cloud_oss",)
            and model_specification["kind"] != "oss"
        ):
            continue
        if (
            hardware_name not in ("cloud_api_only", "cloud_oss")
            and model_specification["kind"] != "oss"
        ):
            continue
        if hardware_name == "cloud_oss":
            for shock_combination in shock_combinations:
                rows.append(
                    (
                        model,
                        "flat",
                        hardware_name,
                        "maxed_out",
                        "pay_per_use",
                        "persona_choice",
                        "eu_cloud_oss",
                        shock_combination,
                    )
                )
            continue
        for token_cost in token_scenarios:
            for access_plan in ACCESS_PLANS:
                fallbacks = (
                    ("persona_choice",)
                    if hardware_name == "cloud_api_only"
                    else LOCAL_FALLBACK_POLICIES
                )
                providers = (
                    ("global_service", "european_service")
                    if hardware_name == "cloud_api_only"
                    else ("local_hardware",)
                )
                for fallback in fallbacks:
                    for service_provider in providers:
                        for shock_combination in shock_combinations:
                            rows.append(
                                (
                                    model,
                                    token_cost,
                                    hardware_name,
                                    "maxed_out",
                                    access_plan,
                                    fallback,
                                    service_provider,
                                    shock_combination,
                                )
                            )
    return rows


def simulate_portfolio_scenario(
    scenario: SweepScenario,
    base_config: SimulationConfig,
    scenario_index: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    hardware_calibration, hardware_name = hardware_selection_for_budget(
        scenario.hardware_budget_usd
    )
    calibration = _hardware_calibration_profile(hardware_calibration)
    config = SimulationConfig(
        users=scenario.users,
        years=base_config.years,
        resolution_months=base_config.resolution_months,
        runs=base_config.runs,
        concurrency=1,
        seed=base_config.seed + scenario_index * 100_003,
        plateau_quarter=(
            base_config.plateau_quarter if scenario.plateau else 10**6
        ),
        max_monthly_service_budget_usd=scenario.service_budget_per_person_usd
        * scenario.users,
        max_upfront_hardware_budget_usd=scenario.hardware_budget_usd,
        confidential_document_fraction=min(scenario.confidential_share, 0.9),
        zero_risk_confidential_work_share=base_config.zero_risk_confidential_work_share,
        counterfactual_skill_growth_enabled=base_config.counterfactual_skill_growth_enabled,
        counterfactual_skill_growth_rate_per_year=base_config.counterfactual_skill_growth_rate_per_year,
        usd_per_service_cost_index_quarter=base_config.usd_per_service_cost_index_quarter,
        usd_per_hardware_capex_index=base_config.usd_per_hardware_capex_index,
        hardware_calibration=hardware_calibration,
        onprem_capability_multiplier=float(
            calibration["onprem_capability_multiplier"]
        ),
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
    # Evaluate all four investable service sleeves under the same organisation
    # scenario; this is required for a covariance-aware service mix.
    asset_names = tuple(
        dict.fromkeys(("cloud_api_only", hardware_name, "cloud_oss"))
    )
    for asset_name in asset_names:
        for model_scenario in simulation_scenarios(
            asset_name,
            scenario.plateau,
            config.hardware_calibration,
            config.stochastic_shocks,
        ):
            # Every candidate asset starts each run from the same random seed.
            # Its run number can therefore be used as a paired future identifier
            # when comparing assets in an external portfolio analysis.
            frames.append(
                _simulate_one_scenario(
                    (
                        *model_scenario[:7],
                        config,
                        config.seed,
                        model_scenario[7],
                    )
                )
            )
    unit_cost_usd = hardware_purchase_cost_usd(
        hardware_calibration, hardware_name
    )
    replicas = (
        int(scenario.hardware_budget_usd // unit_cost_usd)
        if unit_cost_usd
        else 0
    )
    metadata = {
        "hardware_scenario": hardware_name,
        "hardware_calibration": hardware_calibration,
        "hardware_replicas": replicas,
        "hardware_cost_usd": unit_cost_usd * replicas,
    }
    return pd.concat(frames, ignore_index=True), metadata


ASSET_IDENTITY_COLUMNS = [
    "model",
    "token_cost",
    "access_plan",
    "hardware",
    "hardware_refresh",
    "local_fallback",
    "service_provider",
    "shock_combination",
]


def return_change_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Create one virtual representative asset per fixed identity and period.

    Return-rate changes remain available as a descriptive series. Risk is the
    cross-run standard deviation of each asset's return rate at the same month,
    matching the mean-variance definition used for portfolio selection.
    """
    run_cols = [*ASSET_IDENTITY_COLUMNS, "run"]
    ordered = results.sort_values([*run_cols, "period"]).copy()
    ordered["return_rate_change"] = (
        ordered.groupby(run_cols, sort=False)["return_rate"].diff().fillna(0.0)
    )
    group_cols = [*ASSET_IDENTITY_COLUMNS, "period", "month"]
    return (
        ordered.groupby(group_cols, sort=False)
        .agg(
            median_return_rate_change=("return_rate_change", "median"),
            risk_stddev=("return_rate", "std"),
            median_annualized_return_rate=("annualized_return_rate", "median"),
            annualized_return_rate_risk_stddev=(
                "annualized_return_rate",
                "std",
            ),
            median_efficiency_gain=("mean_efficiency_gain", "median"),
            efficiency_gain_risk_stddev=("mean_efficiency_gain", "std"),
            median_zero_risk_efficiency_gain=(
                "zero_risk_efficiency_gain",
                "median",
            ),
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
    finite = np.isfinite(points[risk_column]) & np.isfinite(
        points[reward_column]
    )
    ordered = points.loc[finite].sort_values(
        [risk_column, reward_column], ascending=[True, False]
    )
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
    risk_mean, risk_std = points[risk_column].mean(), points[risk_column].std(
        ddof=0
    )
    gain_mean, gain_std = points[reward_column].mean(), points[
        reward_column
    ].std(ddof=0)
    if portfolio == "low_risk":
        eligible = points[points[risk_column] <= risk_mean - risk_std]
        if eligible.empty:
            return (
                points.sort_values(
                    [risk_column, reward_column], ascending=[True, False]
                ).iloc[0],
                "minimum-risk fallback",
            )
        return (
            eligible.sort_values(
                [reward_column, risk_column], ascending=[False, True]
            ).iloc[0],
            "risk <= mean - 1 sigma",
        )
    if portfolio == "optimum":
        eligible = points[
            points[risk_column].between(
                risk_mean - risk_std, risk_mean + risk_std
            )
        ]
        if eligible.empty:
            eligible = points
            reason = "all-frontier fallback"
        else:
            reason = "risk within mean +/- 1 sigma"
        return (
            eligible.sort_values(
                [reward_column, risk_column], ascending=[False, True]
            ).iloc[0],
            reason,
        )
    eligible = points[points[reward_column] >= gain_mean + 3.0 * gain_std]
    if eligible.empty:
        return (
            points.sort_values(
                [reward_column, risk_column], ascending=[False, True]
            ).iloc[0],
            "maximum-gain fallback",
        )
    return (
        eligible.sort_values(
            [risk_column, reward_column], ascending=[True, False]
        ).iloc[0],
        "gain >= mean + 3 sigma",
    )


def _asset_label(row: pd.Series, scenario: SweepScenario) -> str:
    return (
        f"{scenario.users} employees | ${scenario.service_budget_per_person_usd:g}/person/month service | "
        f"${scenario.hardware_budget_usd:,.0f} upfront hardware | {scenario.confidential_share:.0%} confidential documents | "
        f"{'plateau after 18 months' if scenario.plateau else 'continuous capability improvement'} | "
        f"model={row['model']} | access={row['access_plan']} | hardware={row['hardware']} | "
        f"token={row['token_cost']} | refresh={row['hardware_refresh']} | fallback={row['local_fallback']} | "
        f"provider={row['service_provider']} | shocks={row['shock_combination']}"
    )


def select_portfolios(
    summary: pd.DataFrame, scenario: SweepScenario, metadata: dict[str, object]
) -> pd.DataFrame:
    """Choose three final-period frontier assets and retain their fixed histories."""
    final_period = summary["period"].max()
    frontier = efficient_frontier(
        summary[summary["period"] == final_period],
        "annualized_return_rate_risk_stddev",
        "median_annualized_return_rate",
    )
    if frontier.empty:
        return pd.DataFrame()

    rows = []
    for portfolio in PORTFOLIOS:
        selected, rule = _choose(
            frontier,
            portfolio,
            "annualized_return_rate_risk_stddev",
            "median_annualized_return_rate",
        )
        selected_history = summary.merge(
            selected[ASSET_IDENTITY_COLUMNS].to_frame().T,
            on=ASSET_IDENTITY_COLUMNS,
            how="inner",
        )
        asset_label = _asset_label(selected, scenario)
        for _, row in selected_history.iterrows():
            record = row.to_dict()
            record.update(
                portfolio=portfolio,
                selection_rule=rule,
                selection_period=final_period,
                asset_column_label=asset_label,
                column_label=friendly_label(scenario, portfolio)
                + " | "
                + asset_label,
                users=scenario.users,
                service_budget_per_person_usd=scenario.service_budget_per_person_usd,
                hardware_budget_usd=scenario.hardware_budget_usd,
                confidential_document_share=scenario.confidential_share,
                capability_plateau_after_18_months=scenario.plateau,
                **metadata,
            )
            rows.append(record)
    return pd.DataFrame(rows)


def write_wide_csvs(selected: pd.DataFrame, output_dir: Path) -> None:
    for metric, filename in (
        ("median_return_rate_change", "portfolio_rate_changes.csv"),
        ("risk_stddev", "portfolio_return_rate_risk.csv"),
    ):
        wide = selected.pivot(
            index="month", columns="column_label", values=metric
        ).reset_index()
        wide = wide.reindex(
            columns=["month", *sorted(selected["column_label"].unique())]
        )
        wide.to_csv(output_dir / filename, index=False)


def _best_return_per_risk(
    points: pd.DataFrame, risk_column: str, reward_column: str
) -> pd.Series:
    positive_risk = points[points[risk_column] > 0.0]
    if positive_risk.empty:
        return points.sort_values(
            [reward_column, risk_column], ascending=[False, True]
        ).iloc[0]
    ratios = positive_risk[reward_column] / positive_risk[risk_column]
    return positive_risk.loc[ratios.idxmax()]


def period_frontier_selections(
    summary: pd.DataFrame, access_plan: str | None = None
) -> pd.DataFrame:
    """Select three risk-band frontier assets independently at every period."""
    risk_column = "annualized_return_rate_risk_stddev"
    reward_column = "median_annualized_return_rate"
    selections = []
    if access_plan is not None:
        summary = summary[summary["access_plan"] == access_plan]
    for period, period_points in summary.groupby("period", sort=True):
        frontier = efficient_frontier(
            period_points, risk_column, reward_column
        )
        if frontier.empty:
            continue
        median_risk = frontier[risk_column].median()
        risk_stddev = frontier[risk_column].std(ddof=0)
        lower, upper = median_risk - risk_stddev, median_risk + risk_stddev
        bands = {
            "low_risk": frontier[frontier[risk_column] <= lower],
            "medium_risk": frontier[
                (frontier[risk_column] > lower)
                & (frontier[risk_column] <= upper)
            ],
            "high_risk": frontier[frontier[risk_column] > upper],
        }
        fallbacks = {
            "low_risk": frontier.sort_values(risk_column).head(1),
            "medium_risk": frontier.assign(
                _distance=(frontier[risk_column] - median_risk).abs()
            )
            .sort_values(["_distance", risk_column])
            .head(1),
            "high_risk": frontier.sort_values(
                risk_column, ascending=False
            ).head(1),
        }
        for portfolio in PERIOD_FRONTIER_PORTFOLIOS:
            candidates = bands[portfolio]
            selected = _best_return_per_risk(
                candidates if not candidates.empty else fallbacks[portfolio],
                risk_column,
                reward_column,
            )
            selections.append(
                {
                    "selection_period": period,
                    "portfolio": portfolio,
                    **selected.to_dict(),
                }
            )
    return pd.DataFrame(selections)


def period_frontier_return_rates(
    summary: pd.DataFrame, access_plan: str | None = None
) -> pd.DataFrame:
    """Return full histories for three risk-band frontier selections at every period."""
    reward_column = "median_annualized_return_rate"
    selected = period_frontier_selections(summary, access_plan)
    if selected.empty:
        return pd.DataFrame()
    selection_columns = [
        "selection_period",
        "portfolio",
        *ASSET_IDENTITY_COLUMNS,
    ]
    histories = summary.merge(
        selected[selection_columns], on=ASSET_IDENTITY_COLUMNS, how="inner"
    )
    histories["column_label"] = histories.apply(
        lambda row: f"period_{int(row['selection_period'])}_{row['portfolio']}",
        axis=1,
    )
    wide = histories.pivot(
        index=["period", "month"], columns="column_label", values=reward_column
    ).reset_index()
    columns = [
        "period",
        "month",
        *(
            f"period_{int(period)}_{portfolio}"
            for period in sorted(selected["selection_period"].unique())
            for portfolio in PERIOD_FRONTIER_PORTFOLIOS
        ),
    ]
    return wide.reindex(columns=columns)


def _configuration_path_value(value: float | int) -> str:
    return f"{value:g}".replace("-", "minus").replace(".", "p")


def write_period_frontier_return_rates(
    summary: pd.DataFrame,
    scenario: SweepScenario,
    output_dir: Path,
) -> list[Path]:
    """Write one full-horizon risk-band frontier CSV per access plan."""
    access_plans = (
        sorted(summary["access_plan"].dropna().unique())
        if "access_plan" in summary
        else [None]
    )
    outputs = []
    users = _configuration_path_value(scenario.users)
    hardware = _configuration_path_value(scenario.hardware_budget_usd)
    confidential = _configuration_path_value(scenario.confidential_share)
    service = _configuration_path_value(scenario.service_budget_per_person_usd)
    base_folder = (
        output_dir
        / "period_frontier_return_rates"
        / f"users_{users}"
        / f"max_hardware_invest_usd_{hardware}"
        / f"confidential_document_fraction_{confidential}"
    )
    for access_plan in access_plans:
        rates = period_frontier_return_rates(summary, access_plan)
        if rates.empty:
            continue
        folder = base_folder / (
            f"access_plan_{access_plan}" if access_plan else ""
        )
        filename = (
            f"users_{users}_max_hardware_invest_usd_{hardware}_"
            f"confidential_document_fraction_{confidential}_service_usd_{service}_"
            f"plateau_{str(scenario.plateau).lower()}"
            f"{f'_access_plan_{access_plan}' if access_plan else ''}.csv"
        )
        folder.mkdir(parents=True, exist_ok=True)
        output_path = folder / filename
        rates.to_csv(output_path, index=False)
        outputs.append(output_path)
    return outputs


def final_shock_frontier_return_rates(
    results: pd.DataFrame, access_plan: str | None = None
) -> pd.DataFrame:
    """Return full histories for final-horizon selections in every realized shock state."""
    final_period = results["period"].max()
    run_flags = (
        results.groupby([*ASSET_IDENTITY_COLUMNS, "run"], sort=False)[
            [column for _, column in FINAL_SHOCK_COLUMNS]
        ]
        .max()
        .reset_index()
    )
    final = (
        results[results["period"] == final_period]
        .drop(columns=[column for _, column in FINAL_SHOCK_COLUMNS])
        .merge(run_flags, on=[*ASSET_IDENTITY_COLUMNS, "run"], how="inner")
    )
    if access_plan is not None:
        final = final[final["access_plan"] == access_plan]

    summary = return_change_summary(results)
    histories = []
    columns = []
    for state in np.ndindex(*(2 for _ in FINAL_SHOCK_COLUMNS)):
        state_label = "_".join(
            f"{name}_{'yes' if enabled else 'no'}"
            for (name, _), enabled in zip(FINAL_SHOCK_COLUMNS, state)
        )
        subset = final.copy()
        for (_, column), enabled in zip(FINAL_SHOCK_COLUMNS, state):
            subset = subset[subset[column].astype(bool) == bool(enabled)]
        selected = (
            period_frontier_selections(
                subset.groupby(
                    [*ASSET_IDENTITY_COLUMNS, "period", "month"], sort=False
                )
                .agg(
                    annualized_return_rate_risk_stddev=(
                        "annualized_return_rate",
                        "std",
                    ),
                    median_annualized_return_rate=(
                        "annualized_return_rate",
                        "median",
                    ),
                )
                .reset_index()
                .fillna({"annualized_return_rate_risk_stddev": 0.0})
            )
            if not subset.empty
            else pd.DataFrame()
        )
        for portfolio in PERIOD_FRONTIER_PORTFOLIOS:
            column = f"{portfolio}_{state_label}"
            columns.append(column)
            if selected.empty:
                continue
            match = selected[selected["portfolio"] == portfolio]
            if match.empty:
                continue
            history = summary.merge(
                match.iloc[[0]][ASSET_IDENTITY_COLUMNS],
                on=ASSET_IDENTITY_COLUMNS,
                how="inner",
            )
            history["column_label"] = column
            histories.append(history)
    if not histories:
        return pd.DataFrame(columns=["period", "month", *columns])
    wide = (
        pd.concat(histories, ignore_index=True)
        .pivot(
            index=["period", "month"],
            columns="column_label",
            values="median_annualized_return_rate",
        )
        .reset_index()
    )
    return wide.reindex(columns=["period", "month", *columns])


def write_final_shock_frontier_return_rates(
    results: pd.DataFrame,
    scenario: SweepScenario,
    output_dir: Path,
) -> list[Path]:
    """Write one final-horizon, shock-split frontier CSV per access plan."""
    access_plans = (
        sorted(results["access_plan"].dropna().unique())
        if "access_plan" in results
        else [None]
    )
    outputs = []
    users = _configuration_path_value(scenario.users)
    hardware = _configuration_path_value(scenario.hardware_budget_usd)
    confidential = _configuration_path_value(scenario.confidential_share)
    service = _configuration_path_value(scenario.service_budget_per_person_usd)
    base_folder = (
        output_dir
        / "final_shock_frontier_return_rates"
        / f"users_{users}"
        / f"max_hardware_invest_usd_{hardware}"
        / f"confidential_document_fraction_{confidential}"
    )
    for access_plan in access_plans:
        rates = final_shock_frontier_return_rates(results, access_plan)
        folder = base_folder / (
            f"access_plan_{access_plan}" if access_plan else ""
        )
        filename = (
            f"users_{users}_max_hardware_invest_usd_{hardware}_"
            f"confidential_document_fraction_{confidential}_service_usd_{service}_"
            f"plateau_{str(scenario.plateau).lower()}"
            f"{f'_access_plan_{access_plan}' if access_plan else ''}.csv"
        )
        folder.mkdir(parents=True, exist_ok=True)
        output_path = folder / filename
        rates.to_csv(output_path, index=False)
        outputs.append(output_path)
    return outputs


def hardware_budget_sensitivity_records(
    summary: pd.DataFrame,
    scenario: SweepScenario,
    company_profile: str,
) -> pd.DataFrame:
    """Return final-period frontier selections used for hardware-budget sensitivity plots."""
    selected = period_frontier_selections(
        summary[summary["service_provider"] == "local_hardware"]
    )
    if selected.empty:
        return selected
    final_period = summary["period"].max()
    selected = selected[selected["selection_period"] == final_period].copy()
    selected["company_profile"] = company_profile
    selected["users"] = scenario.users
    selected["service_budget_per_person_usd"] = (
        scenario.service_budget_per_person_usd
    )
    selected["hardware_budget_usd"] = scenario.hardware_budget_usd
    selected["confidential_document_share"] = scenario.confidential_share
    selected["capability_plateau_after_18_months"] = scenario.plateau
    return selected


def hardware_budget_time_sensitivity_records(
    summary: pd.DataFrame,
    scenario: SweepScenario,
    company_profile: str,
) -> pd.DataFrame:
    """Return one local-hardware frontier selection per risk band and simulation period."""
    selected = period_frontier_selections(
        summary[summary["service_provider"] == "local_hardware"]
    )
    if selected.empty:
        return selected
    selected["company_profile"] = company_profile
    selected["users"] = scenario.users
    selected["service_budget_per_person_usd"] = (
        scenario.service_budget_per_person_usd
    )
    selected["hardware_budget_usd"] = scenario.hardware_budget_usd
    selected["confidential_document_share"] = scenario.confidential_share
    selected["capability_plateau_after_18_months"] = scenario.plateau
    return selected


def rebuild_hardware_budget_sensitivity(output_dir: Path) -> pd.DataFrame:
    """Rebuild hardware-only sensitivity records from persisted final-period frontier points."""
    points_path = output_dir / "portfolio_efficiency_risk_points.csv"
    if not points_path.exists():
        return pd.DataFrame()
    points = pd.read_csv(points_path)
    grouping = [
        "company_profile",
        "users",
        "service_budget_per_person_usd",
        "hardware_budget_usd",
        "confidential_document_share",
        "capability_plateau_after_18_months",
    ]
    records = []
    for key, group in points.groupby(grouping, sort=False):
        profile, users, service, hardware, confidential, plateau = key
        plateau_value = str(plateau).lower() == "true"
        scenario = SweepScenario(
            int(users),
            float(service),
            float(hardware),
            float(confidential),
            plateau_value,
        )
        selected = hardware_budget_sensitivity_records(
            group, scenario, str(profile)
        )
        if not selected.empty:
            records.append(selected)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def plot_hardware_budget_sensitivity(
    records: pd.DataFrame,
    output_dir: Path,
    users: int | None = None,
) -> list[Path]:
    """Plot hardware-only return/risk/user against hardware budget."""
    required = {
        "company_profile",
        "portfolio",
        "users",
        "service_budget_per_person_usd",
        "hardware_budget_usd",
        "confidential_document_share",
        "capability_plateau_after_18_months",
        "median_annualized_return_rate",
        "annualized_return_rate_risk_stddev",
    }
    if records.empty or not required.issubset(records.columns):
        return []
    values = records.copy()
    if users is not None:
        values = values[values["users"] == users].copy()
        if values.empty:
            return []
    risk = values["annualized_return_rate_risk_stddev"]
    values = values[np.isfinite(risk) & (risk > 0.0)].copy()
    metric = "return_risk" if users is not None else "return_risk_per_user"
    values[metric] = (
        values["median_annualized_return_rate"]
        / values["annualized_return_rate_risk_stddev"]
    )
    if users is None:
        values[metric] /= values["users"]
    plots_dir = output_dir / "hardware_budget_sensitivity"
    plots_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for column, preferred in (
        (
            "service_budget_per_person_usd",
            values["service_budget_per_person_usd"].min(),
        ),
        (
            "confidential_document_share",
            values["confidential_document_share"].min(),
        ),
        ("capability_plateau_after_18_months", False),
    ):
        if preferred in set(values[column]):
            values = values[values[column] == preferred]
        else:
            values = values[values[column] == values[column].min()]
    colors = dict(
        zip(
            sorted(values["company_profile"].unique()),
            plt.get_cmap("tab10").colors,
            strict=False,
        )
    )
    fig, axes = plt.subplots(
        1, len(PERIOD_FRONTIER_PORTFOLIOS), figsize=(13, 3.8), sharey=True
    )
    for axis, portfolio in zip(axes, PERIOD_FRONTIER_PORTFOLIOS, strict=True):
        panel = values[values["portfolio"] == portfolio]
        for profile, profile_values in panel.groupby(
            "company_profile", sort=True
        ):
            bands = (
                profile_values.groupby("hardware_budget_usd", as_index=False)[
                    metric
                ]
                .agg(
                    median="median",
                    p10=lambda values: values.quantile(0.1),
                    p90=lambda values: values.quantile(0.9),
                )
                .sort_values("hardware_budget_usd")
            )
            axis.fill_between(
                bands["hardware_budget_usd"],
                bands["p10"],
                bands["p90"],
                color=colors[profile],
                alpha=0.16,
            )
            axis.plot(
                bands["hardware_budget_usd"],
                bands["median"],
                color=colors[profile],
                linewidth=1.8,
                label=profile,
            )
        axis.set_title(portfolio.replace("_", " "))
        axis.set_xlabel("Maximum hardware investment (USD)")
        axis.set_xscale("symlog", linthresh=1_000.0)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel(
        "Median annualized return / risk SD"
        + ("" if users is not None else " / user")
    )
    axes[-1].legend(fontsize=7, loc="best")
    title = "Hardware-only budget sensitivity"
    subtitle = (
        f"Fixed at {users} users."
        if users is not None
        else "Lines are medians across company sizes; bands show the 10th–90th percentile."
    )
    fig.suptitle(f"{title}\n{subtitle}", fontsize=10)
    fig.tight_layout()
    path = plots_dir / (
        f"return_risk_users_{users}.png"
        if users is not None
        else "return_risk_per_user.png"
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def plot_hardware_budget_time_sensitivity(
    records: pd.DataFrame,
    output_dir: Path,
    color_limits: tuple[float, float] | None = None,
) -> Path | None:
    """Plot local-hardware return/risk by investment and simulation month."""
    required = {
        "company_profile",
        "portfolio",
        "users",
        "service_budget_per_person_usd",
        "hardware_budget_usd",
        "confidential_document_share",
        "capability_plateau_after_18_months",
        "month",
        "median_annualized_return_rate",
        "annualized_return_rate_risk_stddev",
    }
    if records.empty or not required.issubset(records.columns):
        return None
    values = records.copy()
    risk = values["annualized_return_rate_risk_stddev"]
    values = values[
        np.isfinite(risk)
        & (risk > 0.0)
        & (values["hardware_budget_usd"] > 0.0)
    ].copy()
    if values.empty:
        return None
    values["return_risk"] = (
        values["median_annualized_return_rate"]
        / values["annualized_return_rate_risk_stddev"]
    )
    for column, preferred in (
        (
            "service_budget_per_person_usd",
            values["service_budget_per_person_usd"].min(),
        ),
        (
            "confidential_document_share",
            values["confidential_document_share"].min(),
        ),
        ("capability_plateau_after_18_months", False),
    ):
        values = values[
            values[column]
            == (
                preferred
                if preferred in set(values[column])
                else values[column].min()
            )
        ]
    profiles = sorted(values["company_profile"].unique())
    if not profiles:
        return None
    limits = color_limits or tuple(
        np.nanpercentile(values["return_risk"], [2.0, 98.0])
    )
    figure, axes = plt.subplots(
        len(PERIOD_FRONTIER_PORTFOLIOS),
        len(profiles),
        figsize=(3.0 * len(profiles), 2.4 * len(PERIOD_FRONTIER_PORTFOLIOS)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    image = None
    for row, portfolio in enumerate(PERIOD_FRONTIER_PORTFOLIOS):
        for column, profile in enumerate(profiles):
            axis = axes[row, column]
            panel = values[
                (values["portfolio"] == portfolio)
                & (values["company_profile"] == profile)
            ]
            matrix = (
                panel.pivot_table(
                    index="hardware_budget_usd",
                    columns="month",
                    values="return_risk",
                    aggfunc="median",
                )
                .sort_index()
                .sort_index(axis=1)
            )
            image = axis.pcolormesh(
                matrix.columns.to_numpy(float),
                matrix.index.to_numpy(float),
                matrix.to_numpy(float),
                shading="nearest",
                cmap="viridis",
                vmin=limits[0],
                vmax=limits[1],
            )
            axis.set_yscale("log")
            if row == 0:
                axis.set_title(profile.replace("_", " "), fontsize=8)
            if column == 0:
                axis.set_ylabel(
                    f"{portfolio.replace('_', ' ')}\nHardware USD", fontsize=7
                )
            if row == len(PERIOD_FRONTIER_PORTFOLIOS) - 1:
                axis.set_xlabel("Month", fontsize=7)
            axis.tick_params(labelsize=6)
    figure.colorbar(
        image, ax=axes, label="Median annualized return / risk SD", shrink=0.82
    )
    figure.suptitle("Hardware-only sensitivity over time", fontsize=11)
    figure.tight_layout()
    plots_dir = output_dir / "hardware_budget_sensitivity"
    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / "return_risk_by_hardware_and_month.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def collect_hardware_budget_sensitivity(output_dir: Path) -> pd.DataFrame:
    """Combine per-profile sensitivity files, including independently run array jobs."""
    paths = sorted(
        output_dir.glob("*/portfolio_hardware_budget_sensitivity.csv")
    )
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def collect_hardware_budget_time_sensitivity(output_dir: Path) -> pd.DataFrame:
    paths = sorted(
        output_dir.glob("*/portfolio_hardware_budget_time_sensitivity.csv")
    )
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def paired_future_return_rates(
    results: pd.DataFrame,
    selected: pd.DataFrame,
    scenario: SweepScenario,
    company_profile: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return compact, run-paired annualized return-rate paths for Excel."""
    if selected.empty:
        return pd.DataFrame(), pd.DataFrame()
    final_selected = selected[
        selected["period"] == selected["selection_period"]
    ].copy()
    final_selected = final_selected.drop_duplicates("portfolio")
    selection_columns = ["portfolio", "column_label", *ASSET_IDENTITY_COLUMNS]
    selection = final_selected[selection_columns]
    paired = results.merge(
        selection,
        on=ASSET_IDENTITY_COLUMNS,
        how="inner",
        validate="many_to_many",
    )
    organization_scenario_id = (
        f"{company_profile}|users={scenario.users}|service_usd={scenario.service_budget_per_person_usd:g}|"
        f"hardware_usd={scenario.hardware_budget_usd:g}|confidential={scenario.confidential_share:g}|plateau={scenario.plateau}"
    )
    paired["organization_scenario_id"] = organization_scenario_id
    index = ["organization_scenario_id", "run", "period", "month"]
    wide = (
        paired.pivot(
            index=index, columns="portfolio", values="annualized_return_rate"
        )
        .reindex(columns=PORTFOLIOS)
        .reset_index()
        .rename(columns={"run": "future_id"})
    )
    manifest = final_selected[
        ["portfolio", "column_label", *ASSET_IDENTITY_COLUMNS]
    ].copy()
    manifest.insert(0, "organization_scenario_id", organization_scenario_id)
    return wide, manifest


def write_paired_future_return_rates(
    paths: pd.DataFrame,
    manifest: pd.DataFrame,
    output_dir: Path,
) -> None:
    paths.to_csv(
        output_dir / "portfolio_paired_future_return_rates.csv", index=False
    )
    manifest.to_csv(
        output_dir / "portfolio_paired_future_manifest.csv", index=False
    )


def service_sleeve_statistics(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    scenario: SweepScenario,
    company_profile: str,
    min_incremental_efficiency_gain: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose one efficient asset per service sleeve and retain paired covariance."""
    final_period = int(summary["period"].max())
    final = summary[summary["period"] == final_period].copy()
    final["service_sleeve"] = final["service_provider"].map(SERVICE_SLEEVES)
    baseline = final["median_zero_risk_efficiency_gain"]
    final = final[
        final["median_efficiency_gain"]
        > baseline + min_incremental_efficiency_gain + 1e-12
    ]
    chosen_rows = []
    for sleeve in SERVICE_SLEEVES.values():
        candidates = final[final["service_sleeve"] == sleeve]
        frontier = efficient_frontier(
            candidates,
            "annualized_return_rate_risk_stddev",
            "median_annualized_return_rate",
        )
        if frontier.empty:
            return pd.DataFrame(), pd.DataFrame()
        positive = frontier[
            frontier["annualized_return_rate_risk_stddev"] > 0.0
        ].copy()
        if positive.empty:
            chosen = frontier.sort_values(
                "median_annualized_return_rate", ascending=False
            ).iloc[0]
        else:
            chosen = positive.loc[
                (
                    positive["median_annualized_return_rate"]
                    / positive["annualized_return_rate_risk_stddev"]
                ).idxmax()
            ]
        chosen_rows.append(chosen)
    chosen = pd.DataFrame(chosen_rows)
    selected = chosen[["service_sleeve", *ASSET_IDENTITY_COLUMNS]]
    selected_results = results.merge(
        selected, on=ASSET_IDENTITY_COLUMNS, how="inner"
    )
    scenario_id = (
        f"{company_profile}|users={scenario.users}|service_usd={scenario.service_budget_per_person_usd:g}|"
        f"hardware_usd={scenario.hardware_budget_usd:g}|confidential={scenario.confidential_share:g}|plateau={scenario.plateau}"
    )
    record: dict[str, object] = {
        "organization_scenario_id": scenario_id,
        "company_profile": company_profile,
        "users": scenario.users,
        "service_usd": scenario.service_budget_per_person_usd,
        "hardware_usd": scenario.hardware_budget_usd,
        "confidential": scenario.confidential_share,
        "plateau": str(scenario.plateau),
    }
    records = []
    for (period, month), period_results in selected_results.groupby(
        ["period", "month"], sort=True
    ):
        paired = (
            period_results.pivot(
                index="run",
                columns="service_sleeve",
                values="annualized_return_rate",
            )
            .reindex(columns=list(SERVICE_SLEEVES.values()))
            .dropna()
        )
        if len(paired) < 2:
            continue
        covariance = paired.cov()
        period_record = {
            **record,
            "period": int(period),
            "month": int(month),
            "futures": len(paired),
        }
        for sleeve in SERVICE_SLEEVES.values():
            period_record[f"mean_{sleeve}"] = float(paired[sleeve].mean())
        for left in SERVICE_SLEEVES.values():
            for right in SERVICE_SLEEVES.values():
                period_record[f"cov_{left}_{right}"] = float(
                    covariance.loc[left, right]
                )
        records.append(period_record)
    manifest = chosen[["service_sleeve", *ASSET_IDENTITY_COLUMNS]].copy()
    manifest.insert(0, "organization_scenario_id", scenario_id)
    return pd.DataFrame(records), manifest


def write_fixed_asset_csvs(assets: pd.DataFrame, output_dir: Path) -> None:
    """Export virtual representative-asset histories and their traceability map."""
    for metric, filename in (
        (
            "median_return_rate_change",
            "portfolio_fixed_asset_rate_changes.csv",
        ),
        ("risk_stddev", "portfolio_fixed_asset_rate_change_risk.csv"),
    ):
        wide = assets.pivot(
            index="month", columns="asset_column_label", values=metric
        ).reset_index()
        wide = wide.reindex(
            columns=["month", *sorted(assets["asset_column_label"].unique())]
        )
        wide.to_csv(output_dir / filename, index=False)
    manifest_columns = [
        "asset_column_label",
        *ASSET_IDENTITY_COLUMNS,
        "users",
        "service_budget_per_person_usd",
        "hardware_budget_usd",
        "confidential_document_share",
        "capability_plateau_after_18_months",
        "hardware_calibration",
        "hardware_replicas",
        "hardware_cost_usd",
    ]
    assets[manifest_columns].drop_duplicates().sort_values(
        "asset_column_label"
    ).to_csv(output_dir / "portfolio_fixed_asset_manifest.csv", index=False)


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


def _selected_median_rows(
    selected: pd.DataFrame, month: float | None = None
) -> pd.DataFrame:
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


def _employee_colors(
    values: pd.DataFrame,
) -> dict[int, tuple[float, float, float, float]]:
    users = (
        sorted(int(value) for value in values["users"].dropna().unique())
        if "users" in values
        else []
    )
    palette = plt.get_cmap("tab10").colors
    return {
        user: palette[index % len(palette)] for index, user in enumerate(users)
    }


def _plot_selected_service_medians(
    ax,
    selected: pd.DataFrame,
    x: str,
    y: str,
    colors: dict,
    month: float | None = None,
) -> None:
    medians = _selected_median_rows(selected, month)
    for _, row in medians.iterrows():
        if (
            x not in row
            or y not in row
            or not np.isfinite(row[x])
            or not np.isfinite(row[y])
        ):
            continue
        portfolio = row["portfolio"]
        user_color = (
            colors.get(int(row["users"]))
            if "users" in row and int(row["users"]) in colors
            else None
        )
        marker_color = (
            user_color
            if user_color is not None
            else colors.get(portfolio, "#111827")
        )
        ax.scatter(
            [row[x]],
            [row[y]],
            marker=SERVICE_MARKERS[row["service_type"]],
            s=52,
            color=marker_color,
            edgecolor="black",
            linewidth=0.7,
            zorder=6,
        )


def _add_employee_legend(ax, values: pd.DataFrame, loc="center right") -> None:
    colors = _employee_colors(values)
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=color,
            label=f"{users} employees",
        )
        for users, color in colors.items()
    ]
    legend = ax.legend(
        handles=handles,
        title="Employee bracket",
        fontsize=6,
        title_fontsize=7,
        loc=loc,
    )
    ax.add_artist(legend)


def _add_service_legend(ax, loc="lower right") -> None:
    handles = [
        Line2D(
            [],
            [],
            marker=marker,
            linestyle="",
            color="#374151",
            markeredgecolor="black",
            label=label,
        )
        for label, marker in SERVICE_MARKERS.items()
    ]
    legend = ax.legend(
        handles=handles,
        title="Selected CSV median",
        fontsize=6,
        title_fontsize=7,
        loc=loc,
    )
    ax.add_artist(legend)


def _symlog_threshold(values: pd.Series | np.ndarray) -> float:
    finite = np.abs(np.asarray(values, dtype=float))
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    if finite.size == 0:
        return 1e-6
    return max(1e-6, float(np.percentile(finite, 10)))


def _apply_frontier_scale(
    ax, risk_values, reward_values, symlog: bool
) -> None:
    if not symlog:
        return
    ax.set_xscale("symlog", linthresh=_symlog_threshold(risk_values))
    ax.set_yscale("symlog", linthresh=_symlog_threshold(reward_values))


def _combined_legend(
    ax, values: pd.DataFrame, extra_handles: list[Line2D] | None = None
) -> None:
    colors = _employee_colors(values)
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=color,
            label=f"Employees: {users}",
        )
        for users, color in colors.items()
    ]
    handles.extend(
        Line2D(
            [],
            [],
            marker=marker,
            linestyle="",
            color="#374151",
            markeredgecolor="black",
            label=f"Service: {label}",
        )
        for label, marker in SERVICE_MARKERS.items()
    )
    handles.append(
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color="#9ca3af",
            markeredgecolor="#111827",
            markeredgewidth=2.0,
            label="Per-bracket Pareto asset",
        )
    )
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
    min_incremental_efficiency_gain: float = 0.0,
) -> pd.DataFrame:
    """Select Pareto assets above the free existing-tools efficiency baseline."""
    if points.empty:
        return pd.DataFrame()
    grouping_columns = grouping_columns or []
    group_columns = [
        column
        for column in grouping_columns
        if column in points.columns and column != "users"
    ] + ["users"]
    rows = []
    for group_key, group in points.groupby(
        group_columns, sort=True, dropna=False
    ):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_metadata = dict(zip(group_columns, group_key, strict=True))
        users = group_metadata["users"]

        baseline = (
            group["median_zero_risk_efficiency_gain"]
            if "median_zero_risk_efficiency_gain" in group
            else pd.Series(0.0, index=group.index)
        )
        eligible = group[
            group["median_efficiency_gain"]
            > baseline + min_incremental_efficiency_gain + 1e-12
        ]
        frontier = efficient_frontier(eligible, risk_column, reward_column)
        if frontier.empty:
            continue
        scored = frontier.copy()
        scored["risk_stddev"] = scored[risk_column]
        scored["median_return_rate_change"] = scored[reward_column]
        for portfolio in PORTFOLIOS:
            chosen, rule = _choose(scored, portfolio)
            row = chosen.drop(
                labels=["risk_stddev", "median_return_rate_change"],
                errors="ignore",
            ).to_dict()
            row["portfolio"] = portfolio
            row["selection_rule"] = rule
            row.update(group_metadata)
            row["employee_bracket"] = int(users)
            row["service_type"] = _service_type(chosen)
            rows.append(row)
    return pd.DataFrame(rows)


def write_per_bracket_pareto_csvs(
    assets: pd.DataFrame, output_dir: Path
) -> None:
    assets.to_csv(
        output_dir / "portfolio_per_bracket_pareto_assets.csv", index=False
    )
    for portfolio in PORTFOLIOS:
        assets[assets["portfolio"] == portfolio].to_csv(
            output_dir
            / f"portfolio_per_bracket_{portfolio}_pareto_assets.csv",
            index=False,
        )


def write_pareto_markdown_table(
    points: pd.DataFrame,
    output_path: Path,
    min_incremental_efficiency_gain: float = 0.0,
) -> None:
    """Write a Markdown matrix grouped by shock assumptions and work mix."""
    stochastic_shocks = (
        "shock_combination" in points
        and points["shock_combination"].fillna("none").ne("none").any()
    )
    grouping = (
        ["company_profile", "shock_combination", "users"]
        if stochastic_shocks
        else [
            "company_profile",
            "capability_plateau_after_18_months",
            "token_cost",
            "shock_combination",
            "users",
        ]
    )
    annualized_return_columns = {
        "median_annualized_return_rate",
        "annualized_return_rate_risk_stddev",
    }
    show_annualized_returns = annualized_return_columns.issubset(
        points.columns
    )
    assets = per_bracket_pareto_assets(
        points,
        grouping,
        (
            "annualized_return_rate_risk_stddev"
            if show_annualized_returns
            else "efficiency_gain_risk_stddev"
        ),
        (
            "median_annualized_return_rate"
            if show_annualized_returns
            else "median_efficiency_gain"
        ),
        min_incremental_efficiency_gain,
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
    assets["asset_label"] = (
        assets["service_provider"].map(asset_labels).fillna(assets["hardware"])
    )
    brackets = sorted(
        int(value) for value in assets["employee_bracket"].unique()
    )
    portfolio_labels = {
        "low_risk": "Low risk",
        "optimum": "Optimum",
        "high_gain": "High return",
    }
    lines = [
        "# Per-bracket Pareto assets",
        "",
        (
            "Each cell shows `asset (annualized return rate, annualized return-rate SD)` for the selected per-bracket Pareto asset."
            if show_annualized_returns
            else "Each cell shows `asset (return, risk)` for the selected per-bracket Pareto asset. Return is median per-employee efficiency gain; risk is its standard deviation."
        ),
        "",
        "| Shock assumptions | Work mix | "
        + " | ".join(
            f"{users} employees" for users in brackets for _ in PORTFOLIOS
        )
        + " |",
        "|---|---|"
        + "|".join("---" for _ in range(len(brackets) * len(PORTFOLIOS)))
        + "|",
        "|  |  | "
        + " | ".join(
            portfolio_labels[portfolio]
            for _ in brackets
            for portfolio in PORTFOLIOS
        )
        + " |",
    ]
    for (shock, work_mix), group in assets.groupby(
        ["shock_assumption", "work_mix"], sort=True
    ):
        cells = []
        for users in brackets:
            for portfolio in PORTFOLIOS:
                match = group[
                    (group["employee_bracket"] == users)
                    & (group["portfolio"] == portfolio)
                ]
                if match.empty:
                    cells.append("—")
                    continue
                row = match.iloc[0]
                if show_annualized_returns:
                    cells.append(
                        f"{row['asset_label']} ({row['median_annualized_return_rate']:.3f}, {row['annualized_return_rate_risk_stddev']:.3f})"
                    )
                else:
                    cells.append(
                        f"{row['asset_label']} ({row['median_efficiency_gain']:.3f}, {row['efficiency_gain_risk_stddev']:.3f})"
                    )
        lines.append(f"| {shock} | {work_mix} | " + " | ".join(cells) + " |")
    output_path.write_text("\n".join(lines) + "\n")


def _plot_pareto_assets(
    ax, assets: pd.DataFrame, points: pd.DataFrame, x: str, y: str
) -> None:
    if assets is None or assets.empty or x not in assets or y not in assets:
        return
    colors = _employee_colors(points)
    for _, row in assets.iterrows():
        if not np.isfinite(row[x]) or not np.isfinite(row[y]):
            continue
        ax.scatter(
            [row[x]],
            [row[y]],
            marker=SERVICE_MARKERS[row["service_type"]],
            s=78,
            color=colors.get(int(row["employee_bracket"]), "#111827"),
            edgecolor="#111827",
            linewidth=2.2,
            zorder=8,
        )


def _add_pareto_legend(ax) -> None:
    handle = Line2D(
        [],
        [],
        marker="o",
        linestyle="",
        color="#9ca3af",
        markeredgecolor="#111827",
        markeredgewidth=2.0,
        label="Per-bracket Pareto asset",
    )
    legend = ax.legend(handles=[handle], fontsize=6, loc="lower center")
    ax.add_artist(legend)


def plot_month_grid(selected: pd.DataFrame, output_dir: Path) -> None:
    months = sorted(selected["month"].unique())
    columns = min(4, len(months))
    rows = int(np.ceil(len(months) / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(4.5 * columns, 3.6 * rows), squeeze=False
    )
    colors = {
        "low_risk": "#2563eb",
        "optimum": "#059669",
        "high_gain": "#dc2626",
    }
    bracket_colors = _employee_colors(selected)
    for ax, month in zip(axes.flat, months, strict=False):
        panel = selected[selected["month"] == month]
        for portfolio in PORTFOLIOS:
            values = panel[panel["portfolio"] == portfolio]
            ax.scatter(
                values["annualized_return_rate_risk_stddev"],
                values["median_annualized_return_rate"],
                s=8,
                alpha=0.20,
                color=colors[portfolio],
                label=portfolio.replace("_", " "),
            )
        _plot_selected_service_medians(
            ax,
            panel,
            "annualized_return_rate_risk_stddev",
            "median_annualized_return_rate",
            bracket_colors,
        )
        ax.set_title(f"Month {month}")
        ax.set_xlabel("Annualized return-rate risk (SD)")
        ax.set_ylabel("Median annualized return rate")
    for ax in axes.flat[len(months) :]:
        ax.set_visible(False)
    _combined_legend(
        axes.flat[0],
        selected,
        [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color=color,
                label=label.replace("_", " "),
            )
            for label, color in {
                "low_risk": "#2563eb",
                "optimum": "#059669",
                "high_gain": "#dc2626",
            }.items()
        ],
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
    figure.write_html(
        output_dir / "portfolio_risk_return_animation.html",
        include_plotlyjs="cdn",
    )


def _apply_frontier_x_limits(ax, x_limits: tuple[float, float] | None) -> None:
    if x_limits is not None:
        ax.set_xlim(*x_limits)


def plot_efficiency_risk_frontier(
    points: pd.DataFrame,
    output_dir: Path,
    one_column: bool = False,
    selected: pd.DataFrame | None = None,
    pareto_assets: pd.DataFrame | None = None,
    symlog: bool = False,
    x_limits: tuple[float, float] | None = None,
) -> None:
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
        tangency = positive_risk.loc[
            (positive_risk[reward] / positive_risk[risk]).idxmax()
        ]

    fig, ax = plt.subplots(figsize=(3.5, 3.0) if one_column else (8.2, 5.8))
    ax.scatter(
        points[risk],
        points[reward],
        s=5 if one_column else 8,
        color="#111827",
        alpha=0.18,
        linewidths=0,
        label="Individual simulated portfolios",
    )
    if selected is not None:
        _plot_selected_service_medians(
            ax, selected, risk, reward, _employee_colors(points)
        )
    ax.plot(
        frontier[risk],
        frontier[reward],
        color="#2563eb",
        linewidth=1.5 if one_column else 2.2,
        marker="o",
        markersize=2.5 if one_column else 3.5,
        label="Efficient frontier",
    )
    if selected is not None:
        _plot_pareto_assets(ax, pareto_assets, points, risk, reward)
    if tangency is not None:
        end_risk = max(float(points[risk].max()), float(tangency[risk])) * 1.05
        slope = float(tangency[reward]) / float(tangency[risk])
        ax.plot(
            [0.0, end_risk],
            [0.0, slope * end_risk],
            "--",
            color="#7c3aed",
            linewidth=1.4,
            label="Best risk-adjusted allocation",
        )
        ax.scatter(
            [tangency[risk]],
            [tangency[reward]],
            s=42,
            color="#7c3aed",
            zorder=4,
        )
    _apply_frontier_scale(ax, points[risk], points[reward], symlog)
    _apply_frontier_x_limits(ax, x_limits)
    reward_min, reward_max = float(points[reward].min()), float(
        points[reward].max()
    )
    reward_margin = max(1e-6, (reward_max - reward_min) * 0.06)
    ax.set_ylim(reward_min - reward_margin, reward_max + reward_margin)
    ax.axvline(0.0, color="#6b7280", linewidth=0.8)
    ax.set_xlabel(
        "Annualized return-rate risk (SD)", fontsize=8 if one_column else None
    )
    ax.set_ylabel(
        "Median annualized return rate", fontsize=8 if one_column else None
    )
    if not one_column:
        ax.set_title(
            "Final-period annualized return-rate frontier"
            + (" (symlog axes)" if symlog else "")
        )
    ax.tick_params(labelsize=7 if one_column else None)
    extra_handles = [
        Line2D(
            [], [], color="#111827", linewidth=1.8, label="Efficient frontier"
        ),
    ]
    if tangency is not None:
        extra_handles.append(
            Line2D(
                [],
                [],
                color="#7c3aed",
                linestyle="--",
                label="Best risk-adjusted allocation",
            )
        )
    _combined_legend(ax, points, extra_handles)
    fig.tight_layout()
    suffix = ("_symlog" if symlog else "") + (
        "_one_column" if one_column else ""
    )
    fig.savefig(
        output_dir / f"portfolio_efficiency_risk_frontier{suffix}.png",
        dpi=300 if one_column else 180,
    )
    plt.close(fig)


def plot_efficiency_risk_frontier_by_asset_mix(
    points: pd.DataFrame,
    output_dir: Path,
    simulation_months: float,
    one_column: bool = False,
    selected: pd.DataFrame | None = None,
    pareto_assets: pd.DataFrame | None = None,
    symlog: bool = False,
    x_limits: tuple[float, float] | None = None,
) -> None:
    """Show the final frontier while encoding company size and spending mix."""
    if points.empty:
        return
    risk = "annualized_return_rate_risk_stddev"
    reward = "median_annualized_return_rate"
    values = points.copy()
    service_commitment = (
        values["service_budget_per_person_usd"]
        * values["users"]
        * simulation_months
    )
    hardware_share = values["hardware_budget_usd"] / (
        values["hardware_budget_usd"] + service_commitment
    )
    values["asset_mix"] = np.select(
        [hardware_share >= 2.0 / 3.0, hardware_share <= 1.0 / 3.0],
        ["Hardware-heavy", "Service-heavy"],
        default="Mixed",
    )
    colors = dict(
        zip(
            sorted(values["users"].unique()),
            plt.get_cmap("tab10").colors,
            strict=False,
        )
    )
    markers = {"Hardware-heavy": "^", "Service-heavy": "o", "Mixed": "s"}
    frontier = efficient_frontier(values, risk, reward)

    fig, ax = plt.subplots(figsize=(3.5, 3.0) if one_column else (8.2, 5.8))
    for (users, asset_mix), group in values.groupby(
        ["users", "asset_mix"], sort=True
    ):
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
        _plot_selected_service_medians(
            ax,
            selected,
            risk,
            reward,
            _employee_colors(values if "values" in locals() else points),
        )
    ax.plot(
        frontier[risk],
        frontier[reward],
        color="#111827",
        linewidth=1.4 if one_column else 2.0,
        zorder=3,
    )
    if selected is not None:
        _plot_pareto_assets(ax, pareto_assets, values, risk, reward)
    _apply_frontier_scale(ax, values[risk], values[reward], symlog)
    _apply_frontier_x_limits(ax, x_limits)
    reward_min, reward_max = float(values[reward].min()), float(
        values[reward].max()
    )
    ax.set_ylim(
        reward_min - max(1e-6, (reward_max - reward_min) * 0.06),
        reward_max + max(1e-6, (reward_max - reward_min) * 0.06),
    )
    ax.axvline(0.0, color="#6b7280", linewidth=0.8)
    ax.set_xlabel(
        "Annualized return-rate risk (SD)", fontsize=8 if one_column else None
    )
    ax.set_ylabel(
        "Median annualized return rate", fontsize=8 if one_column else None
    )
    if not one_column:
        ax.set_title(
            "Annualized return-rate frontier by company size and asset mix"
            + (" (symlog axes)" if symlog else "")
        )
    mix_handles = [
        Line2D(
            [],
            [],
            marker=marker,
            linestyle="",
            color="#374151",
            label=f"Asset mix: {asset_mix}",
        )
        for asset_mix, marker in markers.items()
    ]
    _combined_legend(ax, values, mix_handles)
    ax.tick_params(labelsize=7 if one_column else None)
    fig.tight_layout()
    suffix = ("_symlog" if symlog else "") + (
        "_one_column" if one_column else ""
    )
    fig.savefig(
        output_dir
        / f"portfolio_efficiency_risk_frontier_by_asset_mix{suffix}.png",
        dpi=300 if one_column else 180,
    )
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

    def hyperbola(
        y: np.ndarray,
        vertex: float,
        curvature: float,
        center: float,
        spread: float,
    ) -> np.ndarray:
        return vertex - curvature * (
            np.sqrt(1.0 + ((y - center) / spread) ** 2) - 1.0
        )

    risk_span = max(np.ptp(risks), 1e-9)
    reward_span = max(np.ptp(rewards), 1e-9)
    try:
        parameters, _ = curve_fit(
            hyperbola,
            rewards,
            risks,
            p0=(
                float(risks.max()),
                risk_span,
                float(np.median(rewards)),
                reward_span / 2.0,
            ),
            bounds=(
                [-np.inf, 1e-9, rewards.min() - reward_span, 1e-9],
                [np.inf, np.inf, rewards.max() + reward_span, np.inf],
            ),
            maxfev=20_000,
        )
    except (RuntimeError, ValueError):
        return risks, rewards

    curve_rewards = np.linspace(
        float(rewards.min()), float(rewards.max()), 160
    )
    curve_risks = hyperbola(curve_rewards, *parameters)
    return curve_risks, curve_rewards


def plot_efficiency_risk_frontier_by_size(
    points: pd.DataFrame,
    output_dir: Path,
    one_column: bool = False,
    selected: pd.DataFrame | None = None,
    pareto_assets: pd.DataFrame | None = None,
    symlog: bool = False,
    x_limits: tuple[float, float] | None = None,
) -> None:
    """Show one hyperbola-style annualized return-rate frontier per company size."""
    if points.empty:
        return
    risk = "annualized_return_rate_risk_stddev"
    reward = "median_annualized_return_rate"
    colors = dict(
        zip(
            sorted(points["users"].unique()),
            plt.get_cmap("tab10").colors,
            strict=False,
        )
    )
    fig, ax = plt.subplots(figsize=(3.5, 3.0) if one_column else (8.2, 5.8))
    for users, group in points.groupby("users", sort=True):
        color = colors[users]
        ax.scatter(
            group[risk],
            group[reward],
            s=6 if one_column else 11,
            color=color,
            alpha=0.14,
            linewidths=0,
        )
        frontier = efficient_frontier(group, risk, reward)
        curve_risk, curve_reward = _hyperbolic_frontier_curve(
            frontier, risk, reward
        )
        curve_risk = np.array(curve_risk, copy=True)
        fitted_risks = np.interp(
            group[reward].to_numpy(float), curve_reward, curve_risk
        )
        curve_risk -= (
            max(0.0, float((fitted_risks - group[risk].to_numpy(float)).max()))
            + 1e-12
        )
        ax.plot(
            curve_risk,
            curve_reward,
            color=color,
            linewidth=1.5 if one_column else 2.2,
        )
    if selected is not None:
        _plot_selected_service_medians(
            ax,
            selected,
            risk,
            reward,
            _employee_colors(values if "values" in locals() else points),
        )
    reward_min, reward_max = float(points[reward].min()), float(
        points[reward].max()
    )
    reward_margin = max(1e-6, (reward_max - reward_min) * 0.06)
    ax.set_ylim(reward_min - reward_margin, reward_max + reward_margin)
    ax.axvline(0.0, color="#6b7280", linewidth=0.8)
    ax.set_xlabel(
        "Annualized return-rate risk (SD)", fontsize=8 if one_column else None
    )
    ax.set_ylabel(
        "Median annualized return rate", fontsize=8 if one_column else None
    )
    if selected is not None:
        _plot_pareto_assets(ax, pareto_assets, points, risk, reward)
    _apply_frontier_scale(ax, points[risk], points[reward], symlog)
    _apply_frontier_x_limits(ax, x_limits)
    if not one_column:
        ax.set_title(
            "Hyperbolic annualized return-rate frontiers by company size"
            + (" (symlog axes)" if symlog else "")
        )
    ax.tick_params(labelsize=7 if one_column else None)
    _combined_legend(
        ax,
        points,
        [
            Line2D(
                [],
                [],
                color="#111827",
                linewidth=1.8,
                label="Hyperbolic frontier",
            )
        ],
    )
    fig.tight_layout()
    suffix = ("_symlog" if symlog else "") + (
        "_one_column" if one_column else ""
    )
    fig.savefig(
        output_dir / f"portfolio_efficiency_risk_frontier_by_size{suffix}.png",
        dpi=300 if one_column else 180,
    )
    plt.close(fig)


def plot_efficiency_risk_frontier_by_size_empirical(
    points: pd.DataFrame,
    output_dir: Path,
    one_column: bool = False,
    selected: pd.DataFrame | None = None,
    pareto_assets: pd.DataFrame | None = None,
    symlog: bool = False,
    x_limits: tuple[float, float] | None = None,
) -> None:
    """Show the observed annualized return-rate frontier points for each company size."""
    if points.empty:
        return
    risk = "annualized_return_rate_risk_stddev"
    reward = "median_annualized_return_rate"
    colors = dict(
        zip(
            sorted(points["users"].unique()),
            plt.get_cmap("tab10").colors,
            strict=False,
        )
    )
    fig, ax = plt.subplots(figsize=(3.5, 3.0) if one_column else (8.2, 5.8))
    for users, group in points.groupby("users", sort=True):
        color = colors[users]
        ax.scatter(
            group[risk],
            group[reward],
            s=6 if one_column else 11,
            color=color,
            alpha=0.14,
            linewidths=0,
        )
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
        _plot_selected_service_medians(
            ax,
            selected,
            risk,
            reward,
            _employee_colors(values if "values" in locals() else points),
        )
    reward_min, reward_max = float(points[reward].min()), float(
        points[reward].max()
    )
    reward_margin = max(1e-6, (reward_max - reward_min) * 0.06)
    ax.set_ylim(reward_min - reward_margin, reward_max + reward_margin)
    ax.axvline(0.0, color="#6b7280", linewidth=0.8)
    ax.set_xlabel(
        "Annualized return-rate risk (SD)", fontsize=8 if one_column else None
    )
    ax.set_ylabel(
        "Median annualized return rate", fontsize=8 if one_column else None
    )
    if selected is not None:
        _plot_pareto_assets(ax, pareto_assets, points, risk, reward)
    _apply_frontier_scale(ax, points[risk], points[reward], symlog)
    _apply_frontier_x_limits(ax, x_limits)
    if not one_column:
        ax.set_title(
            "Observed annualized return-rate frontiers by company size"
            + (" (symlog axes)" if symlog else "")
        )
    ax.tick_params(labelsize=7 if one_column else None)
    _combined_legend(
        ax,
        points,
        [
            Line2D(
                [],
                [],
                color="#111827",
                linewidth=1.8,
                marker="o",
                markersize=3,
                label="Observed Pareto frontier",
            )
        ],
    )
    fig.tight_layout()
    suffix = ("_symlog" if symlog else "") + (
        "_one_column" if one_column else ""
    )
    fig.savefig(
        output_dir
        / f"portfolio_efficiency_risk_frontier_by_size_empirical{suffix}.png",
        dpi=300 if one_column else 180,
    )
    plt.close(fig)


def _bands(
    values: pd.DataFrame, group_cols: list[str], metric: str
) -> pd.DataFrame:
    return (
        values.groupby(group_cols, sort=True)[metric]
        .agg(
            median="median",
            p10=lambda series: series.quantile(0.10),
            p90=lambda series: series.quantile(0.90),
        )
        .reset_index()
    )


def plot_faceted_timeseries(
    selected: pd.DataFrame,
    output_dir: Path,
    metric: str,
    filename: str,
    ylabel: str,
    title: str,
) -> None:
    """Show the distribution over budgets and confidentiality settings for each size."""
    headcounts = sorted(selected["users"].unique())
    fig, axes = plt.subplots(
        len(PORTFOLIOS),
        len(headcounts),
        figsize=(3.5 * len(headcounts), 2.8 * len(PORTFOLIOS)),
        sharex=True,
    )
    colors = {False: "#2563eb", True: "#dc2626"}
    labels = {False: "Continuous improvement", True: "Plateau after 18 months"}
    bands = _bands(
        selected,
        ["portfolio", "users", "capability_plateau_after_18_months", "month"],
        metric,
    )
    for row, portfolio in enumerate(PORTFOLIOS):
        for column, users in enumerate(headcounts):
            ax = axes[row, column]
            panel = bands[
                (bands["portfolio"] == portfolio) & (bands["users"] == users)
            ]
            for plateau, values in panel.groupby(
                "capability_plateau_after_18_months", sort=False
            ):
                values = values.sort_values("month")
                ax.fill_between(
                    values["month"],
                    values["p10"],
                    values["p90"],
                    color=colors[plateau],
                    alpha=0.14,
                )
                ax.plot(
                    values["month"],
                    values["median"],
                    color=colors[plateau],
                    linewidth=1.7,
                    label=labels[plateau],
                )
            if row == 0:
                ax.set_title(f"{users} employees")
            if column == 0:
                ax.set_ylabel(f"{portfolio.replace('_', ' ')}\n{ylabel}")
            if row == len(PORTFOLIOS) - 1:
                ax.set_xlabel("Month")
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle(
        f"{title} across service, hardware, and confidentiality settings",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_heatmaps(
    selected: pd.DataFrame,
    output_dir: Path,
    metric: str,
    filename: str,
    title: str,
) -> None:
    final_period = selected[selected["period"] == selected["period"].max()]
    values = (
        final_period.groupby(
            [
                "portfolio",
                "users",
                "service_budget_per_person_usd",
                "hardware_budget_usd",
            ],
            sort=True,
        )[metric]
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
            panel = values[
                (values["portfolio"] == portfolio) & (values["users"] == users)
            ]
            pivot = (
                panel.pivot(
                    index="hardware_budget_usd",
                    columns="service_budget_per_person_usd",
                    values=metric,
                )
                .sort_index()
                .sort_index(axis=1)
            )
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
            ax.set_xticks(
                range(len(pivot.columns)),
                [f"${value:g}" for value in pivot.columns],
                rotation=45,
                ha="right",
                fontsize=7,
            )
            ax.set_yticks(
                range(len(pivot.index)),
                [f"${value / 1000:g}k" for value in pivot.index],
                fontsize=7,
            )
            if row == 0:
                ax.set_title(f"{users} employees")
            if column == 0:
                ax.set_ylabel(
                    f"{portfolio.replace('_', ' ')}\nupfront hardware"
                )
            if row == len(PORTFOLIOS) - 1:
                ax.set_xlabel("Service/person/month")
    for column, image in enumerate(images):
        fig.colorbar(
            image,
            ax=axes[:, column],
            location="right",
            shrink=0.82,
            label=title,
        )
    fig.suptitle(
        f"Final-period {title}: median across confidentiality and capability paths"
    )
    fig.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_plateau_comparison(selected: pd.DataFrame, output_dir: Path) -> None:
    bands = _bands(
        selected,
        ["portfolio", "capability_plateau_after_18_months", "month"],
        "median_annualized_return_rate",
    )
    fig, axes = plt.subplots(
        1, len(PORTFOLIOS), figsize=(5 * len(PORTFOLIOS), 4), sharey=True
    )
    colors = {False: "#2563eb", True: "#dc2626"}
    labels = {False: "Continuous improvement", True: "Plateau after 18 months"}
    for ax, portfolio in zip(axes, PORTFOLIOS, strict=True):
        panel = bands[bands["portfolio"] == portfolio]
        for plateau, values in panel.groupby(
            "capability_plateau_after_18_months", sort=False
        ):
            values = values.sort_values("month")
            ax.fill_between(
                values["month"],
                values["p10"],
                values["p90"],
                color=colors[plateau],
                alpha=0.15,
            )
            ax.plot(
                values["month"],
                values["median"],
                color=colors[plateau],
                linewidth=2,
                label=labels[plateau],
            )
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
    linear_x_limits: tuple[float, float] | None = None,
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
    plot_sensitivity_heatmaps(
        selected,
        output_dir,
        "median_annualized_return_rate",
        "portfolio_final_return_sensitivity.png",
        "median annualized return rate",
    )
    plot_sensitivity_heatmaps(
        selected,
        output_dir,
        "annualized_return_rate_risk_stddev",
        "portfolio_final_risk_sensitivity.png",
        "annualized return-rate risk (SD)",
    )
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
        plot_efficiency_risk_frontier(
            frontier_points,
            output_dir,
            one_column,
            selected,
            pareto_assets,
            x_limits=linear_x_limits,
        )
        plot_efficiency_risk_frontier(
            frontier_points,
            output_dir,
            one_column,
            selected,
            pareto_assets,
            symlog=True,
        )
        plot_efficiency_risk_frontier_by_asset_mix(
            frontier_points,
            output_dir,
            simulation_months,
            one_column,
            selected,
            pareto_assets,
            x_limits=linear_x_limits,
        )
        plot_efficiency_risk_frontier_by_asset_mix(
            frontier_points,
            output_dir,
            simulation_months,
            one_column,
            selected,
            pareto_assets,
            symlog=True,
        )
        plot_efficiency_risk_frontier_by_size(
            frontier_points,
            output_dir,
            one_column,
            selected,
            pareto_assets,
            x_limits=linear_x_limits,
        )
        plot_efficiency_risk_frontier_by_size(
            frontier_points,
            output_dir,
            one_column,
            selected,
            pareto_assets,
            symlog=True,
        )
        plot_efficiency_risk_frontier_by_size_empirical(
            frontier_points,
            output_dir,
            one_column,
            selected,
            pareto_assets,
            x_limits=linear_x_limits,
        )
        plot_efficiency_risk_frontier_by_size_empirical(
            frontier_points,
            output_dir,
            one_column,
            selected,
            pareto_assets,
            symlog=True,
        )


def parse_args() -> argparse.Namespace:
    config_path = configuration_path_from_argv()
    try:
        load_simulation_configuration(config_path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    defaults = SWEEP_DEFAULTS["portfolio"]
    global HEADCOUNTS, SERVICE_BUDGETS_USD, HARDWARE_BUDGETS_USD, CONFIDENTIAL_SHARES, COMPANY_PROFILES
    HEADCOUNTS = tuple(defaults["headcounts"])
    SERVICE_BUDGETS_USD = tuple(defaults["service_budgets_usd"])
    HARDWARE_BUDGETS_USD = tuple(defaults["hardware_budgets_usd"])
    CONFIDENTIAL_SHARES = tuple(defaults["confidential_shares"])
    COMPANY_PROFILES = dict(defaults["company_profiles"])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=config_path,
        help="YAML configuration file (default: configs/simulation_config.yaml).",
    )
    parser.add_argument(
        "--replot-only",
        action="store_true",
        help="Regenerate all plots from portfolio_scenario_manifest.csv without resimulating.",
    )
    parser.add_argument(
        "--replot-hardware-budget-sensitivity",
        action="store_true",
        help="Regenerate only the hardware-budget sensitivity plot from persisted frontier CSVs.",
    )
    parser.add_argument(
        "--hardware-budget-sensitivity-users",
        type=int,
        default=None,
        help="Plot only this user count and omit per-user normalization from the sensitivity y-axis.",
    )
    parser.add_argument(
        "--hardware-budget-time-sensitivity-color-limits",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=None,
        help="Fix the time-sensitivity heatmap color scale to MIN and MAX.",
    )
    parser.add_argument(
        "--linear-x-limits",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=None,
        help="Set the x-axis limits (risk SD) for the linear frontier plots only.",
    )
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
    parser.add_argument("--years", type=float, default=defaults["years"])
    parser.add_argument(
        "--resolution-months", type=int, default=defaults["resolution_months"]
    )
    parser.add_argument("--runs", type=int, default=defaults["runs"])
    parser.add_argument(
        "--concurrency",
        type=int,
        default=defaults["concurrency"],
        help="Number of independent portfolio scenarios to simulate locally.",
    )
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--employee-mix", type=str, default=None)
    parser.add_argument(
        "--company-profiles",
        type=str,
        default=",".join(COMPANY_PROFILES),
        help="Comma-separated work-mix profiles to run. Defaults to all built-in profiles.",
    )
    parser.add_argument(
        "--engineering-context",
        choices=["bounded", "maintenance", "mixed"],
        default=SIMULATION_DEFAULTS["engineering_context"],
    )
    parser.add_argument(
        "--backend",
        choices=["numpy", "numba", "torch-mps", "auto"],
        default="auto",
    )
    parser.add_argument(
        "--usd-per-service-cost-index-quarter",
        type=float,
        default=SIMULATION_DEFAULTS["usd_per_service_cost_index_quarter"],
    )
    parser.add_argument(
        "--usd-per-hardware-capex-index", type=float, default=None
    )
    parser.add_argument(
        "--hardware-calibration",
        choices=sorted(HARDWARE_CALIBRATIONS),
        default=SIMULATION_DEFAULTS["hardware_calibration"],
    )
    parser.add_argument(
        "--embargo-shock-probability",
        type=float,
        default=SIMULATION_DEFAULTS["embargo_shock_probability"],
    )
    parser.add_argument(
        "--embargo-rollback-months",
        type=int,
        default=SIMULATION_DEFAULTS["embargo_rollback_months"],
    )
    parser.add_argument(
        "--stochastic-shocks",
        action="store_true",
        help="Replace deterministic token-cost and plateau scenarios with persistent stochastic shock combinations.",
    )
    parser.add_argument(
        "--legacy-period-frontier-return-rates",
        action="store_true",
        help="Write the previous full-horizon period-frontier CSVs instead of final-horizon shock-split CSVs.",
    )
    parser.add_argument(
        "--sudden-break-even-probability-per-month",
        type=float,
        default=SIMULATION_DEFAULTS["sudden_break_even_probability_per_month"],
    )
    parser.add_argument(
        "--gradual-break-even-probability-per-month",
        type=float,
        default=SIMULATION_DEFAULTS[
            "gradual_break_even_probability_per_month"
        ],
    )
    parser.add_argument(
        "--capability-plateau-probability-per-month",
        type=float,
        default=SIMULATION_DEFAULTS[
            "capability_plateau_probability_per_month"
        ],
    )
    parser.add_argument(
        "--confidential-mixing-ratio",
        type=float,
        default=SIMULATION_DEFAULTS["confidential_mixing_ratio"],
    )
    parser.add_argument(
        "--confidential-mixing-penalty",
        type=float,
        default=SIMULATION_DEFAULTS["confidential_mixing_penalty"],
    )
    parser.add_argument(
        "--zero-risk-confidential-work-share",
        type=float,
        default=SIMULATION_DEFAULTS["zero_risk_confidential_work_share"],
    )
    parser.add_argument(
        "--pareto-min-incremental-efficiency-gain",
        type=float,
        default=float(
            defaults.get("pareto_min_incremental_efficiency_gain", 0.0)
        ),
        help="Additional median efficiency gain required above the zero-cost existing-tools baseline for Pareto eligibility.",
    )
    parser.add_argument(
        "--disable-counterfactual-skill-growth", action="store_true"
    )
    parser.add_argument(
        "--counterfactual-skill-growth-rate-per-year",
        type=float,
        default=SIMULATION_DEFAULTS.get(
            "counterfactual_skill_growth_rate_per_year", 0.015
        ),
    )
    parser.add_argument(
        "--cloud-oss-hours-per-usage-unit-period", type=float, default=None
    )
    parser.add_argument(
        "--disable-it-support",
        action="store_true",
        help="Disable the default 0.5-FTE IT-support cost for small on-premise and OSS-cloud companies.",
    )
    parser.add_argument(
        "--it-support-max-users",
        type=int,
        default=SIMULATION_DEFAULTS["it_support_max_users"],
        help="Apply the IT-support cost at or below this employee count.",
    )
    parser.add_argument(
        "--figure-width",
        choices=["default", "one-column"],
        default="default",
        help="Use one-column for the compact 3.5-inch-wide efficiency-risk frontier figure for a two-column paper.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("portfolio_sweep_outputs")
    )
    args = parser.parse_args()
    if args.linear_x_limits is not None:
        args.linear_x_limits = tuple(args.linear_x_limits)
        if args.linear_x_limits[0] >= args.linear_x_limits[1]:
            raise SystemExit(
                "--linear-x-limits requires MIN to be less than MAX."
            )
    if (
        args.hardware_budget_sensitivity_users is not None
        and args.hardware_budget_sensitivity_users <= 0
    ):
        raise SystemExit(
            "--hardware-budget-sensitivity-users must be positive."
        )
    if args.hardware_budget_time_sensitivity_color_limits is not None:
        args.hardware_budget_time_sensitivity_color_limits = tuple(
            args.hardware_budget_time_sensitivity_color_limits
        )
        if (
            args.hardware_budget_time_sensitivity_color_limits[0]
            >= args.hardware_budget_time_sensitivity_color_limits[1]
        ):
            raise SystemExit(
                "--hardware-budget-time-sensitivity-color-limits requires MIN to be less than MAX."
            )
    return args


def parse_employee_mix(value: str | None) -> dict[str, float] | None:
    if value is None:
        return None
    return {
        item.split("=", 1)[0].strip(): float(item.split("=", 1)[1])
        for item in value.split(",")
    }


def selected_company_profiles(
    args: argparse.Namespace,
) -> dict[str, dict[str, float]]:
    if args.employee_mix is not None:
        return {"custom": parse_employee_mix(args.employee_mix) or {}}
    names = [
        name.strip()
        for name in args.company_profiles.split(",")
        if name.strip()
    ]
    unknown = set(names) - set(COMPANY_PROFILES)
    if unknown:
        raise SystemExit(
            f"Unknown company profiles: {', '.join(sorted(unknown))}"
        )
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
    executor_cls = (
        ThreadPoolExecutor
        if config.backend == "torch-mps"
        else ProcessPoolExecutor
    )
    selected_frames = []
    asset_timeseries_frames = []
    frontier_point_frames = []
    paired_future_frames = []
    paired_future_manifest_frames = []
    sleeve_statistics_frames = []
    sleeve_manifest_frames = []
    hardware_sensitivity_frames = []
    hardware_time_sensitivity_frames = []
    seed_groups = dict.fromkeys(
        (
            scenario.users,
            scenario.service_budget_per_person_usd,
            scenario.confidential_share,
            scenario.plateau,
        )
        for scenario in scenarios
    )
    paired_seed_indices = {
        group: index for index, group in enumerate(seed_groups)
    }
    with executor_cls(
        max_workers=min(args.concurrency, len(scenarios))
    ) as executor:
        futures = {
            executor.submit(
                simulate_portfolio_scenario,
                scenario,
                config,
                paired_seed_indices[
                    (
                        scenario.users,
                        scenario.service_budget_per_person_usd,
                        scenario.confidential_share,
                        scenario.plateau,
                    )
                ],
            ): scenario
            for scenario in scenarios
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Sweeping {name}",
            unit="scenario",
        ):
            scenario = futures[future]
            results, metadata = future.result()
            summary = return_change_summary(results)
            if args.legacy_period_frontier_return_rates:
                write_period_frontier_return_rates(
                    summary, scenario, output_dir
                )
            else:
                write_final_shock_frontier_return_rates(
                    results, scenario, output_dir
                )
            hardware_sensitivity_frames.append(
                hardware_budget_sensitivity_records(summary, scenario, name)
            )
            hardware_time_sensitivity_frames.append(
                hardware_budget_time_sensitivity_records(
                    summary, scenario, name
                )
            )
            selected = select_portfolios(summary, scenario, metadata)
            selected["company_profile"] = name
            selected_frames.append(selected)
            paired_paths, paired_manifest = paired_future_return_rates(
                results, selected, scenario, name
            )
            paired_future_frames.append(paired_paths)
            paired_future_manifest_frames.append(paired_manifest)
            sleeve_statistics, sleeve_manifest = service_sleeve_statistics(
                results,
                summary,
                scenario,
                name,
                args.pareto_min_incremental_efficiency_gain,
            )
            if not sleeve_statistics.empty:
                sleeve_statistics_frames.append(sleeve_statistics)
                sleeve_manifest_frames.append(sleeve_manifest)
            assets = summary.copy()
            assets["asset_column_label"] = assets.apply(
                lambda row: _asset_label(row, scenario), axis=1
            )
            assets["users"] = scenario.users
            assets["service_budget_per_person_usd"] = (
                scenario.service_budget_per_person_usd
            )
            assets["hardware_budget_usd"] = scenario.hardware_budget_usd
            assets["confidential_document_share"] = scenario.confidential_share
            assets["capability_plateau_after_18_months"] = scenario.plateau
            assets["company_profile"] = name
            for key, value in metadata.items():
                assets[key] = value
            asset_timeseries_frames.append(assets)
            final_points = summary[
                summary["period"] == summary["period"].max()
            ].copy()
            final_points["users"] = scenario.users
            final_points["service_budget_per_person_usd"] = (
                scenario.service_budget_per_person_usd
            )
            final_points["hardware_budget_usd"] = scenario.hardware_budget_usd
            final_points["confidential_document_share"] = (
                scenario.confidential_share
            )
            final_points["capability_plateau_after_18_months"] = (
                scenario.plateau
            )
            final_points["company_profile"] = name
            frontier_point_frames.append(final_points)
    selected = pd.concat(selected_frames, ignore_index=True).sort_values(
        [
            "users",
            "service_budget_per_person_usd",
            "hardware_budget_usd",
            "confidential_document_share",
            "capability_plateau_after_18_months",
            "month",
            "portfolio",
        ]
    )
    selected.to_csv(
        output_dir / "portfolio_scenario_manifest.csv", index=False
    )
    pd.concat(hardware_sensitivity_frames, ignore_index=True).to_csv(
        output_dir / "portfolio_hardware_budget_sensitivity.csv", index=False
    )
    pd.concat(hardware_time_sensitivity_frames, ignore_index=True).to_csv(
        output_dir / "portfolio_hardware_budget_time_sensitivity.csv",
        index=False,
    )
    asset_timeseries = pd.concat(asset_timeseries_frames, ignore_index=True)
    write_fixed_asset_csvs(asset_timeseries, output_dir)
    write_paired_future_return_rates(
        pd.concat(paired_future_frames, ignore_index=True),
        pd.concat(paired_future_manifest_frames, ignore_index=True),
        output_dir,
    )
    if sleeve_statistics_frames:
        pd.concat(sleeve_statistics_frames, ignore_index=True).to_csv(
            output_dir / "portfolio_service_sleeve_statistics.csv", index=False
        )
        pd.concat(sleeve_manifest_frames, ignore_index=True).to_csv(
            output_dir / "portfolio_service_sleeve_manifest.csv", index=False
        )
    frontier_points = pd.concat(frontier_point_frames, ignore_index=True)
    frontier_points.to_csv(
        output_dir / "portfolio_efficiency_risk_points.csv", index=False
    )
    pareto_assets = per_bracket_pareto_assets(
        frontier_points,
        risk_column="annualized_return_rate_risk_stddev",
        reward_column="median_annualized_return_rate",
        min_incremental_efficiency_gain=args.pareto_min_incremental_efficiency_gain,
    )
    write_per_bracket_pareto_csvs(pareto_assets, output_dir)
    write_wide_csvs(selected, output_dir)
    if not args.no_plots:
        plot_all(
            selected,
            output_dir,
            frontier_points,
            args.figure_width == "one-column",
            config.periods * config.resolution_months,
            pareto_assets,
            args.linear_x_limits,
        )
    print(f"Wrote {name} portfolio sweep outputs to {output_dir}")


def main() -> None:
    args = parse_args()
    profiles = selected_company_profiles(args)
    if args.replot_hardware_budget_sensitivity:
        sensitivity_dirs = sorted(
            path.parent
            for path in args.output_dir.glob(
                "*/portfolio_efficiency_risk_points.csv"
            )
        )
        if not sensitivity_dirs:
            raise SystemExit(
                f"No profile frontier CSVs found below {args.output_dir}; run the portfolio sweep first."
            )
        for sensitivity_dir in sensitivity_dirs:
            sensitivity = rebuild_hardware_budget_sensitivity(sensitivity_dir)
            if not sensitivity.empty:
                sensitivity.to_csv(
                    sensitivity_dir
                    / "portfolio_hardware_budget_sensitivity.csv",
                    index=False,
                )
        sensitivity = collect_hardware_budget_sensitivity(args.output_dir)
        if sensitivity.empty:
            raise SystemExit(
                "No local-hardware frontier selections were found for the sensitivity plot."
            )
        sensitivity.to_csv(
            args.output_dir / "portfolio_hardware_budget_sensitivity.csv",
            index=False,
        )
        time_sensitivity = collect_hardware_budget_time_sensitivity(
            args.output_dir
        )
        if not time_sensitivity.empty:
            time_sensitivity.to_csv(
                args.output_dir
                / "portfolio_hardware_budget_time_sensitivity.csv",
                index=False,
            )
            plot_hardware_budget_time_sensitivity(
                time_sensitivity,
                args.output_dir,
                args.hardware_budget_time_sensitivity_color_limits,
            )
        paths = plot_hardware_budget_sensitivity(
            sensitivity,
            args.output_dir,
            args.hardware_budget_sensitivity_users,
        )
        if not paths:
            raise SystemExit(
                "No sensitivity records matched the requested user count."
            )
        print(f"Wrote {paths[0]}")
        return
    if args.replot_only:
        output_dirs = (
            [args.output_dir]
            if (args.output_dir / "portfolio_scenario_manifest.csv").exists()
            else [args.output_dir / name for name in profiles]
        )
        missing = [
            output_dir
            for output_dir in output_dirs
            if not (output_dir / "portfolio_scenario_manifest.csv").exists()
        ]
        if missing:
            raise SystemExit(
                f"Missing sweep manifest: {missing[0] / 'portfolio_scenario_manifest.csv'}"
            )
        for output_dir in output_dirs:
            manifest = output_dir / "portfolio_scenario_manifest.csv"
            points_path = output_dir / "portfolio_efficiency_risk_points.csv"
            frontier_points = (
                pd.read_csv(points_path) if points_path.exists() else None
            )
            required_annualized_columns = {
                "median_annualized_return_rate",
                "annualized_return_rate_risk_stddev",
            }
            if (
                frontier_points is not None
                and not required_annualized_columns.issubset(
                    frontier_points.columns
                )
            ):
                raise SystemExit(
                    "This sweep predates annualized return-rate frontier data; rerun the simulation before replotting."
                )
            pareto_assets = (
                per_bracket_pareto_assets(
                    frontier_points,
                    risk_column="annualized_return_rate_risk_stddev",
                    reward_column="median_annualized_return_rate",
                    min_incremental_efficiency_gain=args.pareto_min_incremental_efficiency_gain,
                )
                if frontier_points is not None
                else pd.DataFrame()
            )
            if frontier_points is not None:
                write_per_bracket_pareto_csvs(pareto_assets, output_dir)
            if not args.no_plots:
                plot_all(
                    pd.read_csv(manifest),
                    output_dir,
                    frontier_points,
                    args.figure_width == "one-column",
                    args.years * 12,
                    pareto_assets,
                    args.linear_x_limits,
                )
            print(f"Replotted outputs from {manifest}")
        if not args.skip_combined_summary:
            combined_points = pd.concat(
                [
                    pd.read_csv(
                        output_dir / "portfolio_efficiency_risk_points.csv"
                    )
                    for output_dir in output_dirs
                ],
                ignore_index=True,
            )
            write_pareto_markdown_table(
                combined_points,
                args.output_dir / "portfolio_per_bracket_pareto_table.md",
                args.pareto_min_incremental_efficiency_gain,
            )
        sensitivity_dirs = sorted(
            path.parent
            for path in args.output_dir.glob(
                "*/portfolio_efficiency_risk_points.csv"
            )
        )
        for sensitivity_dir in sensitivity_dirs:
            sensitivity = rebuild_hardware_budget_sensitivity(sensitivity_dir)
            if not sensitivity.empty:
                sensitivity.to_csv(
                    sensitivity_dir
                    / "portfolio_hardware_budget_sensitivity.csv",
                    index=False,
                )
        sensitivity_path = (
            args.output_dir / "portfolio_hardware_budget_sensitivity.csv"
        )
        sensitivity = collect_hardware_budget_sensitivity(args.output_dir)
        if sensitivity.empty and sensitivity_path.exists():
            sensitivity = pd.read_csv(sensitivity_path)
        elif not sensitivity.empty:
            sensitivity.to_csv(sensitivity_path, index=False)
        if not args.no_plots:
            plot_hardware_budget_sensitivity(sensitivity, args.output_dir)
            time_sensitivity = collect_hardware_budget_time_sensitivity(
                args.output_dir
            )
            if not time_sensitivity.empty:
                time_sensitivity.to_csv(
                    args.output_dir
                    / "portfolio_hardware_budget_time_sensitivity.csv",
                    index=False,
                )
                plot_hardware_budget_time_sensitivity(
                    time_sensitivity,
                    args.output_dir,
                    args.hardware_budget_time_sensitivity_color_limits,
                )
        return
    if args.resolution_months <= 0 or 18 % args.resolution_months != 0:
        raise SystemExit(
            "--resolution-months must be a positive divisor of 18 so the plateau occurs exactly after 18 months."
        )
    if args.runs < 2:
        raise SystemExit(
            "--runs must be at least 2 to calculate standard deviations."
        )
    if not 0.0 <= args.embargo_shock_probability <= 1.0:
        raise SystemExit(
            "--embargo-shock-probability must be between 0 and 1."
        )
    if args.embargo_rollback_months < 0:
        raise SystemExit("--embargo-rollback-months must be non-negative.")
    for option_name in (
        "sudden_break_even_probability_per_month",
        "gradual_break_even_probability_per_month",
        "capability_plateau_probability_per_month",
    ):
        if not 0.0 <= getattr(args, option_name) <= 1.0:
            raise SystemExit(
                f"--{option_name.replace('_', '-')} must be between 0 and 1."
            )
    if not 0.0 <= args.confidential_mixing_ratio <= 1.0:
        raise SystemExit(
            "--confidential-mixing-ratio must be between 0 and 1."
        )
    if not 0.0 <= args.confidential_mixing_penalty <= 1.0:
        raise SystemExit(
            "--confidential-mixing-penalty must be between 0 and 1."
        )
    if not 0.0 <= args.zero_risk_confidential_work_share <= 1.0:
        raise SystemExit(
            "--zero-risk-confidential-work-share must be between 0 and 1."
        )
    if args.counterfactual_skill_growth_rate_per_year < 0:
        raise SystemExit(
            "--counterfactual-skill-growth-rate-per-year must be non-negative."
        )
    if args.pareto_min_incremental_efficiency_gain < 0:
        raise SystemExit(
            "--pareto-min-incremental-efficiency-gain must be non-negative."
        )
    if (
        args.cloud_oss_hours_per_usage_unit_period is not None
        and args.cloud_oss_hours_per_usage_unit_period <= 0
    ):
        raise SystemExit(
            "--cloud-oss-hours-per-usage-unit-period must be positive."
        )
    if args.it_support_max_users < 0:
        raise SystemExit("--it-support-max-users must be non-negative.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibration = _hardware_calibration_profile(args.hardware_calibration)
    if (
        args.usd_per_hardware_capex_index is not None
        and args.usd_per_hardware_capex_index <= 0
    ):
        raise SystemExit("--usd-per-hardware-capex-index must be positive.")
    hardware_index_usd = (
        args.usd_per_hardware_capex_index
        if args.usd_per_hardware_capex_index is not None
        else float(calibration["target_unit_usd"])
        / HARDWARE_SCENARIOS["onprem_10pct_capacity"]["capex_units"]
    )
    base_config = SimulationConfig(
        years=args.years,
        resolution_months=args.resolution_months,
        runs=args.runs,
        seed=args.seed,
        plateau_quarter=18 // args.resolution_months,
        usd_per_service_cost_index_quarter=args.usd_per_service_cost_index_quarter,
        usd_per_hardware_capex_index=hardware_index_usd,
        hardware_calibration=args.hardware_calibration,
        onprem_capability_multiplier=float(
            calibration["onprem_capability_multiplier"]
        ),
        engineering_context=args.engineering_context,
        backend=_resolve_backend(args.backend),
        output_dir=args.output_dir,
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
        run_company_profile(
            name, employee_mix, base_config, args, company_profile_index(name)
        )
    sensitivity = collect_hardware_budget_sensitivity(args.output_dir)
    sensitivity.to_csv(
        args.output_dir / "portfolio_hardware_budget_sensitivity.csv",
        index=False,
    )
    time_sensitivity = collect_hardware_budget_time_sensitivity(
        args.output_dir
    )
    if not time_sensitivity.empty:
        time_sensitivity.to_csv(
            args.output_dir / "portfolio_hardware_budget_time_sensitivity.csv",
            index=False,
        )
    if not args.no_plots:
        plot_hardware_budget_sensitivity(sensitivity, args.output_dir)
        plot_hardware_budget_time_sensitivity(
            time_sensitivity,
            args.output_dir,
            args.hardware_budget_time_sensitivity_color_limits,
        )
    if not args.skip_combined_summary:
        combined_points = pd.concat(
            [
                pd.read_csv(
                    args.output_dir
                    / name
                    / "portfolio_efficiency_risk_points.csv"
                )
                for name in profiles
            ],
            ignore_index=True,
        )
        write_pareto_markdown_table(
            combined_points,
            args.output_dir / "portfolio_per_bracket_pareto_table.md",
            args.pareto_min_incremental_efficiency_gain,
        )


if __name__ == "__main__":
    main()
