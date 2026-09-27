#!/usr/bin/env python3
"""Run focused local validation cases and compare them with out-of-sample studies.

This is intentionally not a sweep: it runs six evidence-matched work profiles
using the narrow 2024/2025 validation configurations.  It never submits Slurm
jobs; --concurrency is passed only to the simulator's local process pool.
"""

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

import simulate_llm_efficiency as simulation
from simulate_llm_efficiency import (
    SIMULATION_DEFAULTS, SimulationConfig, _hardware_calibration_profile,
    _resolve_backend, load_simulation_configuration, run_simulation)

PROFILES = {
    "administration_heavy": {
        "software_engineering": 0.05,
        "administration": 0.70,
        "manual_labor": 0.10,
        "knowledge_work": 0.15,
        "creative_work": 0.0,
    },
    "knowledge_worker_heavy": {
        "software_engineering": 0.10,
        "administration": 0.20,
        "manual_labor": 0.05,
        "knowledge_work": 0.65,
        "creative_work": 0.0,
    },
    "software_engineering_heavy": {
        "software_engineering": 0.75,
        "administration": 0.10,
        "manual_labor": 0.05,
        "knowledge_work": 0.10,
        "creative_work": 0.0,
    },
}


# These are deliberately out-of-sample values from the evidence note. Values
# are percentage productivity/time effects; None means that the publication did
# not report an uncertainty interval suitable for the same metric.
VALIDATION_CASES = (
    {
        "year": 2024,
        "profile": "administration_heavy",
        "context": "mixed",
        "published_gain_pct": 14.0,
        "published_uncertainty_pct": None,
        "publication": "Brynjolfsson et al. (support): +14%",
    },
    {
        "year": 2024,
        "profile": "knowledge_worker_heavy",
        "context": "mixed",
        "published_gain_pct": 40.0,
        "published_uncertainty_pct": None,
        "publication": "Noy & Zhang (writing time): +40%",
    },
    {
        "year": 2024,
        "profile": "software_engineering_heavy",
        "context": "bounded",
        "published_gain_pct": 25.1,
        "published_uncertainty_pct": None,
        "publication": "Dell'Acqua et al. (bounded tasks): +25.1%",
    },
    {
        "year": 2025,
        "profile": "administration_heavy",
        "context": "mixed",
        "published_gain_pct": 25.0,
        "published_uncertainty_pct": None,
        "publication": "Dillon et al. (email time): +25%",
    },
    {
        "year": 2025,
        "profile": "software_engineering_heavy",
        "context": "bounded",
        "published_gain_pct": 26.08,
        "published_uncertainty_pct": 10.3,
        "publication": "Cui et al. (developers): +26.08% ± 10.3 SE",
    },
    {
        "year": 2025,
        "profile": "software_engineering_heavy",
        "context": "maintenance",
        "published_gain_pct": -19.0,
        "published_uncertainty_pct": 18.5,
        "publication": "Becker et al./METR (maintenance): -19% (95% CI -39 to -2)",
    },
)


# Traceable operational deployments discovered via ZenML's catalogue and then
# verified against the named providers' original case studies. Vendor case
# studies do not provide causal uncertainty.
OPERATIONAL_CASES = (
    {
        "year": 2024,
        "profile": "knowledge_worker_heavy",
        "context": "mixed",
        "published_gain_pct": 20.0,
        "published_low_pct": None,
        "published_high_pct": None,
        "publication": "Agmatix / AWS Bedrock (efficiency): >20%",
    },
    {
        "year": 2025,
        "profile": "administration_heavy",
        "context": "mixed",
        "published_gain_pct": 17.5,
        "published_low_pct": 15.0,
        "published_high_pct": 20.0,
        "publication": "Humach / Claude on Bedrock (operations): 15–20%",
    },
)


ADOPTION_SCENARIOS = (
    {
        "name": "all_personas",
        "label": "All personas",
        "personas": None,
        "color": "#2563eb",
        "marker": "o",
        "offset": -0.20,
    },
    {
        "name": "innovators_to_early_majority",
        "label": "Innovators, early adopters, and early majority",
        "personas": ("innovators", "early_adopters", "early_majority"),
        "color": "#0f766e",
        "marker": "^",
        "offset": 0.0,
    },
    {
        "name": "innovators_and_early_adopters",
        "label": "Innovators + early adopters",
        "personas": ("innovators", "early_adopters"),
        "color": "#7e22ce",
        "marker": "P",
        "offset": 0.20,
    },
    {
        "name": "innovators_only",
        "label": "Innovators",
        "personas": ("innovators",),
        "color": "#dc2626",
        "marker": "+",
        "offset": 0.40,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        type=int,
        default=200,
        help="Monte-Carlo runs per focused case (default: 200).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Local worker-process count per validation case.",
    )
    parser.add_argument(
        "--backend",
        choices=["numpy", "numba", "torch-mps", "auto"],
        default="auto",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("validation_outputs")
    )
    return parser.parse_args()


def build_config(
    config_path: Path,
    profile: dict[str, float],
    context: str,
    runs: int | None,
    concurrency: int,
    backend: str,
) -> SimulationConfig:
    load_simulation_configuration(config_path)
    defaults = SIMULATION_DEFAULTS
    calibration = _hardware_calibration_profile(
        str(defaults["hardware_calibration"])
    )
    # Validation configurations intentionally omit on-prem hardware scenarios,
    # so this unused index merely must remain positive for SimulationConfig.
    hardware_index_usd = float(calibration["target_unit_usd"]) / 165.0
    return SimulationConfig(
        users=int(defaults["users"]),
        years=float(defaults["years"]),
        resolution_months=int(defaults["resolution_months"]),
        runs=int(runs),
        concurrency=concurrency,
        seed=int(defaults["seed"]),
        plateau_quarter=int(defaults["plateau_quarter"]),
        base_cost_index_per_active_user_quarter=float(
            defaults["base_cost_index_per_active_user_quarter"]
        ),
        confidential_document_fraction=float(
            defaults["confidential_document_fraction"]
        ),
        zero_risk_confidential_work_share=float(
            defaults["zero_risk_confidential_work_share"]
        ),
        counterfactual_skill_growth_rate_per_year=float(
            defaults.get("counterfactual_skill_growth_rate_per_year", 0.015)
        ),
        usd_per_service_cost_index_quarter=float(
            defaults["usd_per_service_cost_index_quarter"]
        ),
        usd_per_hardware_capex_index=hardware_index_usd,
        hardware_calibration=str(defaults["hardware_calibration"]),
        onprem_capability_multiplier=float(
            calibration["onprem_capability_multiplier"]
        ),
        employee_mix=profile,
        engineering_context=context,
        adoption_propensity_concentration=float(
            defaults["adoption_propensity_concentration"]
        ),
        adoption_capability_elasticity=float(
            defaults["adoption_capability_elasticity"]
        ),
        effort_expectancy_friction=float(
            defaults["effort_expectancy_friction"]
        ),
        adaptability_productivity_beta=float(
            defaults["adaptability_productivity_beta"]
        ),
        early_majority_effort_expectancy_beta=float(
            defaults["early_majority_effort_expectancy_beta"]
        ),
        early_majority_adaptability_index=float(
            defaults["early_majority_adaptability_index"]
        ),
        improvement_base_probability=float(
            defaults["improvement_base_probability"]
        ),
        improvement_capability_weight=float(
            defaults["improvement_capability_weight"]
        ),
        improvement_probability_min=float(
            defaults["improvement_probability_min"]
        ),
        improvement_probability_max=float(
            defaults["improvement_probability_max"]
        ),
        ai_gain_min=float(defaults["ai_gain_min"]),
        ai_gain_max=float(defaults["ai_gain_max"]),
        zero_risk_feature_gain=float(defaults["zero_risk_feature_gain"]),
        embargo_shock_probability=float(defaults["embargo_shock_probability"]),
        embargo_rollback_months=int(defaults["embargo_rollback_months"]),
        sudden_break_even_probability_per_month=float(
            defaults["sudden_break_even_probability_per_month"]
        ),
        gradual_break_even_probability_per_month=float(
            defaults["gradual_break_even_probability_per_month"]
        ),
        capability_plateau_probability_per_month=float(
            defaults["capability_plateau_probability_per_month"]
        ),
        confidential_mixing_ratio=float(defaults["confidential_mixing_ratio"]),
        confidential_mixing_penalty=float(
            defaults["confidential_mixing_penalty"]
        ),
        cloud_oss_hours_per_usage_unit_period=defaults.get(
            "cloud_oss_hours_per_usage_unit_period"
        ),
        it_support_max_users=int(defaults["it_support_max_users"]),
        backend=_resolve_backend(backend),
        show_progress=False,
    )


def _set_adoption_scenario(adoption_scenario: dict[str, object]) -> None:
    persona_names = adoption_scenario["personas"]
    if persona_names is None:
        return
    personas = simulation.BASE_PERSONAS
    selected = personas[personas["persona"].isin(persona_names)].copy()
    if len(selected) != len(persona_names):
        raise ValueError(
            f"Missing persona in adoption scenario {adoption_scenario['name']}"
        )
    selected["share"] = selected["share"] / selected["share"].sum()
    simulation.BASE_PERSONAS = selected


def run_case(
    case: dict[str, object],
    adoption_scenario: dict[str, object],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config_path = Path(f"validation_config_{case['year']}.yaml")
    config = build_config(
        config_path,
        PROFILES[str(case["profile"])],
        str(case["context"]),
        args.runs,
        args.concurrency,
        args.backend,
    )
    _set_adoption_scenario(adoption_scenario)
    results = run_simulation(config)
    # The validation YAMLs intentionally keep global and European cloud paths.
    # The named published deployments are managed-cloud references, represented
    # by global_service; the European path is a sensitivity case, not evidence.
    results = results[results["service_provider"] == "global_service"].copy()
    final = results[results["period"] == results["period"].max()].copy()
    active_gain = (
        np.divide(
            final["mean_ai_efficiency_gain"].to_numpy(float),
            final["active_user_share"].to_numpy(float),
            out=np.full(len(final), np.nan),
            where=final["active_user_share"].to_numpy(float) > 0.0,
        )
        * config.annualization_factor
        * 100.0
    )
    final["active_user_annualized_gain_pct"] = active_gain
    final["workforce_annualized_gain_pct"] = (
        final["mean_ai_efficiency_gain"].to_numpy(float)
        * config.annualization_factor
        * 100.0
    )
    quantiles = final["workforce_annualized_gain_pct"].quantile(
        [0.05, 0.5, 0.95]
    )
    row = {
        **case,
        "adoption_scenario": adoption_scenario["name"],
        "model_p05_pct": float(quantiles.loc[0.05]),
        "model_median_pct": float(quantiles.loc[0.5]),
        "model_p95_pct": float(quantiles.loc[0.95]),
        "runs": config.runs,
        "users": config.users,
        "service_provider": "global_service",
        "metric": "workforce-wide annualized AI efficiency gain (%)",
    }
    return pd.DataFrame([row]), final


def _plot_comparison(
    ax: plt.Axes, summary: pd.DataFrame, operational: pd.DataFrame
) -> list[Line2D]:
    summary = summary.copy()
    summary["evidence_type"] = "peer-reviewed"
    operational = operational.copy()
    operational["evidence_type"] = "vendor-reported"
    combined = pd.concat([summary, operational], ignore_index=True)
    group_columns = [
        "year",
        "profile",
        "context",
        "publication",
        "evidence_type",
    ]
    groups = list(combined.groupby(group_columns, sort=False))
    groups.reverse()
    y = np.arange(len(groups))
    ax.axvline(0.0, color="#6b7280", linewidth=1.0, zorder=0)
    for index, (_, group) in enumerate(groups):
        for adoption_scenario in ADOPTION_SCENARIOS:
            row = group[
                group["adoption_scenario"] == adoption_scenario["name"]
            ].iloc[0]
            scenario_y = index + float(adoption_scenario["offset"])
            color = str(adoption_scenario["color"])
            ax.hlines(
                scenario_y,
                row["model_p05_pct"],
                row["model_p95_pct"],
                color=color,
                linewidth=1.8,
                zorder=2,
            )
            ax.scatter(
                row["model_median_pct"],
                scenario_y,
                color=color,
                marker=str(adoption_scenario["marker"]),
                s=25,
                zorder=3,
            )
        row = group.iloc[0]
        published = float(row["published_gain_pct"])
        marker = "s" if row["evidence_type"] == "vendor-reported" else "D"
        if "published_low_pct" in row and pd.notna(
            row.get("published_low_pct")
        ):
            low = published - float(row["published_low_pct"])
            high = float(row["published_high_pct"]) - published
            ax.errorbar(
                published,
                index,
                xerr=np.array([[low], [high]]),
                fmt=marker,
                color="#b45309",
                capsize=3,
                zorder=4,
            )
        else:
            uncertainty = row.get("published_uncertainty_pct")
            if pd.notna(uncertainty):
                ax.errorbar(
                    published,
                    index,
                    xerr=float(uncertainty),
                    fmt=marker,
                    color="#b45309",
                    capsize=3,
                    zorder=4,
                )
            else:
                ax.scatter(
                    published,
                    index,
                    marker=marker,
                    color="#b45309",
                    s=28,
                    zorder=4,
                )
    profile_labels = {
        "administration_heavy": "admin.",
        "knowledge_worker_heavy": "knowledge",
        "software_engineering_heavy": "Software. Eng.",
    }
    labels = [
        f"{row['year']} {profile_labels[row['profile']]}\n{str(row['publication']).split(' (')[0]}"
        for _, group in groups
        for row in [group.iloc[0]]
    ]
    ax.set_yticks(y, labels, fontsize=6.5)
    ax.set_xlabel("Workforce-wide annualized AI gain (%)", fontsize=8)
    ax.set_title("Validation comparisons", fontsize=9, pad=12)
    ax.grid(axis="x", alpha=0.25)
    ax.tick_params(axis="x", labelsize=7)
    return [
        *[
            Line2D(
                [0],
                [0],
                color=scenario["color"],
                marker=scenario["marker"],
                linewidth=1.8,
                markersize=4,
                label=scenario["label"],
            )
            for scenario in ADOPTION_SCENARIOS
        ],
        Line2D(
            [0],
            [0],
            color="#b45309",
            marker="D",
            linestyle="None",
            markersize=4,
            label="Peer reviewed",
        ),
        Line2D(
            [0],
            [0],
            color="#b45309",
            marker="s",
            linestyle="None",
            markersize=4,
            label="Vendor reported",
        ),
    ]


def plot_comparison(
    summary: pd.DataFrame, operational: pd.DataFrame, output_path: Path
) -> None:
    # 3.5 in is the conventional width of one column in a two-column paper.
    fig, ax = plt.subplots(figsize=(3.5, 5.8))
    handles = _plot_comparison(ax, summary, operational)
    for handle, label in zip(
        handles,
        (
            "All",
            "Innov.–early maj.",
            "Innov.+early adop.",
            "Innovators",
            "Peer reviewed",
            "Vendor",
        ),
    ):
        handle.set_label(label)
    ax.legend(
        handles=handles,
        loc="lower right",
        ncols=2,
        fontsize=4.8,
        frameon=True,
        framealpha=0.9,
        handlelength=1.0,
        borderpad=0.4,
    )
    fig.subplots_adjust(left=0.40, right=0.97, bottom=0.12, top=0.94)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_comparison_landscape(
    summary: pd.DataFrame, operational: pd.DataFrame, output_path: Path
) -> None:
    """Write a transposed, vertically compact version for a landscape column."""
    summary = summary.copy()
    summary["evidence_type"] = "peer-reviewed"
    operational = operational.copy()
    operational["evidence_type"] = "vendor-reported"
    combined = pd.concat([summary, operational], ignore_index=True)
    group_columns = [
        "year",
        "profile",
        "context",
        "publication",
        "evidence_type",
    ]
    groups = list(combined.groupby(group_columns, sort=False))
    groups.reverse()
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    ax.axhline(0.0, color="#6b7280", linewidth=0.8, zorder=0)
    for index, (_, group) in enumerate(groups):
        for adoption_scenario in ADOPTION_SCENARIOS:
            row = group[
                group["adoption_scenario"] == adoption_scenario["name"]
            ].iloc[0]
            scenario_x = index + float(adoption_scenario["offset"])
            color = str(adoption_scenario["color"])
            ax.vlines(
                scenario_x,
                row["model_p05_pct"],
                row["model_p95_pct"],
                color=color,
                linewidth=1.5,
                zorder=2,
            )
            ax.scatter(
                scenario_x,
                row["model_median_pct"],
                color=color,
                marker=str(adoption_scenario["marker"]),
                s=24,
                zorder=3,
            )
        row = group.iloc[0]
        published = float(row["published_gain_pct"])
        marker = "s" if row["evidence_type"] == "vendor-reported" else "D"
        if "published_low_pct" in row and pd.notna(
            row.get("published_low_pct")
        ):
            ax.errorbar(
                index,
                published,
                yerr=np.array(
                    [
                        [published - float(row["published_low_pct"])],
                        [float(row["published_high_pct"]) - published],
                    ]
                ),
                fmt=marker,
                color="#b45309",
                capsize=2,
                zorder=4,
            )
        elif pd.notna(row.get("published_uncertainty_pct")):
            ax.errorbar(
                index,
                published,
                yerr=float(row["published_uncertainty_pct"]),
                fmt=marker,
                color="#b45309",
                capsize=2,
                zorder=4,
            )
        else:
            ax.scatter(
                index,
                published,
                marker=marker,
                color="#b45309",
                s=24,
                zorder=4,
            )
    profile_labels = {
        "administration_heavy": "admin.",
        "knowledge_worker_heavy": "knowledge",
        "software_engineering_heavy": "Software. Eng.",
    }
    labels = [
        f"{row['year']} {profile_labels[row['profile']]}\n{str(row['publication']).split(' (')[0]}"
        for _, group in groups
        for row in [group.iloc[0]]
    ]
    ax.set_xticks(x, labels, rotation=55, ha="right", fontsize=5.5)
    ax.set_ylabel("Workforce-wide annualized AI gain (%)", fontsize=7)
    ax.set_title("Validation comparisons", fontsize=8, pad=5)
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(axis="y", alpha=0.25)
    handles = [
        *[
            Line2D(
                [0],
                [0],
                color=scenario["color"],
                marker=scenario["marker"],
                linewidth=1.5,
                markersize=4,
                label=scenario["label"],
            )
            for scenario in ADOPTION_SCENARIOS
        ],
        Line2D(
            [0],
            [0],
            color="#b45309",
            marker="D",
            linestyle="None",
            markersize=4,
            label="Peer reviewed",
        ),
        Line2D(
            [0],
            [0],
            color="#b45309",
            marker="s",
            linestyle="None",
            markersize=4,
            label="Vendor reported",
        ),
    ]
    for handle, label in zip(
        handles,
        (
            "All",
            "Innov.–early maj.",
            "Innov.+early adop.",
            "Innovators",
            "Peer reviewed",
            "Vendor",
        ),
    ):
        handle.set_label(label)
    ax.legend(
        handles=handles,
        loc="lower right",
        ncols=2,
        fontsize=4.8,
        frameon=True,
        framealpha=0.9,
        handlelength=1.0,
        borderpad=0.4,
    )
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.38, top=0.90)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_comparison_a4_full_width(
    summary: pd.DataFrame, operational: pd.DataFrame, output_path: Path
) -> None:
    """Write a compact A4-width comparison containing only All and Innovators."""
    scenarios = (ADOPTION_SCENARIOS[0], ADOPTION_SCENARIOS[3])
    summary = summary.copy()
    summary["evidence_type"] = "peer-reviewed"
    operational = operational.copy()
    operational["evidence_type"] = "vendor-reported"
    combined = pd.concat([summary, operational], ignore_index=True)
    group_columns = [
        "year",
        "profile",
        "context",
        "publication",
        "evidence_type",
    ]
    groups = list(combined.groupby(group_columns, sort=False))
    groups.reverse()
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(7.25, 3.1))
    ax.axhline(0.0, color="#6b7280", linewidth=0.8, zorder=0)
    offsets = (-0.12, 0.12)
    for index, (_, group) in enumerate(groups):
        for adoption_scenario, offset in zip(scenarios, offsets):
            row = group[
                group["adoption_scenario"] == adoption_scenario["name"]
            ].iloc[0]
            scenario_x = index + offset
            color = str(adoption_scenario["color"])
            ax.vlines(
                scenario_x,
                row["model_p05_pct"],
                row["model_p95_pct"],
                color=color,
                linewidth=1.5,
                zorder=2,
            )
            ax.scatter(
                scenario_x,
                row["model_median_pct"],
                color=color,
                marker=str(adoption_scenario["marker"]),
                s=24,
                zorder=3,
            )
        row = group.iloc[0]
        published = float(row["published_gain_pct"])
        marker = "s" if row["evidence_type"] == "vendor-reported" else "D"
        if "published_low_pct" in row and pd.notna(
            row.get("published_low_pct")
        ):
            error = np.array(
                [
                    [published - float(row["published_low_pct"])],
                    [float(row["published_high_pct"]) - published],
                ]
            )
            ax.errorbar(
                index,
                published,
                yerr=error,
                fmt=marker,
                color="#b45309",
                capsize=2,
                zorder=4,
            )
        elif pd.notna(row.get("published_uncertainty_pct")):
            ax.errorbar(
                index,
                published,
                yerr=float(row["published_uncertainty_pct"]),
                fmt=marker,
                color="#b45309",
                capsize=2,
                zorder=4,
            )
        else:
            ax.scatter(
                index,
                published,
                marker=marker,
                color="#b45309",
                s=24,
                zorder=4,
            )
    profile_labels = {
        "administration_heavy": "Admin.",
        "knowledge_worker_heavy": "Knowledge",
        "software_engineering_heavy": "Software eng.",
    }
    labels = [
        f"{row['year']} {profile_labels[row['profile']]}\n{str(row['publication']).split(' (')[0]}"
        for _, group in groups
        for row in [group.iloc[0]]
    ]
    ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("Workforce-wide annualized\nAI gain (%)", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    handles = [
        *[
            Line2D(
                [0],
                [0],
                color=scenario["color"],
                marker=scenario["marker"],
                linewidth=1.5,
                markersize=4,
                label=label,
            )
            for scenario, label in zip(scenarios, ("All", "Innovators"))
        ],
        Line2D(
            [0],
            [0],
            color="#b45309",
            marker="D",
            linestyle="None",
            markersize=4,
            label="Peer reviewed",
        ),
        Line2D(
            [0],
            [0],
            color="#b45309",
            marker="s",
            linestyle="None",
            markersize=4,
            label="Vendor reported",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncols=4,
        fontsize=7.5,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.4,
    )
    fig.subplots_adjust(left=0.15, right=0.995, bottom=0.43, top=0.80)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.runs < 2:
        raise SystemExit("--runs must be at least 2 to show uncertainty.")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    run_frames = []
    case_summaries: dict[tuple[int, str, str], pd.DataFrame] = {}
    for case in VALIDATION_CASES:
        for adoption_scenario in ADOPTION_SCENARIOS:
            print(
                f"Running {case['year']} {case['profile']} ({case['context']}; {adoption_scenario['name']})",
                flush=True,
            )
            summary, runs = run_case(case, adoption_scenario, args)
            case_id = f"{case['year']}_{case['profile']}_{case['context']}_{adoption_scenario['name']}"
            runs["case_id"] = case_id
            runs["adoption_scenario"] = adoption_scenario["name"]
            summaries.append(summary)
            run_frames.append(runs)
            case_summaries[
                (
                    int(case["year"]),
                    str(case["profile"]),
                    str(case["context"]),
                    str(adoption_scenario["name"]),
                )
            ] = summary
    comparison = pd.concat(summaries, ignore_index=True)
    operational_rows = []
    for case in OPERATIONAL_CASES:
        for adoption_scenario in ADOPTION_SCENARIOS:
            source = (
                case_summaries[
                    (
                        int(case["year"]),
                        str(case["profile"]),
                        str(case["context"]),
                        str(adoption_scenario["name"]),
                    )
                ]
                .iloc[0]
                .to_dict()
            )
            source.update(case)
            source["adoption_scenario"] = adoption_scenario["name"]
            operational_rows.append(source)
    operational = pd.DataFrame(operational_rows)
    comparison.to_csv(
        args.output_dir / "published_gain_comparison.csv", index=False
    )
    operational.to_csv(
        args.output_dir / "operational_case_comparison.csv", index=False
    )
    pd.concat(run_frames, ignore_index=True).to_csv(
        args.output_dir / "validation_run_level_results.csv", index=False
    )
    plot_comparison(
        comparison,
        operational,
        args.output_dir / "published_gain_comparison.png",
    )
    plot_comparison_landscape(
        comparison,
        operational,
        args.output_dir / "published_gain_comparison_landscape.png",
    )
    plot_comparison_a4_full_width(
        comparison,
        operational,
        args.output_dir / "published_gain_comparison_a4_full_width.png",
    )
    print(f"Wrote validation comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
