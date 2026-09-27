#!/usr/bin/env python3
"""Value LLM productivity gains as labour savings and gross-margin output."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from simulate_llm_efficiency import (
    HARDWARE_CALIBRATIONS, HARDWARE_SCENARIOS, SimulationConfig,
    _hardware_calibration_profile, _hardware_replica_count,
    _it_support_cost_index, _resolve_backend, _simulate_one_scenario,
    hardware_purchase_cost_usd, load_simulation_configuration)

SERVICE_TYPES = {
    "cloud": ("frontier_growth", "flat", "cloud_api_only", "global_service"),
    "european_service": (
        "frontier_growth",
        "flat",
        "cloud_api_only",
        "european_service",
    ),
    "in_prem": (
        "oss_growth_lagged",
        "flat",
        "onprem_full_capacity",
        "local_hardware",
    ),
    # cloud_api_only deliberately keeps this route on the Numba implementation;
    # eu_cloud_oss supplies the European OSS capability and privacy parameters.
    "cloud_oss_europe": (
        "oss_growth_lagged",
        "flat",
        "cloud_api_only",
        "eu_cloud_oss",
    ),
}


def load_economic_configuration(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict) or not isinstance(
        values.get("economics"), dict
    ):
        raise ValueError(
            "Economic configuration requires an 'economics' mapping."
        )
    base = Path(values.get("simulation_config", "simulation_config.yaml"))
    if not base.is_absolute():
        base = path.parent / base
    load_simulation_configuration(base)
    values["_path"] = path.resolve()
    return values


def weighted_value(
    values: dict[str, float], employee_mix: dict[str, float]
) -> float:
    missing = set(employee_mix) - set(values)
    if missing:
        raise ValueError(
            f"Economic values are missing task types: {', '.join(sorted(missing))}"
        )
    if not np.isclose(sum(employee_mix.values()), 1.0):
        raise ValueError("Employee mix must sum to 1.")
    return sum(
        float(employee_mix[task]) * float(values[task])
        for task in employee_mix
    )


def economic_config(
    values: dict,
    *,
    users: int,
    years: float,
    resolution_months: int,
    runs: int,
    seed: int,
    backend: str,
    employee_mix: dict[str, float],
    service_budget_per_person_usd: float | None = None,
    hardware_budget_usd: float | None = None,
    confidential_document_fraction: float = 0.0,
) -> SimulationConfig:
    calibration_name = "rtx6000-pro-gemma4-26b-moe"
    calibration = _hardware_calibration_profile(calibration_name)
    return SimulationConfig(
        users=users,
        years=years,
        resolution_months=resolution_months,
        runs=runs,
        seed=seed,
        concurrency=1,
        backend=backend,
        show_progress=False,
        employee_mix=employee_mix,
        hardware_calibration=calibration_name,
        participating_personas=(
            None
            if values.get("participating_personas") is None
            else tuple(values["participating_personas"])
        ),
        onprem_capability_multiplier=float(
            calibration["onprem_capability_multiplier"]
        ),
        usd_per_hardware_capex_index=float(calibration["target_unit_usd"])
        / HARDWARE_SCENARIOS["onprem_10pct_capacity"]["capex_units"],
        max_monthly_service_budget_usd=(
            None
            if service_budget_per_person_usd is None
            else users * service_budget_per_person_usd
        ),
        max_upfront_hardware_budget_usd=hardware_budget_usd,
        confidential_document_fraction=confidential_document_fraction,
        confidential_document_fraction_by_task=values["economics"].get(
            "confidential_document_fraction_by_task"
        ),
    )


def simulate_service_type(
    config: SimulationConfig, service_type: str
) -> pd.DataFrame:
    if service_type not in SERVICE_TYPES:
        raise ValueError(f"Unknown service type: {service_type}")
    model, token, hardware, provider = SERVICE_TYPES[service_type]
    frame = _simulate_one_scenario(
        (
            model,
            token,
            hardware,
            "maxed_out",
            "pay_per_use",
            "persona_choice",
            provider,
            config,
            config.seed,
        )
    )
    frame["service_type"] = service_type
    return frame


def value_cashflows(
    results: pd.DataFrame,
    economics: dict,
    employee_mix: dict[str, float],
    config: SimulationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_labor = weighted_value(
        economics["tvod_monthly_gross_eur"], employee_mix
    ) * float(economics["employer_cost_factor"])
    annual_margin = weighted_value(
        economics["annual_gross_margin_eur"], employee_mix
    )
    period_years = config.resolution_months / 12.0
    index_to_eur = config.usd_per_service_cost_index_quarter * float(
        economics["eur_per_usd"]
    )
    recurring_investment_eur = (
        results["cloud_cost_index"] + results["topup_cost_index"]
    ) * index_to_eur
    upfront_hardware_eur = 0.0
    if (results["service_type"] == "in_prem").any():
        replicas, budget_scale = _hardware_replica_count(
            config, "onprem_full_capacity"
        )
        upfront_hardware_eur = (
            hardware_purchase_cost_usd(
                config.hardware_calibration, "onprem_full_capacity"
            )
            * replicas
            * budget_scale
            * float(economics["eur_per_usd"])
        )
    administered_service_types = economics.get("administered_service_types")
    if administered_service_types is None:
        administration_mask = np.full(
            len(results), (results["service_type"] == "in_prem").any()
        )
    else:
        administration_mask = (
            results["service_type"].isin(administered_service_types).to_numpy()
        )
    recurring_investment_eur = (
        recurring_investment_eur
        + administration_mask
        * _it_support_cost_index("onprem_full_capacity", config)
        * index_to_eur
    )
    upfront_investment_eur = np.where(
        results["period"].to_numpy() == 0,
        np.where(
            results["service_type"] == "in_prem", upfront_hardware_eur, 0.0
        ),
        0.0,
    )
    gain = results["mean_efficiency_gain"].clip(lower=-1.0)
    frames = []
    for scenario, annual_base, rate in (
        (
            "labor_cost_savings",
            monthly_labor * 12.0,
            float(economics["labor_savings_discount_rate"]),
        ),
        (
            "output_gross_margin",
            annual_margin,
            float(economics["output_uplift_discount_rate"]),
        ),
    ):
        frame = results[["run", "period", "month", "service_type"]].copy()
        frame["scenario"] = scenario
        frame["discount_rate_annual"] = rate
        frame["productivity_gain"] = gain
        frame["gross_benefit_eur"] = (
            gain * config.users * annual_base * period_years
        )
        frame["recurring_ai_investment_eur"] = recurring_investment_eur
        frame["upfront_hardware_investment_eur"] = upfront_investment_eur
        frame["ai_investment_eur"] = (
            frame["recurring_ai_investment_eur"]
            + frame["upfront_hardware_investment_eur"]
        )
        frame["cashflow_eur"] = (
            frame["gross_benefit_eur"] - frame["ai_investment_eur"]
        )
        frame["discount_factor"] = (1.0 + rate) ** (
            -(frame["period"] + 1) * period_years
        )
        frame["discounted_cashflow_eur"] = (
            frame["cashflow_eur"] * frame["discount_factor"]
        )
        frame["horizon_month"] = (
            frame["period"] + 1
        ) * config.resolution_months
        frame["npv_eur"] = frame.groupby(
            ["run", "service_type", "scenario"], sort=False
        )["discounted_cashflow_eur"].cumsum()
        frames.append(frame)
    cashflows = pd.concat(frames, ignore_index=True)
    npv = cashflows[
        [
            "run",
            "service_type",
            "scenario",
            "period",
            "month",
            "horizon_month",
            "discount_rate_annual",
            "npv_eur",
        ]
    ].copy()
    return npv, cashflows


def run_economic_simulation(
    values: dict,
    config: SimulationConfig,
    service_types: tuple[str, ...] = tuple(SERVICE_TYPES),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = pd.concat(
        [simulate_service_type(config, name) for name in service_types],
        ignore_index=True,
    )
    employee_mix = config.employee_mix or {}
    return value_cashflows(results, values["economics"], employee_mix, config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("llm_efficiency_economics_config.yaml"),
    )
    parser.add_argument("--users", type=int, default=100)
    parser.add_argument("--years", type=float, default=3.0)
    parser.add_argument("--resolution-months", type=int, default=3)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--backend", choices=["numba", "numpy", "auto"], default="numba"
    )
    parser.add_argument("--employee-mix", required=True, help="task=share,...")
    parser.add_argument("--service-budget-usd", type=float, default=25.0)
    parser.add_argument("--hardware-budget-usd", type=float, default=15000.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("llm_efficiency_economics_outputs"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = load_economic_configuration(args.config)
    mix = {
        item.split("=", 1)[0]: float(item.split("=", 1)[1])
        for item in args.employee_mix.split(",")
    }
    config = economic_config(
        values,
        users=args.users,
        years=args.years,
        resolution_months=args.resolution_months,
        runs=args.runs,
        seed=args.seed,
        backend=_resolve_backend(args.backend),
        employee_mix=mix,
        service_budget_per_person_usd=args.service_budget_usd,
        hardware_budget_usd=args.hardware_budget_usd,
    )
    npv, cashflows = run_economic_simulation(values, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    npv.to_csv(args.output_dir / "npv_by_horizon.csv", index=False)
    cashflows.to_csv(
        args.output_dir / "cashflows_final_horizon.csv", index=False
    )
    print(
        f"Wrote {args.output_dir / 'npv_by_horizon.csv'} and {args.output_dir / 'cashflows_final_horizon.csv'}"
    )


if __name__ == "__main__":
    main()
