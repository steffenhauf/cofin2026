#!/usr/bin/env python3
"""Monte Carlo toy simulation of time-dependent LLM efficiency return rates."""

from __future__ import annotations

import argparse
import itertools
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from tqdm.auto import tqdm
import yaml


BASE_PERSONAS = pd.DataFrame(
    [
        {
            "persona": "innovators",
            "share": 0.025,
            "adoption_mean": 0.93,
            "usage_intensity": 1.35,
            "wait_tolerance": 0.85,
            "effort_expectancy_beta": 0.00,
            "adaptability_index": 0.20,
        },
        {
            "persona": "early_adopters",
            "share": 0.135,
            "adoption_mean": 0.82,
            "usage_intensity": 1.20,
            "wait_tolerance": 0.72,
            "effort_expectancy_beta": 0.15,
            "adaptability_index": 0.10,
        },
        {
            "persona": "early_majority",
            "share": 0.34,
            "adoption_mean": 0.61,
            "usage_intensity": 1.00,
            "wait_tolerance": 0.55,
            "effort_expectancy_beta": 0.375,
            "adaptability_index": 0.00,
        },
        {
            "persona": "late_majority",
            "share": 0.34,
            "adoption_mean": 0.38,
            "usage_intensity": 0.78,
            "wait_tolerance": 0.36,
            "effort_expectancy_beta": 0.45,
            "adaptability_index": -0.05,
        },
        {
            "persona": "laggards",
            "share": 0.16,
            "adoption_mean": 0.17,
            "usage_intensity": 0.48,
            "wait_tolerance": 0.18,
            "effort_expectancy_beta": 0.45,
            "adaptability_index": -0.10,
        },
    ]
)

DEFAULT_EMPLOYEE_TYPES = pd.DataFrame(
    [
        {
            "task": "software_engineering",
            "share": 0.36,
            "mean_gain": 0.10,
            "sigma": 0.060,
            "capability_fit": 0.85,
            "frontier_quality_work_share": 0.55,
        },
        {
            "task": "administration",
            "share": 0.34,
            "mean_gain": 0.17,
            "sigma": 0.080,
            "capability_fit": 1.00,
            "frontier_quality_work_share": 0.70,
        },
        {
            "task": "creative_work",
            "share": 0.10,
            "mean_gain": 0.10,
            "sigma": 0.120,
            "capability_fit": 0.90,
            "frontier_quality_work_share": 0.70,
        },
        {
            "task": "manual_labor",
            "share": 0.20,
            "mean_gain": 0.020,
            "sigma": 0.020,
            "capability_fit": 0.25,
            "frontier_quality_work_share": 0.15,
        },
        {
            "task": "knowledge_work",
            "share": 0.0,
            "mean_gain": 0.15,
            "sigma": 0.090,
            "capability_fit": 0.90,
            "frontier_quality_work_share": 0.60,
        },
    ]
)

ENGINEERING_CONTEXT_MULTIPLIERS = {
    "mixed": 1.00,
    "bounded": 2.00,
    "maintenance": -0.25,
}

DYNAMICS = {
    "frontier_quarterly_growth": 1.085,
    "frontier_capability_max": 2.60,
    "gradual_token_cost_increase": 0.85,
    "sudden_token_cost_multiplier": 1.85,
}

MODEL_SCENARIOS = {
    "frontier_growth": {"kind": "frontier", "growth": True, "lag_months": 0},
    "frontier_plateau": {"kind": "frontier", "growth": False, "lag_months": 0},
    "oss_growth_lagged": {"kind": "oss", "growth": True, "lag_months": 4},
    "oss_plateau_lagged": {"kind": "oss", "growth": False, "lag_months": 4},
}

SHOCK_SCENARIOS = (
    "sudden_break_even",
    "gradual_break_even",
    "capability_plateau",
)
SHOCK_COMBINATIONS = ("none",) + tuple(
    ",".join(name for index, name in enumerate(SHOCK_SCENARIOS) if mask & (1 << index))
    for mask in range(1, 1 << len(SHOCK_SCENARIOS))
)

SERVICE_PROVIDERS = {
    "global_service": {
        "capability_lag_months": 0,
        "embargo_affected": True,
        "confidential_capacity_share": 0.1,
    },
    "european_service": {
        "capability_lag_months": 6,
        "embargo_affected": False,
        "confidential_capacity_share": 0.2,
    },
    "local_hardware": {
        "capability_lag_months": 6,
        "embargo_affected": False,
        "confidential_capacity_share": 1.0,
    },
    "eu_cloud_oss": {
        "capability_lag_months": 6,
        "embargo_affected": False,
        "confidential_capacity_share": 0.5,
    },
}

TOKEN_SCENARIOS = {
    "flat": "flat",
    "gradual_break_even": "gradual",
    "sudden_break_even": "sudden",
}

ACCESS_PLANS = {
    "pay_per_use": {
        "subscription_cost_per_active_user_quarter": 0.0,
        "five_hour_allowance": np.inf,
        "weekly_allowance": np.inf,
        "topup": False,
    },
    "flatrate_limited": {
        "subscription_cost_per_active_user_quarter": 0.78,
        "five_hour_allowance": 0.82,
        "weekly_allowance": 0.92,
        "topup": False,
    },
    "flatrate_limited_topup": {
        "subscription_cost_per_active_user_quarter": 0.78,
        "five_hour_allowance": 0.82,
        "weekly_allowance": 0.92,
        "topup": True,
    },
}

LOCAL_FALLBACK_POLICIES = ("persona_choice", "local_default")

HARDWARE_SCENARIOS = {
    "cloud_api_only": {"capacity": 0.0, "utilization": 1.0, "capex_units": 0.0},
    "onprem_10pct_capacity": {"capacity": 0.10, "utilization": 1.0, "capex_units": 165.0},
    "onprem_50pct_capacity": {"capacity": 0.50, "utilization": 1.0, "capex_units": 610.0},
    "onprem_full_capacity": {"capacity": 1.00, "utilization": 1.0, "capex_units": 1120.0},
    "onprem_low_30pct_utilization": {"capacity": 1.00, "utilization": 0.30, "capex_units": 1120.0},
    "cloud_oss": {"capacity": 1.00, "utilization": 1.0, "capex_units": 0.0},
}

HARDWARE_REFRESH = {
    "maxed_out": {"refresh_factor": 1.0, "capability_bonus": 1.0},
    "follow_each_generation": {"refresh_factor": 1.45, "capability_bonus": 1.08},
}

HARDWARE_CALIBRATIONS = {
    "rtx6000-ada-gemma4-12b": {
        "target_unit_usd": 8500.0,
        "onprem_capability_multiplier": 0.88,
        "frontier_quality_work_share_multiplier": 0.0,
        "onprem_model_label": "Gemma 4 12B local model",
    },
    "rtx6000-pro-gemma4-26b-moe": {
        "target_unit_usd": 14500.0,
        "onprem_capability_multiplier": 0.94,
        "frontier_quality_work_share_multiplier": 1.0,
        "onprem_model_label": "Gemma 4 26B MoE local model",
    },
    "h100-gemma4-31b": {
        "target_unit_usd": 380000.0,
        "onprem_capability_multiplier": 0.97,
        "frontier_quality_work_share_multiplier": 1.0,
        "onprem_model_label": "Gemma 4 31B local model",
    },
    "b200-gemma4-31b": {
        "target_unit_usd": 750000.0,
        "onprem_capability_multiplier": 0.97,
        "frontier_quality_work_share_multiplier": 1.0,
        "onprem_model_label": "Gemma 4 31B local model",
    },
}
HARDWARE_BENCHMARKS_PATH = Path(__file__).with_name("hardware_benchmarks.csv")
_HARDWARE_BENCHMARKS: pd.DataFrame | None = None
USD_PER_HARDWARE_CAPEX_INDEX_DEFAULT = (
    HARDWARE_CALIBRATIONS["h100-gemma4-31b"]["target_unit_usd"] / HARDWARE_SCENARIOS["onprem_10pct_capacity"]["capex_units"]
)
MONTHS_PER_YEAR = 12
HARDWARE_AMORTIZATION_MONTHS = 36
MAX_CONFIDENTIAL_DOCUMENT_FRACTION = 0.75
DEFAULT_COUNTERFACTUAL_SKILL_GROWTH_RATE_PER_YEAR = 0.015
DEFAULT_CONCURRENCY = max(1, os.cpu_count() or 1)
# TVöD Bund E 12, step 3, from May 2026; employer-cost uplift is an explicit estimate.
TVOD_E12_STEP3_MONTHLY_GROSS_EUR = 5359.50
IT_EMPLOYER_COST_FACTOR = 1.25
IT_SUPPORT_FTE = 0.5
IT_SUPPORT_MONTHLY_COST_EUR = TVOD_E12_STEP3_MONTHLY_GROSS_EUR * IT_EMPLOYER_COST_FACTOR * IT_SUPPORT_FTE

DEFAULT_CONFIG_PATH = Path(__file__).with_name("simulation_config.yaml")
SIMULATION_DEFAULTS: dict[str, object] = {}
SWEEP_DEFAULTS: dict[str, dict[str, object]] = {}


def _required_mapping(config: dict[str, object], key: str) -> dict[str, object]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration key '{key}' must be a mapping.")
    return value


def load_simulation_configuration(path: Path | str = DEFAULT_CONFIG_PATH) -> Path:
    """Load model inputs and command defaults from a YAML configuration file."""
    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except OSError as exc:
        raise ValueError(f"Cannot read configuration file {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping.")

    model = _required_mapping(raw, "model")
    costs = _required_mapping(raw, "costs")
    simulation = _required_mapping(raw, "simulation")
    sweeps = _required_mapping(raw, "sweeps")

    def replace_mapping(target: dict, value: object, name: str) -> None:
        if not isinstance(value, dict):
            raise ValueError(f"Configuration key '{name}' must be a mapping.")
        target.clear()
        target.update(value)

    global DEFAULT_EMPLOYEE_TYPES, BASE_PERSONAS, LOCAL_FALLBACK_POLICIES
    global SHOCK_SCENARIOS, SHOCK_COMBINATIONS, MONTHS_PER_YEAR, HARDWARE_AMORTIZATION_MONTHS
    global MAX_CONFIDENTIAL_DOCUMENT_FRACTION, DEFAULT_COUNTERFACTUAL_SKILL_GROWTH_RATE_PER_YEAR
    global TVOD_E12_STEP3_MONTHLY_GROSS_EUR, IT_EMPLOYER_COST_FACTOR, IT_SUPPORT_FTE, IT_SUPPORT_MONTHLY_COST_EUR

    personas = model.get("personas")
    employee_types = model.get("employee_types")
    if not isinstance(personas, list) or not isinstance(employee_types, list):
        raise ValueError("model.personas and model.employee_types must be lists.")
    BASE_PERSONAS = pd.DataFrame(personas)
    for name, default in (("effort_expectancy_beta", 0.375), ("adaptability_index", 0.0)):
        if name not in BASE_PERSONAS:
            BASE_PERSONAS[name] = default
    DEFAULT_EMPLOYEE_TYPES = pd.DataFrame(employee_types)
    for name, target in (
        ("dynamics", DYNAMICS),
        ("engineering_context_multipliers", ENGINEERING_CONTEXT_MULTIPLIERS),
        ("scenarios", MODEL_SCENARIOS),
        ("service_providers", SERVICE_PROVIDERS),
        ("token_scenarios", TOKEN_SCENARIOS),
        ("access_plans", ACCESS_PLANS),
        ("hardware_scenarios", HARDWARE_SCENARIOS),
        ("hardware_refresh", HARDWARE_REFRESH),
        ("hardware_calibrations", HARDWARE_CALIBRATIONS),
    ):
        replace_mapping(target, model.get(name), f"model.{name}")
    SHOCK_SCENARIOS = tuple(model.get("shock_scenarios", ()))
    LOCAL_FALLBACK_POLICIES = tuple(model.get("local_fallback_policies", ()))
    SHOCK_COMBINATIONS = ("none",) + tuple(
        ",".join(name for index, name in enumerate(SHOCK_SCENARIOS) if mask & (1 << index))
        for mask in range(1, 1 << len(SHOCK_SCENARIOS))
    )
    MONTHS_PER_YEAR = int(costs["months_per_year"])
    HARDWARE_AMORTIZATION_MONTHS = int(costs["hardware_amortization_months"])
    MAX_CONFIDENTIAL_DOCUMENT_FRACTION = float(costs["max_confidential_document_fraction"])
    DEFAULT_COUNTERFACTUAL_SKILL_GROWTH_RATE_PER_YEAR = float(costs["counterfactual_skill_growth_rate_per_year"])
    TVOD_E12_STEP3_MONTHLY_GROSS_EUR = float(costs["tvod_e12_step3_monthly_gross_eur"])
    IT_EMPLOYER_COST_FACTOR = float(costs["it_employer_cost_factor"])
    IT_SUPPORT_FTE = float(costs["it_support_fte"])
    IT_SUPPORT_MONTHLY_COST_EUR = TVOD_E12_STEP3_MONTHLY_GROSS_EUR * IT_EMPLOYER_COST_FACTOR * IT_SUPPORT_FTE
    SIMULATION_DEFAULTS.clear()
    SIMULATION_DEFAULTS.update(simulation)
    SWEEP_DEFAULTS.clear()
    SWEEP_DEFAULTS.update(sweeps)
    return config_path


def configuration_path_from_argv() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_known_args()[0].config


load_simulation_configuration()

_NUMBA_MODULE = None
_NUMBA_IMPORT_ATTEMPTED = False


@dataclass(frozen=True)
class SimulationConfig:
    users: int = int(SIMULATION_DEFAULTS["users"])
    years: float = float(SIMULATION_DEFAULTS["years"])
    resolution_months: int = int(SIMULATION_DEFAULTS["resolution_months"])
    runs: int = int(SIMULATION_DEFAULTS["runs"])
    concurrency: int = DEFAULT_CONCURRENCY
    seed: int = int(SIMULATION_DEFAULTS["seed"])
    plateau_quarter: int = int(SIMULATION_DEFAULTS["plateau_quarter"])
    base_cost_index_per_active_user_quarter: float = float(SIMULATION_DEFAULTS["base_cost_index_per_active_user_quarter"])
    max_monthly_service_budget: float | None = None
    max_upfront_hardware_budget: float | None = None
    max_monthly_service_budget_usd: float | None = None
    max_upfront_hardware_budget_usd: float | None = None
    confidential_document_fraction: float = float(SIMULATION_DEFAULTS["confidential_document_fraction"])
    confidential_document_fraction_by_task: dict[str, float] | None = None
    zero_risk_confidential_work_share: float = float(SIMULATION_DEFAULTS["zero_risk_confidential_work_share"])
    counterfactual_skill_growth_enabled: bool = True
    counterfactual_skill_growth_rate_per_year: float = DEFAULT_COUNTERFACTUAL_SKILL_GROWTH_RATE_PER_YEAR
    usd_per_service_cost_index_quarter: float = float(SIMULATION_DEFAULTS["usd_per_service_cost_index_quarter"])
    usd_per_hardware_capex_index: float = USD_PER_HARDWARE_CAPEX_INDEX_DEFAULT
    hardware_calibration: str = str(SIMULATION_DEFAULTS["hardware_calibration"])
    onprem_capability_multiplier: float = 1.0
    employee_mix: dict[str, float] | None = None
    participating_personas: tuple[str, ...] | None = None
    engineering_context: str = str(SIMULATION_DEFAULTS["engineering_context"])
    adoption_propensity_concentration: float = float(SIMULATION_DEFAULTS["adoption_propensity_concentration"])
    adoption_capability_elasticity: float = float(SIMULATION_DEFAULTS["adoption_capability_elasticity"])
    effort_expectancy_friction: float = float(SIMULATION_DEFAULTS.get("effort_expectancy_friction", 0.20))
    adaptability_productivity_beta: float = float(SIMULATION_DEFAULTS.get("adaptability_productivity_beta", 0.599))
    early_majority_effort_expectancy_beta: float = float(SIMULATION_DEFAULTS.get("early_majority_effort_expectancy_beta", 0.375))
    early_majority_adaptability_index: float = float(SIMULATION_DEFAULTS.get("early_majority_adaptability_index", 0.0))
    improvement_base_probability: float = float(SIMULATION_DEFAULTS["improvement_base_probability"])
    improvement_capability_weight: float = float(SIMULATION_DEFAULTS["improvement_capability_weight"])
    improvement_probability_min: float = float(SIMULATION_DEFAULTS["improvement_probability_min"])
    improvement_probability_max: float = float(SIMULATION_DEFAULTS["improvement_probability_max"])
    ai_gain_min: float = float(SIMULATION_DEFAULTS["ai_gain_min"])
    ai_gain_max: float = float(SIMULATION_DEFAULTS["ai_gain_max"])
    zero_risk_feature_gain: float = float(SIMULATION_DEFAULTS["zero_risk_feature_gain"])
    embargo_shock_probability: float = float(SIMULATION_DEFAULTS["embargo_shock_probability"])
    embargo_rollback_months: int = int(SIMULATION_DEFAULTS["embargo_rollback_months"])
    stochastic_shocks: bool = False
    shock_combination: str = "none"
    sudden_break_even_probability_per_month: float = float(SIMULATION_DEFAULTS["sudden_break_even_probability_per_month"])
    gradual_break_even_probability_per_month: float = float(SIMULATION_DEFAULTS["gradual_break_even_probability_per_month"])
    capability_plateau_probability_per_month: float = float(SIMULATION_DEFAULTS["capability_plateau_probability_per_month"])
    confidential_mixing_ratio: float = float(SIMULATION_DEFAULTS["confidential_mixing_ratio"])
    confidential_mixing_penalty: float = float(SIMULATION_DEFAULTS["confidential_mixing_penalty"])
    cloud_oss_hours_per_usage_unit_period: float | None = None
    it_support_enabled: bool = True
    it_support_max_users: int = int(SIMULATION_DEFAULTS["it_support_max_users"])
    it_support_monthly_cost_eur: float = IT_SUPPORT_MONTHLY_COST_EUR
    backend: str = "numpy"
    show_progress: bool = True
    output_dir: Path = Path("outputs")

    @property
    def periods(self) -> int:
        return int(round(self.years * MONTHS_PER_YEAR / self.resolution_months))

    @property
    def months_per_period(self) -> int:
        return self.resolution_months

    @property
    def period_fraction_of_year(self) -> float:
        return self.months_per_period / MONTHS_PER_YEAR

    @property
    def annualization_factor(self) -> float:
        return 1.0 / self.period_fraction_of_year

    @property
    def quarters_per_period(self) -> float:
        return self.months_per_period / 3

    @property
    def service_budget_per_period(self) -> float | None:
        monthly_budget_index = self.monthly_service_budget_index
        if monthly_budget_index is None:
            return None
        return monthly_budget_index * self.months_per_period

    @property
    def monthly_service_budget_index(self) -> float | None:
        if self.max_monthly_service_budget_usd is not None:
            return self.max_monthly_service_budget_usd / max(
                1e-9, self.usd_per_service_cost_index_quarter / 3.0
            )
        return self.max_monthly_service_budget

    @property
    def upfront_hardware_budget_index(self) -> float | None:
        if self.max_upfront_hardware_budget_usd is not None:
            return self.max_upfront_hardware_budget_usd / max(1e-9, self.usd_per_hardware_capex_index)
        return self.max_upfront_hardware_budget

    @property
    def effective_confidential_document_fraction(self) -> float:
        if self.confidential_document_fraction_by_task is not None and self.employee_mix is not None:
            missing = set(self.employee_mix) - set(self.confidential_document_fraction_by_task)
            if missing:
                raise ValueError(f"Confidential-work shares are missing task types: {sorted(missing)}")
            baseline = sum(
                float(share) * float(self.confidential_document_fraction_by_task[task])
                for task, share in self.employee_mix.items()
            )
            return float(np.clip(
                self.confidential_document_fraction * baseline,
                0.0,
                MAX_CONFIDENTIAL_DOCUMENT_FRACTION,
            ))
        return float(np.clip(self.confidential_document_fraction, 0.0, MAX_CONFIDENTIAL_DOCUMENT_FRACTION))


def _counterfactual_gain_discount(period: int, config: SimulationConfig) -> float:
    if not config.counterfactual_skill_growth_enabled:
        return 1.0
    elapsed_years = period * config.period_fraction_of_year
    return (1.0 + config.counterfactual_skill_growth_rate_per_year) ** (-elapsed_years)


def _zero_risk_gain_access_multiplier(config: SimulationConfig) -> float:
    confidential_share = config.effective_confidential_document_fraction
    return 1.0 - confidential_share * (1.0 - config.zero_risk_confidential_work_share)


def _employee_types(config: SimulationConfig) -> pd.DataFrame:
    employee_types = DEFAULT_EMPLOYEE_TYPES.copy()
    if config.engineering_context not in ENGINEERING_CONTEXT_MULTIPLIERS:
        raise ValueError(f"Unknown engineering context: {config.engineering_context}")
    engineering_multiplier = ENGINEERING_CONTEXT_MULTIPLIERS[config.engineering_context]
    employee_types.loc[
        employee_types["task"] == "software_engineering",
        "mean_gain",
    ] *= engineering_multiplier
    if config.employee_mix is None:
        return employee_types

    missing = set(config.employee_mix) - set(employee_types["task"])
    if missing:
        raise ValueError(f"Unknown employee types in mix: {sorted(missing)}")

    employee_types["share"] = employee_types["task"].map(config.employee_mix).fillna(0.0)
    total_share = float(employee_types["share"].sum())
    if total_share <= 0:
        raise ValueError("Employee mix must assign a positive total share.")
    employee_types["share"] = employee_types["share"] / total_share
    return employee_types


def _hardware_calibration_profile(name: str) -> dict[str, float | str]:
    try:
        return HARDWARE_CALIBRATIONS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown hardware calibration: {name}") from exc


def _hardware_benchmark_row(hardware_calibration: str, hardware_name: str) -> pd.Series:
    global _HARDWARE_BENCHMARKS
    if _HARDWARE_BENCHMARKS is None:
        _HARDWARE_BENCHMARKS = pd.read_csv(HARDWARE_BENCHMARKS_PATH)
    match = _HARDWARE_BENCHMARKS[
        (_HARDWARE_BENCHMARKS["hardware_calibration"] == hardware_calibration)
        & (_HARDWARE_BENCHMARKS["hardware_scenario"] == hardware_name)
    ]
    if len(match) != 1:
        raise ValueError(f"Missing hardware benchmark for {hardware_calibration}/{hardware_name}")
    return match.iloc[0]


def hardware_purchase_cost_usd(hardware_calibration: str, hardware_name: str) -> float:
    if hardware_name in ("cloud_api_only", "cloud_oss"):
        return 0.0
    return float(_hardware_benchmark_row(hardware_calibration, hardware_name)["purchase_cost_usd"])


def _hardware_replica_count(config: SimulationConfig, hardware_name: str) -> tuple[int, float]:
    if hardware_name in ("cloud_api_only", "cloud_oss") or config.max_upfront_hardware_budget_usd is None:
        return 1, 1.0
    unit_cost_usd = hardware_purchase_cost_usd(config.hardware_calibration, hardware_name)
    replicas = int(config.max_upfront_hardware_budget_usd // unit_cost_usd)
    if replicas >= 1:
        return replicas, 1.0
    return 1, config.max_upfront_hardware_budget_usd / unit_cost_usd


def _hardware_benchmark(config: SimulationConfig, hardware_name: str) -> pd.Series:
    return _hardware_benchmark_row(config.hardware_calibration, hardware_name)


def _sample_users(rng: np.random.Generator, users: int, config: SimulationConfig) -> pd.DataFrame:
    tasks = _employee_types(config)
    persona_idx = rng.choice(len(BASE_PERSONAS), size=users, p=BASE_PERSONAS["share"].to_numpy())
    task_idx = rng.choice(len(tasks), size=users, p=tasks["share"].to_numpy())
    users_df = pd.DataFrame(
        {
            "persona": BASE_PERSONAS.iloc[persona_idx]["persona"].to_numpy(),
            "adoption_mean": BASE_PERSONAS.iloc[persona_idx]["adoption_mean"].to_numpy(float),
            "usage_intensity": BASE_PERSONAS.iloc[persona_idx]["usage_intensity"].to_numpy(float),
            "wait_tolerance": BASE_PERSONAS.iloc[persona_idx]["wait_tolerance"].to_numpy(float),
            "effort_expectancy_beta": BASE_PERSONAS.iloc[persona_idx]["effort_expectancy_beta"].to_numpy(float),
            "adaptability_index": BASE_PERSONAS.iloc[persona_idx]["adaptability_index"].to_numpy(float),
            "task": tasks.iloc[task_idx]["task"].to_numpy(),
            "capability_fit": tasks.iloc[task_idx]["capability_fit"].to_numpy(float),
            "mean_gain": tasks.iloc[task_idx]["mean_gain"].to_numpy(float),
            "sigma": tasks.iloc[task_idx]["sigma"].to_numpy(float),
            "frontier_quality_work_share": tasks.iloc[task_idx]["frontier_quality_work_share"].to_numpy(float),
        }
    )
    concentration = config.adoption_propensity_concentration
    means = users_df["adoption_mean"].to_numpy(float)
    users_df["adoption_propensity"] = rng.beta(means * concentration, (1.0 - means) * concentration)
    users_df["gain_moderation"] = (
        (1.0 - users_df["effort_expectancy_beta"] * config.effort_expectancy_friction)
        / (1.0 - config.early_majority_effort_expectancy_beta * config.effort_expectancy_friction)
        * (1.0 + config.adaptability_productivity_beta * (users_df["adaptability_index"] - config.early_majority_adaptability_index))
    )
    if config.participating_personas is None:
        users_df["ai_participant"] = True
    else:
        unknown = set(config.participating_personas) - set(BASE_PERSONAS["persona"])
        if unknown:
            raise ValueError(f"Unknown participating persona: {', '.join(sorted(unknown))}")
        users_df["ai_participant"] = users_df["persona"].isin(config.participating_personas)
    return users_df


def _user_vectors(users_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        users_df["adoption_propensity"].to_numpy(float, copy=True),
        users_df["adoption_mean"].to_numpy(float, copy=True),
        users_df["capability_fit"].to_numpy(float, copy=True),
        users_df["usage_intensity"].to_numpy(float, copy=True),
        users_df["wait_tolerance"].to_numpy(float, copy=True),
        users_df["gain_moderation"].to_numpy(float, copy=True),
        users_df["mean_gain"].to_numpy(float, copy=True),
        users_df["sigma"].to_numpy(float, copy=True),
    )


def _capability_multiplier(
    period: int,
    model_name: str,
    plateau_quarter: int,
    resolution_months: int,
    capability_lag_months: float = 0.0,
    force_plateau: bool = False,
) -> float:
    model = MODEL_SCENARIOS[model_name]
    effective_period = max(
        0.0,
        (period * resolution_months - model["lag_months"] - capability_lag_months) / 3.0,
    )
    if not model["growth"] or force_plateau:
        effective_period = min(effective_period, plateau_quarter)

    quarterly_growth = float(DYNAMICS["frontier_quarterly_growth"])
    multiplier = quarterly_growth**effective_period
    return min(multiplier, float(DYNAMICS["frontier_capability_max"]))


def _service_capability_multiplier(
    period: int,
    model_name: str,
    provider_name: str,
    plateau_quarter: int,
    resolution_months: int,
    embargo_shock: bool,
    config: SimulationConfig,
    force_plateau: bool = False,
) -> float:
    provider = SERVICE_PROVIDERS[provider_name]
    lag_months = provider["capability_lag_months"]
    if embargo_shock and provider["embargo_affected"]:
        lag_months += config.embargo_rollback_months
    return _capability_multiplier(
        period,
        model_name,
        plateau_quarter,
        resolution_months,
        lag_months,
        force_plateau,
    )


def _token_cost_multiplier(period: int, token_scenario: str, periods: int) -> float:
    if token_scenario == "flat":
        return 1.0
    if token_scenario == "gradual":
        return 1.0 + float(DYNAMICS["gradual_token_cost_increase"]) * period / max(1, periods - 1)
    if token_scenario == "sudden":
        return 1.0 if period < max(1, periods // 3) else float(DYNAMICS["sudden_token_cost_multiplier"])
    raise ValueError(f"Unknown token scenario: {token_scenario}")


def _period_probability(probability_per_month: float, resolution_months: int) -> float:
    return 1.0 - (1.0 - probability_per_month) ** resolution_months


def _shock_enabled(shock_combination: str, shock_name: str) -> bool:
    return shock_combination != "none" and shock_name in shock_combination.split(",")


def _stochastic_token_cost(
    period: int,
    periods: int,
    sudden_active: bool,
    gradual_start_period: int | None,
) -> float:
    if sudden_active:
        return float(DYNAMICS["sudden_token_cost_multiplier"])
    if gradual_start_period is None:
        return 1.0
    return 1.0 + float(DYNAMICS["gradual_token_cost_increase"]) * (period - gradual_start_period) / max(1, periods - 1 - gradual_start_period)


def _it_support_cost_index(hardware_name: str, config: SimulationConfig) -> float:
    if not config.it_support_enabled or hardware_name not in ("onprem_10pct_capacity", "onprem_50pct_capacity", "onprem_full_capacity", "onprem_low_30pct_utilization", "cloud_oss"):
        return 0.0
    if config.users > config.it_support_max_users:
        return 0.0
    monthly_cost = config.it_support_monthly_cost_eur * config.months_per_period
    return float(monthly_cost / config.usd_per_service_cost_index_quarter)


def _hardware_adjustments(
    users_df: pd.DataFrame,
    hardware_name: str,
    refresh_name: str,
    config: SimulationConfig,
) -> tuple[np.ndarray, float, float, float, float]:
    refresh = HARDWARE_REFRESH[refresh_name]
    benchmark = _hardware_benchmark(config, hardware_name) if hardware_name != "cloud_api_only" else None
    replicas, hardware_budget_scale = _hardware_replica_count(config, hardware_name)
    upfront_hardware_cost = (
        hardware_purchase_cost_usd(config.hardware_calibration, hardware_name) * replicas / config.usd_per_service_cost_index_quarter
        if benchmark is not None
        else 0.0
    )
    it_support_cost = _it_support_cost_index(hardware_name, config)
    hardware_budget_feasible = 1.0 if hardware_budget_scale >= 0.999999 else 0.0
    if hardware_name in ("cloud_api_only", "cloud_oss"):
        return np.ones(len(users_df)), it_support_cost, hardware_budget_feasible, hardware_budget_scale, 1.0

    available_slots = float(benchmark["max_concurrent_users"]) * replicas * hardware_budget_scale
    if available_slots <= 0.0:
        return np.ones(len(users_df)), 0.0, hardware_budget_feasible, hardware_budget_scale, 1.0

    priority = 0.25 + users_df["wait_tolerance"].to_numpy(float)
    use_multiplier = np.minimum(1.0, available_slots * priority / max(1e-9, float(priority.sum())))

    amortization_fraction = min(1.0, config.months_per_period / HARDWARE_AMORTIZATION_MONTHS)
    hardware_cost = upfront_hardware_cost * hardware_budget_scale * amortization_fraction + it_support_cost
    hardware_bonus = 1.0 + (refresh["capability_bonus"] - 1.0) * hardware_budget_scale
    return (
        np.clip(use_multiplier, 0.0, 1.0),
        hardware_cost,
        hardware_budget_feasible,
        hardware_budget_scale,
        hardware_bonus,
    )


def _user_effective_capability(
    capability: float,
    users_df: pd.DataFrame,
    hardware_name: str,
    config: SimulationConfig,
) -> np.ndarray:
    if hardware_name == "cloud_api_only":
        return np.full(len(users_df), capability)
    calibration = _hardware_calibration_profile(config.hardware_calibration)
    local_multiplier = float(calibration["onprem_capability_multiplier"])
    frontier_share = (
        users_df["frontier_quality_work_share"].to_numpy(float)
        * float(calibration["frontier_quality_work_share_multiplier"])
    )
    return capability * (frontier_share + (1.0 - frontier_share) * local_multiplier)


def _cloud_oss_cost_index(
    cloud_usage: np.ndarray,
    hardware_name: str,
    config: SimulationConfig,
) -> float:
    if hardware_name != "cloud_oss":
        return 0.0
    benchmark = _hardware_benchmark(config, hardware_name)
    hours = config.cloud_oss_hours_per_usage_unit_period
    if hours is None:
        hours = 24.0 * 30.4375 * config.months_per_period
    usd = (
        cloud_usage.sum()
        * float(benchmark["hourly_cloud_price_usd"])
        * hours
        / float(benchmark["max_concurrent_users"])
    )
    return float(usd / config.usd_per_service_cost_index_quarter)


def _apply_service_budget_limit(
    delivered_usage: np.ndarray,
    service_cost: float,
    config: SimulationConfig,
) -> tuple[np.ndarray, float, float]:
    budget = config.service_budget_per_period
    if budget is None or service_cost <= budget:
        return delivered_usage, service_cost, 1.0

    scale = budget / max(service_cost, 1e-9)
    return delivered_usage * scale, budget, scale


def _apply_access_plan(
    requested_usage: np.ndarray,
    active: np.ndarray,
    access_plan_name: str,
    token_cost: float,
    config: SimulationConfig,
) -> tuple[np.ndarray, float, float, float]:
    access_plan = ACCESS_PLANS[access_plan_name]
    if access_plan_name == "pay_per_use":
        cloud_cost = (
            config.base_cost_index_per_active_user_quarter
            * token_cost
            * requested_usage.sum()
            * config.quarters_per_period
        )
        return np.ones_like(requested_usage), float(cloud_cost), 0.0, 0.0

    allowance = min(access_plan["five_hour_allowance"], access_plan["weekly_allowance"]) * config.quarters_per_period
    included_usage = np.minimum(requested_usage, allowance)
    excess_usage = np.maximum(0.0, requested_usage - allowance)
    exhausted_share = float(np.mean((excess_usage > 0) & active))

    subscription_cost = (
        access_plan["subscription_cost_per_active_user_quarter"]
        * active.sum()
        * config.quarters_per_period
    )
    topup_cost = 0.0
    if access_plan["topup"]:
        topup_cost = (
            config.base_cost_index_per_active_user_quarter
            * token_cost
            * excess_usage.sum()
            * config.quarters_per_period
        )
        delivered_usage = requested_usage
    else:
        delivered_usage = included_usage

    access_multiplier = np.divide(
        delivered_usage,
        requested_usage,
        out=np.ones_like(requested_usage),
        where=requested_usage > 0,
    )
    return access_multiplier, float(subscription_cost + topup_cost), float(topup_cost), exhausted_share


def _local_hardware_multiplier(
    wait_multiplier: np.ndarray,
    wait_tolerance: np.ndarray,
    hardware_name: str,
    hardware_budget_scale: float,
    local_fallback_policy: str,
) -> tuple[np.ndarray, np.ndarray]:
    if hardware_name in ("cloud_api_only", "cloud_oss"):
        zeros = np.zeros_like(wait_multiplier)
        return zeros, zeros
    local_capacity = wait_multiplier
    if local_fallback_policy == "local_default":
        fallback = local_capacity
    else:
        fallback = local_capacity * wait_tolerance
    return np.clip(local_capacity, 0.0, 1.0), np.clip(fallback, 0.0, 1.0)


def _effective_usage_with_local_fallback(
    base_requested_usage: np.ndarray,
    wait_multiplier: np.ndarray,
    wait_tolerance: np.ndarray,
    active: np.ndarray,
    hardware_name: str,
    hardware_budget_scale: float,
    local_fallback_policy: str,
    access_plan_name: str,
    token_cost: float,
    service_provider_name: str,
    config: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float, np.ndarray]:
    confidential_requested = base_requested_usage * config.effective_confidential_document_fraction
    non_confidential_requested = base_requested_usage - confidential_requested
    local_multiplier, fallback_multiplier = _local_hardware_multiplier(
        wait_multiplier, wait_tolerance, hardware_name, hardware_budget_scale, local_fallback_policy
    )
    provider_confidential_share = float(SERVICE_PROVIDERS[service_provider_name]["confidential_capacity_share"])
    service_confidential_usage = confidential_requested * provider_confidential_share

    if hardware_name == "cloud_oss":
        frontier_usage = non_confidential_requested
        cloud_usage = frontier_usage + service_confidential_usage
        cloud_cost = _cloud_oss_cost_index(cloud_usage, hardware_name, config)
        frontier_usage, cloud_cost, service_budget_scale = _apply_service_budget_limit(
            frontier_usage, cloud_cost, config
        )
        service_confidential_usage *= service_budget_scale
        topup_cost = 0.0
        exhausted_share = 0.0
    else:
        access_multiplier, cloud_cost, topup_cost, exhausted_share = _apply_access_plan(
            non_confidential_requested, active, access_plan_name, token_cost, config
        )
        frontier_usage = non_confidential_requested * access_multiplier
        frontier_usage, cloud_cost, service_budget_scale = _apply_service_budget_limit(frontier_usage, cloud_cost, config)

    local_confidential_usage = confidential_requested * local_multiplier
    unavailable_confidential = np.maximum(
        0.0, confidential_requested - service_confidential_usage - local_confidential_usage
    )
    fallback_usage = np.maximum(0.0, non_confidential_requested - frontier_usage) * fallback_multiplier
    effective_usage = frontier_usage + service_confidential_usage + local_confidential_usage + fallback_usage
    confidential_penalty = np.clip(
        1.0
        - config.confidential_mixing_ratio
        * config.confidential_mixing_penalty
        * np.divide(unavailable_confidential, base_requested_usage, out=np.zeros_like(base_requested_usage), where=base_requested_usage > 0),
        0.0,
        1.0,
    )
    return effective_usage, confidential_requested, cloud_cost, topup_cost, exhausted_share, service_budget_scale, confidential_penalty

def _torch_module_and_device():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for --backend torch-mps. Install torch in the cofin environment.") from exc

    if not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch is installed, but Apple MPS is not available on this machine/environment.")
    return torch, torch.device("mps")


def _resolve_backend(backend: str) -> str:
    if backend == "numpy":
        return "numpy"
    if backend == "numba":
        _numba_module()
        return "numba"
    if backend == "torch-mps":
        _torch_module_and_device()
        return "torch-mps"
    if backend == "auto":
        try:
            _numba_module()
        except RuntimeError:
            pass
        else:
            return "numba"
        try:
            _torch_module_and_device()
        except RuntimeError:
            return "numpy"
        return "torch-mps"
    raise ValueError(f"Unknown backend: {backend}")


def _simulate_one_scenario(args: tuple) -> pd.DataFrame:
    if args[2] != "cloud_oss" and config_backend(args) == "torch-mps":
        return _simulate_one_scenario_torch_mps(args)
    if args[2] != "cloud_oss" and config_backend(args) == "numba":
        return _simulate_one_scenario_numba(args)

    model_name, token_name, hardware_name, refresh_name, access_plan_name, local_fallback_policy, service_provider_name, config, scenario_seed = args[:9]
    shock_combination = args[9] if len(args) > 9 else "none"
    sudden_enabled = config.stochastic_shocks and _shock_enabled(shock_combination, "sudden_break_even")
    gradual_enabled = config.stochastic_shocks and _shock_enabled(shock_combination, "gradual_break_even")
    plateau_enabled = config.stochastic_shocks and _shock_enabled(shock_combination, "capability_plateau")
    rng = np.random.default_rng(scenario_seed)
    run_rows = []

    for run in range(config.runs):
        users_df = _sample_users(rng, config.users, config)
        return_rates: list[float] = []
        sudden_active = False
        gradual_start_period: int | None = None
        shock_plateau_quarter: float | None = None

        for period in range(config.periods):
            gain_discount = _counterfactual_gain_discount(period, config)
            zero_risk_access_multiplier = _zero_risk_gain_access_multiplier(config)
            # Draw for every asset so paired runs retain the same random-stream
            # alignment even when only global service is embargo-exposed.
            embargo_draw = rng.random()
            embargo_shock = bool(
                service_provider_name == "global_service"
                and embargo_draw < config.embargo_shock_probability
            )
            if sudden_enabled and not sudden_active and rng.random() < _period_probability(config.sudden_break_even_probability_per_month, config.resolution_months):
                sudden_active = True
            if gradual_enabled and gradual_start_period is None and rng.random() < _period_probability(config.gradual_break_even_probability_per_month, config.resolution_months):
                gradual_start_period = period
            if plateau_enabled and shock_plateau_quarter is None and rng.random() < _period_probability(config.capability_plateau_probability_per_month, config.resolution_months):
                provider_lag = SERVICE_PROVIDERS[service_provider_name]["capability_lag_months"]
                rollback = config.embargo_rollback_months if embargo_shock and SERVICE_PROVIDERS[service_provider_name]["embargo_affected"] else 0
                shock_plateau_quarter = max(
                    0.0,
                    (period * config.resolution_months - MODEL_SCENARIOS[model_name]["lag_months"] - provider_lag - rollback) / 3.0,
                )
            capability = _service_capability_multiplier(
                period,
                model_name,
                service_provider_name,
                config.plateau_quarter if shock_plateau_quarter is None else shock_plateau_quarter,
                config.resolution_months,
                embargo_shock,
                config,
                shock_plateau_quarter is not None,
            )
            token_cost = (
                _stochastic_token_cost(period, config.periods, sudden_active, gradual_start_period)
                if config.stochastic_shocks
                else _token_cost_multiplier(period, TOKEN_SCENARIOS[token_name], config.periods)
            )
            (
                wait_multiplier,
                hardware_cost,
                hardware_budget_feasible,
                hardware_budget_scale,
                hardware_bonus,
            ) = _hardware_adjustments(users_df, hardware_name, refresh_name, config)
            effective_capability = _user_effective_capability(capability, users_df, hardware_name, config)

            adoption_multiplier = 1.0 + config.adoption_capability_elasticity * (capability - 1.0)
            adoption_prob = np.clip(users_df["adoption_propensity"].to_numpy(float) * adoption_multiplier, 0.02, 0.98)
            active = (rng.random(config.users) < adoption_prob) & users_df["ai_participant"].to_numpy(bool)
            base_requested_usage = users_df["usage_intensity"].to_numpy(float) * active
            (
                effective_usage,
                confidential_requested,
                cloud_cost,
                topup_cost,
                exhausted_share,
                service_budget_scale,
                confidential_penalty,
            ) = _effective_usage_with_local_fallback(
                base_requested_usage,
                wait_multiplier,
                users_df["wait_tolerance"].to_numpy(float),
                active,
                hardware_name,
                hardware_budget_scale,
                local_fallback_policy,
                access_plan_name,
                token_cost,
                service_provider_name,
                config,
            )
            delivered_usage_multiplier = np.divide(
                effective_usage,
                base_requested_usage,
                out=np.ones_like(base_requested_usage),
                where=base_requested_usage > 0,
            )

            capability_fit = users_df["capability_fit"].to_numpy(float)
            improvement_prob = np.clip(
                config.improvement_base_probability
                + config.improvement_capability_weight * effective_capability * capability_fit,
                config.improvement_probability_min,
                config.improvement_probability_max,
            )
            improved = rng.random(config.users) < improvement_prob

            gain_scale = config.resolution_months / 12.0
            sampled_gains = rng.normal(
                users_df["mean_gain"].to_numpy(float) * gain_scale * effective_capability * hardware_bonus * capability_fit,
                users_df["sigma"].to_numpy(float) * gain_scale,
            )
            ai_realized_gain = np.clip(
                sampled_gains * users_df["gain_moderation"].to_numpy(float),
                config.ai_gain_min * gain_scale,
                config.ai_gain_max * gain_scale,
            ) * active * improved * delivered_usage_multiplier * confidential_penalty * gain_discount
            zero_risk_gain = (
                config.zero_risk_feature_gain
                * gain_scale
                * users_df["adoption_mean"].to_numpy(float)
                * users_df["ai_participant"].to_numpy(float)
                * zero_risk_access_multiplier
                * gain_discount
            )
            realized_gain = zero_risk_gain + ai_realized_gain
            efficiency_index = 1.0 + realized_gain

            total_cost = cloud_cost + hardware_cost
            mean_gain = float(np.mean(efficiency_index - 1.0))
            cost_per_increment = total_cost / max(1e-9, np.sum(efficiency_index - 1.0))
            return_rate = mean_gain / (total_cost / config.users) if total_cost > 1e-9 else np.nan
            annualized_return_rate = return_rate * config.annualization_factor

            if np.isfinite(return_rate):
                return_rates.append(return_rate)
            risk = float(np.std(return_rates, ddof=1)) if len(return_rates) > 1 else 0.0

            run_rows.append(
                {
                    "scenario": f"{model_name}|{token_name}|{access_plan_name}|{hardware_name}|{refresh_name}|{local_fallback_policy}|{service_provider_name}|{shock_combination}",
                    "model": model_name,
                    "token_cost": token_name,
                    "access_plan": access_plan_name,
                    "hardware": hardware_name,
                    "hardware_refresh": refresh_name,
                    "local_fallback": local_fallback_policy,
                    "service_provider": service_provider_name,
                    "shock_combination": shock_combination,
                    "embargo_shock": embargo_shock,
                    "sudden_break_even_shock": sudden_active,
                    "gradual_break_even_shock": gradual_start_period is not None,
                    "capability_plateau_shock": shock_plateau_quarter is not None,
                    "backend": config.backend,
                    "run": run,
                    "period": period,
                    "month": period * config.resolution_months,
                    "capability_multiplier": capability,
                    "mean_efficiency_index": float(np.mean(efficiency_index)),
                    "mean_efficiency_gain": mean_gain,
                    "zero_risk_efficiency_gain": float(np.mean(zero_risk_gain)),
                    "mean_ai_efficiency_gain": float(np.mean(ai_realized_gain)),
                    "negative_ai_gain_share": float(np.mean(ai_realized_gain < 0.0)),
                    "active_user_share": float(np.mean(active)),
                    "delivered_usage_share": float(effective_usage.sum() / max(1e-9, base_requested_usage.sum())),
                    "confidential_usage_share": float(confidential_requested.sum() / max(1e-9, base_requested_usage.sum())),
                    "service_budget_scale": float(service_budget_scale),
                    "hardware_budget_feasible": float(hardware_budget_feasible),
                    "hardware_budget_scale": float(hardware_budget_scale),
                    "rate_limit_exhausted_share": exhausted_share,
                    "cloud_cost_index": float(cloud_cost),
                    "topup_cost_index": float(topup_cost),
                    "hardware_cost_index": float(hardware_cost),
                    "total_cost_index": float(total_cost),
                    "cost_per_efficiency_increment": float(cost_per_increment),
                    "return_rate": float(return_rate),
                    "annualized_return_rate": float(annualized_return_rate),
                    "risk": risk,
                    "annualized_risk": risk * config.annualization_factor,
                }
            )

    return pd.DataFrame(run_rows)


def _numba_module():
    global _NUMBA_MODULE, _NUMBA_IMPORT_ATTEMPTED
    if _NUMBA_MODULE is not None:
        return _NUMBA_MODULE
    if not _NUMBA_IMPORT_ATTEMPTED:
        _NUMBA_IMPORT_ATTEMPTED = True
        try:
            import numba
        except ImportError as exc:
            raise RuntimeError("Numba is required for --backend numba. Install numba in the cofin environment.") from exc
        _NUMBA_MODULE = numba
    return _NUMBA_MODULE


def _model_code(model_name: str, resolution_months: int) -> tuple[int, float]:
    model = MODEL_SCENARIOS[model_name]
    return (1 if model["growth"] else 0), float(model["lag_months"])


def _token_code(token_name: str) -> int:
    if token_name == "flat":
        return 0
    if token_name == "gradual_break_even":
        return 1
    if token_name == "sudden_break_even":
        return 2
    raise ValueError(f"Unknown token scenario: {token_name}")


def _access_plan_code(access_plan_name: str) -> int:
    if access_plan_name == "pay_per_use":
        return 0
    if access_plan_name == "flatrate_limited":
        return 1
    if access_plan_name == "flatrate_limited_topup":
        return 2
    raise ValueError(f"Unknown access plan: {access_plan_name}")


def _local_fallback_policy_code(local_fallback_policy: str) -> int:
    if local_fallback_policy == "persona_choice":
        return 0
    if local_fallback_policy == "local_default":
        return 1
    raise ValueError(f"Unknown local fallback policy: {local_fallback_policy}")


def _numba_simulator():
    numba = _numba_module()

    @numba.njit(cache=True)
    def simulate_run(
        adoption_propensity: np.ndarray,
        adoption_mean: np.ndarray,
        ai_participant: np.ndarray,
        capability_fit: np.ndarray,
        usage_intensity: np.ndarray,
        wait_tolerance: np.ndarray,
        gain_moderation: np.ndarray,
        mean_gain_base: np.ndarray,
        sigma: np.ndarray,
        frontier_quality_work_share: np.ndarray,
        wait_multiplier: np.ndarray,
        adoption_draws: np.ndarray,
        improvement_draws: np.ndarray,
        normal_draws: np.ndarray,
        embargo_draws: np.ndarray,
        sudden_break_even_draws: np.ndarray,
        gradual_break_even_draws: np.ndarray,
        capability_plateau_draws: np.ndarray,
        model_growth: int,
        model_lag: float,
        plateau_quarter: int,
        token_mode: int,
        access_plan_mode: int,
        allowance: float,
        subscription_cost_per_active_user_quarter: float,
        base_cost_index_per_active_user_quarter: float,
        quarters_per_period: float,
        service_budget_per_period: float,
        confidential_document_fraction: float,
        hardware_is_cloud: int,
        local_fallback_default: int,
        scenario_capability_multiplier: float,
        hardware_bonus: float,
        hardware_cost: float,
        hardware_budget_feasible: float,
        hardware_budget_scale: float,
        resolution_months: int,
        adoption_capability_elasticity: float,
        improvement_base_probability: float,
        improvement_capability_weight: float,
        improvement_probability_min: float,
        improvement_probability_max: float,
        ai_gain_min: float,
        ai_gain_max: float,
        zero_risk_feature_gain: float,
        counterfactual_skill_growth_enabled: int,
        counterfactual_skill_growth_rate_per_year: float,
        zero_risk_confidential_access_multiplier: float,
        provider_capability_lag_months: float,
        provider_embargo_affected: int,
        provider_handles_confidential: int,
        embargo_shock_probability: float,
        embargo_rollback_months: int,
        confidential_mixing_ratio: float,
        confidential_mixing_penalty: float,
        stochastic_shocks: int,
        sudden_enabled: int,
        gradual_enabled: int,
        plateau_enabled: int,
        sudden_break_even_probability_per_month: float,
        gradual_break_even_probability_per_month: float,
        capability_plateau_probability_per_month: float,
    ) -> np.ndarray:
        periods, users = adoption_draws.shape
        metrics = np.empty((periods, 23), dtype=np.float64)
        return_count = 0
        return_mean = 0.0
        return_m2 = 0.0
        sudden_active = 0
        gradual_start_period = -1
        shock_plateau_quarter = -1.0

        for period in range(periods):
            gain_discount = 1.0
            if counterfactual_skill_growth_enabled == 1:
                gain_discount = (1.0 + counterfactual_skill_growth_rate_per_year) ** (-(period * resolution_months / 12.0))
            embargo_shock = provider_embargo_affected == 1 and embargo_draws[period] < embargo_shock_probability
            if stochastic_shocks == 1:
                period_sudden_probability = 1.0 - (1.0 - sudden_break_even_probability_per_month) ** resolution_months
                period_gradual_probability = 1.0 - (1.0 - gradual_break_even_probability_per_month) ** resolution_months
                period_plateau_probability = 1.0 - (1.0 - capability_plateau_probability_per_month) ** resolution_months
                if sudden_enabled == 1 and sudden_active == 0 and sudden_break_even_draws[period] < period_sudden_probability:
                    sudden_active = 1
                if gradual_enabled == 1 and gradual_start_period < 0 and gradual_break_even_draws[period] < period_gradual_probability:
                    gradual_start_period = period
                if plateau_enabled == 1 and shock_plateau_quarter < 0.0 and capability_plateau_draws[period] < period_plateau_probability:
                    rollback = embargo_rollback_months if embargo_shock and provider_embargo_affected == 1 else 0
                    shock_plateau_quarter = max(
                        0.0,
                        (period * resolution_months - model_lag - provider_capability_lag_months - rollback) / 3.0,
                    )
            effective_period = (
                period * resolution_months - model_lag - provider_capability_lag_months
            ) / 3.0
            if embargo_shock:
                effective_period -= embargo_rollback_months / 3.0
            if effective_period < 0:
                effective_period = 0
            effective_plateau_quarter = plateau_quarter
            if shock_plateau_quarter >= 0.0:
                effective_plateau_quarter = shock_plateau_quarter
            if (model_growth == 0 or shock_plateau_quarter >= 0.0) and effective_period > effective_plateau_quarter:
                effective_period = effective_plateau_quarter

            capability = 1.085**effective_period
            if capability > 2.60:
                capability = 2.60
            effective_capability = capability * scenario_capability_multiplier

            if stochastic_shocks == 1:
                if sudden_active == 1:
                    token_cost = 1.85
                elif gradual_start_period >= 0:
                    token_cost = 1.0 + 0.85 * (period - gradual_start_period) / max(1, periods - 1 - gradual_start_period)
                else:
                    token_cost = 1.0
            elif token_mode == 0:
                token_cost = 1.0
            elif token_mode == 1:
                token_cost = 1.0 + 0.85 * period / max(1, periods - 1)
            else:
                token_cost = 1.0 if period < max(1, periods // 3) else 1.85

            active_count = 0.0
            frontier_requested_usage_sum = 0.0
            included_usage_sum = 0.0
            excess_usage_sum = 0.0
            exhausted_active = 0.0
            confidential_requested_sum = 0.0
            base_requested_usage_sum = 0.0

            for user in range(users):
                adoption_prob = adoption_propensity[user] * (1.0 + adoption_capability_elasticity * (capability - 1.0))
                if adoption_prob < 0.02:
                    adoption_prob = 0.02
                elif adoption_prob > 0.98:
                    adoption_prob = 0.98

                active = ai_participant[user] and adoption_draws[period, user] < adoption_prob
                if active:
                    active_count += 1.0

                base_requested_usage = usage_intensity[user] * (1.0 if active else 0.0)
                base_requested_usage_sum += base_requested_usage
                confidential_requested = base_requested_usage * confidential_document_fraction
                non_confidential_requested = base_requested_usage - confidential_requested
                confidential_requested_sum += confidential_requested
                frontier_requested_usage_sum += non_confidential_requested

                delivered_usage = non_confidential_requested
                if access_plan_mode != 0:
                    if non_confidential_requested <= allowance:
                        included_usage_sum += non_confidential_requested
                    else:
                        included_usage_sum += allowance
                        excess_usage_sum += non_confidential_requested - allowance
                        if active:
                            exhausted_active += 1.0
                    if access_plan_mode == 1 and non_confidential_requested > allowance:
                        delivered_usage = allowance
            if access_plan_mode == 0:
                cloud_cost = (
                    base_cost_index_per_active_user_quarter
                    * token_cost
                    * frontier_requested_usage_sum
                    * quarters_per_period
                )
                topup_cost = 0.0
                exhausted_share = 0.0
            else:
                subscription_cost = (
                    subscription_cost_per_active_user_quarter
                    * active_count
                    * quarters_per_period
                )
                topup_cost = 0.0
                if access_plan_mode == 2:
                    topup_cost = (
                        base_cost_index_per_active_user_quarter
                        * token_cost
                        * excess_usage_sum
                        * quarters_per_period
                    )
                cloud_cost = subscription_cost + topup_cost
                exhausted_share = exhausted_active / users

            service_budget_scale = 1.0
            if service_budget_per_period >= 0.0 and cloud_cost > service_budget_per_period:
                service_budget_scale = service_budget_per_period / max(cloud_cost, 1e-9)
                cloud_cost = service_budget_per_period

            effective_usage_sum = 0.0
            total_ai_gain = 0.0
            negative_ai_gain_count = 0.0
            zero_risk_gain_sum = 0.0

            for user in range(users):
                adoption_prob = adoption_propensity[user] * (1.0 + adoption_capability_elasticity * (capability - 1.0))
                if adoption_prob < 0.02:
                    adoption_prob = 0.02
                elif adoption_prob > 0.98:
                    adoption_prob = 0.98

                active = ai_participant[user] and adoption_draws[period, user] < adoption_prob
                base_requested_usage = usage_intensity[user] * (1.0 if active else 0.0)
                confidential_requested = base_requested_usage * confidential_document_fraction
                non_confidential_requested = base_requested_usage - confidential_requested

                frontier_delivered = non_confidential_requested
                if access_plan_mode != 0 and access_plan_mode == 1 and non_confidential_requested > allowance:
                    frontier_delivered = allowance
                frontier_delivered *= service_budget_scale

                local_multiplier = 0.0
                fallback_multiplier = 0.0
                if hardware_is_cloud != 1:
                    local_multiplier = wait_multiplier[user]
                    if local_multiplier > 1.0:
                        local_multiplier = 1.0
                    if local_fallback_default == 1:
                        fallback_multiplier = local_multiplier
                    else:
                        fallback_multiplier = local_multiplier * wait_tolerance[user]
                    if fallback_multiplier > 1.0:
                        fallback_multiplier = 1.0

                service_confidential_usage = confidential_requested if provider_handles_confidential == 1 else 0.0
                local_confidential_usage = confidential_requested * local_multiplier
                if provider_handles_confidential == 1:
                    local_confidential_usage = 0.0
                unavailable_confidential = confidential_requested - service_confidential_usage - local_confidential_usage
                fallback_usage = (non_confidential_requested - frontier_delivered) * fallback_multiplier
                if fallback_usage < 0.0:
                    fallback_usage = 0.0
                effective_usage = frontier_delivered + service_confidential_usage + local_confidential_usage + fallback_usage
                effective_usage_sum += effective_usage
                confidential_penalty = 1.0
                if base_requested_usage > 0.0:
                    confidential_penalty = 1.0 - confidential_mixing_ratio * confidential_mixing_penalty * unavailable_confidential / base_requested_usage
                    if confidential_penalty < 0.0:
                        confidential_penalty = 0.0

                user_effective_capability = effective_capability
                if hardware_is_cloud != 1:
                    user_effective_capability = capability * (
                        frontier_quality_work_share[user] + (1.0 - frontier_quality_work_share[user]) * scenario_capability_multiplier
                    )
                improvement_prob = improvement_base_probability + improvement_capability_weight * user_effective_capability * capability_fit[user]
                if improvement_prob < improvement_probability_min:
                    improvement_prob = improvement_probability_min
                elif improvement_prob > improvement_probability_max:
                    improvement_prob = improvement_probability_max

                improved = improvement_draws[period, user] < improvement_prob
                sampled_gain = (
                    normal_draws[period, user] * sigma[user] * resolution_months / 12.0
                    + mean_gain_base[user]
                    * resolution_months
                    / 12.0
                    * user_effective_capability
                    * hardware_bonus
                    * capability_fit[user]
                )
                sampled_gain *= gain_moderation[user]
                scaled_ai_gain_min = ai_gain_min * resolution_months / 12.0
                scaled_ai_gain_max = ai_gain_max * resolution_months / 12.0
                if sampled_gain < scaled_ai_gain_min:
                    sampled_gain = scaled_ai_gain_min
                elif sampled_gain > scaled_ai_gain_max:
                    sampled_gain = scaled_ai_gain_max

                access_multiplier = 1.0
                if base_requested_usage > 0.0:
                    access_multiplier = effective_usage / base_requested_usage

                ai_realized_gain = sampled_gain * (1.0 if active else 0.0) * (1.0 if improved else 0.0) * access_multiplier * confidential_penalty * gain_discount
                total_ai_gain += ai_realized_gain
                if ai_realized_gain < 0.0:
                    negative_ai_gain_count += 1.0
                zero_risk_gain_sum += zero_risk_feature_gain * resolution_months / 12.0 * adoption_mean[user] * ai_participant[user] * zero_risk_confidential_access_multiplier * gain_discount

            total_cost = cloud_cost + hardware_cost
            total_realized_gain = total_ai_gain + zero_risk_gain_sum
            mean_gain = total_realized_gain / users
            cost_per_increment = total_cost / max(1e-9, total_realized_gain)
            return_rate = mean_gain / (total_cost / users) if total_cost > 1e-9 else np.nan

            risk = 0.0
            if not np.isnan(return_rate):
                return_count += 1
                delta = return_rate - return_mean
                return_mean += delta / return_count
                return_m2 += delta * (return_rate - return_mean)
                if return_count > 1:
                    risk = (return_m2 / (return_count - 1)) ** 0.5

            metrics[period, 0] = period
            metrics[period, 1] = period * resolution_months
            metrics[period, 2] = capability
            metrics[period, 3] = 1.0 + mean_gain
            metrics[period, 4] = mean_gain
            metrics[period, 5] = active_count / users
            metrics[period, 6] = effective_usage_sum / max(1e-9, base_requested_usage_sum)
            metrics[period, 7] = confidential_requested_sum / max(1e-9, base_requested_usage_sum)
            metrics[period, 8] = service_budget_scale
            metrics[period, 9] = hardware_budget_feasible
            metrics[period, 10] = hardware_budget_scale
            metrics[period, 11] = exhausted_share
            metrics[period, 12] = cloud_cost
            metrics[period, 13] = topup_cost
            metrics[period, 14] = total_cost
            metrics[period, 15] = cost_per_increment
            metrics[period, 16] = zero_risk_gain_sum / users
            metrics[period, 17] = total_ai_gain / users
            metrics[period, 18] = negative_ai_gain_count / users
            metrics[period, 19] = 1.0 if embargo_shock else 0.0
            metrics[period, 20] = 1.0 if sudden_active == 1 else 0.0
            metrics[period, 21] = 1.0 if gradual_start_period >= 0 else 0.0
            metrics[period, 22] = 1.0 if shock_plateau_quarter >= 0.0 else 0.0
            # return_rate and risk are appended separately to keep the compact matrix small.

        return metrics

    return simulate_run


def _simulate_one_scenario_numba(args: tuple) -> pd.DataFrame:
    model_name, token_name, hardware_name, refresh_name, access_plan_name, local_fallback_policy, service_provider_name, config, scenario_seed = args[:9]
    shock_combination = args[9] if len(args) > 9 else "none"
    rng = np.random.default_rng(scenario_seed)
    users = config.users
    periods = config.periods
    simulate_run = _numba_simulator()
    access_plan = ACCESS_PLANS[access_plan_name]
    allowance = min(access_plan["five_hour_allowance"], access_plan["weekly_allowance"]) * config.quarters_per_period
    service_budget_per_period = config.service_budget_per_period
    model_growth, model_lag = _model_code(model_name, config.resolution_months)
    token_mode = _token_code(token_name)
    access_plan_mode = _access_plan_code(access_plan_name)
    local_fallback_default = _local_fallback_policy_code(local_fallback_policy)
    scenario_label = f"{model_name}|{token_name}|{access_plan_name}|{hardware_name}|{refresh_name}|{local_fallback_policy}|{service_provider_name}|{shock_combination}"
    total_rows = config.runs * periods
    hardware_is_cloud = 1 if hardware_name == "cloud_api_only" else 0

    data: dict[str, np.ndarray] = {
        "scenario": np.full(total_rows, scenario_label, dtype=object),
        "model": np.full(total_rows, model_name, dtype=object),
        "token_cost": np.full(total_rows, token_name, dtype=object),
        "access_plan": np.full(total_rows, access_plan_name, dtype=object),
        "hardware": np.full(total_rows, hardware_name, dtype=object),
        "hardware_refresh": np.full(total_rows, refresh_name, dtype=object),
        "local_fallback": np.full(total_rows, local_fallback_policy, dtype=object),
        "service_provider": np.full(total_rows, service_provider_name, dtype=object),
        "shock_combination": np.full(total_rows, shock_combination, dtype=object),
        "backend": np.full(total_rows, "numba", dtype=object),
        "run": np.empty(total_rows, dtype=np.int64),
        "period": np.empty(total_rows, dtype=np.int64),
        "month": np.empty(total_rows, dtype=np.int64),
        "capability_multiplier": np.empty(total_rows, dtype=np.float64),
        "mean_efficiency_index": np.empty(total_rows, dtype=np.float64),
        "mean_efficiency_gain": np.empty(total_rows, dtype=np.float64),
        "zero_risk_efficiency_gain": np.empty(total_rows, dtype=np.float64),
        "mean_ai_efficiency_gain": np.empty(total_rows, dtype=np.float64),
        "negative_ai_gain_share": np.empty(total_rows, dtype=np.float64),
        "active_user_share": np.empty(total_rows, dtype=np.float64),
        "delivered_usage_share": np.empty(total_rows, dtype=np.float64),
        "confidential_usage_share": np.empty(total_rows, dtype=np.float64),
        "service_budget_scale": np.empty(total_rows, dtype=np.float64),
        "hardware_budget_feasible": np.empty(total_rows, dtype=np.float64),
        "hardware_budget_scale": np.empty(total_rows, dtype=np.float64),
        "rate_limit_exhausted_share": np.empty(total_rows, dtype=np.float64),
        "cloud_cost_index": np.empty(total_rows, dtype=np.float64),
        "topup_cost_index": np.empty(total_rows, dtype=np.float64),
        "hardware_cost_index": np.empty(total_rows, dtype=np.float64),
        "total_cost_index": np.empty(total_rows, dtype=np.float64),
        "cost_per_efficiency_increment": np.empty(total_rows, dtype=np.float64),
        "return_rate": np.empty(total_rows, dtype=np.float64),
        "annualized_return_rate": np.empty(total_rows, dtype=np.float64),
        "risk": np.empty(total_rows, dtype=np.float64),
        "annualized_risk": np.empty(total_rows, dtype=np.float64),
        "embargo_shock": np.empty(total_rows, dtype=np.float64),
        "sudden_break_even_shock": np.empty(total_rows, dtype=np.float64),
        "gradual_break_even_shock": np.empty(total_rows, dtype=np.float64),
        "capability_plateau_shock": np.empty(total_rows, dtype=np.float64),
    }

    for run in range(config.runs):
        users_df = _sample_users(rng, users, config)
        adoption_propensity, adoption_mean, capability_fit, usage_intensity, wait_tolerance, gain_moderation, mean_gain_base, sigma = _user_vectors(users_df)
        frontier_quality_work_share = users_df["frontier_quality_work_share"].to_numpy(float, copy=True)
        (
            wait_multiplier,
            hardware_cost,
            hardware_budget_feasible,
            hardware_budget_scale,
            hardware_bonus,
        ) = _hardware_adjustments(users_df, hardware_name, refresh_name, config)
        scenario_capability_multiplier = config.onprem_capability_multiplier if hardware_name != "cloud_api_only" else 1.0

        adoption_draws = rng.random((periods, users))
        improvement_draws = rng.random((periods, users))
        normal_draws = rng.normal(size=(periods, users))
        embargo_draws = rng.random(periods)
        sudden_break_even_draws = rng.random(periods)
        gradual_break_even_draws = rng.random(periods)
        capability_plateau_draws = rng.random(periods)
        provider = SERVICE_PROVIDERS[service_provider_name]
        metrics = simulate_run(
            adoption_propensity,
            adoption_mean,
            users_df["ai_participant"].to_numpy(bool, copy=True),
            capability_fit,
            usage_intensity,
            wait_tolerance,
            gain_moderation,
            mean_gain_base,
            sigma,
            frontier_quality_work_share,
            wait_multiplier,
            adoption_draws,
            improvement_draws,
            normal_draws,
            embargo_draws,
            sudden_break_even_draws,
            gradual_break_even_draws,
            capability_plateau_draws,
            model_growth,
            model_lag,
            config.plateau_quarter,
            token_mode,
            access_plan_mode,
            allowance,
            access_plan["subscription_cost_per_active_user_quarter"],
            config.base_cost_index_per_active_user_quarter,
            config.quarters_per_period,
            -1.0 if service_budget_per_period is None else service_budget_per_period,
            config.effective_confidential_document_fraction,
            hardware_is_cloud,
            local_fallback_default,
            scenario_capability_multiplier,
            hardware_bonus,
            hardware_cost,
            hardware_budget_feasible,
            hardware_budget_scale,
            config.resolution_months,
            config.adoption_capability_elasticity,
            config.improvement_base_probability,
            config.improvement_capability_weight,
            config.improvement_probability_min,
            config.improvement_probability_max,
            config.ai_gain_min,
            config.ai_gain_max,
            config.zero_risk_feature_gain,
            1 if config.counterfactual_skill_growth_enabled else 0,
            config.counterfactual_skill_growth_rate_per_year,
            _zero_risk_gain_access_multiplier(config),
            provider["capability_lag_months"],
            1 if provider["embargo_affected"] else 0,
            1 if provider["confidential_capacity_share"] >= 0.999999 else 0,
            config.embargo_shock_probability,
            config.embargo_rollback_months,
            config.confidential_mixing_ratio,
            config.confidential_mixing_penalty,
            1 if config.stochastic_shocks else 0,
            1 if _shock_enabled(shock_combination, "sudden_break_even") else 0,
            1 if _shock_enabled(shock_combination, "gradual_break_even") else 0,
            1 if _shock_enabled(shock_combination, "capability_plateau") else 0,
            config.sudden_break_even_probability_per_month,
            config.gradual_break_even_probability_per_month,
            config.capability_plateau_probability_per_month,
        )

        row_slice = slice(run * periods, (run + 1) * periods)
        data["run"][row_slice] = run
        data["period"][row_slice] = metrics[:, 0].astype(np.int64)
        data["month"][row_slice] = metrics[:, 1].astype(np.int64)
        data["capability_multiplier"][row_slice] = metrics[:, 2]
        data["mean_efficiency_index"][row_slice] = metrics[:, 3]
        data["mean_efficiency_gain"][row_slice] = metrics[:, 4]
        data["zero_risk_efficiency_gain"][row_slice] = metrics[:, 16]
        data["mean_ai_efficiency_gain"][row_slice] = metrics[:, 17]
        data["negative_ai_gain_share"][row_slice] = metrics[:, 18]
        data["embargo_shock"][row_slice] = metrics[:, 19]
        data["sudden_break_even_shock"][row_slice] = metrics[:, 20]
        data["gradual_break_even_shock"][row_slice] = metrics[:, 21]
        data["capability_plateau_shock"][row_slice] = metrics[:, 22]
        data["active_user_share"][row_slice] = metrics[:, 5]
        data["delivered_usage_share"][row_slice] = metrics[:, 6]
        data["confidential_usage_share"][row_slice] = metrics[:, 7]
        data["service_budget_scale"][row_slice] = metrics[:, 8]
        data["hardware_budget_feasible"][row_slice] = metrics[:, 9]
        data["hardware_budget_scale"][row_slice] = metrics[:, 10]
        data["rate_limit_exhausted_share"][row_slice] = metrics[:, 11]
        data["cloud_cost_index"][row_slice] = metrics[:, 12]
        data["topup_cost_index"][row_slice] = metrics[:, 13]
        data["hardware_cost_index"][row_slice] = hardware_cost
        data["total_cost_index"][row_slice] = metrics[:, 14]
        data["cost_per_efficiency_increment"][row_slice] = metrics[:, 15]
        data["return_rate"][row_slice] = np.divide(
            metrics[:, 4],
            metrics[:, 14] / users,
            out=np.full(periods, np.nan, dtype=np.float64),
            where=metrics[:, 14] > 1e-9,
        )
        data["annualized_return_rate"][row_slice] = data["return_rate"][row_slice] * config.annualization_factor

        run_returns = data["return_rate"][row_slice]
        run_risk = np.zeros(periods, dtype=np.float64)
        for idx in range(periods):
            finite_returns = run_returns[: idx + 1][np.isfinite(run_returns[: idx + 1])]
            if len(finite_returns) > 1:
                run_risk[idx] = float(np.std(finite_returns, ddof=1))
        data["risk"][row_slice] = run_risk
        data["annualized_risk"][row_slice] = run_risk * config.annualization_factor

    return pd.DataFrame(data)


def config_backend(args: tuple[str, str, str, str, str, str, str, SimulationConfig, int]) -> str:
    return args[7].backend


def _simulate_one_scenario_torch_mps(args: tuple) -> pd.DataFrame:
    model_name, token_name, hardware_name, refresh_name, access_plan_name, local_fallback_policy, service_provider_name, config, scenario_seed = args[:9]
    shock_combination = args[9] if len(args) > 9 else "none"
    torch, device = _torch_module_and_device()
    rng = np.random.default_rng(scenario_seed)
    run_rows = []

    for run in range(config.runs):
        users_df = _sample_users(rng, config.users, config)
        (
            wait_multiplier_np,
            hardware_cost,
            hardware_budget_feasible,
            hardware_budget_scale,
            hardware_bonus,
        ) = _hardware_adjustments(users_df, hardware_name, refresh_name, config)
        scenario_capability_multiplier = config.onprem_capability_multiplier if hardware_name != "cloud_api_only" else 1.0

        adoption_propensity = torch.as_tensor(users_df["adoption_propensity"].to_numpy(float, copy=True), dtype=torch.float32, device=device)
        adoption_mean = torch.as_tensor(users_df["adoption_mean"].to_numpy(float, copy=True), dtype=torch.float32, device=device)
        ai_participant = torch.as_tensor(users_df["ai_participant"].to_numpy(bool, copy=True), dtype=torch.bool, device=device)
        capability_fit = torch.as_tensor(users_df["capability_fit"].to_numpy(float, copy=True), dtype=torch.float32, device=device)
        usage_intensity = torch.as_tensor(users_df["usage_intensity"].to_numpy(float, copy=True), dtype=torch.float32, device=device)
        wait_tolerance = torch.as_tensor(users_df["wait_tolerance"].to_numpy(float, copy=True), dtype=torch.float32, device=device)
        gain_moderation = torch.as_tensor(users_df["gain_moderation"].to_numpy(float, copy=True), dtype=torch.float32, device=device)
        mean_gain_base = torch.as_tensor(users_df["mean_gain"].to_numpy(float, copy=True), dtype=torch.float32, device=device)
        sigma = torch.as_tensor(users_df["sigma"].to_numpy(float, copy=True), dtype=torch.float32, device=device)
        wait_multiplier = torch.as_tensor(wait_multiplier_np, dtype=torch.float32, device=device)

        return_rates: list[float] = []
        sudden_active = False
        gradual_start_period: int | None = None
        shock_plateau_quarter: float | None = None

        for period in range(config.periods):
            gain_discount = _counterfactual_gain_discount(period, config)
            zero_risk_access_multiplier = _zero_risk_gain_access_multiplier(config)
            period_seed = scenario_seed + run * 100003 + period
            cpu_generator = torch.Generator(device="cpu").manual_seed(period_seed)
            embargo_shock = bool(service_provider_name == "global_service" and np.random.default_rng(period_seed + 17).random() < config.embargo_shock_probability)
            shock_rng = np.random.default_rng(period_seed + 31)
            if config.stochastic_shocks and _shock_enabled(shock_combination, "sudden_break_even") and not sudden_active and shock_rng.random() < _period_probability(config.sudden_break_even_probability_per_month, config.resolution_months):
                sudden_active = True
            if config.stochastic_shocks and _shock_enabled(shock_combination, "gradual_break_even") and gradual_start_period is None and shock_rng.random() < _period_probability(config.gradual_break_even_probability_per_month, config.resolution_months):
                gradual_start_period = period
            if config.stochastic_shocks and _shock_enabled(shock_combination, "capability_plateau") and shock_plateau_quarter is None and shock_rng.random() < _period_probability(config.capability_plateau_probability_per_month, config.resolution_months):
                provider = SERVICE_PROVIDERS[service_provider_name]
                rollback = config.embargo_rollback_months if embargo_shock and provider["embargo_affected"] else 0
                shock_plateau_quarter = max(
                    0.0,
                    (period * config.resolution_months - MODEL_SCENARIOS[model_name]["lag_months"] - provider["capability_lag_months"] - rollback) / 3.0,
                )
            capability = _service_capability_multiplier(
                period,
                model_name,
                service_provider_name,
                config.plateau_quarter if shock_plateau_quarter is None else shock_plateau_quarter,
                config.resolution_months,
                embargo_shock,
                config,
                shock_plateau_quarter is not None,
            )
            token_cost = (
                _stochastic_token_cost(period, config.periods, sudden_active, gradual_start_period)
                if config.stochastic_shocks
                else _token_cost_multiplier(period, TOKEN_SCENARIOS[token_name], config.periods)
            )
            effective_capability = capability * scenario_capability_multiplier

            adoption_multiplier = 1.0 + config.adoption_capability_elasticity * (capability - 1.0)
            adoption_prob = torch.clamp(adoption_propensity * adoption_multiplier, 0.02, 0.98)
            active = (torch.rand(config.users, generator=cpu_generator).to(device) < adoption_prob) & ai_participant
            base_requested_usage = usage_intensity * active.to(torch.float32)
            confidential_requested = base_requested_usage * config.effective_confidential_document_fraction
            non_confidential_requested = base_requested_usage - confidential_requested
            access_multiplier, cloud_cost, topup_cost, exhausted_share = _apply_access_plan_torch(
                non_confidential_requested,
                active,
                access_plan_name,
                token_cost,
                config,
                torch,
            )
            frontier_usage = non_confidential_requested * access_multiplier
            frontier_usage, cloud_cost, service_budget_scale = _apply_service_budget_limit_torch(
                frontier_usage,
                cloud_cost,
                config,
            )
            if hardware_name == "cloud_api_only":
                local_multiplier = torch.zeros_like(wait_multiplier)
                fallback_multiplier = torch.zeros_like(wait_multiplier)
            else:
                local_multiplier = torch.clamp(wait_multiplier, 0.0, 1.0)
                if local_fallback_policy == "local_default":
                    fallback_multiplier = local_multiplier
                else:
                    fallback_multiplier = torch.clamp(local_multiplier * wait_tolerance, 0.0, 1.0)
            local_confidential_usage = confidential_requested * local_multiplier
            if SERVICE_PROVIDERS[service_provider_name]["confidential_capacity_share"]:
                service_confidential_usage = confidential_requested
                local_confidential_usage = torch.zeros_like(confidential_requested)
            else:
                service_confidential_usage = torch.zeros_like(confidential_requested)
            unavailable_confidential = torch.clamp(confidential_requested - service_confidential_usage - local_confidential_usage, min=0.0)
            fallback_usage = torch.clamp(non_confidential_requested - frontier_usage, min=0.0) * fallback_multiplier
            effective_usage = frontier_usage + service_confidential_usage + local_confidential_usage + fallback_usage
            delivered_usage_multiplier = torch.where(
                base_requested_usage > 0,
                effective_usage / torch.clamp(base_requested_usage, min=1e-9),
                torch.ones_like(base_requested_usage),
            )

            improvement_prob = torch.clamp(
                config.improvement_base_probability
                + config.improvement_capability_weight * effective_capability * capability_fit,
                config.improvement_probability_min,
                config.improvement_probability_max,
            )
            improved = torch.rand(config.users, generator=cpu_generator).to(device) < improvement_prob
            gain_scale = config.resolution_months / 12.0
            sampled_gains = (
                torch.randn(config.users, generator=cpu_generator).to(device) * sigma * gain_scale
                + mean_gain_base * gain_scale * effective_capability * hardware_bonus * capability_fit
            )
            ai_realized_gain = (
                torch.clamp(
                    sampled_gains * gain_moderation,
                    config.ai_gain_min * gain_scale,
                    config.ai_gain_max * gain_scale,
                )
                * active.to(torch.float32)
                * improved.to(torch.float32)
                * delivered_usage_multiplier
                * torch.clamp(
                    1.0 - config.confidential_mixing_ratio * config.confidential_mixing_penalty * torch.where(
                        base_requested_usage > 0,
                        unavailable_confidential / torch.clamp(base_requested_usage, min=1e-9),
                        torch.zeros_like(base_requested_usage),
                    ),
                    0.0,
                    1.0,
                )
                * gain_discount
            )
            zero_risk_gain = (
                config.zero_risk_feature_gain
                * gain_scale
                * adoption_mean
                * ai_participant.to(torch.float32)
                * zero_risk_access_multiplier
                * gain_discount
            )
            realized_gain = zero_risk_gain + ai_realized_gain
            efficiency_index = 1.0 + realized_gain

            total_cost = cloud_cost + hardware_cost
            mean_gain = float(torch.mean(efficiency_index - 1.0).cpu())
            total_gain = float(torch.sum(efficiency_index - 1.0).cpu())
            base_requested_usage_sum = float(torch.sum(base_requested_usage).cpu())
            effective_usage_sum = float(torch.sum(effective_usage).cpu())
            confidential_requested_sum = float(torch.sum(confidential_requested).cpu())
            cost_per_increment = total_cost / max(1e-9, total_gain)
            return_rate = mean_gain / (total_cost / config.users) if total_cost > 1e-9 else np.nan
            annualized_return_rate = return_rate * config.annualization_factor

            if np.isfinite(return_rate):
                return_rates.append(return_rate)
            risk = float(np.std(return_rates, ddof=1)) if len(return_rates) > 1 else 0.0

            run_rows.append(
                {
                    "scenario": f"{model_name}|{token_name}|{access_plan_name}|{hardware_name}|{refresh_name}|{local_fallback_policy}|{service_provider_name}|{shock_combination}",
                    "model": model_name,
                    "token_cost": token_name,
                    "access_plan": access_plan_name,
                    "hardware": hardware_name,
                    "hardware_refresh": refresh_name,
                    "local_fallback": local_fallback_policy,
                    "service_provider": service_provider_name,
                    "shock_combination": shock_combination,
                    "embargo_shock": embargo_shock,
                    "sudden_break_even_shock": sudden_active,
                    "gradual_break_even_shock": gradual_start_period is not None,
                    "capability_plateau_shock": shock_plateau_quarter is not None,
                    "backend": "torch-mps",
                    "run": run,
                    "period": period,
                    "month": period * config.resolution_months,
                    "capability_multiplier": capability,
                    "mean_efficiency_index": float(torch.mean(efficiency_index).cpu()),
                    "mean_efficiency_gain": mean_gain,
                    "zero_risk_efficiency_gain": float(torch.mean(zero_risk_gain).cpu()),
                    "mean_ai_efficiency_gain": float(torch.mean(ai_realized_gain).cpu()),
                    "negative_ai_gain_share": float(torch.mean((ai_realized_gain < 0.0).to(torch.float32)).cpu()),
                    "active_user_share": float(torch.mean(active.to(torch.float32)).cpu()),
                    "delivered_usage_share": float(effective_usage_sum / max(1e-9, base_requested_usage_sum)),
                    "confidential_usage_share": float(confidential_requested_sum / max(1e-9, base_requested_usage_sum)),
                    "service_budget_scale": float(service_budget_scale),
                    "hardware_budget_feasible": float(hardware_budget_feasible),
                    "hardware_budget_scale": float(hardware_budget_scale),
                    "rate_limit_exhausted_share": exhausted_share,
                    "cloud_cost_index": float(cloud_cost),
                    "topup_cost_index": float(topup_cost),
                    "hardware_cost_index": float(hardware_cost),
                    "total_cost_index": float(total_cost),
                    "cost_per_efficiency_increment": float(cost_per_increment),
                    "return_rate": float(return_rate),
                    "annualized_return_rate": float(annualized_return_rate),
                    "risk": risk,
                    "annualized_risk": risk * config.annualization_factor,
                }
            )

    return pd.DataFrame(run_rows)


def _apply_access_plan_torch(
    requested_usage,
    active,
    access_plan_name: str,
    token_cost: float,
    config: SimulationConfig,
    torch,
) -> tuple[object, float, float, float]:
    access_plan = ACCESS_PLANS[access_plan_name]
    if access_plan_name == "pay_per_use":
        cloud_cost = (
            config.base_cost_index_per_active_user_quarter
            * token_cost
            * float(torch.sum(requested_usage).cpu())
            * config.quarters_per_period
        )
        return torch.ones_like(requested_usage), float(cloud_cost), 0.0, 0.0

    allowance = min(access_plan["five_hour_allowance"], access_plan["weekly_allowance"]) * config.quarters_per_period
    included_usage = torch.minimum(requested_usage, torch.tensor(allowance, dtype=requested_usage.dtype, device=requested_usage.device))
    excess_usage = torch.clamp(requested_usage - allowance, min=0.0)
    exhausted_share = float(torch.mean(((excess_usage > 0) & active).to(torch.float32)).cpu())

    subscription_cost = (
        access_plan["subscription_cost_per_active_user_quarter"]
        * float(torch.sum(active.to(torch.float32)).cpu())
        * config.quarters_per_period
    )
    topup_cost = 0.0
    if access_plan["topup"]:
        topup_cost = (
            config.base_cost_index_per_active_user_quarter
            * token_cost
            * float(torch.sum(excess_usage).cpu())
            * config.quarters_per_period
        )
        delivered_usage = requested_usage
    else:
        delivered_usage = included_usage

    access_multiplier = torch.where(
        requested_usage > 0,
        delivered_usage / torch.clamp(requested_usage, min=1e-9),
        torch.ones_like(requested_usage),
    )
    return access_multiplier, float(subscription_cost + topup_cost), float(topup_cost), exhausted_share


def _apply_service_budget_limit_torch(
    delivered_usage,
    service_cost: float,
    config: SimulationConfig,
) -> tuple[object, float, float]:
    budget = config.service_budget_per_period
    if budget is None or service_cost <= budget:
        return delivered_usage, service_cost, 1.0

    scale = budget / max(service_cost, 1e-9)
    return delivered_usage * scale, budget, scale


def scenario_grid(stochastic_shocks: bool = False) -> Iterable[tuple[str, str, str, str, str, str, str, str]]:
    model_names = (
        tuple(name for name, specification in MODEL_SCENARIOS.items() if specification["growth"])
        if stochastic_shocks
        else tuple(MODEL_SCENARIOS)
    )
    token_names = ("flat",) if stochastic_shocks else tuple(TOKEN_SCENARIOS)
    shock_combinations = SHOCK_COMBINATIONS if stochastic_shocks else ("none",)
    for model_name, token_name, hardware_name, refresh_name, access_plan_name, shock_combination in itertools.product(
        model_names,
        token_names,
        HARDWARE_SCENARIOS.keys(),
        HARDWARE_REFRESH.keys(),
        ACCESS_PLANS.keys(),
        shock_combinations,
    ):
        fallback_policies = ("persona_choice",) if hardware_name == "cloud_api_only" else LOCAL_FALLBACK_POLICIES
        for local_fallback_policy in fallback_policies:
            provider_names = ("local_hardware",) if hardware_name != "cloud_api_only" else ("global_service", "european_service")
            for service_provider_name in provider_names:
                yield model_name, token_name, hardware_name, refresh_name, access_plan_name, local_fallback_policy, service_provider_name, shock_combination


def run_simulation(config: SimulationConfig) -> pd.DataFrame:
    scenario_args = [
        (*scenario[:7], config, config.seed + i * 1009, scenario[7])
        for i, scenario in enumerate(scenario_grid(config.stochastic_shocks))
    ]

    if config.concurrency <= 1:
        frames = [
            _simulate_one_scenario(args)
            for args in tqdm(
                scenario_args,
                desc="Simulating scenarios",
                unit="scenario",
                disable=not config.show_progress,
            )
        ]
    else:
        workers = min(config.concurrency, len(scenario_args))
        frames = []
        executor_cls = ThreadPoolExecutor if config.backend == "torch-mps" else ProcessPoolExecutor
        with executor_cls(max_workers=workers) as executor:
            futures = [executor.submit(_simulate_one_scenario, args) for args in scenario_args]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Simulating scenarios",
                unit="scenario",
                disable=not config.show_progress,
            ):
                frames.append(future.result())
    return pd.concat(frames, ignore_index=True)


def _parse_employee_mix(mix_arg: str | None) -> dict[str, float] | None:
    if mix_arg is None:
        return None

    mix: dict[str, float] = {}
    for item in mix_arg.split(","):
        key, value = item.split("=", 1)
        mix[key.strip()] = float(value)
    return mix


def _validate_budget_args(args: argparse.Namespace) -> None:
    if args.resolution_months < 1:
        raise SystemExit("--resolution-months must be at least 1.")
    if args.max_monthly_service_budget is not None and args.max_monthly_service_budget_usd is not None:
        raise SystemExit("Specify either --max-monthly-service-budget or --max-monthly-service-budget-usd, not both.")
    if args.max_upfront_hardware_budget is not None and args.max_upfront_hardware_budget_usd is not None:
        raise SystemExit("Specify either --max-upfront-hardware-budget or --max-upfront-hardware-budget-usd, not both.")
    if args.usd_per_service_cost_index_quarter <= 0:
        raise SystemExit("--usd-per-service-cost-index-quarter must be positive.")
    if args.usd_per_hardware_capex_index is not None and args.usd_per_hardware_capex_index <= 0:
        raise SystemExit("--usd-per-hardware-capex-index must be positive.")
    if not 0.0 <= args.confidential_document_fraction <= 1.0:
        raise SystemExit("--confidential-document-fraction must be between 0 and 1.")
    if args.adoption_propensity_concentration <= 0:
        raise SystemExit("--adoption-propensity-concentration must be positive.")
    if not 0.0 <= args.improvement_base_probability <= 1.0:
        raise SystemExit("--improvement-base-probability must be between 0 and 1.")
    if args.improvement_capability_weight < 0:
        raise SystemExit("--improvement-capability-weight must be non-negative.")
    if not 0.0 <= args.improvement_probability_min <= args.improvement_probability_max <= 1.0:
        raise SystemExit("Improvement probability clip bounds must be between 0 and 1, with min no greater than max.")
    if args.ai_gain_min > args.ai_gain_max:
        raise SystemExit("--ai-gain-min must be no greater than --ai-gain-max.")
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
    if args.zero_risk_feature_gain < 0:
        raise SystemExit("--zero-risk-feature-gain must be non-negative.")
    if not 0.0 <= args.zero_risk_confidential_work_share <= 1.0:
        raise SystemExit("--zero-risk-confidential-work-share must be between 0 and 1.")
    if args.counterfactual_skill_growth_rate_per_year < 0:
        raise SystemExit("--counterfactual-skill-growth-rate-per-year must be non-negative.")
    if args.it_support_max_users < 0:
        raise SystemExit("--it-support-max-users must be non-negative.")
    if args.cloud_oss_hours_per_usage_unit_period is not None and args.cloud_oss_hours_per_usage_unit_period <= 0:
        raise SystemExit("--cloud-oss-hours-per-usage-unit-period must be positive.")


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "scenario",
        "model",
        "token_cost",
        "access_plan",
        "hardware",
        "hardware_refresh",
        "local_fallback",
        "service_provider",
        "shock_combination",
        "backend",
        "period",
        "month",
    ]
    metrics = [
        "mean_efficiency_gain",
        "zero_risk_efficiency_gain",
        "mean_ai_efficiency_gain",
        "negative_ai_gain_share",
        "embargo_shock",
        "sudden_break_even_shock",
        "gradual_break_even_shock",
        "capability_plateau_shock",
        "active_user_share",
        "delivered_usage_share",
        "confidential_usage_share",
        "service_budget_scale",
        "hardware_budget_feasible",
        "hardware_budget_scale",
        "rate_limit_exhausted_share",
        "cloud_cost_index",
        "topup_cost_index",
        "hardware_cost_index",
        "total_cost_index",
        "cost_per_efficiency_increment",
        "return_rate",
        "annualized_return_rate",
        "risk",
        "annualized_risk",
    ]

    rows = []
    for keys, group in results.groupby(group_cols, sort=False):
        row = dict(zip(group_cols, keys, strict=True))
        for metric in metrics:
            values = group[metric].to_numpy(float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_p05"] = float(np.quantile(values, 0.05))
            row[f"{metric}_p50"] = float(np.quantile(values, 0.50))
            row[f"{metric}_p95"] = float(np.quantile(values, 0.95))
        rows.append(row)
    return pd.DataFrame(rows)


def fit_curves(summary: pd.DataFrame) -> pd.DataFrame:
    fit_rows = []
    for scenario, group in summary.groupby("scenario", sort=False):
        group = group.sort_values("month")
        x = group["month"].to_numpy(float)
        y = group["annualized_return_rate_mean"].to_numpy(float)
        smoothing = max(1e-8, len(x) * float(np.var(y)) * 0.03)
        spline = UnivariateSpline(x, y, s=smoothing, k=min(3, len(x) - 1))
        dense_x = np.linspace(float(x.min()), float(x.max()), 80)
        dense_y = spline(dense_x)
        meta = group.iloc[0][["model", "token_cost", "access_plan", "hardware", "hardware_refresh", "shock_combination", "backend"]].to_dict()
        for month, fitted_return in zip(dense_x, dense_y, strict=True):
            fit_rows.append(
                {
                    "scenario": scenario,
                    **meta,
                    "month": float(month),
                    "fitted_annualized_return_rate": float(fitted_return),
                }
            )
    return pd.DataFrame(fit_rows)


def _top_scenarios(summary: pd.DataFrame, limit: int = 10) -> list[str]:
    final_period = int(summary["period"].max())
    final = summary[summary["period"] == final_period].copy()
    final = final.sort_values("annualized_return_rate_mean", ascending=False)
    return final["scenario"].head(limit).tolist()


def _slice_grid_shape(slice_count: int) -> tuple[int, int]:
    cols = min(4, max(1, slice_count))
    rows = int(np.ceil(slice_count / cols))
    return rows, cols


def _classify_llm_access(group: pd.DataFrame) -> pd.Series:
    local_budget_scale = (
        group["hardware_budget_scale_mean"]
        if "hardware_budget_scale_mean" in group
        else group["hardware_budget_feasible_mean"]
    )
    return np.where(
        group["hardware"] == "cloud_api_only",
        "Service LLM",
        np.where(local_budget_scale > 0.0, "Local LLM", "Budget-blocked Local"),
    )


def _plot_all_risk_return_slices(summary: pd.DataFrame, plots_dir: Path) -> None:
    months = sorted(summary["month"].unique())
    rows, cols = _slice_grid_shape(len(months))
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 4.0 * rows), sharey=False)
    axes_array = np.atleast_1d(axes).ravel()

    for ax, month in zip(axes_array, months, strict=False):
        group = summary[summary["month"] == month]
        ax.scatter(group["annualized_risk_mean"], group["annualized_return_rate_mean"], s=16, alpha=0.60)
        ax.set_title(f"Month {month}")
        ax.set_xlabel("Annualized return-rate risk (SD)")
        ax.set_ylabel("Mean annualized return rate")

    for ax in axes_array[len(months) :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(plots_dir / "risk_return_2d_slices_all_months.png", dpi=180)
    plt.close(fig)


def _plot_all_risk_return_slices_by_access(summary: pd.DataFrame, plots_dir: Path) -> None:
    months = sorted(summary["month"].unique())
    rows, cols = _slice_grid_shape(len(months))
    access_colors = {
        "Service LLM": "#2f6fbb",
        "Local LLM": "#d97706",
        "Budget-blocked Local": "#7c3aed",
    }
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 4.0 * rows), sharey=False)
    axes_array = np.atleast_1d(axes).ravel()

    for ax, month in zip(axes_array, months, strict=False):
        group = summary[summary["month"] == month].copy()
        group["llm_access_type"] = _classify_llm_access(group)
        for access_type, color in access_colors.items():
            access_group = group[group["llm_access_type"] == access_type]
            ax.scatter(
                access_group["annualized_risk_mean"],
                access_group["annualized_return_rate_mean"],
                s=18,
                alpha=0.62,
                color=color,
                label=access_type,
            )
        ax.set_title(f"Month {month}")
        ax.set_xlabel("Annualized return-rate risk (SD)")
        ax.set_ylabel("Mean annualized return rate")

    for ax in axes_array[len(months) :]:
        ax.axis("off")

    handles, labels = axes_array[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(plots_dir / "risk_return_2d_slices_all_months_by_llm_access.png", dpi=180)
    plt.close(fig)


def plot_outputs(summary: pd.DataFrame, fits: pd.DataFrame, output_dir: Path) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    chosen = _top_scenarios(summary, limit=8)

    fig, ax = plt.subplots(figsize=(12, 7))
    for scenario in chosen:
        group = summary[summary["scenario"] == scenario].sort_values("month")
        fit = fits[fits["scenario"] == scenario].sort_values("month")
        ax.plot(fit["month"], fit["fitted_annualized_return_rate"], linewidth=1.8, label=scenario)
        ax.scatter(group["month"], group["annualized_return_rate_mean"], s=12)
    ax.set_title("Fitted Annualized Return-Rate Curves")
    ax.set_xlabel("Month")
    ax.set_ylabel("Annualized return rate")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(plots_dir / "fitted_return_rate_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7))
    for scenario in chosen:
        group = summary[summary["scenario"] == scenario].sort_values("month")
        ax.plot(group["month"], group["cost_per_efficiency_increment_mean"], linewidth=1.8, label=scenario)
    ax.set_title("Cost Per Realized Efficiency Increment")
    ax.set_xlabel("Month")
    ax.set_ylabel("Cost-index units per efficiency increment")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(plots_dir / "cost_per_efficiency_increment.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    final_months = sorted(summary["month"].unique())[-3:]
    for ax, month in zip(axes, final_months, strict=True):
        group = summary[summary["month"] == month]
        ax.scatter(group["annualized_risk_mean"], group["annualized_return_rate_mean"], s=18, alpha=0.65)
        ax.set_title(f"Month {month}")
        ax.set_xlabel("Annualized return-rate risk (SD)")
        ax.set_ylabel("Mean annualized return rate")
    fig.tight_layout()
    fig.savefig(plots_dir / "risk_return_2d_slices.png", dpi=180)
    plt.close(fig)

    access_colors = {
        "Service LLM": "#2f6fbb",
        "Local LLM": "#d97706",
        "Budget-blocked Local": "#7c3aed",
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for ax, month in zip(axes, final_months, strict=True):
        group = summary[summary["month"] == month].copy()
        group["llm_access_type"] = _classify_llm_access(group)
        for access_type, color in access_colors.items():
            access_group = group[group["llm_access_type"] == access_type]
            ax.scatter(
                access_group["annualized_risk_mean"],
                access_group["annualized_return_rate_mean"],
                s=20,
                alpha=0.65,
                color=color,
                label=access_type,
            )
        ax.set_title(f"Month {month}")
        ax.set_xlabel("Annualized return-rate risk (SD)")
        ax.set_ylabel("Mean annualized return rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(plots_dir / "risk_return_2d_slices_by_llm_access.png", dpi=180)
    plt.close(fig)

    _plot_all_risk_return_slices(summary, plots_dir)
    _plot_all_risk_return_slices_by_access(summary, plots_dir)

    fig = plt.figure(figsize=(12, 8))
    ax3d = fig.add_subplot(111, projection="3d")
    reduced = summary.groupby(["scenario", "month"], sort=False).head(1)
    ax3d.scatter(
        reduced["month"],
        reduced["annualized_risk_mean"],
        reduced["annualized_return_rate_mean"],
        c=reduced["annualized_return_rate_mean"],
        cmap="viridis",
        s=16,
        alpha=0.7,
    )
    ax3d.set_title("Annualized Return Rate vs Risk vs Time")
    ax3d.set_xlabel("Month")
    ax3d.set_ylabel("Annualized return-rate risk (SD)")
    ax3d.set_zlabel("Mean annualized return rate")
    fig.tight_layout()
    fig.savefig(plots_dir / "revenue_risk_time_3d.png", dpi=180)
    plt.close(fig)

    try:
        import plotly.express as px

        fig_px = px.scatter_3d(
            reduced,
            x="month",
            y="annualized_risk_mean",
            z="annualized_return_rate_mean",
            color="model",
            symbol="token_cost",
            hover_name="scenario",
            title="Annualized Return Rate vs Risk vs Time",
        )
        fig_px.write_html(plots_dir / "revenue_risk_time_3d.html")
    except Exception as exc:
        (plots_dir / "plotly_skipped.txt").write_text(f"Plotly export skipped: {exc}\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    config_path = configuration_path_from_argv()
    try:
        load_simulation_configuration(config_path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=config_path, help="YAML configuration file (default: simulation_config.yaml).")
    parser.add_argument("--users", type=int, default=SIMULATION_DEFAULTS["users"])
    parser.add_argument("--years", type=float, default=SIMULATION_DEFAULTS["years"])
    parser.add_argument("--resolution-months", type=int, default=SIMULATION_DEFAULTS["resolution_months"])
    parser.add_argument("--runs", type=int, default=SIMULATION_DEFAULTS["runs"])
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--seed", type=int, default=SIMULATION_DEFAULTS["seed"])
    parser.add_argument("--plateau-quarter", type=int, default=SIMULATION_DEFAULTS["plateau_quarter"])
    parser.add_argument(
        "--employee-mix",
        type=str,
        default=None,
        help="Comma-separated employee-type shares, e.g. software_engineering=0.3,administration=0.4,manual_labor=0.3",
    )
    parser.add_argument(
        "--engineering-context",
        choices=sorted(ENGINEERING_CONTEXT_MULTIPLIERS),
        default=SIMULATION_DEFAULTS["engineering_context"],
        help="Software-engineering task context: mixed baseline, bounded enterprise tasks, or mature-codebase maintenance.",
    )
    parser.add_argument(
        "--adoption-propensity-concentration",
        type=float,
        default=SIMULATION_DEFAULTS["adoption_propensity_concentration"],
        help="Beta-distribution concentration around each adopter group's adoption prior; lower values mean more person-to-person variation.",
    )
    parser.add_argument(
        "--adoption-capability-elasticity",
        type=float,
        default=SIMULATION_DEFAULTS["adoption_capability_elasticity"],
        help="Sensitivity of AI-use probability to capability above its baseline (a calibration scenario parameter).",
    )
    parser.add_argument(
        "--improvement-base-probability",
        type=float,
        default=SIMULATION_DEFAULTS["improvement_base_probability"],
        help="Baseline chance that delivered AI use improves a task (a calibration scenario parameter).",
    )
    parser.add_argument(
        "--improvement-capability-weight",
        type=float,
        default=SIMULATION_DEFAULTS["improvement_capability_weight"],
        help="Additional improvement probability from model capability times work-type capability fit.",
    )
    parser.add_argument("--improvement-probability-min", type=float, default=SIMULATION_DEFAULTS["improvement_probability_min"])
    parser.add_argument("--improvement-probability-max", type=float, default=SIMULATION_DEFAULTS["improvement_probability_max"])
    parser.add_argument("--ai-gain-min", type=float, default=SIMULATION_DEFAULTS["ai_gain_min"])
    parser.add_argument("--ai-gain-max", type=float, default=SIMULATION_DEFAULTS["ai_gain_max"])
    parser.add_argument("--embargo-shock-probability", type=float, default=SIMULATION_DEFAULTS["embargo_shock_probability"])
    parser.add_argument("--embargo-rollback-months", type=int, default=SIMULATION_DEFAULTS["embargo_rollback_months"])
    parser.add_argument(
        "--stochastic-shocks",
        action="store_true",
        help="Replace deterministic token-cost and plateau scenarios with persistent stochastic shock combinations.",
    )
    parser.add_argument("--sudden-break-even-probability-per-month", type=float, default=SIMULATION_DEFAULTS["sudden_break_even_probability_per_month"])
    parser.add_argument("--gradual-break-even-probability-per-month", type=float, default=SIMULATION_DEFAULTS["gradual_break_even_probability_per_month"])
    parser.add_argument("--capability-plateau-probability-per-month", type=float, default=SIMULATION_DEFAULTS["capability_plateau_probability_per_month"])
    parser.add_argument("--confidential-mixing-ratio", type=float, default=SIMULATION_DEFAULTS["confidential_mixing_ratio"])
    parser.add_argument("--confidential-mixing-penalty", type=float, default=SIMULATION_DEFAULTS["confidential_mixing_penalty"])
    parser.add_argument("--cloud-oss-hours-per-usage-unit-period", type=float, default=None)
    parser.add_argument("--disable-it-support", action="store_true", help="Disable the default 0.5-FTE IT-support cost for small on-premise and OSS-cloud companies.")
    parser.add_argument("--it-support-max-users", type=int, default=SIMULATION_DEFAULTS["it_support_max_users"], help="Apply the IT-support cost at or below this employee count.")
    parser.add_argument(
        "--zero-risk-feature-gain",
        type=float,
        default=SIMULATION_DEFAULTS["zero_risk_feature_gain"],
        help="Deterministic per-employee efficiency gain from no-cost integrated AI features, scaled by the persona adoption-prior mean.",
    )
    parser.add_argument(
        "--zero-risk-confidential-work-share",
        type=float,
        default=SIMULATION_DEFAULTS["zero_risk_confidential_work_share"],
        help="Share of confidential work eligible for zero-risk integrated-AI gains.",
    )
    parser.add_argument(
        "--disable-counterfactual-skill-growth",
        action="store_true",
        help="Do not discount AI-attributed gains for counterfactual employee skill growth.",
    )
    parser.add_argument(
        "--counterfactual-skill-growth-rate-per-year",
        type=float,
        default=DEFAULT_COUNTERFACTUAL_SKILL_GROWTH_RATE_PER_YEAR,
        help="Annual compound counterfactual employee skill-growth rate used to discount AI-attributed gains.",
    )
    parser.add_argument("--max-monthly-service-budget", type=float, default=None)
    parser.add_argument("--max-upfront-hardware-budget", type=float, default=None)
    parser.add_argument("--max-monthly-service-budget-usd", type=float, default=None)
    parser.add_argument("--max-upfront-hardware-budget-usd", type=float, default=None)
    parser.add_argument(
        "--confidential-document-fraction",
        type=float,
        default=SIMULATION_DEFAULTS["confidential_document_fraction"],
        help="Fraction of organization documents/workflows that must stay on-prem and cannot be served by cloud-only scenarios.",
    )
    parser.add_argument(
        "--usd-per-service-cost-index-quarter",
        type=float,
        default=SIMULATION_DEFAULTS["usd_per_service_cost_index_quarter"],
        help="Dollar calibration for one service cost-index unit per active user per quarter.",
    )
    parser.add_argument(
        "--usd-per-hardware-capex-index",
        type=float,
        default=None,
        help="Dollar calibration for one upfront hardware capex index unit. Defaults to the selected hardware calibration anchor.",
    )
    parser.add_argument(
        "--hardware-calibration",
        choices=sorted(HARDWARE_CALIBRATIONS),
        default=SIMULATION_DEFAULTS["hardware_calibration"],
        help="Hardware/model calibration profile for local on-prem runs.",
    )
    parser.add_argument(
        "--backend",
        choices=["numpy", "numba", "torch-mps", "auto"],
        default="auto",
        help="Simulation backend. auto prefers numba, then torch-mps, then numpy.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_budget_args(args)
    try:
        backend = _resolve_backend(args.backend)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    calibration = _hardware_calibration_profile(args.hardware_calibration)
    hardware_capex_index = (
        args.usd_per_hardware_capex_index
        if args.usd_per_hardware_capex_index is not None
        else float(calibration["target_unit_usd"]) / HARDWARE_SCENARIOS["onprem_10pct_capacity"]["capex_units"]
    )

    config = SimulationConfig(
        users=args.users,
        years=args.years,
        resolution_months=args.resolution_months,
        runs=args.runs,
        concurrency=args.concurrency,
        seed=args.seed,
        plateau_quarter=args.plateau_quarter,
        max_monthly_service_budget=args.max_monthly_service_budget,
        max_upfront_hardware_budget=args.max_upfront_hardware_budget,
        max_monthly_service_budget_usd=args.max_monthly_service_budget_usd,
        max_upfront_hardware_budget_usd=args.max_upfront_hardware_budget_usd,
        confidential_document_fraction=args.confidential_document_fraction,
        usd_per_service_cost_index_quarter=args.usd_per_service_cost_index_quarter,
        usd_per_hardware_capex_index=hardware_capex_index,
        hardware_calibration=args.hardware_calibration,
        onprem_capability_multiplier=float(calibration["onprem_capability_multiplier"]),
        employee_mix=_parse_employee_mix(args.employee_mix),
        engineering_context=args.engineering_context,
        adoption_propensity_concentration=args.adoption_propensity_concentration,
        adoption_capability_elasticity=args.adoption_capability_elasticity,
        improvement_base_probability=args.improvement_base_probability,
        improvement_capability_weight=args.improvement_capability_weight,
        improvement_probability_min=args.improvement_probability_min,
        improvement_probability_max=args.improvement_probability_max,
        ai_gain_min=args.ai_gain_min,
        ai_gain_max=args.ai_gain_max,
        embargo_shock_probability=args.embargo_shock_probability,
        embargo_rollback_months=args.embargo_rollback_months,
        stochastic_shocks=args.stochastic_shocks,
        sudden_break_even_probability_per_month=args.sudden_break_even_probability_per_month,
        gradual_break_even_probability_per_month=args.gradual_break_even_probability_per_month,
        capability_plateau_probability_per_month=args.capability_plateau_probability_per_month,
        confidential_mixing_ratio=args.confidential_mixing_ratio,
        confidential_mixing_penalty=args.confidential_mixing_penalty,
        cloud_oss_hours_per_usage_unit_period=args.cloud_oss_hours_per_usage_unit_period,
        it_support_enabled=not args.disable_it_support,
        it_support_max_users=args.it_support_max_users,
        zero_risk_feature_gain=args.zero_risk_feature_gain,
        zero_risk_confidential_work_share=args.zero_risk_confidential_work_share,
        counterfactual_skill_growth_enabled=not args.disable_counterfactual_skill_growth,
        counterfactual_skill_growth_rate_per_year=args.counterfactual_skill_growth_rate_per_year,
        backend=backend,
        output_dir=args.output_dir,
    )
    if config.periods < 3:
        raise SystemExit("Use at least three simulated periods so risk and curve fitting are meaningful.")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using simulation backend: {config.backend}", flush=True)
    results = run_simulation(config)
    summary = summarize(results)
    fits = fit_curves(summary)

    results.to_csv(config.output_dir / "monte_carlo_results.csv", index=False)
    summary.to_csv(config.output_dir / "scenario_summary.csv", index=False)
    fits.to_csv(config.output_dir / "fitted_return_rate_curves.csv", index=False)
    plot_outputs(summary, fits, config.output_dir)

    best = summary[summary["period"] == summary["period"].max()].sort_values("annualized_return_rate_mean", ascending=False).head(5)
    print(f"Wrote outputs to {config.output_dir}")
    print("Top final-period scenarios by mean annualized return rate:")
    print(best[["scenario", "annualized_return_rate_mean", "annualized_risk_mean", "cost_per_efficiency_increment_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
