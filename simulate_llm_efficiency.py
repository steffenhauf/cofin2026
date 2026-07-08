#!/usr/bin/env python3
"""Monte Carlo toy simulation of time-dependent LLM efficiency return rates."""

from __future__ import annotations

import argparse
import itertools
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
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


BASE_PERSONAS = pd.DataFrame(
    [
        {
            "persona": "innovators",
            "share": 0.025,
            "adoption_prob": 0.93,
            "task_fit": 1.18,
            "usage_intensity": 1.35,
            "wait_tolerance": 0.85,
        },
        {
            "persona": "early_adopters",
            "share": 0.135,
            "adoption_prob": 0.82,
            "task_fit": 1.10,
            "usage_intensity": 1.20,
            "wait_tolerance": 0.72,
        },
        {
            "persona": "early_majority",
            "share": 0.34,
            "adoption_prob": 0.61,
            "task_fit": 1.00,
            "usage_intensity": 1.00,
            "wait_tolerance": 0.55,
        },
        {
            "persona": "late_majority",
            "share": 0.34,
            "adoption_prob": 0.38,
            "task_fit": 0.87,
            "usage_intensity": 0.78,
            "wait_tolerance": 0.36,
        },
        {
            "persona": "laggards",
            "share": 0.16,
            "adoption_prob": 0.17,
            "task_fit": 0.68,
            "usage_intensity": 0.48,
            "wait_tolerance": 0.18,
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
        },
        {
            "task": "administration",
            "share": 0.44,
            "mean_gain": 0.17,
            "sigma": 0.080,
        },
        {
            "task": "manual_labor",
            "share": 0.20,
            "mean_gain": 0.020,
            "sigma": 0.020,
        },
    ]
)

ENGINEERING_CONTEXT_MULTIPLIERS = {
    "mixed": 1.00,
    "bounded": 2.00,
    "maintenance": -0.25,
}

MODEL_SCENARIOS = {
    "frontier_growth": {"kind": "frontier", "growth": True, "lag": 0},
    "frontier_plateau": {"kind": "frontier", "growth": False, "lag": 0},
    "oss_growth_lagged": {"kind": "oss", "growth": True, "lag": 3},
    "oss_plateau_lagged": {"kind": "oss", "growth": False, "lag": 3},
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

HARDWARE_SCENARIOS = {
    "cloud_api_only": {"capacity": 0.0, "utilization": 1.0, "capex_units": 0.0},
    "onprem_10pct_capacity": {"capacity": 0.10, "utilization": 1.0, "capex_units": 165.0},
    "onprem_50pct_capacity": {"capacity": 0.50, "utilization": 1.0, "capex_units": 610.0},
    "onprem_full_capacity": {"capacity": 1.00, "utilization": 1.0, "capex_units": 1120.0},
    "onprem_low_30pct_utilization": {"capacity": 1.00, "utilization": 0.30, "capex_units": 1120.0},
}

HARDWARE_REFRESH = {
    "maxed_out": {"refresh_factor": 1.0, "capability_bonus": 1.0},
    "follow_each_generation": {"refresh_factor": 1.45, "capability_bonus": 1.08},
}

RTX_6000_ADA_TARGET_USD = 8565.0 / 1.26
USD_PER_HARDWARE_CAPEX_INDEX_DEFAULT = (
    RTX_6000_ADA_TARGET_USD / HARDWARE_SCENARIOS["onprem_10pct_capacity"]["capex_units"]
)


@dataclass(frozen=True)
class SimulationConfig:
    users: int = 500
    years: float = 3.0
    resolution_months: int = 3
    runs: int = 1000
    concurrency: int = 1
    seed: int = 20260703
    plateau_quarter: int = 5
    base_cost_index_per_active_user_quarter: float = 1.0
    max_monthly_service_budget: float | None = None
    max_upfront_hardware_budget: float | None = None
    max_monthly_service_budget_usd: float | None = None
    max_upfront_hardware_budget_usd: float | None = None
    usd_per_service_cost_index_quarter: float = 60.0
    usd_per_hardware_capex_index: float = USD_PER_HARDWARE_CAPEX_INDEX_DEFAULT
    employee_mix: dict[str, float] | None = None
    engineering_context: str = "mixed"
    backend: str = "numpy"
    output_dir: Path = Path("outputs")

    @property
    def periods(self) -> int:
        return int(round(self.years * 12 / self.resolution_months))

    @property
    def quarters_per_period(self) -> float:
        return self.resolution_months / 3

    @property
    def service_budget_per_period(self) -> float | None:
        monthly_budget_index = self.monthly_service_budget_index
        if monthly_budget_index is None:
            return None
        return monthly_budget_index * self.resolution_months

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


def _sample_users(rng: np.random.Generator, users: int, config: SimulationConfig) -> pd.DataFrame:
    tasks = _employee_types(config)
    persona_idx = rng.choice(len(BASE_PERSONAS), size=users, p=BASE_PERSONAS["share"].to_numpy())
    task_idx = rng.choice(len(tasks), size=users, p=tasks["share"].to_numpy())
    users_df = pd.DataFrame(
        {
            "persona": BASE_PERSONAS.iloc[persona_idx]["persona"].to_numpy(),
            "adoption_prob": BASE_PERSONAS.iloc[persona_idx]["adoption_prob"].to_numpy(float),
            "task_fit": BASE_PERSONAS.iloc[persona_idx]["task_fit"].to_numpy(float),
            "usage_intensity": BASE_PERSONAS.iloc[persona_idx]["usage_intensity"].to_numpy(float),
            "wait_tolerance": BASE_PERSONAS.iloc[persona_idx]["wait_tolerance"].to_numpy(float),
            "task": tasks.iloc[task_idx]["task"].to_numpy(),
            "mean_gain": tasks.iloc[task_idx]["mean_gain"].to_numpy(float),
            "sigma": tasks.iloc[task_idx]["sigma"].to_numpy(float),
        }
    )
    return users_df


def _capability_multiplier(period: int, model_name: str, plateau_quarter: int) -> float:
    model = MODEL_SCENARIOS[model_name]
    effective_period = max(0, period - model["lag"])
    if not model["growth"]:
        effective_period = min(effective_period, plateau_quarter)

    quarterly_growth = 1.085
    multiplier = quarterly_growth**effective_period
    if model["kind"] == "oss":
        multiplier *= 0.82
    return min(multiplier, 2.60)


def _token_cost_multiplier(period: int, token_scenario: str, periods: int) -> float:
    if token_scenario == "flat":
        return 1.0
    if token_scenario == "gradual":
        return 1.0 + 0.85 * period / max(1, periods - 1)
    if token_scenario == "sudden":
        return 1.0 if period < max(1, periods // 3) else 1.85
    raise ValueError(f"Unknown token scenario: {token_scenario}")


def _hardware_adjustments(
    users_df: pd.DataFrame,
    hardware_name: str,
    refresh_name: str,
    config: SimulationConfig,
) -> tuple[np.ndarray, float, float]:
    hardware = HARDWARE_SCENARIOS[hardware_name]
    refresh = HARDWARE_REFRESH[refresh_name]
    upfront_hardware_cost = hardware["capex_units"] * refresh["refresh_factor"]
    hardware_budget_feasible = (
        1.0
        if config.upfront_hardware_budget_index is None or upfront_hardware_cost <= config.upfront_hardware_budget_index
        else 0.0
    )
    if hardware_budget_feasible == 0.0:
        return np.ones(len(users_df)), 0.0, hardware_budget_feasible

    effective_capacity = hardware["capacity"] * hardware["utilization"]
    if effective_capacity <= 0:
        return np.ones(len(users_df)), 0.0, hardware_budget_feasible

    queue_pressure = max(0.0, 1.0 - effective_capacity)
    use_multiplier = 1.0 - queue_pressure * (1.0 - users_df["wait_tolerance"].to_numpy(float)) * 0.55

    amortization_periods = max(1.0, 36 / config.resolution_months)
    hardware_cost = upfront_hardware_cost / amortization_periods
    return np.clip(use_multiplier, 0.05, 1.0), hardware_cost, hardware_budget_feasible


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
    if backend == "torch-mps":
        _torch_module_and_device()
        return "torch-mps"
    if backend == "auto":
        try:
            _torch_module_and_device()
        except RuntimeError:
            return "numpy"
        return "torch-mps"
    raise ValueError(f"Unknown backend: {backend}")


def _simulate_one_scenario(args: tuple[str, str, str, str, str, SimulationConfig, int]) -> pd.DataFrame:
    if config_backend(args) == "torch-mps":
        return _simulate_one_scenario_torch_mps(args)

    model_name, token_name, hardware_name, refresh_name, access_plan_name, config, scenario_seed = args
    rng = np.random.default_rng(scenario_seed)
    run_rows = []

    for run in range(config.runs):
        users_df = _sample_users(rng, config.users, config)
        previous_return = None
        return_changes: list[float] = []

        for period in range(config.periods):
            capability = _capability_multiplier(period, model_name, config.plateau_quarter)
            token_cost = _token_cost_multiplier(period, TOKEN_SCENARIOS[token_name], config.periods)
            wait_multiplier, hardware_cost, hardware_budget_feasible = _hardware_adjustments(users_df, hardware_name, refresh_name, config)
            hardware_bonus = (
                HARDWARE_REFRESH[refresh_name]["capability_bonus"]
                if hardware_name != "cloud_api_only" and hardware_budget_feasible > 0.0
                else 1.0
            )

            adoption_prob = np.clip(users_df["adoption_prob"].to_numpy(float) * (0.86 + 0.16 * capability), 0.02, 0.98)
            active = rng.random(config.users) < adoption_prob
            requested_usage = users_df["usage_intensity"].to_numpy(float) * wait_multiplier * active
            access_multiplier, cloud_cost, topup_cost, exhausted_share = _apply_access_plan(
                requested_usage,
                active,
                access_plan_name,
                token_cost,
                config,
            )
            effective_usage = requested_usage * access_multiplier
            effective_usage, cloud_cost, service_budget_scale = _apply_service_budget_limit(effective_usage, cloud_cost, config)
            access_multiplier = np.divide(
                effective_usage,
                requested_usage,
                out=np.ones_like(requested_usage),
                where=requested_usage > 0,
            )

            improvement_prob = np.clip(0.42 + 0.20 * capability * users_df["task_fit"].to_numpy(float), 0.05, 0.92)
            improved = rng.random(config.users) < improvement_prob

            sampled_gains = rng.normal(
                users_df["mean_gain"].to_numpy(float) * capability * hardware_bonus * users_df["task_fit"].to_numpy(float),
                users_df["sigma"].to_numpy(float),
            )
            realized_gain = np.clip(sampled_gains, -0.08, 0.70) * active * improved * wait_multiplier * access_multiplier
            efficiency_index = 1.0 + realized_gain

            total_cost = cloud_cost + hardware_cost
            mean_gain = float(np.mean(efficiency_index - 1.0))
            cost_per_increment = total_cost / max(1e-9, np.sum(efficiency_index - 1.0))
            return_rate = mean_gain / max(1e-9, total_cost / config.users)

            if previous_return is not None:
                return_changes.append(return_rate - previous_return)
            previous_return = return_rate
            risk = float(np.std(return_changes, ddof=1)) if len(return_changes) > 1 else 0.0

            run_rows.append(
                {
                    "scenario": f"{model_name}|{token_name}|{access_plan_name}|{hardware_name}|{refresh_name}",
                    "model": model_name,
                    "token_cost": token_name,
                    "access_plan": access_plan_name,
                    "hardware": hardware_name,
                    "hardware_refresh": refresh_name,
                    "backend": config.backend,
                    "run": run,
                    "period": period,
                    "month": period * config.resolution_months,
                    "capability_multiplier": capability,
                    "mean_efficiency_index": float(np.mean(efficiency_index)),
                    "mean_efficiency_gain": mean_gain,
                    "active_user_share": float(np.mean(active)),
                    "delivered_usage_share": float(effective_usage.sum() / max(1e-9, requested_usage.sum())),
                    "service_budget_scale": float(service_budget_scale),
                    "hardware_budget_feasible": float(hardware_budget_feasible),
                    "rate_limit_exhausted_share": exhausted_share,
                    "cloud_cost_index": float(cloud_cost),
                    "topup_cost_index": float(topup_cost),
                    "hardware_cost_index": float(hardware_cost),
                    "total_cost_index": float(total_cost),
                    "cost_per_efficiency_increment": float(cost_per_increment),
                    "return_rate": float(return_rate),
                    "risk": risk,
                }
            )

    return pd.DataFrame(run_rows)


def config_backend(args: tuple[str, str, str, str, str, SimulationConfig, int]) -> str:
    return args[5].backend


def _simulate_one_scenario_torch_mps(args: tuple[str, str, str, str, str, SimulationConfig, int]) -> pd.DataFrame:
    model_name, token_name, hardware_name, refresh_name, access_plan_name, config, scenario_seed = args
    torch, device = _torch_module_and_device()
    rng = np.random.default_rng(scenario_seed)
    run_rows = []

    for run in range(config.runs):
        users_df = _sample_users(rng, config.users, config)
        wait_multiplier_np, hardware_cost, hardware_budget_feasible = _hardware_adjustments(users_df, hardware_name, refresh_name, config)

        adoption_base = torch.as_tensor(users_df["adoption_prob"].to_numpy(float, copy=True), dtype=torch.float32, device=device)
        task_fit = torch.as_tensor(users_df["task_fit"].to_numpy(float, copy=True), dtype=torch.float32, device=device)
        usage_intensity = torch.as_tensor(users_df["usage_intensity"].to_numpy(float, copy=True), dtype=torch.float32, device=device)
        mean_gain_base = torch.as_tensor(users_df["mean_gain"].to_numpy(float, copy=True), dtype=torch.float32, device=device)
        sigma = torch.as_tensor(users_df["sigma"].to_numpy(float, copy=True), dtype=torch.float32, device=device)
        wait_multiplier = torch.as_tensor(wait_multiplier_np, dtype=torch.float32, device=device)

        previous_return = None
        return_changes: list[float] = []

        for period in range(config.periods):
            period_seed = scenario_seed + run * 100003 + period
            cpu_generator = torch.Generator(device="cpu").manual_seed(period_seed)
            capability = _capability_multiplier(period, model_name, config.plateau_quarter)
            token_cost = _token_cost_multiplier(period, TOKEN_SCENARIOS[token_name], config.periods)
            hardware_bonus = (
                HARDWARE_REFRESH[refresh_name]["capability_bonus"]
                if hardware_name != "cloud_api_only" and hardware_budget_feasible > 0.0
                else 1.0
            )

            adoption_prob = torch.clamp(adoption_base * (0.86 + 0.16 * capability), 0.02, 0.98)
            active = torch.rand(config.users, generator=cpu_generator).to(device) < adoption_prob
            requested_usage = usage_intensity * wait_multiplier * active.to(torch.float32)
            access_multiplier, cloud_cost, topup_cost, exhausted_share = _apply_access_plan_torch(
                requested_usage,
                active,
                access_plan_name,
                token_cost,
                config,
                torch,
            )
            effective_usage = requested_usage * access_multiplier
            effective_usage, cloud_cost, service_budget_scale = _apply_service_budget_limit_torch(
                effective_usage,
                cloud_cost,
                config,
            )
            access_multiplier = torch.where(
                requested_usage > 0,
                effective_usage / torch.clamp(requested_usage, min=1e-9),
                torch.ones_like(requested_usage),
            )

            improvement_prob = torch.clamp(0.42 + 0.20 * capability * task_fit, 0.05, 0.92)
            improved = torch.rand(config.users, generator=cpu_generator).to(device) < improvement_prob
            sampled_gains = (
                torch.randn(config.users, generator=cpu_generator).to(device) * sigma
                + mean_gain_base * capability * hardware_bonus * task_fit
            )
            realized_gain = (
                torch.clamp(sampled_gains, -0.08, 0.70)
                * active.to(torch.float32)
                * improved.to(torch.float32)
                * wait_multiplier
                * access_multiplier
            )
            efficiency_index = 1.0 + realized_gain

            total_cost = cloud_cost + hardware_cost
            mean_gain = float(torch.mean(efficiency_index - 1.0).cpu())
            total_gain = float(torch.sum(efficiency_index - 1.0).cpu())
            requested_usage_sum = float(torch.sum(requested_usage).cpu())
            effective_usage_sum = float(torch.sum(effective_usage).cpu())
            cost_per_increment = total_cost / max(1e-9, total_gain)
            return_rate = mean_gain / max(1e-9, total_cost / config.users)

            if previous_return is not None:
                return_changes.append(return_rate - previous_return)
            previous_return = return_rate
            risk = float(np.std(return_changes, ddof=1)) if len(return_changes) > 1 else 0.0

            run_rows.append(
                {
                    "scenario": f"{model_name}|{token_name}|{access_plan_name}|{hardware_name}|{refresh_name}",
                    "model": model_name,
                    "token_cost": token_name,
                    "access_plan": access_plan_name,
                    "hardware": hardware_name,
                    "hardware_refresh": refresh_name,
                    "backend": "torch-mps",
                    "run": run,
                    "period": period,
                    "month": period * config.resolution_months,
                    "capability_multiplier": capability,
                    "mean_efficiency_index": float(torch.mean(efficiency_index).cpu()),
                    "mean_efficiency_gain": mean_gain,
                    "active_user_share": float(torch.mean(active.to(torch.float32)).cpu()),
                    "delivered_usage_share": float(effective_usage_sum / max(1e-9, requested_usage_sum)),
                    "service_budget_scale": float(service_budget_scale),
                    "hardware_budget_feasible": float(hardware_budget_feasible),
                    "rate_limit_exhausted_share": exhausted_share,
                    "cloud_cost_index": float(cloud_cost),
                    "topup_cost_index": float(topup_cost),
                    "hardware_cost_index": float(hardware_cost),
                    "total_cost_index": float(total_cost),
                    "cost_per_efficiency_increment": float(cost_per_increment),
                    "return_rate": float(return_rate),
                    "risk": risk,
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


def scenario_grid() -> Iterable[tuple[str, str, str, str, str]]:
    return itertools.product(
        MODEL_SCENARIOS.keys(),
        TOKEN_SCENARIOS.keys(),
        HARDWARE_SCENARIOS.keys(),
        HARDWARE_REFRESH.keys(),
        ACCESS_PLANS.keys(),
    )


def run_simulation(config: SimulationConfig) -> pd.DataFrame:
    scenario_args = [
        (*scenario, config, config.seed + i * 1009)
        for i, scenario in enumerate(scenario_grid())
    ]

    if config.concurrency <= 1:
        frames = [
            _simulate_one_scenario(args)
            for args in tqdm(scenario_args, desc="Simulating scenarios", unit="scenario")
        ]
    else:
        workers = min(config.concurrency, len(scenario_args))
        frames = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_simulate_one_scenario, args) for args in scenario_args]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Simulating scenarios", unit="scenario"):
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
    if args.max_monthly_service_budget is not None and args.max_monthly_service_budget_usd is not None:
        raise SystemExit("Specify either --max-monthly-service-budget or --max-monthly-service-budget-usd, not both.")
    if args.max_upfront_hardware_budget is not None and args.max_upfront_hardware_budget_usd is not None:
        raise SystemExit("Specify either --max-upfront-hardware-budget or --max-upfront-hardware-budget-usd, not both.")
    if args.usd_per_service_cost_index_quarter <= 0:
        raise SystemExit("--usd-per-service-cost-index-quarter must be positive.")
    if args.usd_per_hardware_capex_index <= 0:
        raise SystemExit("--usd-per-hardware-capex-index must be positive.")


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "scenario",
        "model",
        "token_cost",
        "access_plan",
        "hardware",
        "hardware_refresh",
        "backend",
        "period",
        "month",
    ]
    metrics = [
        "mean_efficiency_gain",
        "active_user_share",
        "delivered_usage_share",
        "service_budget_scale",
        "hardware_budget_feasible",
        "rate_limit_exhausted_share",
        "cloud_cost_index",
        "topup_cost_index",
        "hardware_cost_index",
        "total_cost_index",
        "cost_per_efficiency_increment",
        "return_rate",
        "risk",
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
        y = group["return_rate_mean"].to_numpy(float)
        smoothing = max(1e-8, len(x) * float(np.var(y)) * 0.03)
        spline = UnivariateSpline(x, y, s=smoothing, k=min(3, len(x) - 1))
        dense_x = np.linspace(float(x.min()), float(x.max()), 80)
        dense_y = spline(dense_x)
        meta = group.iloc[0][["model", "token_cost", "access_plan", "hardware", "hardware_refresh", "backend"]].to_dict()
        for month, fitted_return in zip(dense_x, dense_y, strict=True):
            fit_rows.append(
                {
                    "scenario": scenario,
                    **meta,
                    "month": float(month),
                    "fitted_return_rate": float(fitted_return),
                }
            )
    return pd.DataFrame(fit_rows)


def _top_scenarios(summary: pd.DataFrame, limit: int = 10) -> list[str]:
    final_period = int(summary["period"].max())
    final = summary[summary["period"] == final_period].copy()
    final = final.sort_values("return_rate_mean", ascending=False)
    return final["scenario"].head(limit).tolist()


def _slice_grid_shape(slice_count: int) -> tuple[int, int]:
    cols = min(4, max(1, slice_count))
    rows = int(np.ceil(slice_count / cols))
    return rows, cols


def _classify_llm_access(group: pd.DataFrame) -> pd.Series:
    return np.where(
        group["hardware"] == "cloud_api_only",
        "Service LLM",
        np.where(group["hardware_budget_feasible_mean"] > 0.0, "Local LLM", "Budget-blocked Local"),
    )


def _plot_all_risk_return_slices(summary: pd.DataFrame, plots_dir: Path) -> None:
    months = sorted(summary["month"].unique())
    rows, cols = _slice_grid_shape(len(months))
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 4.0 * rows), sharey=False)
    axes_array = np.atleast_1d(axes).ravel()

    for ax, month in zip(axes_array, months, strict=False):
        group = summary[summary["month"] == month]
        ax.scatter(group["risk_mean"], group["return_rate_mean"], s=16, alpha=0.60)
        ax.set_title(f"Month {month}")
        ax.set_xlabel("Risk")
        ax.set_ylabel("Return rate")

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
                access_group["risk_mean"],
                access_group["return_rate_mean"],
                s=18,
                alpha=0.62,
                color=color,
                label=access_type,
            )
        ax.set_title(f"Month {month}")
        ax.set_xlabel("Risk")
        ax.set_ylabel("Return rate")

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
        ax.plot(fit["month"], fit["fitted_return_rate"], linewidth=1.8, label=scenario)
        ax.scatter(group["month"], group["return_rate_mean"], s=12)
    ax.set_title("Fitted Efficient Return Rate Curves")
    ax.set_xlabel("Month")
    ax.set_ylabel("Efficiency gain per cost-index unit per person")
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
        ax.scatter(group["risk_mean"], group["return_rate_mean"], s=18, alpha=0.65)
        ax.set_title(f"Month {month}")
        ax.set_xlabel("Risk: std. dev. of return-rate changes")
        ax.set_ylabel("Return rate")
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
                access_group["risk_mean"],
                access_group["return_rate_mean"],
                s=20,
                alpha=0.65,
                color=color,
                label=access_type,
            )
        ax.set_title(f"Month {month}")
        ax.set_xlabel("Risk: std. dev. of return-rate changes")
        ax.set_ylabel("Return rate")
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
        reduced["risk_mean"],
        reduced["return_rate_mean"],
        c=reduced["return_rate_mean"],
        cmap="viridis",
        s=16,
        alpha=0.7,
    )
    ax3d.set_title("Revenue-Rate vs Risk vs Time")
    ax3d.set_xlabel("Month")
    ax3d.set_ylabel("Risk")
    ax3d.set_zlabel("Return rate")
    fig.tight_layout()
    fig.savefig(plots_dir / "revenue_risk_time_3d.png", dpi=180)
    plt.close(fig)

    try:
        import plotly.express as px

        fig_px = px.scatter_3d(
            reduced,
            x="month",
            y="risk_mean",
            z="return_rate_mean",
            color="model",
            symbol="token_cost",
            hover_name="scenario",
            title="Revenue-Rate vs Risk vs Time",
        )
        fig_px.write_html(plots_dir / "revenue_risk_time_3d.html")
    except Exception as exc:
        (plots_dir / "plotly_skipped.txt").write_text(f"Plotly export skipped: {exc}\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=500)
    parser.add_argument("--years", type=float, default=3.0)
    parser.add_argument("--resolution-months", type=int, default=3)
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--plateau-quarter", type=int, default=5)
    parser.add_argument(
        "--employee-mix",
        type=str,
        default=None,
        help="Comma-separated employee-type shares, e.g. software_engineering=0.3,administration=0.4,manual_labor=0.3",
    )
    parser.add_argument(
        "--engineering-context",
        choices=sorted(ENGINEERING_CONTEXT_MULTIPLIERS),
        default="mixed",
        help="Software-engineering task context: mixed baseline, bounded enterprise tasks, or mature-codebase maintenance.",
    )
    parser.add_argument("--max-monthly-service-budget", type=float, default=None)
    parser.add_argument("--max-upfront-hardware-budget", type=float, default=None)
    parser.add_argument("--max-monthly-service-budget-usd", type=float, default=None)
    parser.add_argument("--max-upfront-hardware-budget-usd", type=float, default=None)
    parser.add_argument(
        "--usd-per-service-cost-index-quarter",
        type=float,
        default=60.0,
        help="Dollar calibration for one service cost-index unit per active user per quarter.",
    )
    parser.add_argument(
        "--usd-per-hardware-capex-index",
        type=float,
        default=USD_PER_HARDWARE_CAPEX_INDEX_DEFAULT,
        help="Dollar calibration for one upfront hardware capex index unit. Defaults to the 10%% local deployment target anchor.",
    )
    parser.add_argument(
        "--backend",
        choices=["numpy", "torch-mps", "auto"],
        default="numpy",
        help="Simulation backend. torch-mps requires PyTorch with Apple MPS support.",
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
        usd_per_service_cost_index_quarter=args.usd_per_service_cost_index_quarter,
        usd_per_hardware_capex_index=args.usd_per_hardware_capex_index,
        employee_mix=_parse_employee_mix(args.employee_mix),
        engineering_context=args.engineering_context,
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

    best = summary[summary["period"] == summary["period"].max()].sort_values("return_rate_mean", ascending=False).head(5)
    print(f"Wrote outputs to {config.output_dir}")
    print("Top final-period scenarios by mean return rate:")
    print(best[["scenario", "return_rate_mean", "risk_mean", "cost_per_efficiency_increment_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
