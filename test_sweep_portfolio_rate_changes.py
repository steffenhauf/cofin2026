import pandas as pd

from simulate_llm_efficiency import (
    SHOCK_COMBINATIONS, SimulationConfig, _capability_multiplier,
    _hardware_adjustments, _period_probability, _stochastic_token_cost)
from sweep_portfolio_rate_changes import (
    COMPANY_PROFILES, SweepScenario, _apply_frontier_x_limits, _choose,
    company_profile_index, final_shock_frontier_return_rates, friendly_label,
    hardware_budget_sensitivity_records,
    hardware_budget_time_sensitivity_records, hardware_for_budget,
    hardware_selection_for_budget, paired_future_return_rates,
    period_frontier_return_rates, select_portfolios,
    write_final_shock_frontier_return_rates, write_pareto_markdown_table,
    write_period_frontier_return_rates, write_wide_csvs)


def test_linear_frontier_x_limits_are_applied_only_when_requested():
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    _apply_frontier_x_limits(axis, (-0.1, 0.5))
    assert axis.get_xlim() == (-0.1, 0.5)
    plt.close(figure)


def test_company_profiles_are_complete_workforce_mixes():
    assert set(COMPANY_PROFILES) == {
        "administration_heavy",
        "mixed",
        "manual_labor_heavy",
        "software_engineering_heavy",
        "knowledge_worker_heavy",
        "creative_worker_heavy",
    }
    for mix in COMPANY_PROFILES.values():
        assert abs(sum(mix.values()) - 1.0) < 1e-12
        assert "knowledge_work" in mix


def test_company_profile_seed_indices_are_stable_for_array_jobs():
    assert company_profile_index("administration_heavy") == 0
    assert company_profile_index("creative_worker_heavy") == 5
    assert company_profile_index("custom") == 0


def test_oss_capability_matches_frontier_one_year_earlier():
    assert _capability_multiplier(
        4 / 3, "oss_growth_lagged", 8, 3
    ) == _capability_multiplier(0, "frontier_growth", 8, 3)
    assert _capability_multiplier(
        12, "oss_plateau_lagged", 8, 3
    ) == _capability_multiplier(12 - 4 / 3, "frontier_plateau", 8, 3)


def test_rtx_pro_tier_scales_to_public_seed_concurrency():
    users = pd.DataFrame({"wait_tolerance": [0.5] * 25})
    config = SimulationConfig(
        users=25,
        max_upfront_hardware_budget_usd=14_500.0,
        usd_per_hardware_capex_index=14500.0 / 165.0,
        hardware_calibration="rtx6000-pro-gemma4-26b-moe",
    )
    allocation, *_ = _hardware_adjustments(
        users, "onprem_full_capacity", "maxed_out", config
    )
    assert abs(allocation.sum() - 32.0) < 1e-12


def test_hardware_selection_uses_strongest_affordable_tier():
    assert (
        hardware_for_budget(0.0, "rtx6000-pro-gemma4-26b-moe")
        == "cloud_api_only"
    )
    assert (
        hardware_for_budget(9_000.0, "rtx6000-pro-gemma4-26b-moe")
        == "cloud_api_only"
    )
    assert (
        hardware_for_budget(14_500.0, "rtx6000-pro-gemma4-26b-moe")
        == "onprem_full_capacity"
    )
    assert (
        hardware_for_budget(100_000.0, "rtx6000-pro-gemma4-26b-moe")
        == "onprem_full_capacity"
    )
    assert (
        hardware_for_budget(500_000.0, "rtx6000-pro-gemma4-26b-moe")
        == "onprem_full_capacity"
    )
    assert hardware_selection_for_budget(3_100_000.0) == (
        "b200-gemma4-31b",
        "onprem_full_capacity",
    )


def test_portfolio_sigma_rules_and_fallbacks():
    frontier = pd.DataFrame(
        {
            "risk_stddev": [1.0, 2.0, 3.0, 4.0],
            "median_return_rate_change": [1.0, 2.0, 3.0, 20.0],
        }
    )
    low_risk, low_rule = _choose(frontier, "low_risk")
    optimum, optimum_rule = _choose(frontier, "optimum")
    high_gain_frontier = pd.DataFrame(
        {
            "risk_stddev": list(range(1, 11)),
            "median_return_rate_change": [0.0] * 9 + [100.0],
        }
    )
    high_gain, high_rule = _choose(high_gain_frontier, "high_gain")
    assert low_risk["risk_stddev"] == 1.0
    assert low_rule == "risk <= mean - 1 sigma"
    assert optimum["median_return_rate_change"] == 3.0
    assert optimum_rule == "risk within mean +/- 1 sigma"
    assert high_gain["median_return_rate_change"] == 100.0
    assert high_rule == "gain >= mean + 3 sigma"


def test_wide_csvs_share_column_layout(tmp_path):
    scenario = SweepScenario(50, 10.0, 50_000.0, 0.25, True)
    label = friendly_label(scenario, "low_risk")
    selected = pd.DataFrame(
        {
            "month": [0, 3],
            "column_label": [label, label],
            "median_return_rate_change": [0.0, 0.2],
            "risk_stddev": [0.0, 0.1],
        }
    )
    write_wide_csvs(selected, tmp_path)
    returns = pd.read_csv(tmp_path / "portfolio_rate_changes.csv")
    risks = pd.read_csv(tmp_path / "portfolio_return_rate_risk.csv")
    assert returns.columns.tolist() == risks.columns.tolist()


def test_stochastic_shock_combinations_cover_all_subsets():
    assert len(SHOCK_COMBINATIONS) == 8
    assert SHOCK_COMBINATIONS[0] == "none"
    assert (
        SHOCK_COMBINATIONS[-1]
        == "sudden_break_even,gradual_break_even,capability_plateau"
    )


def test_monthly_shock_probability_is_resolution_invariant():
    monthly = 0.1
    quarterly = _period_probability(monthly, 3)
    assert abs(1.0 - (1.0 - quarterly) ** 4 - (1.0 - monthly) ** 12) < 1e-12


def test_stochastic_token_costs_are_persistent_and_ramp_after_occurrence():
    assert _stochastic_token_cost(2, 12, False, None) == 1.0
    assert _stochastic_token_cost(2, 12, True, None) == 1.85
    assert _stochastic_token_cost(3, 12, False, 2) == 1.0
    assert _stochastic_token_cost(11, 12, False, 2) == 1.85


def test_stochastic_pareto_table_omits_deterministic_baseline_labels(tmp_path):
    points = pd.DataFrame(
        {
            "company_profile": ["mixed"],
            "users": [50],
            "shock_combination": ["capability_plateau"],
            "token_cost": ["flat"],
            "capability_plateau_after_18_months": [False],
            "hardware": ["cloud_api_only"],
            "service_provider": ["global_service"],
            "efficiency_gain_risk_stddev": [0.1],
            "median_efficiency_gain": [0.2],
        }
    )
    output = tmp_path / "pareto.md"
    write_pareto_markdown_table(points, output)
    text = output.read_text()
    assert "shocks: capability_plateau" in text
    assert "flat;" not in text
    assert "continuous capability growth" not in text


def test_monthly_period_scaling_and_annualized_pareto_labels(tmp_path):
    monthly = SimulationConfig(resolution_months=1)
    quarterly = SimulationConfig(resolution_months=3)
    assert monthly.annualization_factor == 12.0
    assert quarterly.annualization_factor == 4.0

    points = pd.DataFrame(
        {
            "company_profile": ["mixed"],
            "users": [50],
            "shock_combination": ["none"],
            "token_cost": ["flat"],
            "capability_plateau_after_18_months": [False],
            "hardware": ["cloud_api_only"],
            "service_provider": ["global_service"],
            "median_annualized_return_rate": [0.8],
            "annualized_return_rate_risk_stddev": [0.2],
            "efficiency_gain_risk_stddev": [0.1],
            "median_efficiency_gain": [0.2],
        }
    )
    output = tmp_path / "annualized_pareto.md"
    write_pareto_markdown_table(points, output)
    text = output.read_text()
    assert "annualized return rate" in text
    assert "Global API (0.800, 0.200)" in text
    assert "cloud_api_only / cloud" not in text


def test_pareto_selection_excludes_zero_risk_existing_tools_baseline():
    from sweep_portfolio_rate_changes import per_bracket_pareto_assets

    points = pd.DataFrame(
        {
            "users": [50, 50],
            "hardware": ["cloud_api_only", "onprem_full_capacity"],
            "service_provider": ["global_service", "local_hardware"],
            "median_efficiency_gain": [0.01, 0.03],
            "median_zero_risk_efficiency_gain": [0.01, 0.01],
            "efficiency_gain_risk_stddev": [0.0, 0.2],
        }
    )
    selected = per_bracket_pareto_assets(points)
    assert selected["hardware"].eq("onprem_full_capacity").all()


def test_portfolio_selection_keeps_one_final_frontier_asset_for_all_months():
    identities = {
        "model": ["frontier_growth", "frontier_growth"] * 2,
        "token_cost": ["flat", "flat"] * 2,
        "access_plan": ["pay_per_use", "pay_per_use"] * 2,
        "hardware": ["cloud_api_only", "onprem_full_capacity"] * 2,
        "hardware_refresh": ["maxed_out", "maxed_out"] * 2,
        "local_fallback": ["persona_choice", "persona_choice"] * 2,
        "service_provider": ["global_service", "global_service"] * 2,
        "shock_combination": ["none", "none"] * 2,
    }
    summary = pd.DataFrame(
        {
            **identities,
            "period": [0, 0, 1, 1],
            "month": [0, 0, 3, 3],
            "median_return_rate_change": [0.1, 0.2, 0.4, 0.3],
            "risk_stddev": [0.1, 0.2, 0.1, 0.3],
            "median_annualized_return_rate": [0.2, 0.3, 0.9, 0.4],
            "annualized_return_rate_risk_stddev": [0.2, 0.3, 0.1, 0.4],
            "median_efficiency_gain": [0.1] * 4,
            "efficiency_gain_risk_stddev": [0.1] * 4,
            "median_zero_risk_efficiency_gain": [0.0] * 4,
        }
    )
    selected = select_portfolios(
        summary, SweepScenario(50, 10.0, 0.0, 0.0, False), {}
    )
    assert selected.groupby("portfolio")["hardware"].nunique().eq(1).all()
    assert selected.groupby("portfolio")["month"].nunique().eq(2).all()


def test_paired_future_paths_keep_each_run_aligned_across_selected_assets():
    identities = {
        "model": ["frontier_growth", "frontier_growth"],
        "token_cost": ["flat", "flat"],
        "access_plan": ["pay_per_use", "pay_per_use"],
        "hardware": ["cloud_api_only", "onprem_full_capacity"],
        "hardware_refresh": ["maxed_out", "maxed_out"],
        "local_fallback": ["persona_choice", "persona_choice"],
        "service_provider": ["global_service", "local_hardware"],
        "shock_combination": ["none", "none"],
    }
    selected = pd.DataFrame(
        {
            **identities,
            "portfolio": ["low_risk", "optimum"],
            "period": [1, 1],
            "selection_period": [1, 1],
            "column_label": ["low asset", "optimum asset"],
        }
    )
    rows = []
    for run in (0, 1):
        for period, month in ((0, 0), (1, 3)):
            for asset, base_return in enumerate((0.2, 0.4)):
                row = {
                    column: values[asset]
                    for column, values in identities.items()
                }
                row.update(
                    run=run,
                    period=period,
                    month=month,
                    annualized_return_rate=base_return + run + period,
                )
                rows.append(row)
    paths, manifest = paired_future_return_rates(
        pd.DataFrame(rows),
        selected,
        SweepScenario(50, 10.0, 0.0, 0.0, False),
        "mixed",
    )
    assert paths.columns.tolist()[-3:] == ["low_risk", "optimum", "high_gain"]
    assert paths.shape[0] == 4
    assert (
        paths.loc[
            (paths["future_id"] == 1) & (paths["month"] == 3), "optimum"
        ].item()
        == 2.4
    )
    assert manifest["portfolio"].tolist() == ["low_risk", "optimum"]


def test_period_frontier_return_rates_keep_three_full_horizon_series_per_period(
    tmp_path,
):
    identities = {
        "model": ["model_a", "model_b", "model_c"] * 2,
        "token_cost": ["flat"] * 6,
        "access_plan": ["pay_per_use"] * 6,
        "hardware": [
            "cloud_api_only",
            "cloud_api_only",
            "onprem_full_capacity",
        ]
        * 2,
        "hardware_refresh": ["maxed_out"] * 6,
        "local_fallback": ["persona_choice"] * 6,
        "service_provider": [
            "global_service",
            "european_service",
            "local_hardware",
        ]
        * 2,
        "shock_combination": ["none"] * 6,
    }
    summary = pd.DataFrame(
        {
            **identities,
            "period": [0, 0, 0, 1, 1, 1],
            "month": [0, 0, 0, 3, 3, 3],
            "annualized_return_rate_risk_stddev": [0.1, 0.2, 0.4] * 2,
            "median_annualized_return_rate": [0.2, 0.5, 0.7, 0.3, 0.4, 1.2],
        }
    )
    rates = period_frontier_return_rates(summary)
    assert rates.columns.tolist() == [
        "period",
        "month",
        "period_0_low_risk",
        "period_0_medium_risk",
        "period_0_high_risk",
        "period_1_low_risk",
        "period_1_medium_risk",
        "period_1_high_risk",
    ]
    assert rates.shape == (2, 8)
    assert (
        rates.loc[rates["period"] == 1, "period_0_medium_risk"].item() == 0.4
    )
    assert rates.loc[rates["period"] == 0, "period_1_high_risk"].item() == 0.7

    scenario = SweepScenario(50, 10.0, 50_000.0, 0.25, False)
    outputs = write_period_frontier_return_rates(summary, scenario, tmp_path)
    output = (
        tmp_path
        / "period_frontier_return_rates"
        / "users_50"
        / "max_hardware_invest_usd_50000"
        / "confidential_document_fraction_0p25"
        / "access_plan_pay_per_use"
        / "users_50_max_hardware_invest_usd_50000_confidential_document_fraction_0p25_service_usd_10_plateau_false_access_plan_pay_per_use.csv"
    )
    assert outputs == [output]
    assert pd.read_csv(output).equals(rates)


def test_period_frontier_return_rates_fan_out_by_access_plan(tmp_path):
    summary = pd.DataFrame(
        {
            "model": ["model_a", "model_b"] * 3,
            "token_cost": ["flat"] * 6,
            "access_plan": ["pay_per_use"] * 2
            + ["flatrate_limited"] * 2
            + ["flatrate_limited_topup"] * 2,
            "hardware": ["cloud_api_only"] * 6,
            "hardware_refresh": ["maxed_out"] * 6,
            "local_fallback": ["persona_choice"] * 6,
            "service_provider": ["global_service"] * 6,
            "shock_combination": ["none"] * 6,
            "period": [0, 0] * 3,
            "month": [0, 0] * 3,
            "annualized_return_rate_risk_stddev": [0.1, 0.2] * 3,
            "median_annualized_return_rate": [0.2, 0.5] * 3,
        }
    )
    outputs = write_period_frontier_return_rates(
        summary, SweepScenario(50, 10.0, 50_000.0, 0.25, False), tmp_path
    )

    assert [path.parent.name for path in outputs] == [
        "access_plan_flatrate_limited",
        "access_plan_flatrate_limited_topup",
        "access_plan_pay_per_use",
    ]
    assert all("access_plan_" in path.name for path in outputs)
    assert all(pd.read_csv(path).shape == (1, 5) for path in outputs)


def test_final_shock_frontier_return_rates_have_48_final_horizon_columns(
    tmp_path,
):
    rows = []
    for run in (0, 1):
        for asset, reward in enumerate((0.2, 0.4, 0.7)):
            for period, month in ((0, 0), (1, 3)):
                rows.append(
                    {
                        "model": f"model_{asset}",
                        "token_cost": "flat",
                        "access_plan": "pay_per_use",
                        "hardware": "cloud_api_only",
                        "hardware_refresh": "maxed_out",
                        "local_fallback": "persona_choice",
                        "service_provider": "global_service",
                        "shock_combination": "none",
                        "run": run,
                        "period": period,
                        "month": month,
                        "annualized_return_rate": (
                            reward + run * 0.1 + period * 0.05
                            if run == 0
                            else float("nan")
                        ),
                        "return_rate": (
                            reward + run * 0.1 + period * 0.05
                            if run == 0
                            else float("nan")
                        ),
                        "mean_efficiency_gain": reward,
                        "zero_risk_efficiency_gain": 0.0,
                        "embargo_shock": run == 1,
                        "sudden_break_even_shock": False,
                        "gradual_break_even_shock": False,
                        "capability_plateau_shock": False,
                    }
                )
    results = pd.DataFrame(rows)
    rates = final_shock_frontier_return_rates(results)
    assert rates.shape == (2, 50)
    assert (
        rates.columns[2]
        == "low_risk_plateau_no_price_gradual_no_price_abrupt_no_embargo_no"
    )
    assert (
        rates.columns[-1]
        == "high_risk_plateau_yes_price_gradual_yes_price_abrupt_yes_embargo_yes"
    )
    assert rates.iloc[:, 2:5].notna().all().all()
    assert rates.iloc[:, 5:].isna().all().all()

    outputs = write_final_shock_frontier_return_rates(
        results, SweepScenario(50, 10.0, 50_000.0, 0.25, False), tmp_path
    )
    assert len(outputs) == 1
    assert pd.read_csv(outputs[0]).equals(rates)


def test_hardware_budget_sensitivity_records_keep_final_period_risk_bands():
    summary = pd.DataFrame(
        {
            "model": ["model_a", "model_b", "model_c"] * 2,
            "token_cost": ["flat"] * 6,
            "access_plan": ["pay_per_use"] * 6,
            "hardware": [
                "cloud_api_only",
                "cloud_api_only",
                "onprem_full_capacity",
            ]
            * 2,
            "hardware_refresh": ["maxed_out"] * 6,
            "local_fallback": ["persona_choice"] * 6,
            "service_provider": [
                "global_service",
                "european_service",
                "local_hardware",
            ]
            * 2,
            "shock_combination": ["none"] * 6,
            "period": [0, 0, 0, 1, 1, 1],
            "month": [0, 0, 0, 3, 3, 3],
            "annualized_return_rate_risk_stddev": [0.1, 0.2, 0.4] * 2,
            "median_annualized_return_rate": [0.2, 0.5, 0.7, 0.3, 0.4, 1.2],
        }
    )
    records = hardware_budget_sensitivity_records(
        summary, SweepScenario(50, 10.0, 50_000.0, 0.25, False), "mixed"
    )
    assert records["portfolio"].tolist() == [
        "low_risk",
        "medium_risk",
        "high_risk",
    ]
    assert records["selection_period"].eq(1).all()
    assert records["hardware_budget_usd"].eq(50_000.0).all()
    assert records["company_profile"].eq("mixed").all()


def test_hardware_budget_time_sensitivity_records_keep_every_period():
    summary = pd.DataFrame(
        {
            "model": ["model_a", "model_a"],
            "token_cost": ["flat", "flat"],
            "access_plan": ["pay_per_use", "pay_per_use"],
            "hardware": ["onprem_full_capacity", "onprem_full_capacity"],
            "hardware_refresh": ["maxed_out", "maxed_out"],
            "local_fallback": ["persona_choice", "persona_choice"],
            "service_provider": ["local_hardware", "local_hardware"],
            "shock_combination": ["none", "none"],
            "period": [0, 1],
            "month": [0, 3],
            "annualized_return_rate_risk_stddev": [0.1, 0.2],
            "median_annualized_return_rate": [0.2, 0.4],
        }
    )
    records = hardware_budget_time_sensitivity_records(
        summary, SweepScenario(50, 10.0, 50_000.0, 0.25, False), "mixed"
    )
    assert set(records["selection_period"]) == {0, 1}
    assert set(records["month"]) == {0, 3}
    assert (
        records.groupby("selection_period")["portfolio"].nunique().eq(3).all()
    )
