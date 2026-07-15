import pandas as pd

from simulate_llm_efficiency import SimulationConfig, _capability_multiplier, _hardware_adjustments
from sweep_portfolio_rate_changes import (
    COMPANY_PROFILES,
    SweepScenario,
    _choose,
    friendly_label,
    hardware_for_budget,
    hardware_selection_for_budget,
    write_wide_csvs,
)


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


def test_oss_capability_matches_frontier_one_year_earlier():
    assert _capability_multiplier(4 / 3, "oss_growth_lagged", 8, 3) == _capability_multiplier(0, "frontier_growth", 8, 3)
    assert _capability_multiplier(12, "oss_plateau_lagged", 8, 3) == _capability_multiplier(12 - 4 / 3, "frontier_plateau", 8, 3)


def test_rtx_pro_tier_scales_to_public_seed_concurrency():
    users = pd.DataFrame({"wait_tolerance": [0.5] * 25})
    config = SimulationConfig(
        users=25,
        max_upfront_hardware_budget_usd=14_500.0,
        usd_per_hardware_capex_index=14500.0 / 165.0,
        hardware_calibration="rtx6000-pro-gemma4-26b-moe",
    )
    allocation, *_ = _hardware_adjustments(users, "onprem_full_capacity", "maxed_out", config)
    assert abs(allocation.sum() - 32.0) < 1e-12


def test_hardware_selection_uses_strongest_affordable_tier():
    assert hardware_for_budget(0.0, "rtx6000-pro-gemma4-26b-moe") == "cloud_api_only"
    assert hardware_for_budget(9_000.0, "rtx6000-pro-gemma4-26b-moe") == "cloud_api_only"
    assert hardware_for_budget(14_500.0, "rtx6000-pro-gemma4-26b-moe") == "onprem_full_capacity"
    assert hardware_for_budget(100_000.0, "rtx6000-pro-gemma4-26b-moe") == "onprem_full_capacity"
    assert hardware_for_budget(500_000.0, "rtx6000-pro-gemma4-26b-moe") == "onprem_full_capacity"
    assert hardware_selection_for_budget(3_100_000.0) == ("b200-gemma4-31b", "onprem_full_capacity")


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
    risks = pd.read_csv(tmp_path / "portfolio_rate_change_risk.csv")
    assert returns.columns.tolist() == risks.columns.tolist()
