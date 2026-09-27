import pandas as pd
import pytest
from simulate_llm_efficiency import SimulationConfig
from simulate_llm_efficiency_economics import value_cashflows


def test_upfront_investment_is_not_discounted():
    results = pd.DataFrame(
        {
            "run": [0, 0],
            "period": [0, 1],
            "month": [0, 12],
            "service_type": ["in_prem", "in_prem"],
            "cloud_cost_index": [0.0, 0.0],
            "topup_cost_index": [0.0, 0.0],
            "mean_efficiency_gain": [0.1, 0.1],
        }
    )
    economics = {
        "tvod_monthly_gross_eur": {"test": 100.0},
        "employer_cost_factor": 1.0,
        "annual_gross_margin_eur": {"test": 1_200.0},
        "eur_per_usd": 1.0,
        "administered_service_types": [],
        "labor_savings_discount_rate": 0.1,
        "output_uplift_discount_rate": 0.1,
    }
    config = SimulationConfig(
        users=1,
        years=2,
        resolution_months=12,
        max_upfront_hardware_budget_usd=1_000_000.0,
    )

    _, cashflows = value_cashflows(results, economics, {"test": 1.0}, config)
    labor = cashflows[
        cashflows["scenario"] == "labor_cost_savings"
    ].sort_values("period")
    first, second = (row for _, row in labor.iterrows())

    assert first["upfront_hardware_investment_eur"] > 0.0
    assert first["discount_factor"] == pytest.approx(1.0 / 1.1)
    assert first["discounted_cashflow_eur"] == pytest.approx(
        (first["gross_benefit_eur"] - first["recurring_ai_investment_eur"])
        * first["discount_factor"]
        - first["upfront_hardware_investment_eur"]
    )
    assert second["upfront_hardware_investment_eur"] == 0.0
    assert second["discounted_cashflow_eur"] == pytest.approx(
        second["cashflow_eur"] * second["discount_factor"]
    )
