from copy import deepcopy

import numpy as np
from simulate_llm_efficiency import CONFIG_DIR, SimulationConfig, _sample_users
from simulate_llm_efficiency_economics import (
    economic_config, load_economic_configuration, run_economic_simulation)


def test_sampled_users_mark_only_configured_tam_personas_as_participants():
    config = SimulationConfig(
        users=500, participating_personas=("innovators", "early_majority")
    )

    users = _sample_users(np.random.default_rng(7), config.users, config)

    assert users["ai_participant"].equals(
        users["persona"].isin(("innovators", "early_majority"))
    )


def test_nonparticipants_generate_no_ai_cost_or_productivity():
    values = load_economic_configuration(
        CONFIG_DIR / "llm_efficiency_economics_config_v2_detail.yaml"
    )
    config = SimulationConfig(
        users=10,
        years=1 / 12,
        resolution_months=1,
        runs=1,
        seed=11,
        backend="numpy",
        show_progress=False,
        participating_personas=(),
        employee_mix=values["sweep"]["company_profiles"]["mixed"],
    )

    npv, cashflows = run_economic_simulation(values, config, ("cloud",))

    assert (cashflows["gross_benefit_eur"] == 0).all()
    assert (cashflows["recurring_ai_investment_eur"] == 0).all()
    assert (npv["npv_eur"] == 0).all()


def test_cloud_administration_surcharge_is_explicit_and_thresholded():
    values = load_economic_configuration(
        CONFIG_DIR
        / "llm_efficiency_economics_config_v2_detail_50_70_tam_all.yaml"
    )
    without_administration = deepcopy(values)
    without_administration["economics"]["administered_service_types"] = []
    mix = values["sweep"]["company_profiles"]["mixed"]

    def cashflows(configuration, users):
        config = economic_config(
            configuration,
            users=users,
            years=1 / 12,
            resolution_months=1,
            runs=1,
            seed=13,
            backend="numpy",
            employee_mix=mix,
            service_budget_per_person_usd=10.0,
            hardware_budget_usd=0.0,
        )
        return run_economic_simulation(configuration, config, ("cloud",))[1]

    with_support = cashflows(values, 50)
    without_support = cashflows(without_administration, 50)
    surcharge = (
        with_support["recurring_ai_investment_eur"]
        - without_support["recurring_ai_investment_eur"]
    )
    assert np.allclose(surcharge, 3081.7125)

    above_threshold = cashflows(values, 55)
    above_threshold_without_support = cashflows(without_administration, 55)
    assert np.allclose(
        above_threshold["recurring_ai_investment_eur"],
        above_threshold_without_support["recurring_ai_investment_eur"],
    )
