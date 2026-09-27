# Simulation pseudocode

The simulator evaluates one fixed model, token-cost path, access plan,
hardware/refresh choice, fallback policy, provider, and shock combination over
repeated Monte Carlo runs. A period spans `resolution_months` calendar months
(three by default in the base simulation and one by default in the portfolio
sweep).

```text
INPUT: scenario configuration, simulation configuration, random seed

periods = round(years * 12 / resolution_months)
period_probability(p_month) = 1 - (1 - p_month) ** resolution_months

for each Monte Carlo run:
    sample employees from enabled adopter personas and work types
    initialise persistent shock state and the run's return-rate history

    for each simulation period:
        month = period * resolution_months
        compute the counterfactual-skill-growth discount

        sample a global-service embargo for this period, if applicable

        if stochastic shocks are enabled:
            test each enabled, inactive shock using period_probability(...)
            once triggered, these shocks remain active:
                sudden break-even: token-cost multiplier is 1.85
                gradual break-even: multiplier ramps from 1.0 to 1.85
                    over the remaining periods
                capability plateau: capability freezes at the trigger's
                    lag-adjusted frontier position

        compute provider- and model-adjusted capability
            apply model/provider lags and any current embargo rollback
            apply the configured deterministic plateau or persistent shock
                plateau and cap capability at the configured maximum
        compute the deterministic or shock-driven token-cost multiplier
        compute affordable hardware replicas, local concurrent-user capacity,
            period CAPEX amortization, eligible IT support cost, and any
            refresh capability bonus

        for each employee:
            compute capability-adjusted adoption probability
            sample active AI use and compute requested usage

        split requested usage into confidential and non-confidential demand
        allocate usage across the provider and local hardware:
            apply access-plan allowances and optional paid top-ups
            apply the per-period service budget
            route eligible confidential work to the provider/local capacity
            route unserved non-confidential work through the configured local
                fallback policy and local capacity
            calculate the gain penalty caused by unavailable confidential work

        for each employee:
            compute effective capability for the selected hardware/model
            compute capability- and work-fit-adjusted improvement probability
            sample whether AI use improves the task
            sample and bound the period-scaled AI productivity gain
            scale it by participation, improvement, delivered usage,
                confidential-work penalty, and skill-growth discount
            add the zero-cost integrated-AI gain, scaled by baseline adoption,
                participation, confidential eligibility, period length, and
                skill-growth discount

        aggregate employees:
            period efficiency gain = mean(realised gain)
            period cost = provider/cloud-OSS cost + hardware/IT cost
            return rate = mean gain / per-employee period cost
            annualized return rate = return rate * 12 / resolution_months
            within-run risk = sample SD of finite return rates observed in
                this run through the current period
            annualized within-run risk = risk * 12 / resolution_months

        record scenario metadata, current shock states, capability, usage,
            costs, gains, return rates, and within-run risk

OUTPUT: one row per Monte Carlo run and simulation period
```

Annual gain parameters are multiplied by `resolution_months / 12`. Service
prices and access allowances are specified per quarter and multiplied by
`resolution_months / 3`; monthly budgets and IT costs are multiplied by the
period length. Hardware CAPEX is amortized over 36 calendar months.

## Compact mathematical formulation

For run \(r\), period \(j\), and employee \(i\), let
\(\Delta\) be `resolution_months`, \(t_j=j\Delta\) be elapsed calendar months,
\(U\) be the employee count, \(J\) be the number of periods, and \(q=1.085\)
be the quarterly capability-growth factor. For each enabled persistent shock
\(k\), the per-period trigger probability is

$$
p_k^{(\Delta)}=1-(1-p_k^{\text{month}})^\Delta,
\qquad
S_{k,j}=S_{k,j-1}\lor\mathrm{Bernoulli}
\!\left(p_k^{(\Delta)}\right).
$$

The embargo indicator \(B_j\) is different: it is independently sampled each
period for the global provider using the configured per-period probability.
It is not persistent.

With model lag \(l_m\), provider lag \(l_p\), embargo rollback \(l_b\), and
plateau frontier position \(\pi_j\), capability is

$$
\begin{aligned}
e_j &= \max\!\left(0,\frac{t_j-l_m-l_p-B_jl_b}{3}\right),\\
e'_j &=
\begin{cases}
\min(e_j,\pi_j), & \text{deterministic or shock plateau active},\\
e_j, & \text{otherwise},
\end{cases}\\
C_j &= \min\!\left(C_{\max},q^{e'_j}\right).
\end{aligned}
$$

For deterministic token scenarios, the multiplier is flat, ramps over the
full simulation, or jumps after the first third. In stochastic mode it is

$$
K_j=
\begin{cases}
1.85, & \text{sudden break-even active},\\
1+0.85\dfrac{j-j_g}{\max(1,J-1-j_g)},
    & \text{gradual break-even triggered at }j_g,\\
1, & \text{otherwise}.
\end{cases}
$$

Employee adoption, requested usage, allocation, and improvement are

$$
\begin{aligned}
A_{i,j} &\sim \mathrm{Bernoulli}\!\left(
  \mathrm{clip}\!\left(a_i[1+\varepsilon(C_j-1)],.02,.98\right)
  \right)\,\mathbf 1_{\{i\text{ participates}\}},\\
R_{i,j} &= u_iA_{i,j},\\
D_{i,j} &= \mathrm{Allocate}\!\left(
  R_{i,j};\text{access, provider, local fallback, capacity, budget,
  confidentiality}\right),\\
P_{i,j} &= \mathrm{clip}\!\left(
  p_0+\beta C^*_{i,j}f_i,p_{\min},p_{\max}\right),\\
I_{i,j} &\sim \mathrm{Bernoulli}(P_{i,j}).
\end{aligned}
$$

Here \(C^*_{i,j}\) includes the selected local-model capability adjustment.
If \(h\) is the hardware-refresh bonus, \(m_i\) is the persona/work-type gain
moderation, \(c_{i,j}\) is the confidential-mixing penalty, and
\(d_j=(1+g_s)^{-j\Delta/12}\) is the optional counterfactual skill-growth
discount, the AI-attributed gain is

$$
\begin{aligned}
Y_{i,j} &\sim \mathcal N\!\left(
  \mu_i\frac{\Delta}{12}C^*_{i,j}hf_i,
  \sigma_i\frac{\Delta}{12}\right),\\
G^{AI}_{i,j} &=
  \mathrm{clip}\!\left(
    m_iY_{i,j},g_{\min}\frac{\Delta}{12},
    g_{\max}\frac{\Delta}{12}\right)
  A_{i,j}I_{i,j}\frac{D_{i,j}}{R_{i,j}}c_{i,j}d_j.
\end{aligned}
$$

For an employee with zero requested usage, the implementation defines the
delivered/requested multiplier as one; the active-use factor is then zero, so
the AI-attributed gain remains zero without a division by zero.

The zero-cost integrated-AI baseline and total gain are

$$
\begin{aligned}
z(c)&=1-c(1-z_{\mathrm{conf}}),\\
G^0_{i,j}&=g_0\frac{\Delta}{12}
  a_i^{\mathrm{mean}}\mathbf 1_{\{i\text{ participates}\}}z(c)d_j,\\
G_{i,j}&=G^{AI}_{i,j}+G^0_{i,j}.
\end{aligned}
$$

The effective confidential share \(c\) is either the configured organisation
share or the employee-mix-weighted task shares, capped at 0.90.
`zero_risk_confidential_work_share` supplies \(z_{\mathrm{conf}}\) (0.20 by
default).

For total period cost \(X_{r,j}=X^{\mathrm{service}}_{r,j}+
X^{\mathrm{hardware}}_{r,j}\), the recorded return measures are

$$
\begin{aligned}
\bar G_{r,j}&=U^{-1}\sum_iG_{r,i,j},\\
\rho_{r,j}&=\frac{\bar G_{r,j}}{X_{r,j}/U},\\
\rho^{\mathrm{ann}}_{r,j}&=\frac{12}{\Delta}\rho_{r,j},\\
s_{r,j}&=\mathrm{SD}\!\left(
  \{\rho_{r,k}:0\leq k\leq j,\ \rho_{r,k}\text{ finite}\}\right),\\
s^{\mathrm{ann}}_{r,j}&=\frac{12}{\Delta}s_{r,j}.
\end{aligned}
$$

The within-run sample standard deviation is zero until at least two finite
returns exist. If period cost is zero, return and annualized return are
undefined; those observations are excluded from return/risk frontiers.

## Portfolio-sweep aggregation

The portfolio sweep does not use the simulator's expanding within-run risk for
mean-variance selection. It first groups rows by fixed asset identity and
period, then calculates:

```text
median return       = median across Monte Carlo runs at the same period
portfolio risk      = sample SD across runs at the same period
return-rate change  = within-run period-to-period difference, then median
                      across runs
```

At the final period, the sweep builds an efficient frontier using median
annualized return as reward and the cross-run standard deviation of annualized
return as risk. It selects low-risk, balanced-optimum, and high-gain assets on
that frontier, then retains each selected fixed asset's complete history.
Separate period-frontier outputs repeat selection independently at every
period. Per-bracket Pareto outputs additionally require median efficiency gain
to exceed the zero-cost integrated-AI baseline by the configured incremental
gain threshold.

Backend selection supports NumPy, Numba, and optional Torch/MPS execution.
Cloud-OSS scenarios always use the NumPy path because their usage-priced
hosting cost has separate allocation logic.
