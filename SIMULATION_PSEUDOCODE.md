# Simulation pseudocode

The simulation evaluates one model, access, hardware, provider, and shock
configuration over repeated Monte Carlo runs. Each period represents
`resolution_months` calendar months (minimum: one month).

```text
INPUT: scenario configuration, simulation configuration, random seed

for each Monte Carlo run:
    sample employees from adopter personas and work types
    initialise persistent shocks and previous return rate

    for each simulation period:
        sample global-service embargo shock, if applicable

        if stochastic shocks are enabled:
            sample each enabled shock using its monthly probability,
            converted to the current period length
            once triggered, a shock remains active:
                sudden break-even: token-cost multiplier jumps to 1.85
                gradual break-even: token-cost multiplier ramps to 1.85
                capability plateau: capability freezes at its trigger level

        compute provider- and model-adjusted capability
            apply model/provider lags, embargo rollback, and any plateau
        compute token cost under the deterministic or shock-driven path
        compute local-hardware capacity, CAPEX amortization, IT support cost,
            and any hardware capability bonus

        for each employee:
            compute capability-adjusted adoption probability
            sample active AI use
            compute requested usage

        allocate requested usage across the selected service, local fallback,
            and confidential-data path
            apply access-plan allowances, token costs, service budget caps,
            local capacity limits, and confidential-work penalties

        for each employee:
            compute capability- and work-fit-adjusted improvement probability
            sample whether AI use improves the task
            sample and bound the AI productivity gain
            scale the realised gain by activity, delivered usage, and any
            confidential-work penalty
            discount AI-attributed gains for counterfactual skill growth
            add the deterministic integrated-AI gain, with confidential work
                receiving only its configured eligible share

        aggregate employees:
            period efficiency gain = mean(realised gain)
            period cost = service cost + amortized hardware cost
            return rate = per-employee efficiency gain / per-employee cost
            annualized return rate = return rate × 12 / resolution_months
            risk = SD of prior within-run period-to-period return-rate changes
            annualized risk = risk × 12 / resolution_months

        record scenario metadata, active shocks, capability, usage, costs,
            efficiency outcomes, return rates, and risk

OUTPUT: one row per Monte Carlo run and simulation period
```

Service costs, cloud-OSS hours, IT support, gains, and CAPEX amortization all
scale with the simulated period length. Hardware CAPEX is amortized over 36
calendar months.

## Compact mathematical formulation

For run \(r\), period \(t\), and employee \(i\), let \(m\) be months per
period, \(U\) the number of employees, and \(q=1.085\) the quarterly
capability-growth factor.

$$
\begin{aligned}
p_m(p) &= 1-(1-p)^m && \text{monthly shock probability → period probability},\\
S_{k,t} &= S_{k,t-1}\lor\operatorname{Bernoulli}\!\left(p_m(p_k)\right)
  && \text{persistent enabled shock }k,\\[2pt]
L_t &= \text{model lag}+\text{provider lag}
       +\text{embargo}_t\,\text{rollback},\\
e_t &= \max\!\left(0,\frac{tm-L_t}{3}\right),\\
e'_t &=
\begin{cases}
\min(e_t,\pi_t), & \text{plateau model or plateau shock active},\\
e_t, & \text{otherwise},
\end{cases}\\
C_t &= \min\!\left(2.60, q^{e'_t}\right),\\
K_t &=
\begin{cases}
1.85, & \text{sudden break-even active},\\
1+0.85\dfrac{t-t_g}{T-1-t_g}, & \text{gradual break-even began at }t_g,\\
\text{deterministic token-cost path}, & \text{otherwise},
\end{cases}\\[2pt]
A_{i,t} &\sim \operatorname{Bernoulli}\!\left(\operatorname{clip}\left(a_i[1+\varepsilon(C_t-1)],.02,.98\right)\right),\\
R_{i,t} &= u_iA_{i,t},\qquad
D_{i,t}=\operatorname{Allocate}(R_{i,t};\text{access, capacity, budget, confidentiality}),\\
P_{i,t} &= \operatorname{clip}(p_0+\beta C^*_{i,t}f_i,p_{\min},p_{\max}),\\
I_{i,t} &\sim \operatorname{Bernoulli}(P_{i,t}),\\
G^{AI}_{i,t} &= \operatorname{clip}\!\left(
  \mathcal{N}\!\left(\mu_i\frac{m}{12}C^*_{i,t}hf_i,\;\sigma_i\frac{m}{12}\right),
  g_{\min}\frac{m}{12},g_{\max}\frac{m}{12}\right)\\
&\quad\times A_{i,t}I_{i,t}\frac{D_{i,t}}{R_{i,t}}c_{i,t}d_t,\\
G_{i,t} &= G^{AI}_{i,t}+g_0\frac{m}{12}a_i^{\text{mean}}z(c)d_t,\\[2pt]
\bar G_t &= U^{-1}\sum_iG_{i,t},\qquad X_t=X_t^{\text{service}}+X_t^{\text{hardware}},\\
\rho_t &= \frac{\bar G_t}{X_t/U},\qquad
\rho_t^{\text{ann}}=\frac{12}{m}\rho_t,\\
\sigma_t &= \operatorname{SD}_{\tau\le t}(\rho_\tau-\rho_{\tau-1}),\qquad
\sigma_t^{\text{ann}}=\frac{12}{m}\sigma_t.
\end{aligned}
$$

Here \(a_i,u_i,f_i,\mu_i,\sigma_i\) are sampled employee adoption, usage,
capability-fit, mean-gain, and gain-uncertainty parameters; \(C^*_{i,t}\)
includes local-hardware capability effects; \(h\) is the hardware-refresh
bonus; and \(c_{i,t}\) is the confidential-work penalty. `Allocate` applies
the selected access plan, local fallback, capacity constraints, token prices,
and the service-budget limit. Hardware cost includes CAPEX amortized over 36
months and period-scaled IT support. \(\pi_t\) is the configured plateau
threshold, replaced at the trigger period when the capability-plateau shock
is enabled; the shock threshold applies to both growing and plateau model
scenarios. AI-attributed gains are discounted by
\(d_t=(1+0.015)^{-tm/12}\) by default to account for counterfactual employee
skill growth. Confidential share \(c\) is capped at 0.90, and the zero-risk
gain multiplier is \(z(c)=1-c+0.20c\); its 0.20 confidential-work credit is
configurable. If \(X_t=0\), \(\rho_t\) and its annualized form are undefined;
such observations are excluded from return/risk frontier selection.
