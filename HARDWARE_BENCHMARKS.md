# Hardware benchmark and local-model-quality inputs

`hardware_benchmarks.csv` is the costing and capacity input used by both the
single simulation and the portfolio sweep. It deliberately models ideal IT
operations: no host, networking, electricity, staff, or idle-capacity charge.
All euro amounts are used as dollar-denominated planning inputs, matching the
existing simulation cost unit.

## Hardware tiers

The following values are transcribed from the table supplied for this project
(14 July 2026). Its system price is the up-front CAPEX; its private-cloud price
is the hourly cost of one complete system. H100 and B200 refer to 8-GPU systems.

| Tier | largest local model | CAPEX | parallel users | private-cloud rate |
| --- | --- | ---: | ---: | ---: |
| RTX 6000 Ada | Gemma 4 12B | 8,500 | 14 | 0.80/hour |
| RTX 6000 Pro | Gemma 4 26B MoE | 14,500 | 32 | 2.40/hour |
| H100 | Gemma 4 31B | 380,000 | 65 | 2.80/hour |
| B200 | Gemma 4 31B | 750,000 | 210 | 5.00/hour |

The sweep selects the highest tier whose CAPEX fits the asset's hardware
budget, then buys replicas of that tier when the budget permits. The table's
parallel-user figure is the capacity of one replica at the stated 100%
utilisation assumption.

## EU cloud-OSS asset

`cloud_oss` uses the same selected model tier and hence the same model-quality
logic as the corresponding on-premise asset. It has no upfront cost, is not
embargo-affected, and accepts half of confidential requests (the in-EU-server
assumption). It rents a fractional number of systems, so it has no queueing
loss or unused capacity. Cost each simulation period is:

`requested concurrent-user units × period-hours × hourly system rate / parallel-user capacity`.

Thus a normalized requested-usage unit represents an occupied interactive slot
for the period. The default period-hours are calendar hours (24 × 30.4375 ×
resolution months); `cloud_oss_hours_per_usage_unit_period` can override that
mapping if the workload model should represent less than continuous occupancy.

Google Cloud's official GPU price documentation supports the use of hourly
rather than purchase pricing for cloud hardware, while warning that VM, storage,
and network charges are additional and are intentionally excluded here:
https://cloud.google.com/products/compute/gpus-pricing

## Local model quality

For Gemma 4 26B MoE and larger, the simulation treats a persona-specific share
of work as frontier-quality. The remaining share uses the tier capability
multiplier. These are sensitivity-analysis estimates, not measurements of a
universal quality ratio:

| Persona / work type | frontier-quality work share |
| --- | ---: |
| Software engineering | 55% |
| Administration | 70% |
| Knowledge work | 60% |
| Manual labour | 15% |

The calibration is deliberately higher for routine administrative language work
and lower for physical work. It is moderate for software and knowledge work,
where task difficulty and verification needs are heterogeneous. This follows
Dell’Acqua et al.'s experimental evidence that generative AI has a *jagged*
frontier—nearby tasks can improve or worsen—and their warning against treating
model performance as uniform across a job:
https://pubsonline.informs.org/doi/10.1287/orsc.2025.21838

The model-size threshold is supported only in direction, not as proof of the
specific shares: Google's Gemma model card reports materially stronger coding,
reasoning, and knowledge-task benchmark scores for its larger open models, and
documents their task limitations:
https://ai.google.dev/gemma/docs/core/model_card_3

These estimates should be varied in sensitivity runs before being used for a
procurement decision.


## Small-company IT support cost

On-premise and `cloud_oss` assets include a default 0.5-FTE IT position when the
company has at most 50 employees. The salary basis is TVöD Bund E 12, Stufe 3:
€5,359.50 gross per month from May 2026. The simulator applies a 1.25 employer
load estimate and half-time fraction, yielding €3,349.69 per month, or
€10,049.06 per quarter at the default three-month resolution. This is converted
to the existing service-cost index using `usd_per_service_cost_index_quarter`.

The cost is controlled by `--disable-it-support` and
`--it-support-max-users`; it is enabled by default with a threshold of 50. The
TVöD salary table source is: https://oeffentlicher-dienst.org/tvoed/bund/rechner/e-12/stufe-3

## Serving-capacity context

The table is the authoritative numerical input for this simulation. vLLM's
benchmarking guidance explains why concurrency depends on request mix and SLA;
it should not be read as a portable throughput guarantee:
https://docs.vllm.ai/en/latest/benchmarking/cli/
