# 2024–2025 validation evidence

These two configurations are intentionally **case-calibrated cloud-service
scenarios**, not estimates of an average firm or a causal validation of every
model parameter. The reported outcomes below are task-specific operational
measures. They should be compared to the corresponding simulated support or
administrative-work outputs, not to an organisation-wide productivity number.

## Evidence selection rule

An observation is usable only when the public source identifies all of:

1. a named organisation and a measurable efficiency outcome;
2. the LLM provision approach or named service; and
3. whether local-model capital expenditure is disclosed.

None of the selected sources discloses a purchase price, GPU count, or other
auditable CAPEX for a local model. It would therefore be misleading to infer
one from a company’s total CAPEX, cloud spend, fundraising, or an unrelated
hardware price. Both validation files consequently restrict the simulation to
`cloud_api_only`; local-model CAPEX is **USD 0 in the simulated scenario**,
not an estimate of the company’s undisclosed spending.

## 2024: cloud LLM customer support

### Klarna — nearest 2,500-employee bracket, external customer service

OpenAI’s February 2024 customer story states that Klarna’s OpenAI-powered
assistant handled 2.3 million conversations in its first month, equivalent to
the work of 700 full-time agents; it also reports a 25% decline in repeat
inquiries and a reduction in customer errand resolution from 11 minutes to
under two minutes. Klarna also made ChatGPT Enterprise available throughout
the organisation. [OpenAI case study](https://openai.com/index/klarna/)

Provision classification: managed OpenAI service / ChatGPT Enterprise, not an
on-prem or company-owned local model. The source does not disclose the number
of agents in the relevant baseline, so **700 FTE-equivalent must not be
converted into a percentage productivity gain**. It is retained as a
directional external-support validation check and as evidence for the
cloud-only, high-adoption provision assumption.

### Microsoft support — large-enterprise reference, internal support

Microsoft reports that, in a study of nearly 10,000 support agents, several
teams using Copilot achieved a 12% reduction in case-handling time and a 10%
increase in case resolution; it separately reports 90 minutes saved per week
for its salespeople. [Microsoft WorkLab case report](https://www.microsoft.com/en-us/worklab/our-year-with-copilot-what-microsoft-has-learned-about-ai-at-work)

Provision classification: Microsoft Copilot / cloud service. This is the
numeric calibration anchor for `configs/validation_config_2024.yaml`:
`administration.mean_gain: 0.12`, with all workers represented as the
support/administration task. It is deliberately not used to assert a 12%
gain for all Microsoft work or all firms.

## 2025: managed-cloud RAG support

### Orion Health — 500-employee validation bracket, internal support

AWS documents that Orion Health’s internal RAG chatbot, Oribot, reclaimed
approximately 50 support-staff hours per day, searches more than 500,000
internal records, and was launched as a working prototype in two months. The
implementation uses Amazon Bedrock, Lambda, DynamoDB, a vector database, and
an Amazon VPC. AWS also reports that Bedrock was roughly ten times more
cost-effective at scale than other commercial chatbot platforms for this use
case. [AWS case study](https://aws.amazon.com/solutions/case-studies/orion-health/)

Provision classification: managed AWS cloud RAG service, not local hardware.
The case study does not publish the support-team size or hours worked, so its
50 staff-hours/day result cannot defensibly be converted to a percentage. The
2025 configuration therefore sets a conservative 10% annual administrative
gain as an **explicit modelling inference**, with lower variance than the
generic baseline. The 500-user setting is the project’s validation bracket,
not a claim about Orion Health’s headcount.

## How to use the configurations

Run the two annual cases separately:

```bash
python scripts/simulate_llm_efficiency.py --config configs/validation_config_2024.yaml --output-dir validation_2024
python scripts/simulate_llm_efficiency.py --config configs/validation_config_2025.yaml --output-dir validation_2025
```

Each has `years: 1.0` and `resolution_months: 1`, yielding 12 monthly periods.
The YAML files deliberately retain two cloud-provider rows because the current
simulator enumerates `global_service` and `european_service` for cloud-only
scenarios. For the evidence comparison, use the `global_service` rows as the
managed-service proxy and treat the European rows as an internal sensitivity
case, not an empirical claim about Klarna or Orion.

For the focused out-of-sample comparison, run the local-only runner instead:

```bash
python scripts/validate_published_gains.py --runs 200 --concurrency 4 --output-dir validation_outputs
```

It runs the evidence-matched administration, knowledge-work, bounded-engineering,
and maintenance-engineering profiles sequentially; `--concurrency` controls
only the simulator's local process pool. It writes `published_gain_comparison.csv`,
run-level results, and the single forest-style plot
`published_gain_comparison.png`.

Each evidence-matched profile is also simulated for three adoption populations.
The all-personas estimate retains the configured Rogers (2003, ch. 7) shares.
The early-population estimate retains innovators, early adopters, and early
majority only, renormalizing their 2.5%, 13.5%, and 34.0% population shares to
5%, 27%, and 68%, respectively. The narrower estimate retains innovators and
early adopters only, renormalizing their shares to 15.625% and 84.375%. These
are alternative populations—not assumed adoption rates for the full workforce.
The plot shows their simulation intervals with distinct colours and shapes
against each common published comparator.

The plotted simulation metric is the **workforce-wide annualized AI gain**:
the mean gain over all employees, including inactive employees. This makes the
adoption-population scenarios interpretable as rollout-stage estimates. The
run-level CSV also retains `active_user_annualized_gain_pct` for diagnosing the
per-active-user effect, but that conditional metric is not plotted because it
largely removes the adoption-share difference. Published estimates remain
task-specific and should therefore be interpreted as directional comparators,
not like-for-like estimates of workforce-wide gain.

### Cohort moderation calibration

Persona-specific task gains use a normalized moderation factor rather than an
ad-hoc gain multiplier:

\[
M_c = \frac{1-\beta^{EE}_c f}{1-\beta^{EE}_{EM} f}
      \left[1 + \beta^A(a_c-a_{EM})\right].
\]

Here \(f\) is configured effort friction, \(\beta^{EE}\) is cohort effort-
expectancy sensitivity, and \(a\) is an adaptability index. The early-majority
anchors make \(M_{EM}=1\), preserving that cohort’s previous AI-task-gain
calibration. The 0.599 adaptability coefficient and the 0.30–0.45 early-majority
effort-sensitivity range are user-supplied scenario inputs, not generalizable
effects. UTAUT supports experience as a moderator of effort expectancy and
behavioural intention, but does not itself estimate an LLM productivity
multiplier; see [Venkatesh et al. (2003)](https://doi.org/10.2307/30036540).

## Traceable operational cases screened from the ZenML catalogue

ZenML’s 2025 catalogue is used here as a **discovery index**, not as the
evidence source. Each selected result below was traced to the deployment
provider’s original case study before being added to
`published_gain_comparison.png`. Those studies are vendor-reported operational
outcomes, not causal estimates, and their stated ranges do not include
sampling uncertainty.

| Candidate | Screening decision | Deployment and reported outcome | Plot treatment |
|---|---|---|---|
| Airbnb GraphQL mock generation | Exclude from numeric panel | Gemini 2.5 Pro is embedded in GraphQL mock-data generation with schema validation and retries; more than 700 mocks were merged, but no baseline time, percentage gain, or uncertainty is published. [ZenML entry](https://www.zenml.io/llmops-database/llm-powered-graphql-mock-data-generation-for-developer-productivity) | Qualitative developer-workflow/provisioning evidence only. |
| Agmatix Leafy | Include | Anthropic Claude is called through Amazon Bedrock for field-trial analysis; AWS reports more than 20% efficiency improvement and threefold potential analysis throughput. [AWS case study](https://aws.amazon.com/blogs/machine-learning/generative-ai-for-agriculture-how-agmatix-is-improving-agriculture-with-amazon-bedrock/) | 2024 knowledge-worker-heavy profile; plot the stated 20% efficiency figure only, not the threefold throughput result. |
| Humach customer experience | Include | Claude through Amazon Bedrock supports live-agent assistance and digital workers; Anthropic reports 15–20% operational-efficiency improvement and about 20% call automation. [Anthropic case study](https://www.anthropic.com/customers/humach) | 2025 administration-heavy profile; plot the reported 15–20% range only, not call automation. |
| Accenture Knowledge Assist | Exclude pending source trace | ZenML reports training and escalation reductions, but this screening did not identify a primary source substantiating that exact LLM deployment and those figures. | Do not plot. |

The runner writes `operational_case_comparison.csv` alongside the existing
peer-reviewed comparison. In the single plot, peer-reviewed comparators use
orange diamonds and vendor-reported outcomes use orange squares; both carry an
interval only when the source reports one.

### Self-hosted / on-premises LLM screen

I screened ZenML entries containing *self-hosted*, *local*, and *on-premise*
terminology. None provides enough evidence to add an **on-premises** validation
case: self-hosting is not evidence that the hardware is located on company
premises, and neither CAPEX nor hardware utilisation is reported.

| Candidate | What the source establishes | Decision |
|---|---|---|
| Fuzzy Labs developer-documentation RAG | Quantized Mistral-7B, vLLM and Ray Serve improved single-user latency from 11 s to about 3 s without additional compute. The detailed entry identifies the model servers as GPU instances in AWS, so it is explicitly cloud-hosted rather than on-premises. [ZenML entry](https://www.zenml.io/llmops-database/scaling-self-hosted-llms-with-gpu-optimization-and-load-testing) | Exclude from the on-premises comparison; retain as qualitative evidence about serving optimisation. |
| Faire search-relevance prediction | Fine-tuned Llama3-8B was served on Faire’s GPU cluster across 16 A100 GPUs, achieving 70 million backfill predictions/day and a 28% relevance-accuracy improvement. The source says the GPUs were already procured, but does not state their physical location, purchase cost, or utilisation. [Faire engineering post](https://craft.faire.com/fine-tuning-llama3-to-measure-semantic-relevance-in-search-86a7b13c24ea) | Strong self-hosted hardware case, but exclude from an on-premises or employee-efficiency validation until location and comparable outcome are published. |

This leaves the validation plot focused on published employee/operational
efficiency effects, rather than treating throughput, latency, or accuracy as
interchangeable with labour productivity.

### OSS-model screen (broader criterion)

With the location requirement relaxed, Faire is the strongest 2024–25 case for
the model’s **local/OSS capability path**: it uses an open-weight Llama3-8B,
fine-tunes on 8 A100 GPUs, and serves quantized batch inference on its GPU
cluster. Its reported 28% effect is a *relative relevance-accuracy* change,
not an employee-efficiency gain, so it remains a technical benchmark rather
than another point in the labour-efficiency plot.

| Candidate | OSS-model evidence and quantitative result | Use in this project |
|---|---|---|
| Faire | Fine-tuned Llama3-8B; 28% relative improvement in relevance prediction over its production GPT model; 70m batch predictions/day on 16 GPUs. [Primary engineering post](https://craft.faire.com/fine-tuning-llama3-to-measure-semantic-relevance-in-search-86a7b13c24ea) | Best hardware/throughput benchmark for an OSS local-model scenario; do not compare its accuracy percentage against simulated labour gain. |
| Fuzzy Labs | Quantized Mistral-7B with vLLM and Ray Serve; 11 s to about 3 s latency and about 10x throughput using the same compute. It ran on AWS GPU instances. [ZenML entry](https://www.zenml.io/llmops-database/scaling-self-hosted-llms-with-gpu-optimization-and-load-testing) | Serving-efficiency sensitivity evidence; not a local-hardware or labour-productivity comparator. |
| LinkedIn | Open-source vLLM serving across thousands of hosts; ZenML reports about 10% tokens/s improvement and savings of more than 60 GPUs for some workloads. [ZenML entry](https://www.zenml.io/llmops-database/scaling-genai-applications-with-vllm-for-high-throughput-llm-serving) | Large-scale OSS serving benchmark; exclude from current validation because it reports infrastructure rather than workforce outcomes. |
| Checkr | Fine-tuned Llama2-7B reportedly reduced classification latency below 0.5 s and cost below $800 while maintaining task-quality results; deployment/provider and time basis of cost need primary-source confirmation. [ZenML entry](https://www.zenml.io/llmops-database/streamlining-background-check-classification-with-fine-tuned-small-language-models) | Promising task-efficiency lead, but excluded until traced to the original case study. |

## Out-of-sample validation comparators — do not calibrate to these

The following publications were **not used to set either YAML file**. Keep the
current configurations fixed, run them, and report the simulated gain alongside
these task-specific estimates. Agreement would be a useful plausibility check;
disagreement should prompt an explanation of task mix, adoption, model, and
outcome definition rather than a retrospective change to the configuration.

| Comparator | Published result | Appropriate comparison | Why it remains out of sample |
|---|---:|---|---|
| Brynjolfsson, Li & Raymond (2023/2025 QJE) | 14% average increase in issues resolved per hour among 5,172 customer-support agents; the paper’s staffing calculation is 12% fewer worker-hours. [QJE article](https://academic.oup.com/qje/article/140/2/889/7990658) | 2024 and 2025 administrative/support annual gain; compare direction and magnitude, not the exact customer-service workflow. | The 2024 YAML instead uses Microsoft’s separately reported 12% case-handling-time outcome. |
| Noy & Zhang (2023) | Professional-writing task time fell 40% and quality rose 18% with ChatGPT. [Science article](https://doi.org/10.1126/science.adh2586) | A high-end writing/knowledge-work bound, not the support-only configuration’s central estimate. | Different task, participant population, and outcome measure. |
| Dell’Acqua et al. (2023) | Consultants working within the tool’s effective frontier completed 12.2% more tasks and completed them 25.1% faster; the paper also documents task dependence. [HBS working paper](https://www.hbs.edu/ris/download.aspx?name=24-013.pdf) | Check that the model’s capability-fit mechanism can produce positive but task-dependent gains. | The study’s task bundle and model context differ from both company cases. |
| Dillon et al. (2025) | In a cross-industry field experiment, users spent 3 fewer hours per week (25% less) on email; the intent-to-treat estimate was 1.4 hours, and document work was moderately faster. [Microsoft Research paper](https://www.microsoft.com/en-us/research/publication/shifting-work-patterns-with-generative-ai/) | 2025 administrative-work time-allocation sensitivity; do not treat email reduction as total output gain. | This is an independent comparison, not the Orion/BEDROCK calibration anchor. |
| Cui et al. (2025) | Combined developer experiments at Microsoft, Accenture, and a Fortune 100 company estimate a 26.08% increase in completed tasks (SE 10.3%). [Management Science article](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2025.00535) | A positive 2025 software-engineering comparator if a separate engineering validation run is added. | The present configurations set software-engineering share to zero. |
| Becker et al. / METR (2025) | In mature repositories, experienced open-source developers took 19% longer with early-2025 AI tools. [Study paper](https://metr.org/Early_2025_AI_Experienced_OS_Devs_Study-paper.pdf) | A negative engineering boundary check and reason to preserve the maintenance context. | Different population and work; it should not lower the support-work calibration. |

For each annual run, record: simulated median annualized efficiency gain,
run-level risk, active-user share, delivered-usage share, and the exact
work-type mix. Compare only like measures: task completion/time estimates to
the simulated task-level gain proxy, and do not compare a customer-facing
resolution-time measure directly with a firm-wide profit or headcount result.

## References

### Comparative publications and operational case studies

The first eight references below are the publications and case studies entered
as comparators by `scripts/validate_published_gains.py`. They are listed in APA 7th
edition style. The operational case studies are vendor-reported and should not
be interpreted as independent causal estimates.

Amazon Web Services. (2024, November 12). *Generative AI for agriculture: How
Agmatix is improving agriculture with Amazon Bedrock*. AWS Machine Learning
Blog. https://aws.amazon.com/blogs/machine-learning/generative-ai-for-agriculture-how-agmatix-is-improving-agriculture-with-amazon-bedrock/

Anthropic. (n.d.). *Humach enhances AI-powered customer experience solutions
with Claude*. Claude. Retrieved July 19, 2026, from
https://www.anthropic.com/customers/humach

Becker, J., Rush, N., Barnes, B., & Rein, D. (2025). *Measuring the impact of
early-2025 AI on experienced open-source developer productivity*. Model
Evaluation & Threat Research. https://metr.org/Early_2025_AI_Experienced_OS_Devs_Study-paper.pdf

Brynjolfsson, E., Li, D., & Raymond, L. (2025). Generative AI at work. *The
Quarterly Journal of Economics, 140*(2), 889–942.
https://doi.org/10.1093/qje/qjae044

Cui, K., Demirer, M., Jaffe, S., Musolff, L., Peng, S., & Salz, T. (2026). The
effects of generative AI on high-skilled work: Evidence from three field
experiments with software developers. *Management Science*.
https://doi.org/10.1287/mnsc.2025.00535

Dell’Acqua, F., McFowland, E., III, Mollick, E., Lifshitz-Assaf, H., Kellogg,
K. C., Rajendran, S., Krayer, L., Candelon, F., & Lakhani, K. R. (2023).
*Navigating the jagged technological frontier: Field experimental evidence of
the effects of artificial intelligence on knowledge worker productivity and
quality* (Harvard Business School Working Paper No. 24-013). Harvard Business
School. https://www.hbs.edu/ris/download.aspx?name=24-013.pdf

Dillon, E. W., Jaffe, S., Immorlica, N., & Stanton, C. T. (2025). *Shifting
work patterns with generative AI* (NBER Working Paper No. 33795). National
Bureau of Economic Research. https://doi.org/10.3386/w33795

Noy, S., & Zhang, W. (2023). Experimental evidence on the productivity effects
of generative artificial intelligence. *Science, 381*(6654), 187–192.
https://doi.org/10.1126/science.adh2586

### Validation-configuration and provisioning sources

These sources define the cloud-service context and calibration boundaries in
the validation configurations; they are not additional numeric comparators in
the focused comparison plot.

Amazon Web Services. (n.d.). *Orion Health: Using generative AI to transform
healthcare*. AWS. Retrieved July 19, 2026, from
https://aws.amazon.com/solutions/case-studies/orion-health/

Microsoft. (2024). *Our year with Copilot: What Microsoft has learned about AI
at work*. WorkLab. https://www.microsoft.com/en-us/worklab/our-year-with-copilot-what-microsoft-has-learned-about-ai-at-work

OpenAI. (2024, February). *Klarna’s AI assistant does the work of 700
full-time agents*. https://openai.com/index/klarna/

Venkatesh, V., Morris, M. G., Davis, G. B., & Davis, F. D. (2003). User
acceptance of information technology: Toward a unified view. *MIS Quarterly,
27*(3), 425–478. https://doi.org/10.2307/30036540

## Interpretation limits

- Vendor case studies are useful implementation records but are not
  independent causal studies.
- A time saving, faster case resolution, reduced repeat contact, and a profit
  estimate are different outcomes. Do not substitute one for another.
- The 2024 and 2025 files change the empirical gain anchor and technology
  provision description; they do not claim to measure the year-on-year change
  in foundation-model capability.
- A validation run that needs local-model CAPEX should wait for a public case
  that discloses both a measurable work outcome and the local hardware or
  contract cost. The supplied `configs/hardware_benchmarks.csv` is suitable for a
  sensitivity analysis, but it is not evidence that a selected company
  incurred that CAPEX.
