# High-Stakes ML Safeguards

## Contents

- [When this applies](#when-this-applies)
- [Required gate](#required-gate)
- [Evidence standard](#evidence-standard)
- [Operational safeguards](#operational-safeguards)
- [Required reporting](#required-reporting)

## When this applies

Treat a use case as high stakes when model output can materially affect health,
safety, liberty, employment, credit, insurance, education, housing, legal
rights, or access to essential services. Also treat low-frequency catastrophic
harm as high stakes even if average impact is modest.

The skill provides technical workflow guidance, not legal, clinical, actuarial,
credit-risk, or regulatory approval.

## Required gate

Before recommending deployment, require and record:

- accountable domain owner;
- intended decision and prohibited uses;
- applicable legal/regulatory/governance review status;
- human oversight, override and appeal/escalation path;
- critical harm and cost matrix, including rare-class misses;
- label provenance, adjudication quality and known historical-policy bias;
- prospective/external validation plan;
- incident owner, monitoring cadence and rollback/disable mechanism.

If these are missing, limit the result to research, retrospective analysis, or
silent prospective validation. Do not recommend autonomous action.

## Evidence standard

- Evaluate at the independent unit: patient, person, account, policyholder,
  site, or time block—not merely rows.
- Report effective independent sample size and rare-outcome support.
- Use uncertainty bounds on harm-critical metrics, not only average accuracy.
- Refuse a safety claim when rare classes cannot support discrimination,
  calibration or miss-rate estimates.
- Require future/external evidence when local cross-validation cannot reproduce
  deployment shift.
- Audit label delay, selective labeling, treatment/policy feedback and
  historical inequity.
- Assess calibration, abstention/coverage, worst-supported subgroups and
  out-of-distribution behavior.

Statistical non-significance is not proof of equal safety. A good overall score
cannot compensate for an unacceptable critical-harm slice.

## Operational safeguards

Prefer decision support with meaningful human review. Define:

- hard domain rules that the model cannot override;
- abstention and manual-review triggers;
- behavior for missing critical inputs, unseen values and drift;
- output explanations appropriate to the reviewer;
- logging of inputs, versions, decisions, overrides and outcomes;
- rate/capacity limits and fallback procedure;
- monitoring after labels mature;
- change control and revalidation requirements.

Do not describe human review as a safeguard when reviewers lack time,
information, authority, training, or a practical override path.

## Required reporting

`config.json` must record `governance.risk_tier: "high"` and the domain owner,
oversight, approval status, deployment decision and prohibited uses.

The model card and results must state:

- intended and prohibited uses;
- population, sites, periods and exclusions represented;
- validation design and independence limitations;
- critical metric point estimates, uncertainty and support;
- calibration/abstention and subgroup findings;
- label and causal limitations;
- known failure modes and out-of-distribution policy;
- whether use is research, silent validation, decision support, or autonomous;
- remaining approvals and evidence required.

When evidence is insufficient, lead with that conclusion. Do not bury it below
model scores.
