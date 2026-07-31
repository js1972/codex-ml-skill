# SAP RPT Track

## Contents

- [Role and production positioning](#role-and-production-positioning)
- [Approval and access](#approval-and-access)
- [When to use](#when-to-use)
- [Fair evaluation](#fair-evaluation)
- [Context and query packaging](#context-and-query-packaging)
- [Response validation](#response-validation)
- [Scale and reproducibility](#scale-and-reproducibility)
- [Inference and artifacts](#inference-and-artifacts)
- [Selection](#selection)

## Role and production positioning

Treat SAP RPT as a production-capable pretrained tabular foundation model for
supported classification and regression tasks.

SAP RPT does not require project model training, fitting, Optuna, or
hyperparameter search. The project prepares and manages:

- a valid labelled context from permitted development rows;
- query rows;
- schema and task semantics;
- batching and request budgets;
- response validation;
- evaluation and inference adapters.

Separate the model from its access route:

- The internal SAP RPT CLI provides convenient internal managed access without requiring
  the user to provision a BTP subaccount, SAP AI Core instance, service keys,
  or an RPT deployment.
- Paying customers can use the same RPT model through SAP AI Core and manage
  their production deployment, authentication, quotas, and billing there.

The CLI's internal-use status describes the access channel, not the model's
quality or production capability. Treat RPT as a first-class selectable
backend.

## Approval and access

Include SAP RPT in the mandatory experiment approval. Present:

- include or decline;
- accessible RPT model IDs and proposed access route;
- target and eligible features;
- whether the full fold-valid labelled context fits; otherwise the distinct
  context sizes and retrieval strategies to compare;
- `sap-rpt[retrieval]` readiness and any approved installation;
- input format and reproducibility seed;
- maximum context rows, context-plus-query rows per request, query rows per
  call, and transmitted columns;
- maximum calls/retries, request timeout, CPU count, parallel jobs, GPU flag,
  memory, and latency/cost allowance;
- data sent to the remote service.

Do not infer exclusion from “train the best model.” Recommend whether to include
RPT in the consolidated experiment approval. If the user explicitly requested
RPT, treat the include/decline decision as already answered and show RPT as
selected.

When the internal CLI is selected:

1. Check whether `sap-rpt` is available without modifying the environment.
2. If absent or unconfigured, direct the user to the current private
   `rpt-cli` repository and its installation/configuration instructions.
3. Inspect accessible models with `sap-rpt configure` or the current
   machine-readable equivalent; never assume standard or large model IDs.
4. Check whether the retrieval extra imports successfully. If the approved
   plan needs `vectorsearch` and the extra is absent, disclose the private
   installation action in the consolidated approval.
5. Keep credentials and interactive authentication user-managed; never
   request, display, log, or store tokens.
6. Inspect CLI help/version and perform non-secret readiness checks after the
   user confirms setup.
7. Do not provision BTP, AI Core, service keys, or an RPT deployment for this
   route; the CLI handles access under the hood.

Run `sap-rpt auth status -o json` during read-only readiness checks. Treat exit
code 1 as expired authentication and disclose the user-managed
`sap-rpt auth login` prerequisite before the consolidated approval. For a
missing model, refresh accessible choices with `sap-rpt configure` or pass the
approved `-m <model-id>`; never silently substitute another model. These are
preflight blockers for the RPT track, not reasons to interrupt an otherwise
approved run after execution starts.

Include the named endpoint and feature/label/query/identifier transfer scope in
the single consolidated experiment approval. That approval covers the first
and subsequent requests within the disclosed scope; do not ask a second
RPT-specific confirmation. Pause only this track when setup or consolidated
transfer approval is missing. Ask again only if the destination, purpose,
fields, sensitivity, or row volume materially expands, or an independent
external policy requires it.

Record the consolidated permission as a structured
`approval.remote_transfers` entry and reference its ID from
`backends.sap_rpt.transfer_confirmation`. This backend field is execution
evidence that the request matched the approved scope, not evidence of a second
user prompt.

## When to use

Use SAP RPT for supported supervised tabular classification or regression when:

- row-level supervision is scientifically valid;
- a representative fold-valid labelled context can be built;
- deployed limits support the columns, classes, context, and query batches;
- remote transfer, latency, request volume, and access are acceptable.

Do not force the tabular interface onto native forecasting, causal/uplift,
survival, or unlabeled anomaly tasks.

Do not run when:

- the user declines;
- no leakage-safe representative context exists;
- response probability/value semantics cannot support the approved metric;
- offline/edge serving is mandatory and no acceptable RPT route exists;
- required calls exceed the approved budget.

These are task or access limitations, not limitations of the underlying RPT
model.

## Fair evaluation

Share information boundaries with other tracks:

- source population, target, and eligible prediction-time features;
- development/evaluation row IDs, groups, times, and weights;
- primary metric implementation;
- final evidence boundary.

For every fold, build labelled context only from its permitted training rows
and query the corresponding validation rows. Never include validation,
holdout, active outer-fold, or future labels in context.

For temporal evaluation, use only features and matured labels available at the
fold scoring origin. For grouped evaluation, preserve the group boundary in
context and queries. Under nested CV, repeat any RPT model/context-policy
selection inside the outer training partition or label the result
development-only.

Freeze the RPT model, access/deployment route, schema, context policy, query
batching, probability interpretation, and optional calibration before final
evaluation.

## Context and query packaging

Treat context design as data management, not model training.

Use the full fold-training population when supported. Do not default to 512
context rows or confuse `max_query_batch_rows` with `max_context_rows`.

If the full context exceeds a deployed or approved limit, compare distinct
useful sizes on development folds. Use up to 512 and 2,048 rows as diagnostic
anchors when the available population makes them distinct, and always include
the largest permitted/practical context. Deduplicate sizes that collapse to
the same value. These anchors are not universal caps.

Use the RPT CLI's documented context strategies:

- `random::N`: seeded, reproducible uniform context sampling with no retrieval
  dependency;
- `vectorsearch::N`: FAISS plus character n-gram hashing, when the retrieval
  extra is installed and approved;
- no `--context-strategy`: full eligible context when it fits.

Treat `random::N` as the dependency-free truncation baseline. Compare
`vectorsearch::N` when it is available and scientifically relevant, but do not
assume it will win: character n-grams are especially natural for
identifier-like and categorical values and may be less meaningful for
predominantly continuous numeric tables. Query rows are preserved by the CLI
regardless of context strategy.

Create every strategy's candidate population from fold-training rows only.
Query features may choose neighbors at inference time; query labels and query
rows must never enter the retrieval corpus. “Predate the query” is mandatory
when time has real prediction semantics; for IID data without a meaningful
time axis, enforce fold/group/duplicate isolation instead.

For each resulting context:

- classification: preserve representative priors and include adequate support
  for every evaluated class; record intentional rebalancing;
- regression: preserve the target range, tails, important groups, and time
  regimes;
- grouped data: select groups before rows;
- temporal data: use a declared as-of or recency policy.

Record context row IDs, order, selector logic, seed, size, class/target support,
as-of boundary, schema, and SHA-256 fingerprint. Keep the same context for all
query batches in a fold unless query-dependent `vectorsearch` was explicitly
approved.

Keep identifiers outside model features and use them to verify response
alignment. Represent numeric-looking codes as categorical/string fields when
they are not quantities. Do not apply the classical preprocessing pipeline.

Compare model IDs, context sizes, and retrieval policies on development
evidence within the approved request budget. Freeze one selected configuration
before final evaluation; do not multiply final-holdout requests across
candidate configurations. Do not describe these choices as RPT
hyperparameters or tune them on final evidence.

## Response validation

Invoke the configured route fail-closed. Require:

- process/request success and a successful parsed status;
- exact response count and unique row-ID alignment;
- finite task-valid predictions;
- actual model and deployment/routing identifiers;
- request IDs, timings, retries, and failure reasons.

For classification, verify known unique classes, finite confidences in
`[0, 1]`, and complete distributions that approximately sum to one. If the
service returns truncated top-k results, use only metrics supported by those
semantics; do not invent missing probabilities for log loss, Brier score,
calibration, average precision, or AUC.

Fit any calibration only from fold-valid out-of-fold RPT scores. For
regression, verify finite outputs and treat intervals as unspecified until
their semantics are documented and empirical coverage is measured.

Test repeated-call stability and singleton-versus-batch behavior on a small
development sample. Record the production-equivalent batch size.

## Scale and reproducibility

Discover actual deployed limits rather than assuming an old model-card or CLI
limit. Keep four capacities distinct:

- `max_context_rows`: labelled context rows included in a request;
- `max_request_rows`: total context plus query rows accepted in one request;
- `max_query_batch_rows`: query rows accepted in one call;
- `max_columns`: all transmitted columns, including target and internal row ID.

Require `max_context_rows + max_query_batch_rows <= max_request_rows` for the
planned maximum request. Do not reuse “rows per request” to mean query rows
only.

Use column-oriented JSON, CSV, or Parquet input for CLI context strategies;
never use row-oriented `{"rows": [...]}` input with them. Prefer Parquet when
the input exceeds 1 MB. Parquet improves packaging and transport efficiency;
it does not override deployed row, column, or request limits.

Calculate planned calls before execution:

```text
sum(ceil(query_rows_per_fold / max_query_batch_rows))
```

Include retries and the required representative/single-row inference
validation repeats in the approved request allowance.

Make the retained adapter batch-safe for arbitrary new input sizes. Split query
rows into sequential chunks no larger than `max_query_batch_rows`; before each
transfer, enforce `context rows + chunk rows <= max_request_rows` and the
`max_columns` limit. Preserve stable row IDs and original input order when
combining responses. Fail before the first request if the complete operation,
including allowed retries, would exceed `max_requests`.

Record one configuration-ledger row for every attempted combination, including
model/deployment ID, candidate/planned/transmitted context rows, exact CLI
strategy, seed, input format, fold/time eligibility policy, status/failure
reason, comparable score, request count, median/p95 latency, throughput, and
whether it was selected. Also record context/query rows and columns,
approximate request bytes, failures/retries, and wall time.

Remote results may have limited point-in-time reproducibility. Record:

- client version and source revision;
- auth/access mode and endpoint;
- RPT model and actual deployment/routing ID;
- context/query fingerprints and batch size;
- request IDs and timestamps;
- raw response hashes when retained.

Never store service credentials. Do not download a huge remote source merely
to use RPT; build bounded fold-valid contexts and queries where the data lives.

## Inference and artifacts

Retain:

```text
backends/sap_rpt/
├── context.parquet
└── adapter.py                 # only when needed
```

Record the context policy, context hash, schema, model/access metadata, request
summary, result, and limitations in root `run.json`. Do not create:

- an RPT `train.py`;
- a local model file;
- Optuna trials;
- training hyperparameters;
- copied classical or AutoGluon artifacts;
- redundant request/response trees.

Implement and test:

```text
python infer.py --backend sap-rpt \
  --input new_wines.csv --output predictions.csv
```

The adapter must load the frozen context, validate new-row schema, package
queries, invoke the configured route, validate row alignment and response
semantics, and emit the same task-level output contract as other retained
backends.

## Selection

Report SAP RPT in the inclusive comparison table with its development and any
permitted final metric, uncertainty, context coverage, probability limitations,
request failures, latency, throughput, and access requirements.

Include the full RPT configuration ledger and explicitly summarize whether
context scale, retrieval, model variants, and full context were tested. Use
“evaluated under the approved configurations,” not an unqualified “fully
evaluated.” When full fold-valid context fits, explain that retrieval
comparison was unnecessary. When truncation was required, identify any
available context scale, retrieval strategy, or model variant that was not
tested.

Permit SAP RPT to be:

- the best predictive result;
- the recommended operational backend;
- the default inference backend.

Base operational selection on the user's actual deployment constraints. When
customer production use requires SAP AI Core, state the remaining deployment
work—exact deployment, authentication, context storage, quotas, resilience,
observability, residency, monitoring, and validation on that route—without
downgrading the RPT model or invalidating the measured comparison.
