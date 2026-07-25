# Governance and Execution

## Contents

- [Decision policy](#decision-policy)
- [Question design](#question-design)
- [Progress reporting](#progress-reporting)
- [Managed processes](#managed-processes)
- [Budget and dependency control](#budget-and-dependency-control)
- [Audit trail](#audit-trail)

## Decision policy

Classify decisions by consequence:

- **Ask** — obtain explicit approval for choices that change the business
  question, target, prediction moment, split semantics, irreversible data
  handling, evaluation interpretation, large dependency installation, or
  deployment decision.
- **Recommend** — present a grouped table of material data-quality/modeling
  findings with a recommended action and let the user approve or override the
  group or individual rows.
- **Default** — apply low-risk reversible mechanics such as seed 42, artifact
  names, plot sampling seed, and safe output directories; record them.
- **Required safeguard** — do not allow user convenience to silently introduce
  target leakage, holdout reuse, unsafe deserialization, or false performance
  claims. Explain the constraint and offer a valid alternative.

Do not encode universal thresholds as scientific truths. A percentage is a
screening trigger that must be interpreted in context.

## Question design

Ask the smallest number of questions needed to avoid a wrong workflow.

1. Ask blocking problem-framing questions before loading data.
2. Profile data.
3. Present related findings in a concise decision table:

   | Finding | Why it matters | Recommendation | Decision |
   |---|---|---|---|

4. Separate unrelated high-impact questions, such as redefining the target or
   changing the prediction moment.
5. Explain jargon in one sentence and state the consequence of each option.

Avoid dozens of one-column-at-a-time prompts. Transparency means visible
reasoning and recorded decisions, not unnecessary interruption.

## Progress reporting

Use the host task/plan UI when available (`update_plan` in Codex or
`TodoWrite` in Claude Code). Maintain one item per major workflow stage, not
one item per chart or model trial.

Send user-visible updates:

```text
▶ Stage N — <name>
✓ Stage N complete — <decision or result>
```

During long work, update at least once per minute with useful information such
as completed trials, best validation score, elapsed budget, or current model
family. Do not repeat unchanged status.

## Managed processes

Write long-running Python entry points to disk and make them persist progress
and results. Launch them through the host's managed-process/session mechanism:

- Codex: retain the command session ID and poll it.
- Claude Code: use background Bash and poll the task ID.

Use managed execution for model search, stacking, AutoGluon, large SHAP runs,
or any operation likely to exceed a foreground tool timeout. Record
`execution_mode: managed_process` and the session/task ID when available.

Do not shorten the user-approved ML budget merely to fit a tool timeout.

## Budget and dependency control

- Ask for a compute or elapsed-time budget before expensive search.
- Use a conservative default when the user has no preference: enough to cover
  each eligible family at least once, then adaptive search within a bounded
  budget.
- Control CPU threads, process parallelism, memory, and GPU use explicitly.
- Ask before installing large optional systems such as AutoGluon.
- Install dependencies only inside the project environment.
- Record installation failures and continue with valid alternatives when
  possible.

## Audit trail

Record in `artefacts/config.json`:

- user choices and defaults;
- assumptions and unresolved domain questions;
- rejected alternatives and reasons;
- task route and references used;
- data/split fingerprints;
- compute budgets and actual usage;
- failed trials and dependency limitations;
- any override of a leakage, no-signal, or evaluation warning.

The artifacts on disk are authoritative. If a chat summary conflicts with
them, fix the summary or artifacts before handoff.
