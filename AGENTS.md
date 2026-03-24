# PersonaMem Benchmark Revision Notes

This file records the current design constraints for privacy-policy benchmark revision in this repository.

Working terminology:

- `no_store`: use now, do not retain afterward
- `deletion`: remove or stop using previously shared information
- `no_use`: answer the current request without relying on restricted earlier context

These three behaviors are distinct and should not share the same candidate-selection logic.

## Core Problem Statement

The current benchmark often selects targets that look like "concrete facts" in a shallow way, but do not match natural user privacy behavior.

In particular, the current candidate-picking logic tends to select:

- scaffold-like user turns
- broad descriptive statements
- turns that are memory-bearing only in a loose lexical sense

This is not sufficient for a realistic `no_store` benchmark.

## Preferred World Definitions

### 1. No-Store World

User intent:

- "You may use this information right now."
- "Do not keep it as persistent memory afterward."

Natural pattern:

- user asks the model to do something
- the user provides temporary context needed for the current task
- that context is sensitive, private, or situational

Typical examples:

- financial details provided for one-time budgeting advice
- medical symptoms or medication details provided for immediate triage/help
- legal dispute facts provided for immediate procedural advice
- travel identifiers, private logistics, or personal constraints provided for immediate planning

Implication:

- `no_store` targets should usually be task-bearing turns with sensitive temporary context
- they should not be selected mainly because they look like standalone "facts"

### 2. Deletion World

User intent:

- "I already told you this."
- "Now delete / forget / stop using it."

Natural pattern:

- previously shared information is later withdrawn, corrected, or revoked

Typical examples:

- wrong address or identifier
- overshared personal detail
- outdated diagnosis/status
- old preference or policy that user no longer wants retained

Implication:

- deletion targets should usually be previously shared items that are later retracted or invalidated

### 3. No-Use World

User intent:

- "Answer this without using the previous message/context."

Natural pattern:

- user wants scope control over the current answer
- user does not necessarily request deletion or memory erasure

Implication:

- `no_use` should test response-time dependence control, not persistent memory removal

## Baseline World Concerns

The current baseline conversations often look like:

- user dumps a sequence of facts
- assistant reacts with polished suggestions

This is weaker than a realistic help-seeking conversation.

Preferred baseline shape:

- user comes to the assistant for concrete help
- user provides context because it is needed for the task
- assistant performs an actual task: triage, prioritization, planning, rewriting, advising, checking, comparing, filtering, summarizing

Less preferred baseline shape:

- user narrates a benchmark-like autobiography
- assistant only echoes preference-aware planning language

Benchmark revision should consider changing baseline generation so that many turns are:

- request-first
- context-for-task-second
- naturally privacy-sensitive when appropriate

## Candidate Selection Guidance

Do not optimize only for "concrete memory-bearing facts."

Instead, use separate candidate schemas by policy type.

### No-Store Candidate Schema

Good no-store candidates usually contain:

- a direct user request
- temporary context needed to answer that request
- privacy-sensitive content

Examples of useful signals:

- debt, income, balance, account, repayment
- symptoms, medication, diagnosis, clinician notes
- legal dispute details, evidence, named parties, contract terms
- addresses, contact info, IDs, booking details, private schedules

Bad no-store candidates:

- pure opener/scaffold turns
- generic preference narration
- benchmark-style "let me walk you through..." turns
- turns whose main content should naturally become long-term memory

### Deletion Candidate Schema

Good deletion candidates usually involve:

- previously shared information
- later correction, revocation, or explicit withdrawal

Strong sources:

- history records with `[Old Event]` plus `[Reasons of Change]`
- turns showing reversal, suspension, withdrawal, cancellation, disabling, shutdown, or correction

### No-Use Candidate Schema

Good no-use candidates usually involve:

- a follow-up question
- a plausible temptation to use previous context
- a user instruction that narrows the allowed evidence source

## History Signals Worth Prioritizing

For revision work, the best structured signals often come from contextual history records that contain:

- `Event`
- `[Old Event]`
- `[Reasons of Change]`

These are especially valuable for `deletion` and sometimes `no_store` because they encode:

- prior state
- updated state
- reason for change

They often capture facts a user might later want to:

- revise
- retract
- stop relying on
- replace with newer information

## Immediate Revision Priorities

1. Revisit baseline conversation generation so interactions are more task-oriented.
2. Replace current candidate-picking heuristics with policy-specific selection.
3. Avoid selecting targets solely because they are long, adjacent to side notes, or lexically "fact-like."
4. Prefer sensitive temporary context for `no_store`.
5. Prefer revoked or corrected prior content for `deletion`.
6. Prefer scope-controlled answer requests for `no_use`.

## Migration Notes

The codebase may still contain legacy `retention` names. Future revisions should standardize behavior and naming around `no_store`.

Recommended migration order:

1. finalize behavior definition
2. update baseline/data construction assumptions
3. update builder logic and candidate selection
4. update QA generation assumptions
5. rename scripts, docs, and output paths

Do not assume that script renaming alone fixes conceptual misalignment.
