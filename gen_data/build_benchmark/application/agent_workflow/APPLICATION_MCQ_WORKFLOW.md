# Application MCQ Agent Workflow

This workflow standardizes construction and review of MemoryCtrl application
MCQs. Application MCQs test whether a model uses a previously seen lasting user
preference, constraint, or habit in a new natural task. They must not be recall
questions for one-off target-turn facts.

## Core Standard

An application MCQ is valid only when all of the following are true:

- The target memory is a lasting preference, stable constraint, reusable habit,
  or durable user tendency.
- The target memory is target-unique within the source stage after removing the
  target user turn and its immediate assistant reply.
- The new MCQ question is a concrete user task that could be answered naturally
  without seeing answer choices.
- The question does not disclose the target preference or target-only facts.
- The choices are natural assistant responses, not benchmark instructions.
- With the target turn present, the plain solver should select the memory-using
  answer.
- Without the target user turn and immediate assistant reply, the plain solver
  should select the default answer that does not use that memory.

The source context for both baseline conditions must be the original
`Conversation Stage XX` from `data/generated/.../conversation_package.json`.
Synthetic single-stage contexts are not allowed.

## Lasting Preference Versus Turn-Specific Fact

Acceptable target memories include:

- dietary restrictions and safety constraints, such as seafood allergy,
  vegetarian preference, or cross-contact concern
- mobility, accessibility, pacing, transit, or time-of-day constraints
- stable communication style preferences, such as short, friendly, non-formal
  messages
- durable budgeting or decision habits, such as preferring a strict cap,
  prioritizing low-friction choices, or avoiding fragile cash-flow plans
- health-management habits, refill-planning preferences, appointment access
  constraints, or clinician-communication preferences

Reject target memories that are only turn-specific facts, including:

- one-off prices, balances, ticket quantities, dates, pickup windows, or lab
  values
- one specific message draft detail, reservation time, invoice amount, or form
  field
- assistant-side calculations, recommendations, summaries, or inferred advice
- facts that matter only for the original turn and do not transfer to a new
  user task

Some facts can be usable only if they reveal a stable preference or constraint.
For example, "the user has a seafood allergy" is a reusable constraint; "the
user asked for 3 bags to be stored from 1:00 PM to 3:00 PM" is a one-off task
fact and should not be used for application MCQ targeting.

## Natural Question Standard

The MCQ question should be a real user request. It should not mention memory,
prior conversation, target turn, hidden context, or benchmark conditions.

Good question shape:

```text
I am going to Hainan and want a few local restaurant ideas. What would you
recommend for a first dinner?
```

Bad question shape:

```text
Which response best uses the remembered dietary constraint from the prior
conversation?
```

The question may introduce a new situation where the hidden preference matters.
It should contain enough background to be specific, but it must not contain the
target preference itself.

## Choice Roles

Each item should have three choices unless a specific export format requires
two:

- `use_memory`: a natural, high-quality answer that applies the target lasting
  preference or stable constraint.
- `without_memory`: the natural default answer when the target memory is not
  known. This should be plausible and often slightly more obvious from the new
  question alone.
- `plausible_wrong`: a weaker answer that is not absurd, but is clearly worse
  than the other two.

Choices must not contain phrases such as:

- "if the prior conversation established"
- "based on remembered context"
- "use the memory"
- "only if target turn is present"
- "without memory"

## Baseline Conditions

`with_target_baseline`:

- Uses the original source stage.
- Keeps the target user turn and immediate assistant reply.
- Expected answer: `use_memory`.

`without_target_baseline`:

- Uses the same original source stage.
- Removes the target user turn and its immediate assistant reply.
- Expected answer: `without_memory`.

If other turns in the same stage still support the same target preference after
target removal, the item is contaminated. Replan the target within the same
stage before dropping the stage.

## Roles

The application MCQ workflow uses six agent roles:

1. `application_mcq_question_planner`
2. `application_mcq_questioner`
3. `application_mcq_solver`
4. `application_mcq_analyst`
5. `application_mcq_fixer`

Run two solver instances for each item or batch: solver A for
`with_target_baseline` and solver B for `without_target_baseline`.

## Iteration Loop

1. The question planner selects and ranks usable target-turn candidates for a
   stage.
2. The questioner writes a natural application MCQ from the selected candidate.
3. Solver A answers the MCQ with the target present.
4. Solver B answers the MCQ with the target removed.
5. The analyst judges the results and diagnoses failures.
6. The fixer edits the question or choices first when the target is valid. If
   the target is invalid or contaminated, the fixer sends the stage back to the
   planner.

Prefer preserving one item per stage by replanning the target. Drop a stage only
when the stage has no target-unique lasting preference or stable constraint.

## Output Expectations

Every accepted item should preserve enough provenance to audit it later:

- topic, persona, and stage id
- target user turn text and recomputed source-stage turn id
- target lasting preference or stable constraint
- rejected or backup candidates when useful
- natural MCQ question
- choice text and choice roles
- expected labels for both baseline conditions
- solver A/B choices and rationales
- analyst verdict and failure/fix notes
