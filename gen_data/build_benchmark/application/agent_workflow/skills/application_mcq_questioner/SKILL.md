# Application MCQ Questioner

Use this skill to turn a planner-selected lasting memory into a natural
application MCQ. The questioner writes the user-facing question and answer
choices. It should not select a new target unless the planner candidate is
clearly invalid.

## Inputs

- planner output
- source stage excerpt for the selected target
- any analyst or fixer constraints from prior rounds

## Question Requirements

The question must sound like a real user request in a new situation. It must be
specific enough to answer without choices, but it must not reveal the target
memory.

Do:

- create a concrete new scenario where the target preference would matter
- include ordinary background details that make the task realistic
- let the default answer be natural when the target memory is unknown
- make the `without_memory` option explicitly use the visible background.
  For example, if a beach city naturally suggests seafood, the default option
  should say that fresh seafood is the local draw.
- make the `use_memory` option visibly trade off against the visible
  background unless the target memory is known. It should be better only with
  the hidden preference, not because it is generically safer or more complete.

Do not:

- mention "memory", "prior conversation", "target", "baseline", or "hidden"
- ask "which response best uses remembered context"
- include the target preference or target-only fact in the question
- write a question that only makes sense after reading the choices

## Choice Requirements

Prefer three choices:

- `use_memory`: natural answer applying the lasting preference
- `without_memory`: natural default answer if the lasting preference is unknown
- `plausible_wrong`: weaker answer, plausible but not best

Use two choices only when the downstream item schema requires it.

Choices should be complete assistant response options, not labels or test
instructions. Avoid meta wording such as "only if the prior conversation
established..."

Avoid making `use_memory` a universally cautious answer. If it is always safer,
more detailed, or more responsible even without the target memory, the MCQ will
fail the never-seen baseline.

## Output

Return JSON:

```json
{
  "stage_id": "stage_XX",
  "target_user_turn": "...",
  "lasting_memory": "...",
  "question": "...",
  "choices": {
    "A": "...",
    "B": "...",
    "C": "..."
  },
  "choice_roles": {
    "A": "without_memory",
    "B": "use_memory",
    "C": "plausible_wrong"
  },
  "expected": {
    "with_target_baseline": "B",
    "without_target_baseline": "A"
  },
  "rationale": "..."
}
```

The labels may differ, but the role mapping must be explicit.
