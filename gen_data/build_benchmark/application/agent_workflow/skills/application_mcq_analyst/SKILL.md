# Application MCQ Analyst

Use this skill to judge application MCQ results after solver A and solver B
have answered. The analyst diagnoses whether failures are caused by target
selection, question wording, choices, contamination, or solver ambiguity.

## Inputs

- planner output
- generated MCQ item
- with-target solver result and rationale
- without-target solver result and rationale
- source stage when needed

## Pass Criteria

An item passes only if:

- the target is a lasting preference, stable constraint, reusable habit, or
  durable tendency
- the target is target-unique after removing the target user turn and immediate
  assistant reply
- the question is a natural concrete user request
- the question does not leak the target memory
- the choices are natural answers without benchmark/meta wording
- solver A selects the `use_memory` role
- solver B selects the `without_memory` role

## Failure Types

Use these failure labels:

- `not_lasting_memory`: target is a turn-specific fact
- `contaminated_stage`: other turns support the same preference
- `question_leakage`: question reveals the target memory
- `meta_wording`: question or choices expose benchmark mechanics
- `weak_use_memory_choice`: memory answer is not clearly better when seen
- `weak_without_memory_choice`: default answer is implausible or too weak
- `ambiguous_question`: question cannot be answered naturally without choices
- `unsafe_or_overconfident`: medical, financial, legal, or travel answer is too
  strong for the domain
- `better_candidate_available`: another turn in the same stage is more valid

## Actions

Recommend exactly one:

- `accept`
- `fix_question_or_choices`
- `replan_target`
- `drop_stage`

Prefer `fix_question_or_choices` when the target is valid and unique. Prefer
`replan_target` when the target is turn-specific or contaminated. Use
`drop_stage` only when no target-unique lasting memory appears available.

## Output

Return JSON:

```json
{
  "id": "...",
  "verdict": "accept | needs_fix | replan | drop",
  "failure_types": [],
  "solver_summary": {
    "with_target_choice": "B",
    "without_target_choice": "A"
  },
  "analysis": "...",
  "recommended_action": "accept",
  "fix_guidance": null,
  "best_stage_candidate": null
}
```
