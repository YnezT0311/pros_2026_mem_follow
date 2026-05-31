# Application MCQ Question Planner

Use this skill to inspect one source conversation stage and decide which user
turns can support valid application MCQs. The planner does not write final MCQs.
It selects target memories and explains why candidates are valid or invalid.

## Inputs

- `data/generated/<topic>/persona0_sample0/conversation_package.json`
- one `Conversation Stage XX`
- existing item or failed item, if replanning
- analyst report, if available

## Planning Standard

Select only target user turns that contain a lasting preference, stable
constraint, reusable habit, or durable user tendency. The memory must be useful
in a new task outside the original turn.

Reject candidates when the target is only:

- a one-off price, deadline, quantity, appointment time, pickup window, ticket
  tier, lab value, bill amount, or route detail
- a task-local message draft fact
- an assistant-side recommendation, calculation, or wording choice
- a fact that cannot naturally affect a future application decision
- a whole task workflow compressed into a "preference" (for example,
  medication-management, budget-review, or trip-planning steps bundled
  together). If a turn contains a workflow, extract only a narrow durable
  preference or habit inside it; otherwise reject or replan.
- a broad domain theme already present throughout the stage, such as generic
  medication timing, budgeting, training, diet, or travel logistics. The
  selected memory must be narrower than the stage topic.

Good targets are usually one crisp durable preference:

- prefers gentle reminders over loud alarms
- wants written summaries for partner decisions
- uses a spreadsheet plus notebook instead of linked budgeting apps
- avoids a specific allergen or diet category

Bad targets are usually bundles:

- "the user needs a medication reminder routine"
- "the user wants a budget reset"
- "the user needs a travel food plan"

## Contamination Check

For every candidate, mentally remove the target user turn and its immediate
assistant reply. Then inspect the rest of the same stage.

Mark the candidate contaminated if another turn still supports the same target
preference or stable constraint strongly enough that a solver could choose the
memory answer without the target.

Do not mark a candidate contaminated merely because the stage has broad topic
similarity. The question is whether the same actionable preference is still
recoverable.

When a broad topic appears repeatedly, do not rescue it by writing a broader
memory. Narrow the memory until it is target-unique, or send the stage back to
planning/drop. If the remaining stage still contains direct evidence for the
narrow memory, the candidate is contaminated.

## Scoring

Score each candidate qualitatively:

- `lastingness`: whether the memory is reusable beyond the original turn
- `target_uniqueness`: whether removing the target removes the evidence
- `application_usefulness`: whether it can change a future answer
- `natural_question_potential`: whether a concrete new user task can use it
- `domain_safety`: whether the resulting MCQ can be safe and appropriate

## Output

Return structured JSON:

```json
{
  "stage_id": "stage_XX",
  "verdict": "has_candidate | weak_candidate_only | drop_stage",
  "best_candidate": {
    "target_user_turn": "...",
    "target_turn_id_if_known": "stage_XX_HNNN",
    "lasting_memory": "...",
    "why_lasting": "...",
    "why_target_unique": "...",
    "downstream_question_idea": "..."
  },
  "backup_candidates": [],
  "rejected_candidates": [
    {
      "target_user_turn": "...",
      "reason": "turn_specific_fact | contaminated | assistant_derived | not_application_useful | unsafe"
    }
  ],
  "drop_reason": null
}
```

If no candidate is strong, provide the best weak candidate only so the analyst
or fixer can decide whether to drop or attempt a constrained repair.
