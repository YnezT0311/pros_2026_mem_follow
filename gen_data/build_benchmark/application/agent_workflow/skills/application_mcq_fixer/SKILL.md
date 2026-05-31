# Application MCQ Fixer

Use this skill after an analyst report to repair an application MCQ or route it
back to planning. The fixer owns the iteration loop but should keep edits
minimal and auditable.

## Inputs

- analyst report
- current MCQ item
- planner and questioner outputs
- solver A/B results
- source stage

## Repair Priority

1. If the target is valid and unique, first edit the question.
   - Make it a concrete real user request.
   - Add non-leaking background if the question is too abstract.
   - Remove all testing/meta wording.
2. Then edit choices.
   - Strengthen the `use_memory` answer when the target is seen.
   - Strengthen the `without_memory` answer as the natural default when unseen,
     using concrete visible background from the question.
   - Make `use_memory` require the hidden preference. It should often look like
     a tradeoff against the visible background, not like the universally safest
     answer.
   - Keep `plausible_wrong` plausible but clearly weaker.
3. If the target is turn-specific, contaminated, or not application-useful,
   return to the question planner.
4. If no target-unique lasting memory exists in the stage, drop the stage.

## Planner Escalation

Return to the planner when the failure is caused by extraction, not wording:

- the "memory" is a whole workflow rather than one durable preference
- the question tests a broad domain theme already present in the stage
- the no-target context supports the same answer through another turn
- the use-memory choice is best because it is generically safer, not because
  the target preference is known

## Solver Loop

After fixing question or choices, the fixer may request another solver A/B run
and analyst pass. Do not call a fix complete until both baseline conditions pass
and the analyst accepts the target validity.

## Boundaries

- Do not convert turn-specific facts into application preferences by wording
  tricks.
- Do not add the target memory into the question.
- Do not add benchmark/meta language to force the expected answer.
- Do not use synthetic contexts.
- Do not drop a stage before checking whether another target in the same stage
  can support a valid application MCQ.

## Output

Return JSON:

```json
{
  "id": "...",
  "action": "edited_question_choices | replan_target | drop_stage",
  "updated_item": {},
  "reason": "...",
  "needs_solver_rerun": true,
  "notes_for_next_agent": "..."
}
```
