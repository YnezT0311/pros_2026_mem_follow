# Application MCQ Solver

Use this skill to simulate a plain API baseline answer for application MCQs.
Run one solver instance for `with_target_baseline` and one for
`without_target_baseline`.

## Inputs

- baseline solver input JSON for exactly one condition
- original messages provided in that solver input
- MCQ question and choices

## Rules

- Use only the messages provided in the solver input.
- Do not inspect expected labels.
- Do not use external knowledge beyond ordinary language understanding.
- Answer as the assistant would under the given conversation condition.
- Choose exactly one choice label.
- Keep rationales concise and evidence-grounded.

## Conditions

For `with_target_baseline`, the target user turn and immediate assistant reply
are present. A valid item should usually lead to the `use_memory` choice.

For `without_target_baseline`, the target user turn and immediate assistant
reply are removed. A valid item should usually lead to the `without_memory`
choice.

If the conversation without the target still supports the memory choice, choose
the memory choice and explain the visible evidence. That is useful evidence for
the analyst.

## Output

For a batch, write JSON:

```json
{
  "items": [
    {
      "id": "...",
      "choice": "A",
      "rationale": "..."
    }
  ]
}
```

Choices should be uppercase labels.
