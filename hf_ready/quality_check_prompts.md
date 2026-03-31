# Quality Check Prompts

This file focuses on the quality-review prompts we would use to audit generated conversations and MCQs.

These prompts are based on the concrete issues that repeatedly showed up during development, even when they were not always captured in a formal automated checker at the time.

## 1. Problems We Actually Ran Into

### Conversation problems

These were the recurring issues we saw in generated conversations:

- **Too general / too vague**
  - user says things like broad summaries instead of concrete, grounded requests
  - assistant replies with generic advice rather than specific next steps
- **Not like a real user conversation**
  - user utterances sound like prompt instructions or compressed summaries rather than normal chat
  - turns feel over-written for the benchmark instead of naturally spoken
- **Too clipped / too compressed**
  - user gives headline-style statements without practical constraints, motivations, or feelings
  - interaction does not feel like a genuine help-seeking exchange
- **Assistant responds by asking instead of helping**
  - assistant ends with a question before offering any substantive help
  - answer feels incomplete, as if the assistant has not actually responded yet
- **Lack of concrete details**
  - conversation mentions a task but not the specific constraints that make the request believable
  - event and interaction blocks feel under-specified
- **Awkward local transitions**
  - a line before the next `Side_Note` does not transition naturally
  - the section reads like pasted fragments rather than one coherent exchange

### MCQ problems

These were the recurring issues we saw in rendered recall questions:

- **Question wording was too templatic**
  - examples:
    - `For the 'Family island getaway insurance' request...`
    - `In the earlier interaction labeled ...`
  - this sounds like annotation language, not a natural later user question
- **Identifier label was over-forced into wording**
  - quoted labels appeared directly in the question too often
  - the question sounded benchmark-like instead of conversational
- **Whole recall leaked slot-level details**
  - whole-recall questions asked about dates, budgets, emails, schedules, etc.
  - this blurred the boundary between whole recall and slot recall
- **Distractor choices were too unrelated**
  - wrong answers sometimes pointed to a different conversation entirely
  - they were not plausible competitors to the correct answer
- **Choices were mismatched in form**
  - `not_remember` was often a full sentence
  - but `remember_correct` / `distractor` could be bare values like dates, dollar amounts, or emails
- **Choices were mismatched in length or specificity**
  - one option could be much longer, more concrete, or more obviously right than the others
- **Slot recall answers looked like raw fields**
  - choices sometimes looked like extracted database values, not natural responses
- **Identifier ambiguity**
  - generated `identifier_label` could be broad enough to match more than one interaction

## 2. Conversation Quality-Check Prompt

This prompt is meant for auditing whether a generated conversation sounds concrete, realistic, and naturally help-seeking, rather than just structurally valid.

### Intended use

Use this after a conversation stage has already been generated and basic timestamp coverage is complete.

Its job is to answer:

- Does this sound like a real user talking to a helpful assistant?
- Is the dialogue concrete enough?
- Are any turns too generic, too compressed, or too benchmark-like?
- Does the assistant actually help before asking for more?

### Prompt

```text
You are reviewing a generated conversation for quality. Your job is not to check timestamps or JSON formatting. Your job is to judge whether the conversation sounds like a realistic, concrete, naturally flowing interaction between a user and a helpful assistant.

Please review the conversation section below and identify any problems in the following categories:

1. The user sounds too general, vague, compressed, or unnatural.
2. The assistant sounds too generic, unhelpful, or responds with a question before offering real help.
3. The dialogue lacks concrete details, practical constraints, motivations, or grounded specifics that would make the exchange believable.
4. The wording sounds like prompt instructions, summaries, or dataset annotations rather than ordinary conversation.
5. The local transition is awkward, especially near the last utterance before the next Side_Note.

When judging quality, prefer realistic everyday dialogue. A good conversation should feel like:
- a normal user asking for help in natural language
- a helpful assistant giving a concrete answer, suggestion, or next step
- enough detail to make the request feel specific and believable
- wording that is conversational rather than abstract or benchmark-like

For each problematic line, return:
- the exact original line
- a short explanation of what is wrong
- a revised version that keeps the same meaning but makes the conversation more natural, concrete, and helpful

If a line is already good, do not include it.

Return valid JSON only in the following format:
{
  "has_issues": true,
  "issues": [
    {
      "line": "...",
      "problem_type": "...",
      "reason": "...",
      "suggested_revision": "..."
    }
  ],
  "overall_assessment": "..."
}

If the section is already strong, return:
{
  "has_issues": false,
  "issues": [],
  "overall_assessment": "..."
}

Here is the conversation section to review:

{conversation_section}
```

## 3. MCQ Quality-Check Prompt

This prompt is meant for auditing rendered recall questions and answer choices after generation.

### Intended use

Use this after a whole-recall or slot-recall MCQ has already been rendered.

Its job is to answer:

- Does the question sound like a natural later-conversation question?
- Is the wording overly templatic or annotation-like?
- Does whole recall stay at the right abstraction level?
- Are the distractors plausible competitors to the correct answer?
- Are all three choices similar in form, length, and specificity?

### Prompt

```text
You are reviewing a rendered multiple-choice recall question for quality. Your job is not to solve the question. Your job is to judge whether the question and answer choices feel natural, balanced, and well-designed.

Please review the question and its three answer choices for the following issues:

1. The question sounds templatic, benchmark-like, or annotation-like rather than like a natural later user question.
   Examples of bad style include:
   - "For the ... request ..."
   - "In the earlier interaction labeled ..."
   - overuse of quoted identifier labels

2. The question forces the identifier_label into the wording in an unnatural way.

3. A whole-recall question incorrectly asks about slot-level details such as dates, budgets, schedules, contact information, or other specific sensitive values.

4. A distractor is too unrelated to the correct answer and sounds like it belongs to a completely different conversation rather than the same request space.

5. The answer choices are mismatched in length, sentence form, or specificity.

6. Some choices are raw values or field fragments rather than natural assistant responses.

7. The correct answer is too obviously right even without context because the distractors are too weak.

For each issue you find, return:
- the issue type
- a short explanation
- a suggested rewrite

If possible, provide a fully revised version of:
- the question
- the remember_correct answer
- the distractor answer
- the not_remember answer

The revised answers should:
- all be natural responses
- all be similar in length and sentence form
- keep the distractor in the same topical neighborhood as the correct answer
- make it harder to distinguish the correct answer without the conversation context

Return valid JSON only in the following format:
{
  "has_issues": true,
  "issues": [
    {
      "issue_type": "...",
      "reason": "...",
      "suggested_fix": "..."
    }
  ],
  "revised_version": {
    "question": "...",
    "remember_correct": "...",
    "distractor_irrelevant": "...",
    "not_remember": "..."
  },
  "overall_assessment": "..."
}

If the MCQ is already strong, return:
{
  "has_issues": false,
  "issues": [],
  "revised_version": null,
  "overall_assessment": "..."
}

Here is the MCQ to review:

Question: {question}
Remember-correct answer: {remember_correct}
Distractor answer: {distractor_irrelevant}
Not-remember answer: {not_remember}
QA family: {qa_family}
Identifier label: {identifier_label}
```

## 4. MCQ Structural Completeness Check

Besides semantic quality review, the renderer also applies a strict structural completeness check before a rendered MCQ file is written to disk.

### Intended use

Use this immediately after rendering and before exporting the final benchmark QA files.

Its job is to catch broken MCQs such as:

- empty `choices`
- empty `choice_to_answer_type`
- missing `remember_correct_choice`
- fewer than three answer candidates
- missing one of the required answer types:
  - `remember_correct`
  - `distractor_irrelevant`
  - `not_remember`
- duplicated answer types

This check is especially important because these failures can silently corrupt downstream evaluation and make baseline/test-world utility hard to interpret.

### Current logic in the renderer

For `whole_recall`:

- if the generated question leaks slot-level details such as budgets, dates, contact info, schedules, arrivals, departures, prices, or costs, the item is sent to repair
- if the generated answer bank does not contain exactly the three required answer types, the item is sent to repair
- if the repaired item still does not contain a complete three-answer bank, rendering fails instead of writing a broken question

For `slot_recall`:

- each rendered item must contain exactly three answer candidates
- the answer bank must contain exactly one each of:
  - `remember_correct`
  - `distractor_irrelevant`
  - `not_remember`
- otherwise rendering fails instead of writing a partial item

For the final rendered file:

- every whole-recall item must have non-empty:
  - `choices`
  - `choice_to_answer_type`
  - `remember_correct_choice`
- every slot-recall item must satisfy the same condition
- every finalized `choice_to_answer_type` mapping must cover exactly the three required answer types

### Prompt-style checker for manual audits

```text
You are reviewing rendered MCQs for structural completeness. Your job is not to judge style or naturalness. Your job is to verify that every rendered item is complete enough to support reliable evaluation.

Please check the rendered MCQ payload for the following issues:

1. Missing or empty `choices`
2. Missing or empty `choice_to_answer_type`
3. Missing or empty `remember_correct_choice`
4. Fewer than three answer candidates
5. Missing one of the required answer types:
   - remember_correct
   - distractor_irrelevant
   - not_remember
6. Duplicated answer types
7. Any mismatch between visible choices and answer-type mappings

Return valid JSON only in the following format:
{
  "has_issues": true,
  "issues": [
    {
      "item_id": "...",
      "issue_type": "...",
      "reason": "..."
    }
  ],
  "overall_assessment": "..."
}

If the payload is structurally complete, return:
{
  "has_issues": false,
  "issues": [],
  "overall_assessment": "..."
}

Here is the rendered MCQ payload to review:

{rendered_payload}
```
