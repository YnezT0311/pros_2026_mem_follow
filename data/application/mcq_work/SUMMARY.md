# Application MCQ Source Summary

This directory contains the editable source items for the application benchmark.
Each topic has one `persona0_sample0/application_items.json` file.

## Coverage

- 36 deployed application MCQs
- 3 topics: `travelPlanning`, `financialConsultation`, `medicalConsultation`
- 6 evaluation worlds per item: `seen_baseline`, `never_seen_baseline`,
  `no_store`, `forget`, `no_use_active`, and `no_use_release`
- 216 rendered world records in `data/application/mcq/application_mcq.json`

## Source Policy

- Source conversations come from
  `data/generated/<topic>/persona0_sample0/conversation_package.json`.
- Each MCQ is rendered from one original `Conversation Stage XX`, not from a
  synthetic full-history context.
- `target_user_turn` must exactly match a user turn in the source stage.
- Each item carries a `forget_reference` so the shared recall-style forget
  transform can point to the relevant application target turn naturally.

## Deployed Data

The deployed API/web data lives in:

```text
data/application/mcq/application_mcq.json
data/application/mcq/by_world/<world>.json
```

Regenerate those files with:

```text
gen_data/build_benchmark/application/build_worlds.py
gen_data/build_benchmark/application/strip_contamination.py
```

The application control worlds reuse `memory_control_tests.transforms` while
targeting application MCQ turns.
