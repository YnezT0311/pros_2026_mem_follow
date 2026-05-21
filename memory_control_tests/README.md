# Memory Control Evaluation

This package contains evaluation code for MemoryCtrl.

Data-generation principles live in `../gen_data/DATA_GENERATION_GUIDE.md`. This README
is intentionally limited to evaluation concepts and entry points.

## Evaluation Goal

The evaluation asks whether a system can answer recall questions correctly
under different memory-control worlds.

The supported worlds are:

- `baseline`: ordinary conversation memory
- `no_store`: information may be used for the current turn but should not be retained
- `forget`: previously shared information should be removed or no longer recalled
- `no_use`: retained information should not be used for the current answer

The supported methods are:

- `plain`
- `mem0`
- `langmem`
- `amem`
- `memoryos`
- `memtree`

## Main Entry Points

Use `memory_control_tests.evaluation.mem_evals` to run raw evaluations.

Use `memory_control_tests.evaluation.scores` to parse answer labels and compute
summary metrics from saved raw outputs.

Use `memory_control_tests.analysis.build_report` and related files under
`memory_control_tests/analysis/` for aggregate reports and research-question
analysis.

## Evaluation Flow

The recommended evaluation flow is two-stage:

1. Run the evaluator and save raw model outputs.
2. Run scoring over the saved outputs.

This keeps model execution separate from answer parsing and lets scoring logic
be improved without rerunning model calls.

## Result Shape

Evaluation outputs usually contain:

- source conversation and rendered QA metadata
- world and method configuration
- raw model answers
- parsed answer labels
- per-family recall results, usually `whole_recall` and `slot_recall`
- summary metrics after scoring

The `application` QA family is still treated as deferred unless a generated
dataset explicitly provides those items.

## Notes For Agents

Do not put source-data generation rules in this README. Keep this file focused
on evaluation behavior, method entry points, and result interpretation.

When changing evaluation code, avoid regenerating source conversations as a side
effect. Evaluation should consume already generated conversation and QA files.
