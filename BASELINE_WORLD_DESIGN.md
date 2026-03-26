# Baseline World Design

## Goal

The baseline world should support:

- long-horizon update utility
- realistic help-seeking interactions
- later construction of memory-control worlds
- direct-cue memorization and utility evaluation

## Core Structure

Baseline generation uses three linked representations:

- `event history`
- `interaction history`
- `conversation history`

Recommended ratio:

- `Initial Stage`: 18 event-history items and 6 derived interaction items
- each later stage: 9 event-history items and 3 derived interaction items

Event history remains the broader longitudinal world state. Interaction history is a smaller, derived help-seeking layer. Conversation history is the combined, time-ordered realization skeleton used for conversation generation.

## Event History

Event history records the broader longitudinal world state for a period.

Each event is an `update` with one of two subtypes:

- `init`
  - establishes a currently valid state
- `change`
  - records a change from an earlier state

There is no standalone `continuation` subtype in the current pipeline.

Change-tracking fields should be preserved verbatim when present:

- `[Old Event]`
- `[Old Event Date]`
- `[Reasons of Change]`
- `[Old Fact] Likes`
- `[Old Fact] Dislikes`
- `[Updated Fact] Likes`
- `[Updated Fact] Dislikes`

These fields are part of the baseline design because they support long-memory update reasoning.

## Interaction History

Interaction history contains only the interaction items derived from suitable event-history items.

Each interaction item is `help_seek` and should include:

- `task_goal`
- `context_can_add`
- `sensitive_info`

The interaction timestamp is derived from its source event and should look like:

- `MM/DD/YYYY-I01`

This keeps the interaction on the same date axis while making it independently traceable.

## Help-Seek Design

Help-seeking interactions should arise only from events that naturally support an immediate request.

Each `help_seek` interaction should:

- represent a concrete current task
- extend the source event rather than replace it
- include 3--5 concrete background items in `context_can_add`
- include at least some potentially sensitive background when it is natural and useful
- explain what private details those sensitive context items would involve
- include concrete sensitive values in `sensitive_info` only for the sensitive background items in `context_can_add`
- avoid leaving abstract pool placeholders like `departure window` or `guesthouse contact` as final sensitive values

Examples of task types:

- prioritization
- triage
- comparison
- itinerary repair
- document preparation
- risk review
- workflow support

## Sensitive Information Schema

Sensitive information is represented as a dictionary:

- key: the private detail named in the explanation of a sensitive `context_can_add` item
- value: a list of concrete sensitive values for that item

The interaction-detail prompt may reuse persona-level synthetic anchors as inspiration, but those anchors are treated as abstract hints rather than final values. If Persona PII provides a relevant concrete value, the model should use it; otherwise it should synthesize a persona-consistent concrete value instead of returning the abstract placeholder itself.

Example:

```json
{
  "my lodging contact can confirm the late arrival": ["guesthouse contact"],
  "my current departure window is already fixed": ["departure window"]
}
```

The persona-level sensitive-information pool provides recurring synthetic anchors such as:

- `email`
- `phone_number`
- `address`
- `synthetic_id`
- `named_contact`
- `private_schedule`
- `document_or_record_reference`

Operational rule:

- the pool is available context, not mandatory content
- interaction derivation should reuse pool values when they are genuinely needed
- if a suitable private detail is not in the pool, a new synthetic detail may be generated as long as it remains persona-consistent

## Relation and Lineage

Each event or interaction item should support explicit relation metadata.

Required fields:

- `event_id`
- `relations`
  - each relation record includes:
    - `type`: `evolves_from` or `derived_from`
    - `source_event_id`

Interpretation:

- `evolves_from`
  - used for direct longitudinal change
- `derived_from`
  - used when an interaction item is derived from a base event

## Conversation History

Conversation history is the combined, time-ordered skeleton used by conversation generation.

It contains:

- all event-history items
- all derived interaction-history items

Each interaction item is inserted immediately after its source event.

Event items in conversation history preserve the original event record fields and add:

- `timestamp`
- `kind = "event"`
- `[Sensitive Info]`

For event items, `[Sensitive Info]` is optional and should stay empty unless the event text itself already contains concrete sensitive values.

Interaction items in conversation history use explicit bracketed keys:

- `timestamp`
- `kind = "interaction"`
- `[Prev Event]`
- `[Task Goal]`
- `[Context Can Add]`
- `[Sensitive Info]`

## Conversation Construction

Conversation is generated from conversation history.

The intended format is:

- an optional brief topic-related intro without `Side_Note`, written as one `User` line followed by one `Assistant` line
- then, for every history item:
  - one `Side_Note`
  - one user line
  - one assistant line

So each history-related block should follow:

- `Side_Note`
- `User`
- `Assistant`

The intro may exist, but it should not invent a timestamped `Side_Note`. User turns should not feel overly compressed; they should usually share enough context, motivation, or practical detail to sound like a real person who is willing to explain what is going on.

## Minimal Schema

Base event history item:

```json
{
  "event_id": "E12",
  "turn_type": "update",
  "update_subtype": "change",
  "timestamp": "MM/DD/YYYY",
  "Event": "...",
  "[Old Event]": "...",
  "[Old Event Date]": "MM/DD/YYYY",
  "[Reasons of Change]": "...",
  "sensitive_info": {
    "document_or_record_reference": ["route note"]
  },
  "relations": [
    {
      "type": "evolves_from",
      "source_event_id": "E03"
    }
  ]
}
```

Derived interaction-history item:

```json
{
  "event_id": "I05",
  "turn_type": "help_seek",
  "update_subtype": null,
  "timestamp": "MM/DD/YYYY-I01",
  "Event": "...",
  "source_event_id": "E12",
  "source_event_date": "MM/DD/YYYY",
  "[Prev Event]": "...",
  "[Task Goal]": "...",
  "[Context Can Add]": {
    "...": "..."
  },
  "[Sensitive Info]": {
    "...": ["..."]
  },
  "relations": [
    {
      "type": "derived_from",
      "source_event_id": "E12"
    }
  ]
}
```

## Pipeline

The baseline pipeline follows this order:

1. expand persona
2. generate general personal history for the four canonical stages
3. generate contextual personal history for each topic and stage
4. build a persona-level sensitive-information pool
5. normalize contextual histories into event histories
6. select source events for help-seeking interactions
7. derive interaction history with `task_goal`, `context_can_add`, and `sensitive_info`
8. assemble conversation history by combining event history and interaction history
9. generate conversation from conversation history
10. reflect, repair, and refine the conversation while preserving timestamp alignment

## Regeneration

The current pipeline also supports:

- `--regenerate_conversation_only`

This mode reuses the existing persona, histories, and conversation-history sections from an existing output file and rewrites only the final `Conversation ... Stage` sections.
