# Baseline World Design

## Goal

The baseline world should support:

- long-horizon update utility
- realistic help-seeking interactions
- later construction of memory-control worlds
- direct-cue memorization and utility evaluation

## Core Structure

Baseline generation should use two linked layers:

- `event history`
- `interaction history`
- `conversation history`

Recommended ratio:

- `Init`: 15 event-history items and 5 derived interaction items
- later periods: 9 event-history items and 3 derived interaction items

This means event history remains the broader hobby- and update-driven world state, while interaction history contains only the smaller set of derived interaction items.

## Event History

Event history records the broader longitudinal world state for a period.

Each event should be one of:

- `update.init`
  - establishes the currently valid state for the period
- `update.change`
  - records a change from an earlier state
- `update.continuation`
  - extends an already active state

Event history should preserve change-tracking fields such as:

- `[Old Event]`
- `[Old Event Date]`
- `[Reasons of Change]`

These are part of the baseline design because they support long-memory update reasoning.

## Interaction History

Interaction history contains only the interaction items derived from suitable event-history items.

Each interaction event should be `help_seek` and should include:

- `task_goal`
- `needed_context`
- `sensitive_info`

## Help-Seek Design

Help-seeking interactions should arise only from events that naturally support an immediate request.

Each `help_seek` interaction should:

- represent a concrete current task
- reveal only the context needed for that task
- include specific details naturally
- allow sensitive details when they are relevant to solving the task

Examples of task types:

- prioritization
- triage
- comparison
- itinerary repair
- document preparation
- risk review
- workflow support

## Sensitive Information Schema

Sensitive information should be represented as a dictionary:

- key: information type
- value: list of concrete values mentioned or implied

Example:

```json
{
  "email": ["alex.test@example.com"],
  "booking_identifier": ["AF39K2", "reservation 7QPL"],
  "named_contact": ["Maria from the guesthouse"]
}
```

Suggested information-type keys include:

- `email`
- `phone_number`
- `address`
- `account_or_balance`
- `booking_identifier`
- `legal_dispute_detail`
- `medical_symptom`
- `medication_or_dosing`
- `named_contact`
- `private_schedule`
- `family_or_relationship_detail`
- `document_or_record_reference`

Sensitive information should be produced by combining:

- a persona-level sensitive information pool
- event-level, context-driven realization

The persona-level pool should provide recurring synthetic anchors such as:

- synthetic email
- synthetic phone
- synthetic address
- synthetic ID
- recurring named contacts
- recurring account, booking, or project identifiers
- a small set of family or work entities when relevant

Operational rule:

- the pool is available context, not mandatory content
- event generation and conversation generation should draw from the pool only when the event naturally needs a concrete sensitive anchor
- event-specific generation may omit pool values when no sensitive detail is needed
- event-specific generation may add contextual sensitive details when they are needed, but should prefer pool-consistent values for recurring anchors

TODO:

- the current information-type set is expected to evolve when new topics are added
- when new topics are added, extend the topic design, the information-type schema, and the persona-level pool design together

## Relation and Lineage

Each event or interaction-bearing event should support explicit relation metadata.

Required fields:

- `event_id`
- `relations`
  - list of relation records
  - each relation record should include:
    - `type`: `evolves_from` or `derived_from`
    - `source_event_id`

Interpretation:

- `evolves_from`
  - used for direct longitudinal change
- `derived_from`
  - used when an interaction item is derived from a base event

The relation graph should remain traceable across derived and changed content.

## Interaction Mapping

Interaction history should be built by deriving interaction items from event-history items that naturally support:

- an immediate user task
- concrete context disclosure
- realistic consultation behavior

Every interaction item should remain linked to its source event through:

- `relations`

## Conversation History

Conversation history should be constructed by combining:

- all event-history items
- all derived interaction-history items

Each interaction-history item should be inserted near its source event so that the final conversation remains coherent and locally grounded.

## Conversation Construction

Conversation should be generated from conversation history.

All conversation-history items should be explicitly expanded into conversation.

Each conversation-history item should appear as:

- `Side_Note`
- user turn
- assistant turn

The conversation should therefore fully cover conversation history, while event history remains the broader world state and interaction history remains the derived consultation layer.

## Minimal Schema

Base event history item:

```json
{
  "event_id": "E12",
  "turn_type": "update",
  "update_subtype": "change",
  "timestamp": "MM/DD/YYYY",
  "event": "...",
  "old_event": "...",
  "old_event_date": "MM/DD/YYYY",
  "reasons_of_change": "...",
  "sensitive_info": {
    "document_or_record_reference": ["ledger snapshot"],
    "family_or_relationship_detail": ["retired bookkeeper cousin"]
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
  "timestamp": "MM/DD/YYYY",
  "event": "...",
  "task_goal": "...",
  "needed_context": ["...", "..."],
  "sensitive_info": {
    "email": ["alex.test@example.com"],
    "named_contact": ["Maria from the guesthouse"]
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

The baseline pipeline should follow this order:

1. generate event history
2. assign update subtype and change metadata
3. derive interaction history from suitable event-history items
4. for each interaction item, add:
   - `task_goal`
   - `needed_context`
   - `sensitive_info`
5. attach relation metadata
6. construct conversation history by combining event history and interaction history
7. generate conversation from conversation history
