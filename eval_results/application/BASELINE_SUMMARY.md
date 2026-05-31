# Application MCQ — baseline results (39 items, per-stage)

Run date: 2026-05-25. Data: canonical regeneration via `build_worlds.py`
+ `strip_contamination.py` (see `gen_data/build_benchmark/application/REGENERATION.md`).
Deployed to `data/application/mcq/by_world/`. 39 items = travelPlanning 15,
financialConsultation 10, medicalConsultation 14.

Each item is evaluated on its own per-stage context in two conditions:
- **seen_baseline** — target turn present → correct = use-memory option
- **never_seen_baseline** — target turn removed → correct = without-memory option

An item **differentiates** when the model is correct in BOTH conditions.

## Headline (after the rewrite pass)

| model | seen | never | differentiating |
|---|---|---|---|
| gpt-5.4-mini | 36/39 | 25/39 | **23/39** |
| opus-4.7 | 39/39 | 37/39 | **37/39** |
| gpt-5.5 | 39/39 | 38/39 | **38/39** |

Seven items that originally failed on multiple models were rewritten (see
"Rewrite pass" below). After the rewrite, **every item differentiates on at
least 2 of the 3 models** — there are no remaining 2-model or 3-model
failures. The residual is entirely single-model behavioral quirks.

Initial state, for reference: mini 21 / opus 29 / gpt-5.5 32.

## Rewrite pass (7 items)

The original 🔴 (3-model-fail) and 🟠 (2-model-fail) items shared one flaw:
the use-memory option read as universally good advice, so a no-preference
model picked it in the never condition too. The fix (validated per item with
a blind Claude subagent in both conditions, then the real API on all three
models) follows one pattern:

> **Make the without-memory option the default that a holder of the hidden
> preference would actively reject.** The use-memory option then only makes
> sense once the preference is known, and the never condition lands on the
> default.

| item | hidden preference | fix | result |
|---|---|---|---|
| financial_23 | old-school paper budgeting | without-mem = auto-import app (the thing a manual tracker rejects) | 3/3 ✓ |
| medical_05 | gentle/visual reminders | without-mem = several loud alarms | 3/3 ✓ |
| financial_11 | homemade/personalized gifts | without-mem = store-bought shopping list; question no longer leaks "personal" | 3/3 ✓ |
| medical_12 | free after ~10am training | without-mem = earliest slot (8:30) | opus+5.5 ✓, mini ✗ |
| medical_13 | written > phone | billing-dispute framing where phone is the default | opus+5.5 ✓, mini ✗ |
| travel_12 | family nut allergy | "make the popular IG breakfast bowl" — default recipe includes nuts; **also neutralized a non-target leak** (see audit below) | 3/3 ✓ |
| financial_08 | transit/cycling, cut €180 cost | apartment choice: 10 km hilly 40-50 min bike vs 20 min subway — bike is no longer the frugal default | opus+5.5 ✓, mini ✗ |

## Residual: items where one model still fails (21)

All are single-model failures; the item differentiates on the other two.

- **gpt-5.4-mini only (16)**: financial_01, financial_08*, medical_02, _03,
  _06, _08, _09, _12*, _13*, travel_03, _05, _06, _11, _12*, _16, _21
  — almost all are dietary/medical-safety or frugality items where mini's
  safety/frugal prior picks the "cautious" option in the never condition
  regardless of the (absent) preference. (* = rewritten items mini still
  can't crack.)
- **opus-4.7 only (3)**: medical_01, medical_14, financial_12.
- **gpt-5.5 only (2)**: travel_09, financial_14.

These are model behavior, not MCQ flaws: each was confirmed sound by a blind
strict-reasoner subagent, and differentiates on the other two models.

## Contamination audit (via mini rationales)

Reading gpt-5.4-mini's `model_response` on its never-condition failures is an
effective contamination detector: if mini cites *specific context facts* that
restate the hidden preference, the never world leaks; if it merely echoes the
chosen option's own "Since you mentioned X" claim, that's claim-acceptance
behavior.

- **travel_12 had a soft non-target leak.** A non-target reunion turn used
  generic accommodate-everyone wording — *"unfortunate experiences with past
  allergies … consider diverse meal options this time so everyone feels
  included"* — which mini (quoting it verbatim) generalized into "avoid nuts".
  It was not an exact nut-allergy restatement (the only nut-allergy mention is
  the target turn), so rather than strip whole turns, that one clause is
  **neutralized application-side** (via `strip_contamination.py`'s REWRITE op,
  non-seen worlds only) to *"…I just want to make sure everyone eats well"*;
  the shared `conversation_package.json` is left untouched. The remaining
  food turns name only different restrictions (a dairy-free guest, the user's
  own allergy meds) which mini correctly does *not* generalize to nuts. After
  the edit, travel_12 differentiates on all three models; mini's never
  rationale reads *"the conversation did NOT mention a nut allergy … the
  earlier dietary constraints were 'dairy-free' and 'some vegetarian family
  members', so the generic recipe is the most appropriate fit"* → (a).
- **All other mini never-fails are clean.** Their rationales quote the option's
  own claim text (e.g. medical_08 "computer-based record system…", travel_16
  "gluten-free constraint") — verified against each never context, no
  independent restatement of the target preference. These are genuine model
  behavior (claim-acceptance + safety/efficiency priors), not flaws.

### The hardest preference axis

`financial_08` (prefers transit/cycling + cutting cost) is the most stubborn:
frugality is a near-universal default, so when the memory-aligned option also
saves money, weak models pick it without needing the memory. It was only
crackable for opus/gpt-5.5 by making the bike commute genuinely effortful
(10 km, hills, 40-50 min vs a 20-min subway), so biking stops being the
no-preference default. gpt-5.4-mini remains incoherent on it either way.

## gpt-5.5 policy-block note (initial full run)

The initial full baseline hit intermittent account-level
`Policy Violation … blocked for a previous policy violation` (HTTP 502)
errors on the gpt-5.5 route — transient route-level moderation throttling,
not per-item content failures and not a permanent ban. Resolved by re-running
and merging successful predictions until 0 blocked. The per-item rewrite
re-tests (single-item) were unaffected.

## Artifacts

```
eval_results/application/<world>/<model_tag>/<world>.plain_api_<model_tag>.json
```
Raw `model_response` is persisted per item. The 7 rewritten items' results
were merged into these files from the per-item re-tests (no full re-run).
Scorer: `scripts/evaluation/api_models/score_application_baselines.py`.
