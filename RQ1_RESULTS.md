# RQ1 Overall Capability

Question:
- Can language models correctly follow different types of user-directed memory instructions (`no_store`, `deletion`, and `no_use`), while preserving utility on allowed information?

Current setup:
- sampling: `stratified`
- `max_questions = 200` for the main tasks
- `no_use direct-cue` also uses `200` sampled questions
- filled model: `GPT-5.1`
- s reserved for the rest

Main metrics:
- `constraint_score`
- `forbidden_leakage`
- `constraint_follow_increase`
- `forbidden_leakage_drop`
- `clean_utility`
- `robust_utility`
- `clean_utility_drop`
- `robust_utility_drop`

## Table 1. Main Capability Snapshot

This table is the main RQ1 comparison.
It answers:
- Across instruction types, how much allowed information is still usable?
- Across instruction types, how often does the model directly choose the forbidden-content option?

| Model | no_store clean utility | no_store forbidden leakage | deletion clean utility | deletion forbidden leakage | no_use clean utility (direct-cue) | no_use forbidden leakage (direct-cue) |
|---|---:|---:|---:|---:|---:|---:|
| GPT-5.1 | 0.9167 | 0.1316 | 0.9324 | 0.0000 | 0.5538 | 0.6857 |
| GPT-5-mini | 1.0000 | 0.7619 | 1.0000 | 0.6512 | 0.5538 | 0.9571 |
| GPT-4o |  |  |  |  |  |  |
| GPT-4o-mini |  |  |  |  |  |  |
| gemini-2.5-pro |  |  |  |  |  |  |
| gemini-2.5-flash |  |  |  |  |  |  |
| claude-haiku-4.5 |  |  |  |  |  |  |
| claude-3.7-sonnet |  |  |  |  |  |  |
| deepseek-chat-v3.1 |  |  |  |  |  |  |
| grok-3-fast |  |  |  |  |  |  |
| grok-3-mini |  |  |  |  |  |  |

Analysis:

- The main three instruction families already separate clearly in the current `GPT-5.1` run.
- `deletion` currently has the lowest `forbidden_leakage` (`0.0000`) while keeping high `clean_utility` (`0.9324`).
- `no_store` also looks strong, combining high `clean_utility` (`0.9167`) with low `forbidden_leakage` (`0.1316`).
- Under the original `direct-cue` formulation, `no_use` remains qualitatively different from the other two:
  - `clean_utility` is much lower for `GPT-5.1` (`0.5538`)
  - `forbidden_leakage` is much higher (`0.6857`)
- At the current sample level, the cleanest headline from Table 1 is:
  - `deletion` and `no_store` still look substantially more effective than `no_use` on their main instruction-following metrics
  - `deletion` still looks best on direct forbidden-memory leakage
- `GPT-5-mini` already reveals a different profile from `GPT-5.1`.
  - It preserves allowed-memory utility extremely well in `no_store` and `deletion`.
  - Under `direct-cue`, its `no_use` clean utility is still much lower than `no_store/deletion`.

## Table 2. Trade-off View

This table is for the RQ1 trade-off story.
It answers:
- How much instruction-following improvement is achieved?
- What utility cost is paid for that suppression?

| Model | no_store constraint follow increase | no_store clean drop | no_store robust drop | deletion constraint follow increase | deletion clean drop | deletion robust drop | no_use constraint follow increase (direct-cue) | no_use clean drop (direct-cue) | no_use robust drop (direct-cue) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-5.1 | 0.7935 | 0.0372 | 0.0823 | 0.8372 | -0.0135 | 0.1500 | 0.2714 |  | -0.0154 |
| GPT-5-mini | 0.2381 | -0.0147 | 0.0000 | 0.3372 | 0.0000 | 0.0000 | 0.0286 |  |  |
| GPT-4o |  |  |  |  |  |  |  |  |  |
| GPT-4o-mini |  |  |  |  |  |  |  |  |  |
| gemini-2.5-pro |  |  |  |  |  |  |  |  |  |
| gemini-2.5-flash |  |  |  |  |  |  |  |  |  |
| claude-haiku-4.5 |  |  |  |  |  |  |  |  |  |
| claude-3.7-sonnet |  |  |  |  |  |  |  |  |  |
| deepseek-chat-v3.1 |  |  |  |  |  |  |  |  |  |
| grok-3-fast |  |  |  |  |  |  |  |  |  |
| grok-3-mini |  |  |  |  |  |  |  |  |  |

Analysis:

- Table 2 shows that stronger instruction following does not automatically imply a strong clean-utility penalty.
- `deletion` currently has the strongest `constraint_follow_increase` (`0.8372`), but its `clean_utility_drop` is slightly negative rather than positive.
- The more visible cost appears in `robust_utility_drop`:
  - `no_store robust_utility_drop = 0.0823`
  - `deletion robust_utility_drop = 0.1500`
  - `no_use direct-cue robust_utility_drop = -0.0154`
- This suggests the main trade-off is not “suppression vs all utility.”
  - It is more specifically “suppression vs robustness under misleading refusal-style distractors.”
- Under the original `direct-cue` setup, `no_use` shows weaker improvement than `no_store/deletion`, but still a visibly positive signal for `GPT-5.1` (`0.2714`).
- This makes the direct-cue branch useful as a strong-cue suppression probe, even though it is not the best test of downstream memory use.
- `GPT-5-mini` reinforces this point.
  - It pays little or no utility cost in `no_store/deletion`.
  - But even under direct-cue, its `no_use` improvement remains much smaller (`0.0286`) than `GPT-5.1`.

## Table 3. no_use Evaluation Branches

This table separates the two `no_use` evaluation branches.
It answers:
- What does `no_use` look like under a direct-cue suppression probe?
- What does `no_use` look like under the newer reasoning / blocked-use probe?

### Table 3a. no_use scope

| Model | branch | clean utility | forbidden leakage | constraint follow increase | clean drop | robust drop |
|---|---|---:|---:|---:|---:|---:|
| GPT-5.1 | direct-cue | 0.5538 | 0.6857 | 0.2714 | 0.0308 | 0.0308 |
| GPT-5.1 | reasoning | 0.5385 | 0.1286 | 0.0000 | -0.0308 | 0.0154 |
| GPT-5-mini | direct-cue | 0.5538 | 0.9571 | 0.0286 | 0.1026 | 0.0000 |
| GPT-5-mini | reasoning | 0.6154 | 0.2286 | 0.0286 | 0.0000 | 0.0308 |

### Table 3b. no_use temporal scope

| Model | branch | constraint score | forbidden leakage | clean utility | robust utility | clean drop | robust drop | recovery rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-5.1 | direct-cue | 0.2174 | 0.7609 | 0.7065 | 0.6129 | 0.0000 | -0.0323 | 1.0000 |
| GPT-5.1 | reasoning | 0.8043 | 0.1522 | 0.6500 | 0.4426 | -0.0333 | 0.0820 | 0.2121 |
| GPT-5-mini | direct-cue | 0.0000 | 1.0000 | 0.7065 | 0.5806 | 0.0652 | 0.0000 | 1.0000 |
| GPT-5-mini | reasoning | 0.7174 | 0.2391 | 0.7000 | 0.5082 | -0.0500 | 0.0492 | 0.3939 |

Analysis:

- `no_use` should now be read as two complementary probes.
- `direct-cue` measures strong-cue suppression under explicit recall pressure.
  - `reasoning` measures blocked-use behavior on more natural downstream questions.
- The two branches do not tell the same story.
- Under `direct-cue`, `GPT-5.1` still shows substantial forbidden leakage on the full 200-question branch.
  - Under the filtered `reasoning` branch, forbidden leakage drops sharply, but `scope` instruction-following improvement also collapses to near zero.
- That divergence is useful rather than contradictory:
  - the `direct-cue` branch is a stronger probe of raw forbidden recall under cueing
  - the `reasoning` branch is a better probe of downstream memory gating once the task allows a safe alternative answer
- The refreshed `reasoning` results now show a cleaner split:
  - `scope reasoning` mostly reduces forbidden-answer selection, but does not clearly separate baseline from no-use for `GPT-5.1`
  - `temporal reasoning` shows a more visible positive `constraint_follow_increase` for both models
- The updated results also make the model gap clearer:
  - `GPT-5.1` improves substantially under `direct-cue` (`constraint_follow_increase = 0.2714`)
  - `GPT-5-mini` improves only weakly on the same sampled questions (`0.0286`)
- `temporal_scope` remains a mechanism probe rather than a fourth main instruction condition.
- For now, the safest interpretation is:
  - keep the main RQ1 tables tied to `direct-cue`
  - use the reasoning branch as a separate `no_use` analysis layer

## Takeaways

- `deletion` currently looks strongest on direct instruction following.
  - It has the highest `constraint_follow_increase` and the lowest `forbidden_leakage`.
- `no_store` currently looks like the cleanest balance between instruction following and utility preservation.
- `no_use` should be interpreted through two branches rather than one.
  - `direct-cue` says more about strong-cue suppression.
  - `reasoning` says more about downstream use-time reasoning under a memory restriction.
- Under the new reasoning branch, `GPT-5.1` shows almost no `scope` improvement, while `temporal_scope` still shows a positive improvement.
- The filtered reasoning branch therefore looks less like “massive leakage” and more like “weak baseline-vs-no_use separation in scope, moderate separation in temporal scope.”
- The updated direct-cue branch now gives a cleaner comparison point:
  - `GPT-5.1` improves meaningfully on `no_use`
  - `GPT-5-mini` still leaks heavily and improves much less
- `GPT-5.1` and `GPT-5-mini` already show a meaningful divergence.
  - `GPT-5-mini` preserves utility extremely well.
  - But it follows `no_store` / `deletion` instructions much more weakly.
- The main trade-off appears in `robust_utility_drop`, not in `clean_utility_drop`.
- These results are still provisional because:
  - only two models are filled
  - evaluation is sampled rather than full
  - `retention` currently uses the stable 22-world subset rather than the latest full baseline pool

## Sources

- `no_store`: [retention_eval_summary.json](/mnt/yao_data/proj_2026_agent/PersonaMem-main/data/retention/retention_eval_summary.json)
- `deletion`: [deletion_eval_summary.json](/mnt/yao_data/proj_2026_agent/PersonaMem-main/data/deletion/deletion_eval_summary.json)
- `no_use direct_cue` (`200-question archived result`): [9de25c4a-bfb1-484d-a64d-13a668b8a064_Results_for_RQ1_Overall_Capability.pdf](/mnt/yao_data/proj_2026_agent/PersonaMem-main/9de25c4a-bfb1-484d-a64d-13a668b8a064_Results_for_RQ1_Overall_Capability.pdf)
- `no_use scope` (`reasoning`): [no_use_eval_summary_scope.json](/mnt/yao_data/proj_2026_agent/PersonaMem-main/data/no_use/no_use_eval_summary_scope.json)
- `no_use temporal_scope` (`reasoning`): [no_use_eval_summary_temporal_scope.json](/mnt/yao_data/proj_2026_agent/PersonaMem-main/data/no_use/no_use_eval_summary_temporal_scope.json)
- `no_store` (`GPT-5-mini`): [retention_eval_summary_gpt5mini.json](/mnt/yao_data/proj_2026_agent/PersonaMem-main/data/retention/retention_eval_summary_gpt5mini.json)
- `deletion` (`GPT-5-mini`): [deletion_eval_summary_gpt5mini.json](/mnt/yao_data/proj_2026_agent/PersonaMem-main/data/deletion/deletion_eval_summary_gpt5mini.json)
- `no_use scope` (`reasoning`, `GPT-5-mini`): [no_use_eval_summary_scope_gpt5mini.json](/mnt/yao_data/proj_2026_agent/PersonaMem-main/data/no_use/no_use_eval_summary_scope_gpt5mini.json)
- `no_use temporal_scope` (`reasoning`, `GPT-5-mini`): [no_use_eval_summary_temporal_scope_gpt5mini.json](/mnt/yao_data/proj_2026_agent/PersonaMem-main/data/no_use/no_use_eval_summary_temporal_scope_gpt5mini.json)
