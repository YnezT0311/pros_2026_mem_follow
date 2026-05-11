# Interim Instruction-Control Report (Fast Cut)

Generated: 2026-04-06 11:05

This version is formatted for presentation. `plain`, `mem0`, `LangMem`, and `A-Mem` are now complete for the intended scope. `Zep` is still incomplete and is not included in the summary tables below.

## Coverage Status

- `plain`: complete
- `mem0`: complete
- `LangMem`: complete
- `A-Mem`: complete
- `Zep`: incomplete, excluded from summary tables

## Metric Convention

- `TP`: allowed to be recalled, and recalled
- `FP`: not allowed to be recalled, but recalled
- `TN`: not allowed to be recalled, and not recalled
- `FN`: allowed to be recalled, but not recalled
- `TPR = TP / (TP + FN)`: success rate on allowed recall
- `FPR = FP / (FP + TN)`: violation rate on forbidden recall

In the tables below, probe turns are best read through `TPR`, while forbidden key turns in controlled worlds are best read through `FPR`.

## Research Questions (RQs)

1. **What is the average performance of each system under `baseline`, `no_store`, `forget`, and `no_use`?**
   Notes: `baseline` is still strongest overall. Among the completed memory systems, `mem0` is the weakest on raw utility, `LangMem` is the most conservative, and `A-Mem` sits in the middle with better raw utility than `mem0` but weaker key suppression than `LangMem` on several settings.
2. **Which instruction type is easier or harder for each model or memory system to follow?**
   Notes: `forget` is still the strongest instruction overall. `gpt-4o` is more instruction-sensitive than `gpt-5.4-mini`; `LangMem` remains the most conservative, while `A-Mem` shows moderate suppression with less probe damage than `mem0`.
3. **How much does each instruction type hurt `probe` utility?**
   Notes: `gpt-4o + forget` hurts probe utility the most among the plain models. `LangMem` and `A-Mem` generally preserve probe utility better than `mem0`, with `LangMem` still the strongest on probe preservation.
4. **For `no_store`, how does the test stage (`Early`, `Intermediate`, `Late`) change the behavior?**
   Notes: The stage effect exists but is not dramatic. It is clearer for `gpt-4o` than for `gpt-5.4-mini`.
5. **For `forget`, what do we see when we test the forgotten key at different stages?**
   - **Does recall drop immediately after the matching `forget` instruction?** Evaluated with `key1 forget@E ask@E`, `key2 forget@I ask@I`, and `key3 forget@L ask@L`.
     Notes: Yes. The target key is suppressed immediately after the matching `forget` instruction, especially for `gpt-4o`; `gpt-5.4-mini` shows the same effect more weakly.
   - **If we forget early, does the effect still remain when we ask later?** Evaluated by tracking `key1 forget@E` across `ask@E`, `ask@I`, and `ask@L`.
     Notes: Yes. The suppression can remain when the forgotten key is tested later, although it is more stable on `gpt-4o` than on `gpt-5.4-mini`.
   - **Does forgetting earlier versus later make a clear difference?** Evaluated by comparing `key1@E ask@E`, `key2@I ask@I`, and `key3@L ask@L`.
     Notes: Not clearly. There are timing differences, but they are not sharp or clean enough to be the main takeaway in the current results.
6. **For `no_use`, what do we see when we change the restriction, release, and test stages?**
   - **Does `no_use` suppress memory use immediately once the restriction is given?** Evaluated with `no_use@E test@E`, `no_use@I test@I`, and `no_use@L test@L`.
     Notes: Yes. `no_use` can suppress use of earlier memory immediately, but the effect is generally weaker than `forget`.
   - **If `no_use` is given early, does the suppression still remain when we test later?** Evaluated with `no_use@E test@I` and `no_use@E test@L`.
     Notes: Yes. Some suppression remains at later test stages, although the pattern is weaker and less consistent than `forget`.
   - **After a release instruction, does memory use come back?** Evaluated with `no_use@E release@E test@E`, `no_use@E release@E test@I`, and `no_use@E release@E test@L`.
     Notes: There are early signs of recovery after release, but this result is still preliminary and should be treated cautiously.
7. **In `slot_recall`, which kinds of sensitive information are easier to remember or forget?**
   Notes: This section is intentionally deferred to the full report. It depends on LLM post-processing over the plain slot-recall results.

## Average Performance by Instruction Type

Columns are shown as:
- `key TPR/FPR`: for forbidden key facts in controlled worlds, the main safety number is `FPR`; for `baseline`, these rows mostly behave like utility rows.
- `probe TPR/FPR`: for allowed probe facts, the main utility number is `TPR`.

### **baseline**

| System | whole key TPR/FPR | slot key TPR/FPR | whole probe TPR/FPR | slot probe TPR/FPR |
|---|---:|---:|---:|---:|
| **gpt-5.4-mini** | 1.000 / 0.000 | 0.937 / 0.000 | 1.000 / 0.000 | 0.944 / 0.040 |
| **gpt-4o** | 0.861 / 0.056 | 0.770 / 0.086 | 0.889 / 0.111 | 0.904 / 0.030 |
| **gpt-5.4-mini + mem0** | 0.611 / 0.389 | 0.309 / 0.580 | 0.833 / 0.167 | 0.308 / 0.679 |
| **gpt-4o + mem0** | 0.167 / 0.778 | 0.198 / 0.741 | 0.222 / 0.778 | 0.231 / 0.705 |
| **gpt-5.4-mini + A-Mem** | 0.889 / 0.083 | 0.685 / 0.216 | 0.944 / 0.056 | 0.717 / 0.232 |
| **gpt-4o + A-Mem** | 0.194 / 0.389 | 0.599 / 0.266 | 0.333 / 0.611 | 0.616 / 0.263 |
| **gpt-5.4-mini + LangMem** | 0.778 / 0.222 | 0.864 / 0.074 | 0.833 / 0.111 | 0.808 / 0.192 |
| **gpt-4o + LangMem** | 0.267 / 0.533 | 0.774 / 0.129 | 0.533 / 0.333 | 0.803 / 0.148 |

### no_store

| System | whole key TPR/FPR | slot key TPR/FPR | whole probe TPR/FPR | slot probe TPR/FPR |
|---|---:|---:|---:|---:|
| gpt-5.4-mini | 0.944 / 0.056 | 0.937 / 0.018 | 1.000 / 0.000 | 0.960 / 0.025 |
| gpt-4o | 0.722 / 0.250 | 0.743 / 0.144 | 0.750 / 0.250 | 0.808 / 0.101 |
| gpt-5.4-mini + mem0 | 0.500 / 0.500 | 0.173 / 0.728 | 0.778 / 0.222 | 0.256 / 0.692 |
| gpt-4o + mem0 | 0.167 / 0.778 | 0.136 / 0.840 | 0.278 / 0.722 | 0.128 / 0.846 |
| gpt-5.4-mini + A-Mem | 0.917 / 0.000 | 0.662 / 0.243 | 0.944 / 0.056 | 0.747 / 0.192 |
| gpt-4o + A-Mem | 0.111 / 0.444 | 0.559 / 0.306 | 0.306 / 0.611 | 0.601 / 0.263 |
| gpt-5.4-mini + LangMem | 0.722 / 0.278 | 0.877 / 0.037 | 0.833 / 0.167 | 0.833 / 0.167 |
| gpt-4o + LangMem | 0.200 / 0.800 | 0.790 / 0.129 | 0.533 / 0.333 | 0.836 / 0.131 |

### forget

| System | whole key TPR/FPR | slot key TPR/FPR | whole probe TPR/FPR | slot probe TPR/FPR |
|---|---:|---:|---:|---:|
| gpt-5.4-mini | 0.778 / 0.222 | 0.909 / 0.050 | 0.922 / 0.078 | 0.943 / 0.040 |
| gpt-4o | 0.333 / 0.633 | 0.572 / 0.357 | 0.456 / 0.533 | 0.670 / 0.292 |
| gpt-5.4-mini + mem0 | 0.444 / 0.556 | 0.148 / 0.790 | 0.556 / 0.444 | 0.141 / 0.808 |
| gpt-4o + mem0 | 0.222 / 0.778 | 0.099 / 0.877 | 0.222 / 0.778 | 0.064 / 0.923 |
| gpt-5.4-mini + A-Mem | 0.744 / 0.122 | 0.638 / 0.274 | 0.867 / 0.100 | 0.703 / 0.227 |
| gpt-4o + A-Mem | 0.211 / 0.456 | 0.557 / 0.326 | 0.356 / 0.489 | 0.627 / 0.256 |
| gpt-5.4-mini + LangMem | 0.722 / 0.278 | 0.877 / 0.049 | 0.833 / 0.167 | 0.821 / 0.179 |
| gpt-4o + LangMem | 0.133 / 0.733 | 0.742 / 0.161 | 0.600 / 0.333 | 0.803 / 0.115 |

### no_use

| System | whole key TPR/FPR | slot key TPR/FPR | whole probe TPR/FPR | slot probe TPR/FPR |
|---|---:|---:|---:|---:|
| gpt-5.4-mini | 0.885 / 0.115 | 0.875 / 0.086 | 0.896 / 0.104 | 0.877 / 0.102 |
| gpt-4o | 0.823 / 0.125 | 0.694 / 0.216 | 0.729 / 0.271 | 0.811 / 0.142 |
| gpt-5.4-mini + mem0 | 0.513 / 0.487 | 0.145 / 0.742 | 0.538 / 0.462 | 0.217 / 0.745 |
| gpt-4o + mem0 | 0.156 / 0.800 | 0.152 / 0.797 | 0.178 / 0.800 | 0.152 / 0.806 |
| gpt-5.4-mini + A-Mem | 0.802 / 0.094 | 0.679 / 0.230 | 0.979 / 0.021 | 0.752 / 0.195 |
| gpt-4o + A-Mem | 0.198 / 0.344 | 0.588 / 0.280 | 0.365 / 0.562 | 0.614 / 0.258 |
| gpt-5.4-mini + LangMem | 0.733 / 0.267 | 0.912 / 0.039 | 0.933 / 0.067 | 0.821 / 0.170 |
| gpt-4o + LangMem | 0.542 / 0.458 | 0.875 / 0.031 | 0.833 / 0.167 | 0.750 / 0.181 |

## Instruction Difficulty and Probe Cost

Higher `key suppression` means the instruction pushes forbidden key facts toward `not_remember` more strongly. Higher `probe cost` means more damage to allowed probe recall relative to baseline.

| System | no_store key suppression | forget key suppression | no_use key suppression | no_store probe cost | forget probe cost | no_use probe cost |
|---|---:|---:|---:|---:|---:|---:|
| gpt-5.4-mini | 0.037 | 0.136 | 0.100 | -0.008 | 0.040 | 0.086 |
| gpt-4o | 0.197 | 0.495 | 0.171 | 0.117 | 0.333 | 0.127 |
| gpt-5.4-mini + mem0 | 0.614 | 0.673 | 0.615 | 0.053 | 0.222 | 0.193 |
| gpt-4o + mem0 | 0.809 | 0.827 | 0.798 | 0.024 | 0.083 | 0.062 |
| gpt-5.4-mini + A-Mem | 0.122 | 0.198 | 0.162 | -0.015 | 0.046 | -0.035 |
| gpt-4o + A-Mem | 0.375 | 0.391 | 0.312 | 0.021 | -0.016 | -0.014 |
| gpt-5.4-mini + LangMem | 0.157 | 0.164 | 0.153 | -0.013 | -0.006 | -0.057 |
| gpt-4o + LangMem | 0.465 | 0.447 | 0.245 | -0.016 | -0.033 | -0.123 |

## `no_store` Test-Stage Effect

| System | Stage | whole key TPR/FPR | slot key TPR/FPR | whole probe TPR/FPR | slot probe TPR/FPR |
|---|---|---:|---:|---:|---:|
| gpt-5.4-mini | Early | 0.917 / 0.083 | 0.946 / 0.014 | 1.000 / 0.000 | 0.955 / 0.030 |
| gpt-5.4-mini | Intermediate | 1.000 / 0.000 | 0.946 / 0.014 | 1.000 / 0.000 | 0.955 / 0.030 |
| gpt-5.4-mini | Late | 0.917 / 0.083 | 0.919 / 0.027 | 1.000 / 0.000 | 0.970 / 0.015 |
| gpt-4o | Early | 0.667 / 0.250 | 0.770 / 0.135 | 0.750 / 0.250 | 0.833 / 0.076 |
| gpt-4o | Intermediate | 0.667 / 0.333 | 0.716 / 0.162 | 0.750 / 0.250 | 0.818 / 0.106 |
| gpt-4o | Late | 0.833 / 0.167 | 0.743 / 0.135 | 0.750 / 0.250 | 0.773 / 0.121 |
| gpt-5.4-mini + mem0 | Early | 0.500 / 0.500 | 0.148 / 0.741 | 0.833 / 0.167 | 0.346 / 0.615 |
| gpt-5.4-mini + mem0 | Intermediate | 0.667 / 0.333 | 0.185 / 0.667 | 0.500 / 0.500 | 0.115 / 0.846 |
| gpt-5.4-mini + mem0 | Late | 0.333 / 0.667 | 0.185 / 0.778 | 1.000 / 0.000 | 0.308 / 0.615 |
| gpt-4o + mem0 | Early | 0.167 / 0.833 | 0.148 / 0.852 | 0.333 / 0.667 | 0.115 / 0.885 |
| gpt-4o + mem0 | Intermediate | 0.167 / 0.833 | 0.037 / 0.889 | 0.167 / 0.833 | 0.077 / 0.885 |
| gpt-4o + mem0 | Late | 0.167 / 0.667 | 0.222 / 0.778 | 0.333 / 0.667 | 0.192 / 0.769 |
| gpt-5.4-mini + LangMem | Early | 0.667 / 0.333 | 0.815 / 0.037 | 0.833 / 0.167 | 0.846 / 0.154 |
| gpt-5.4-mini + LangMem | Intermediate | 0.667 / 0.333 | 0.889 / 0.037 | 0.833 / 0.167 | 0.846 / 0.154 |
| gpt-5.4-mini + LangMem | Late | 0.833 / 0.167 | 0.926 / 0.037 | 0.833 / 0.167 | 0.808 / 0.192 |
| gpt-4o + LangMem | Early | 0.000 / 1.000 | 0.741 / 0.148 | 0.333 / 0.500 | 0.846 / 0.154 |
| gpt-4o + LangMem | Intermediate | 0.333 / 0.667 | 0.815 / 0.148 | 0.500 / 0.333 | 0.846 / 0.115 |
| gpt-4o + LangMem | Late | 0.333 / 0.667 | 0.875 / 0.000 | 1.000 / 0.000 | 0.778 / 0.111 |

## `forget` Questions

Immediate effect: averaged over `key1 forget@E ask@E`, `key2 forget@I ask@I`, and `key3 forget@L ask@L`.
Persistence: tracks `key1 forget@E` as the question moves from `E -> I -> L`.
Timing sensitivity: compares `key1@E ask@E`, `key2@I ask@I`, and `key3@L ask@L`.

### gpt-5.4-mini

Immediate effect:
```json
{
  "whole_recall_key_turns": {
    "num_questions": 30,
    "remember_correct_rate": 0.6666666666666666,
    "not_remember_rate": 0.3333333333333333,
    "distractor_irrelevant_rate": 0.0
  },
  "slot_recall_key_turns": {
    "num_questions": 186,
    "remember_correct_rate": 0.9247311827956989,
    "not_remember_rate": 0.04838709677419355,
    "distractor_irrelevant_rate": 0.026881720430107527
  }
}
```
Persistence:
```json
{
  "Conversation Early Stage": {
    "whole_recall_key_turns": {
      "num_questions": 10,
      "remember_correct_rate": 0.9,
      "not_remember_rate": 0.1,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 51,
      "remember_correct_rate": 0.8823529411764706,
      "not_remember_rate": 0.0784313725490196,
      "distractor_irrelevant_rate": 0.0392156862745098
    }
  },
  "Conversation Intermediate Stage": {
    "whole_recall_key_turns": {
      "num_questions": 10,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 51,
      "remember_correct_rate": 0.7843137254901961,
      "not_remember_rate": 0.11764705882352941,
      "distractor_irrelevant_rate": 0.09803921568627451
    }
  },
  "Conversation Late Stage": {
    "whole_recall_key_turns": {
      "num_questions": 10,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 51,
      "remember_correct_rate": 0.9019607843137255,
      "not_remember_rate": 0.0392156862745098,
      "distractor_irrelevant_rate": 0.058823529411764705
    }
  }
}
```
Timing sensitivity:
```json
{
  "Conversation Early Stage": {
    "whole_recall_key_turns": {
      "num_questions": 10,
      "remember_correct_rate": 0.9,
      "not_remember_rate": 0.1,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 51,
      "remember_correct_rate": 0.8823529411764706,
      "not_remember_rate": 0.0784313725490196,
      "distractor_irrelevant_rate": 0.0392156862745098
    }
  },
  "Conversation Intermediate Stage": {
    "whole_recall_key_turns": {
      "num_questions": 10,
      "remember_correct_rate": 0.5,
      "not_remember_rate": 0.5,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 60,
      "remember_correct_rate": 0.9166666666666666,
      "not_remember_rate": 0.05,
      "distractor_irrelevant_rate": 0.03333333333333333
    }
  },
  "Conversation Late Stage": {
    "whole_recall_key_turns": {
      "num_questions": 10,
      "remember_correct_rate": 0.6,
      "not_remember_rate": 0.4,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 75,
      "remember_correct_rate": 0.96,
      "not_remember_rate": 0.02666666666666667,
      "distractor_irrelevant_rate": 0.013333333333333334
    }
  }
}
```

### gpt-4o

Immediate effect:
```json
{
  "whole_recall_key_turns": {
    "num_questions": 30,
    "remember_correct_rate": 0.16666666666666666,
    "not_remember_rate": 0.8333333333333334,
    "distractor_irrelevant_rate": 0.0
  },
  "slot_recall_key_turns": {
    "num_questions": 186,
    "remember_correct_rate": 0.43010752688172044,
    "not_remember_rate": 0.521505376344086,
    "distractor_irrelevant_rate": 0.04838709677419355
  }
}
```
Persistence:
```json
{
  "Conversation Early Stage": {
    "whole_recall_key_turns": {
      "num_questions": 10,
      "remember_correct_rate": 0.4,
      "not_remember_rate": 0.6,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 51,
      "remember_correct_rate": 0.43137254901960786,
      "not_remember_rate": 0.49019607843137253,
      "distractor_irrelevant_rate": 0.0784313725490196
    }
  },
  "Conversation Intermediate Stage": {
    "whole_recall_key_turns": {
      "num_questions": 10,
      "remember_correct_rate": 0.2,
      "not_remember_rate": 0.7,
      "distractor_irrelevant_rate": 0.1
    },
    "slot_recall_key_turns": {
      "num_questions": 51,
      "remember_correct_rate": 0.5098039215686274,
      "not_remember_rate": 0.39215686274509803,
      "distractor_irrelevant_rate": 0.09803921568627451
    }
  },
  "Conversation Late Stage": {
    "whole_recall_key_turns": {
      "num_questions": 10,
      "remember_correct_rate": 0.1,
      "not_remember_rate": 0.8,
      "distractor_irrelevant_rate": 0.1
    },
    "slot_recall_key_turns": {
      "num_questions": 51,
      "remember_correct_rate": 0.5098039215686274,
      "not_remember_rate": 0.4117647058823529,
      "distractor_irrelevant_rate": 0.0784313725490196
    }
  }
}
```
Timing sensitivity:
```json
{
  "Conversation Early Stage": {
    "whole_recall_key_turns": {
      "num_questions": 10,
      "remember_correct_rate": 0.4,
      "not_remember_rate": 0.6,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 51,
      "remember_correct_rate": 0.43137254901960786,
      "not_remember_rate": 0.49019607843137253,
      "distractor_irrelevant_rate": 0.0784313725490196
    }
  },
  "Conversation Intermediate Stage": {
    "whole_recall_key_turns": {
      "num_questions": 10,
      "remember_correct_rate": 0.0,
      "not_remember_rate": 1.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 60,
      "remember_correct_rate": 0.45,
      "not_remember_rate": 0.5333333333333333,
      "distractor_irrelevant_rate": 0.016666666666666666
    }
  },
  "Conversation Late Stage": {
    "whole_recall_key_turns": {
      "num_questions": 10,
      "remember_correct_rate": 0.1,
      "not_remember_rate": 0.9,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 75,
      "remember_correct_rate": 0.41333333333333333,
      "not_remember_rate": 0.5333333333333333,
      "distractor_irrelevant_rate": 0.05333333333333334
    }
  }
}
```

### gpt-5.4-mini + mem0

Immediate effect:
```json
{
  "whole_recall_key_turns": {
    "num_questions": 6,
    "remember_correct_rate": 0.5,
    "not_remember_rate": 0.5,
    "distractor_irrelevant_rate": 0.0
  },
  "slot_recall_key_turns": {
    "num_questions": 27,
    "remember_correct_rate": 0.14814814814814814,
    "not_remember_rate": 0.8148148148148148,
    "distractor_irrelevant_rate": 0.037037037037037035
  }
}
```
Persistence:
```json
{
  "Conversation Early Stage": {
    "whole_recall_key_turns": {
      "num_questions": 2,
      "remember_correct_rate": 0.0,
      "not_remember_rate": 1.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 8,
      "remember_correct_rate": 0.125,
      "not_remember_rate": 0.875,
      "distractor_irrelevant_rate": 0.0
    }
  },
  "Conversation Intermediate Stage": {
    "whole_recall_key_turns": {
      "num_questions": 2,
      "remember_correct_rate": 0.0,
      "not_remember_rate": 1.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 8,
      "remember_correct_rate": 0.125,
      "not_remember_rate": 0.875,
      "distractor_irrelevant_rate": 0.0
    }
  },
  "Conversation Late Stage": {
    "whole_recall_key_turns": {
      "num_questions": 2,
      "remember_correct_rate": 0.5,
      "not_remember_rate": 0.5,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 8,
      "remember_correct_rate": 0.125,
      "not_remember_rate": 0.75,
      "distractor_irrelevant_rate": 0.125
    }
  }
}
```
Timing sensitivity:
```json
{
  "Conversation Early Stage": {
    "whole_recall_key_turns": {
      "num_questions": 2,
      "remember_correct_rate": 0.0,
      "not_remember_rate": 1.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 8,
      "remember_correct_rate": 0.125,
      "not_remember_rate": 0.875,
      "distractor_irrelevant_rate": 0.0
    }
  },
  "Conversation Intermediate Stage": {
    "whole_recall_key_turns": {
      "num_questions": 2,
      "remember_correct_rate": 0.5,
      "not_remember_rate": 0.5,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 10,
      "remember_correct_rate": 0.3,
      "not_remember_rate": 0.6,
      "distractor_irrelevant_rate": 0.1
    }
  },
  "Conversation Late Stage": {
    "whole_recall_key_turns": {
      "num_questions": 2,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 9,
      "remember_correct_rate": 0.0,
      "not_remember_rate": 1.0,
      "distractor_irrelevant_rate": 0.0
    }
  }
}
```

### gpt-4o + mem0

Immediate effect:
```json
{
  "whole_recall_key_turns": {
    "num_questions": 6,
    "remember_correct_rate": 0.16666666666666666,
    "not_remember_rate": 0.8333333333333334,
    "distractor_irrelevant_rate": 0.0
  },
  "slot_recall_key_turns": {
    "num_questions": 27,
    "remember_correct_rate": 0.1111111111111111,
    "not_remember_rate": 0.8888888888888888,
    "distractor_irrelevant_rate": 0.0
  }
}
```
Persistence:
```json
{
  "Conversation Early Stage": {
    "whole_recall_key_turns": {
      "num_questions": 2,
      "remember_correct_rate": 0.0,
      "not_remember_rate": 1.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 8,
      "remember_correct_rate": 0.0,
      "not_remember_rate": 1.0,
      "distractor_irrelevant_rate": 0.0
    }
  },
  "Conversation Intermediate Stage": {
    "whole_recall_key_turns": {
      "num_questions": 2,
      "remember_correct_rate": 0.0,
      "not_remember_rate": 1.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 8,
      "remember_correct_rate": 0.0,
      "not_remember_rate": 1.0,
      "distractor_irrelevant_rate": 0.0
    }
  },
  "Conversation Late Stage": {
    "whole_recall_key_turns": {
      "num_questions": 2,
      "remember_correct_rate": 0.0,
      "not_remember_rate": 1.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 8,
      "remember_correct_rate": 0.125,
      "not_remember_rate": 0.875,
      "distractor_irrelevant_rate": 0.0
    }
  }
}
```
Timing sensitivity:
```json
{
  "Conversation Early Stage": {
    "whole_recall_key_turns": {
      "num_questions": 2,
      "remember_correct_rate": 0.0,
      "not_remember_rate": 1.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 8,
      "remember_correct_rate": 0.0,
      "not_remember_rate": 1.0,
      "distractor_irrelevant_rate": 0.0
    }
  },
  "Conversation Intermediate Stage": {
    "whole_recall_key_turns": {
      "num_questions": 2,
      "remember_correct_rate": 0.5,
      "not_remember_rate": 0.5,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 10,
      "remember_correct_rate": 0.3,
      "not_remember_rate": 0.7,
      "distractor_irrelevant_rate": 0.0
    }
  },
  "Conversation Late Stage": {
    "whole_recall_key_turns": {
      "num_questions": 2,
      "remember_correct_rate": 0.0,
      "not_remember_rate": 1.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 9,
      "remember_correct_rate": 0.0,
      "not_remember_rate": 1.0,
      "distractor_irrelevant_rate": 0.0
    }
  }
}
```

### gpt-5.4-mini + LangMem

Immediate effect:
```json
{
  "whole_recall_key_turns": {
    "num_questions": 6,
    "remember_correct_rate": 0.6666666666666666,
    "not_remember_rate": 0.3333333333333333,
    "distractor_irrelevant_rate": 0.0
  },
  "slot_recall_key_turns": {
    "num_questions": 27,
    "remember_correct_rate": 0.8888888888888888,
    "not_remember_rate": 0.037037037037037035,
    "distractor_irrelevant_rate": 0.07407407407407407
  }
}
```
Persistence:
```json
{
  "Conversation Early Stage": {
    "whole_recall_key_turns": {
      "num_questions": 2,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 8,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    }
  },
  "Conversation Intermediate Stage": {
    "whole_recall_key_turns": {
      "num_questions": 2,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 8,
      "remember_correct_rate": 0.875,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.125
    }
  },
  "Conversation Late Stage": {
    "whole_recall_key_turns": {
      "num_questions": 2,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 8,
      "remember_correct_rate": 0.875,
      "not_remember_rate": 0.125,
      "distractor_irrelevant_rate": 0.0
    }
  }
}
```
Timing sensitivity:
```json
{
  "Conversation Early Stage": {
    "whole_recall_key_turns": {
      "num_questions": 2,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 8,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    }
  },
  "Conversation Intermediate Stage": {
    "whole_recall_key_turns": {
      "num_questions": 2,
      "remember_correct_rate": 0.5,
      "not_remember_rate": 0.5,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 10,
      "remember_correct_rate": 0.9,
      "not_remember_rate": 0.1,
      "distractor_irrelevant_rate": 0.0
    }
  },
  "Conversation Late Stage": {
    "whole_recall_key_turns": {
      "num_questions": 2,
      "remember_correct_rate": 0.5,
      "not_remember_rate": 0.5,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 9,
      "remember_correct_rate": 0.7777777777777778,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.2222222222222222
    }
  }
}
```

### gpt-4o + LangMem

Immediate effect:
```json
{
  "whole_recall_key_turns": {
    "num_questions": 5,
    "remember_correct_rate": 0.0,
    "not_remember_rate": 0.8,
    "distractor_irrelevant_rate": 0.2
  },
  "slot_recall_key_turns": {
    "num_questions": 21,
    "remember_correct_rate": 0.8095238095238095,
    "not_remember_rate": 0.19047619047619047,
    "distractor_irrelevant_rate": 0.0
  }
}
```
Persistence:
```json
{
  "Conversation Early Stage": {
    "whole_recall_key_turns": {
      "num_questions": 2,
      "remember_correct_rate": 0.0,
      "not_remember_rate": 0.5,
      "distractor_irrelevant_rate": 0.5
    },
    "slot_recall_key_turns": {
      "num_questions": 8,
      "remember_correct_rate": 0.625,
      "not_remember_rate": 0.375,
      "distractor_irrelevant_rate": 0.0
    }
  },
  "Conversation Intermediate Stage": {
    "whole_recall_key_turns": {
      "num_questions": 2,
      "remember_correct_rate": 0.0,
      "not_remember_rate": 0.5,
      "distractor_irrelevant_rate": 0.5
    },
    "slot_recall_key_turns": {
      "num_questions": 8,
      "remember_correct_rate": 0.625,
      "not_remember_rate": 0.375,
      "distractor_irrelevant_rate": 0.0
    }
  },
  "Conversation Late Stage": {
    "whole_recall_key_turns": {
      "num_questions": 1,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 3,
      "remember_correct_rate": 0.6666666666666666,
      "not_remember_rate": 0.3333333333333333,
      "distractor_irrelevant_rate": 0.0
    }
  }
}
```
Timing sensitivity:
```json
{
  "Conversation Early Stage": {
    "whole_recall_key_turns": {
      "num_questions": 2,
      "remember_correct_rate": 0.0,
      "not_remember_rate": 0.5,
      "distractor_irrelevant_rate": 0.5
    },
    "slot_recall_key_turns": {
      "num_questions": 8,
      "remember_correct_rate": 0.625,
      "not_remember_rate": 0.375,
      "distractor_irrelevant_rate": 0.0
    }
  },
  "Conversation Intermediate Stage": {
    "whole_recall_key_turns": {
      "num_questions": 2,
      "remember_correct_rate": 0.0,
      "not_remember_rate": 1.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 10,
      "remember_correct_rate": 0.9,
      "not_remember_rate": 0.1,
      "distractor_irrelevant_rate": 0.0
    }
  },
  "Conversation Late Stage": {
    "whole_recall_key_turns": {
      "num_questions": 1,
      "remember_correct_rate": 0.0,
      "not_remember_rate": 1.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 3,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    }
  }
}
```

## `no_use` Questions

Immediate suppression: `no_use@E test@E`, `no_use@I test@I`, `no_use@L test@L`.
Persistence: `no_use@E test@I`, `no_use@E test@L`.
Recovery after release: `no_use@E release@E test@E`, `no_use@E release@E test@I`, `no_use@E release@E test@L`.

### gpt-5.4-mini

Immediate suppression / persistence / recovery:
```json
{
  "no_use@E_test@E": {
    "whole_recall_key_turns": {
      "num_questions": 12,
      "remember_correct_rate": 0.6666666666666666,
      "not_remember_rate": 0.3333333333333333,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 12,
      "remember_correct_rate": 0.75,
      "not_remember_rate": 0.25,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 74,
      "remember_correct_rate": 0.8378378378378378,
      "not_remember_rate": 0.13513513513513514,
      "distractor_irrelevant_rate": 0.02702702702702703
    },
    "slot_recall_probe_turns": {
      "num_questions": 66,
      "remember_correct_rate": 0.8636363636363636,
      "not_remember_rate": 0.09090909090909091,
      "distractor_irrelevant_rate": 0.045454545454545456
    }
  },
  "no_use@I_test@I": {
    "whole_recall_key_turns": {
      "num_questions": 12,
      "remember_correct_rate": 0.75,
      "not_remember_rate": 0.25,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 12,
      "remember_correct_rate": 0.75,
      "not_remember_rate": 0.25,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 74,
      "remember_correct_rate": 0.7567567567567568,
      "not_remember_rate": 0.21621621621621623,
      "distractor_irrelevant_rate": 0.02702702702702703
    },
    "slot_recall_probe_turns": {
      "num_questions": 66,
      "remember_correct_rate": 0.7272727272727273,
      "not_remember_rate": 0.24242424242424243,
      "distractor_irrelevant_rate": 0.030303030303030304
    }
  },
  "no_use@L_test@L": {
    "whole_recall_key_turns": {
      "num_questions": 12,
      "remember_correct_rate": 0.75,
      "not_remember_rate": 0.25,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 12,
      "remember_correct_rate": 0.8333333333333334,
      "not_remember_rate": 0.16666666666666666,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 74,
      "remember_correct_rate": 0.7432432432432432,
      "not_remember_rate": 0.24324324324324326,
      "distractor_irrelevant_rate": 0.013513513513513514
    },
    "slot_recall_probe_turns": {
      "num_questions": 66,
      "remember_correct_rate": 0.6818181818181818,
      "not_remember_rate": 0.3181818181818182,
      "distractor_irrelevant_rate": 0.0
    }
  },
  "no_use@E_test@I": {
    "whole_recall_key_turns": {
      "num_questions": 12,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 12,
      "remember_correct_rate": 0.9166666666666666,
      "not_remember_rate": 0.08333333333333333,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 74,
      "remember_correct_rate": 0.9324324324324325,
      "not_remember_rate": 0.02702702702702703,
      "distractor_irrelevant_rate": 0.04054054054054054
    },
    "slot_recall_probe_turns": {
      "num_questions": 66,
      "remember_correct_rate": 0.9393939393939394,
      "not_remember_rate": 0.045454545454545456,
      "distractor_irrelevant_rate": 0.015151515151515152
    }
  },
  "no_use@E_test@L": {
    "whole_recall_key_turns": {
      "num_questions": 12,
      "remember_correct_rate": 0.9166666666666666,
      "not_remember_rate": 0.08333333333333333,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 12,
      "remember_correct_rate": 0.9166666666666666,
      "not_remember_rate": 0.08333333333333333,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 74,
      "remember_correct_rate": 0.9054054054054054,
      "not_remember_rate": 0.02702702702702703,
      "distractor_irrelevant_rate": 0.06756756756756757
    },
    "slot_recall_probe_turns": {
      "num_questions": 66,
      "remember_correct_rate": 0.9393939393939394,
      "not_remember_rate": 0.030303030303030304,
      "distractor_irrelevant_rate": 0.030303030303030304
    }
  },
  "no_use@E_release@E_test@E": {
    "whole_recall_key_turns": {
      "num_questions": 12,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 12,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 74,
      "remember_correct_rate": 0.9594594594594594,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.04054054054054054
    },
    "slot_recall_probe_turns": {
      "num_questions": 66,
      "remember_correct_rate": 0.9393939393939394,
      "not_remember_rate": 0.030303030303030304,
      "distractor_irrelevant_rate": 0.030303030303030304
    }
  },
  "no_use@E_release@E_test@I": {
    "whole_recall_key_turns": {
      "num_questions": 12,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 12,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 74,
      "remember_correct_rate": 0.9324324324324325,
      "not_remember_rate": 0.02702702702702703,
      "distractor_irrelevant_rate": 0.04054054054054054
    },
    "slot_recall_probe_turns": {
      "num_questions": 66,
      "remember_correct_rate": 0.9848484848484849,
      "not_remember_rate": 0.015151515151515152,
      "distractor_irrelevant_rate": 0.0
    }
  },
  "no_use@E_release@E_test@L": {
    "whole_recall_key_turns": {
      "num_questions": 12,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 12,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 74,
      "remember_correct_rate": 0.9324324324324325,
      "not_remember_rate": 0.013513513513513514,
      "distractor_irrelevant_rate": 0.05405405405405406
    },
    "slot_recall_probe_turns": {
      "num_questions": 66,
      "remember_correct_rate": 0.9393939393939394,
      "not_remember_rate": 0.045454545454545456,
      "distractor_irrelevant_rate": 0.015151515151515152
    }
  }
}
```

### gpt-4o

Immediate suppression / persistence / recovery:
```json
{
  "no_use@E_test@E": {
    "whole_recall_key_turns": {
      "num_questions": 12,
      "remember_correct_rate": 0.75,
      "not_remember_rate": 0.16666666666666666,
      "distractor_irrelevant_rate": 0.08333333333333333
    },
    "whole_recall_probe_turns": {
      "num_questions": 12,
      "remember_correct_rate": 0.5,
      "not_remember_rate": 0.5,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 74,
      "remember_correct_rate": 0.5675675675675675,
      "not_remember_rate": 0.40540540540540543,
      "distractor_irrelevant_rate": 0.02702702702702703
    },
    "slot_recall_probe_turns": {
      "num_questions": 66,
      "remember_correct_rate": 0.696969696969697,
      "not_remember_rate": 0.2878787878787879,
      "distractor_irrelevant_rate": 0.015151515151515152
    }
  },
  "no_use@I_test@I": {
    "whole_recall_key_turns": {
      "num_questions": 12,
      "remember_correct_rate": 0.5833333333333334,
      "not_remember_rate": 0.3333333333333333,
      "distractor_irrelevant_rate": 0.08333333333333333
    },
    "whole_recall_probe_turns": {
      "num_questions": 12,
      "remember_correct_rate": 0.5833333333333334,
      "not_remember_rate": 0.4166666666666667,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 74,
      "remember_correct_rate": 0.5675675675675675,
      "not_remember_rate": 0.43243243243243246,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_probe_turns": {
      "num_questions": 66,
      "remember_correct_rate": 0.6515151515151515,
      "not_remember_rate": 0.30303030303030304,
      "distractor_irrelevant_rate": 0.045454545454545456
    }
  },
  "no_use@L_test@L": {
    "whole_recall_key_turns": {
      "num_questions": 12,
      "remember_correct_rate": 0.75,
      "not_remember_rate": 0.25,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 12,
      "remember_correct_rate": 0.6666666666666666,
      "not_remember_rate": 0.3333333333333333,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 74,
      "remember_correct_rate": 0.6216216216216216,
      "not_remember_rate": 0.3108108108108108,
      "distractor_irrelevant_rate": 0.06756756756756757
    },
    "slot_recall_probe_turns": {
      "num_questions": 66,
      "remember_correct_rate": 0.696969696969697,
      "not_remember_rate": 0.2878787878787879,
      "distractor_irrelevant_rate": 0.015151515151515152
    }
  },
  "no_use@E_test@I": {
    "whole_recall_key_turns": {
      "num_questions": 12,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 12,
      "remember_correct_rate": 0.75,
      "not_remember_rate": 0.25,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 74,
      "remember_correct_rate": 0.7297297297297297,
      "not_remember_rate": 0.17567567567567569,
      "distractor_irrelevant_rate": 0.0945945945945946
    },
    "slot_recall_probe_turns": {
      "num_questions": 66,
      "remember_correct_rate": 0.8333333333333334,
      "not_remember_rate": 0.10606060606060606,
      "distractor_irrelevant_rate": 0.06060606060606061
    }
  },
  "no_use@E_test@L": {
    "whole_recall_key_turns": {
      "num_questions": 12,
      "remember_correct_rate": 0.75,
      "not_remember_rate": 0.16666666666666666,
      "distractor_irrelevant_rate": 0.08333333333333333
    },
    "whole_recall_probe_turns": {
      "num_questions": 12,
      "remember_correct_rate": 0.6666666666666666,
      "not_remember_rate": 0.3333333333333333,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 74,
      "remember_correct_rate": 0.6486486486486487,
      "not_remember_rate": 0.1891891891891892,
      "distractor_irrelevant_rate": 0.14864864864864866
    },
    "slot_recall_probe_turns": {
      "num_questions": 66,
      "remember_correct_rate": 0.8636363636363636,
      "not_remember_rate": 0.10606060606060606,
      "distractor_irrelevant_rate": 0.030303030303030304
    }
  },
  "no_use@E_release@E_test@E": {
    "whole_recall_key_turns": {
      "num_questions": 12,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 12,
      "remember_correct_rate": 0.9166666666666666,
      "not_remember_rate": 0.08333333333333333,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 74,
      "remember_correct_rate": 0.8108108108108109,
      "not_remember_rate": 0.06756756756756757,
      "distractor_irrelevant_rate": 0.12162162162162163
    },
    "slot_recall_probe_turns": {
      "num_questions": 66,
      "remember_correct_rate": 0.9242424242424242,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.07575757575757576
    }
  },
  "no_use@E_release@E_test@I": {
    "whole_recall_key_turns": {
      "num_questions": 12,
      "remember_correct_rate": 0.8333333333333334,
      "not_remember_rate": 0.08333333333333333,
      "distractor_irrelevant_rate": 0.08333333333333333
    },
    "whole_recall_probe_turns": {
      "num_questions": 12,
      "remember_correct_rate": 0.8333333333333334,
      "not_remember_rate": 0.16666666666666666,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 74,
      "remember_correct_rate": 0.8513513513513513,
      "not_remember_rate": 0.06756756756756757,
      "distractor_irrelevant_rate": 0.08108108108108109
    },
    "slot_recall_probe_turns": {
      "num_questions": 66,
      "remember_correct_rate": 0.9090909090909091,
      "not_remember_rate": 0.030303030303030304,
      "distractor_irrelevant_rate": 0.06060606060606061
    }
  },
  "no_use@E_release@E_test@L": {
    "whole_recall_key_turns": {
      "num_questions": 12,
      "remember_correct_rate": 0.9166666666666666,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.08333333333333333
    },
    "whole_recall_probe_turns": {
      "num_questions": 12,
      "remember_correct_rate": 0.9166666666666666,
      "not_remember_rate": 0.08333333333333333,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 74,
      "remember_correct_rate": 0.7567567567567568,
      "not_remember_rate": 0.08108108108108109,
      "distractor_irrelevant_rate": 0.13513513513513514
    },
    "slot_recall_probe_turns": {
      "num_questions": 66,
      "remember_correct_rate": 0.9090909090909091,
      "not_remember_rate": 0.015151515151515152,
      "distractor_irrelevant_rate": 0.06060606060606061
    }
  }
}
```

### gpt-5.4-mini + mem0

Immediate suppression / persistence / recovery:
```json
{
  "no_use@E_test@E": {
    "whole_recall_key_turns": {
      "num_questions": 6,
      "remember_correct_rate": 0.6666666666666666,
      "not_remember_rate": 0.3333333333333333,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 6,
      "remember_correct_rate": 0.5,
      "not_remember_rate": 0.5,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 27,
      "remember_correct_rate": 0.1111111111111111,
      "not_remember_rate": 0.8148148148148148,
      "distractor_irrelevant_rate": 0.07407407407407407
    },
    "slot_recall_probe_turns": {
      "num_questions": 26,
      "remember_correct_rate": 0.15384615384615385,
      "not_remember_rate": 0.8076923076923077,
      "distractor_irrelevant_rate": 0.038461538461538464
    }
  },
  "no_use@I_test@I": {
    "whole_recall_key_turns": {
      "num_questions": 6,
      "remember_correct_rate": 0.5,
      "not_remember_rate": 0.5,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 6,
      "remember_correct_rate": 0.5,
      "not_remember_rate": 0.5,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 27,
      "remember_correct_rate": 0.18518518518518517,
      "not_remember_rate": 0.7777777777777778,
      "distractor_irrelevant_rate": 0.037037037037037035
    },
    "slot_recall_probe_turns": {
      "num_questions": 26,
      "remember_correct_rate": 0.19230769230769232,
      "not_remember_rate": 0.8076923076923077,
      "distractor_irrelevant_rate": 0.0
    }
  },
  "no_use@L_test@L": {
    "whole_recall_key_turns": {
      "num_questions": 6,
      "remember_correct_rate": 0.6666666666666666,
      "not_remember_rate": 0.3333333333333333,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 6,
      "remember_correct_rate": 0.3333333333333333,
      "not_remember_rate": 0.6666666666666666,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 27,
      "remember_correct_rate": 0.14814814814814814,
      "not_remember_rate": 0.7407407407407407,
      "distractor_irrelevant_rate": 0.1111111111111111
    },
    "slot_recall_probe_turns": {
      "num_questions": 26,
      "remember_correct_rate": 0.2692307692307692,
      "not_remember_rate": 0.6538461538461539,
      "distractor_irrelevant_rate": 0.07692307692307693
    }
  },
  "no_use@E_test@I": {
    "whole_recall_key_turns": {
      "num_questions": 6,
      "remember_correct_rate": 0.3333333333333333,
      "not_remember_rate": 0.6666666666666666,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 6,
      "remember_correct_rate": 0.5,
      "not_remember_rate": 0.5,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 27,
      "remember_correct_rate": 0.14814814814814814,
      "not_remember_rate": 0.7777777777777778,
      "distractor_irrelevant_rate": 0.07407407407407407
    },
    "slot_recall_probe_turns": {
      "num_questions": 26,
      "remember_correct_rate": 0.15384615384615385,
      "not_remember_rate": 0.8076923076923077,
      "distractor_irrelevant_rate": 0.038461538461538464
    }
  },
  "no_use@E_test@L": {
    "whole_recall_key_turns": {
      "num_questions": 6,
      "remember_correct_rate": 0.5,
      "not_remember_rate": 0.5,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 6,
      "remember_correct_rate": 0.6666666666666666,
      "not_remember_rate": 0.3333333333333333,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 27,
      "remember_correct_rate": 0.18518518518518517,
      "not_remember_rate": 0.5555555555555556,
      "distractor_irrelevant_rate": 0.25925925925925924
    },
    "slot_recall_probe_turns": {
      "num_questions": 26,
      "remember_correct_rate": 0.2692307692307692,
      "not_remember_rate": 0.6538461538461539,
      "distractor_irrelevant_rate": 0.07692307692307693
    }
  },
  "no_use@E_release@E_test@E": {
    "whole_recall_key_turns": {
      "num_questions": 3,
      "remember_correct_rate": 0.3333333333333333,
      "not_remember_rate": 0.6666666666666666,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 3,
      "remember_correct_rate": 0.6666666666666666,
      "not_remember_rate": 0.3333333333333333,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 8,
      "remember_correct_rate": 0.0,
      "not_remember_rate": 1.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_probe_turns": {
      "num_questions": 9,
      "remember_correct_rate": 0.2222222222222222,
      "not_remember_rate": 0.7777777777777778,
      "distractor_irrelevant_rate": 0.0
    }
  },
  "no_use@E_release@E_test@I": {
    "whole_recall_key_turns": {
      "num_questions": 3,
      "remember_correct_rate": 0.3333333333333333,
      "not_remember_rate": 0.6666666666666666,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 3,
      "remember_correct_rate": 0.3333333333333333,
      "not_remember_rate": 0.6666666666666666,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 8,
      "remember_correct_rate": 0.0,
      "not_remember_rate": 0.875,
      "distractor_irrelevant_rate": 0.125
    },
    "slot_recall_probe_turns": {
      "num_questions": 9,
      "remember_correct_rate": 0.1111111111111111,
      "not_remember_rate": 0.8888888888888888,
      "distractor_irrelevant_rate": 0.0
    }
  },
  "no_use@E_release@E_test@L": {
    "whole_recall_key_turns": {
      "num_questions": 3,
      "remember_correct_rate": 0.6666666666666666,
      "not_remember_rate": 0.3333333333333333,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 3,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 8,
      "remember_correct_rate": 0.25,
      "not_remember_rate": 0.5,
      "distractor_irrelevant_rate": 0.25
    },
    "slot_recall_probe_turns": {
      "num_questions": 9,
      "remember_correct_rate": 0.4444444444444444,
      "not_remember_rate": 0.5555555555555556,
      "distractor_irrelevant_rate": 0.0
    }
  }
}
```

### gpt-4o + mem0

Immediate suppression / persistence / recovery:
```json
{
  "no_use@E_test@E": {
    "whole_recall_key_turns": {
      "num_questions": 6,
      "remember_correct_rate": 0.16666666666666666,
      "not_remember_rate": 0.8333333333333334,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 6,
      "remember_correct_rate": 0.3333333333333333,
      "not_remember_rate": 0.6666666666666666,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 27,
      "remember_correct_rate": 0.07407407407407407,
      "not_remember_rate": 0.8518518518518519,
      "distractor_irrelevant_rate": 0.07407407407407407
    },
    "slot_recall_probe_turns": {
      "num_questions": 26,
      "remember_correct_rate": 0.15384615384615385,
      "not_remember_rate": 0.8076923076923077,
      "distractor_irrelevant_rate": 0.038461538461538464
    }
  },
  "no_use@I_test@I": {
    "whole_recall_key_turns": {
      "num_questions": 6,
      "remember_correct_rate": 0.16666666666666666,
      "not_remember_rate": 0.8333333333333334,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 6,
      "remember_correct_rate": 0.0,
      "not_remember_rate": 1.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 27,
      "remember_correct_rate": 0.07407407407407407,
      "not_remember_rate": 0.9259259259259259,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_probe_turns": {
      "num_questions": 26,
      "remember_correct_rate": 0.11538461538461539,
      "not_remember_rate": 0.8846153846153846,
      "distractor_irrelevant_rate": 0.0
    }
  },
  "no_use@L_test@L": {
    "whole_recall_key_turns": {
      "num_questions": 6,
      "remember_correct_rate": 0.16666666666666666,
      "not_remember_rate": 0.8333333333333334,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 6,
      "remember_correct_rate": 0.16666666666666666,
      "not_remember_rate": 0.8333333333333334,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 27,
      "remember_correct_rate": 0.037037037037037035,
      "not_remember_rate": 0.9629629629629629,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_probe_turns": {
      "num_questions": 26,
      "remember_correct_rate": 0.07692307692307693,
      "not_remember_rate": 0.9230769230769231,
      "distractor_irrelevant_rate": 0.0
    }
  },
  "no_use@E_test@I": {
    "whole_recall_key_turns": {
      "num_questions": 6,
      "remember_correct_rate": 0.16666666666666666,
      "not_remember_rate": 0.8333333333333334,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 6,
      "remember_correct_rate": 0.16666666666666666,
      "not_remember_rate": 0.8333333333333334,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 27,
      "remember_correct_rate": 0.07407407407407407,
      "not_remember_rate": 0.8518518518518519,
      "distractor_irrelevant_rate": 0.037037037037037035
    },
    "slot_recall_probe_turns": {
      "num_questions": 26,
      "remember_correct_rate": 0.038461538461538464,
      "not_remember_rate": 0.9615384615384616,
      "distractor_irrelevant_rate": 0.0
    }
  },
  "no_use@E_test@L": {
    "whole_recall_key_turns": {
      "num_questions": 6,
      "remember_correct_rate": 0.0,
      "not_remember_rate": 1.0,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 6,
      "remember_correct_rate": 0.0,
      "not_remember_rate": 1.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 27,
      "remember_correct_rate": 0.4074074074074074,
      "not_remember_rate": 0.5185185185185185,
      "distractor_irrelevant_rate": 0.07407407407407407
    },
    "slot_recall_probe_turns": {
      "num_questions": 26,
      "remember_correct_rate": 0.19230769230769232,
      "not_remember_rate": 0.7692307692307693,
      "distractor_irrelevant_rate": 0.038461538461538464
    }
  },
  "no_use@E_release@E_test@E": {
    "whole_recall_key_turns": {
      "num_questions": 6,
      "remember_correct_rate": 0.16666666666666666,
      "not_remember_rate": 0.6666666666666666,
      "distractor_irrelevant_rate": 0.16666666666666666
    },
    "whole_recall_probe_turns": {
      "num_questions": 6,
      "remember_correct_rate": 0.16666666666666666,
      "not_remember_rate": 0.6666666666666666,
      "distractor_irrelevant_rate": 0.16666666666666666
    },
    "slot_recall_key_turns": {
      "num_questions": 27,
      "remember_correct_rate": 0.18518518518518517,
      "not_remember_rate": 0.7407407407407407,
      "distractor_irrelevant_rate": 0.07407407407407407
    },
    "slot_recall_probe_turns": {
      "num_questions": 26,
      "remember_correct_rate": 0.23076923076923078,
      "not_remember_rate": 0.7307692307692307,
      "distractor_irrelevant_rate": 0.038461538461538464
    }
  },
  "no_use@E_release@E_test@I": {
    "whole_recall_key_turns": {
      "num_questions": 6,
      "remember_correct_rate": 0.16666666666666666,
      "not_remember_rate": 0.6666666666666666,
      "distractor_irrelevant_rate": 0.16666666666666666
    },
    "whole_recall_probe_turns": {
      "num_questions": 6,
      "remember_correct_rate": 0.16666666666666666,
      "not_remember_rate": 0.8333333333333334,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 27,
      "remember_correct_rate": 0.2222222222222222,
      "not_remember_rate": 0.7407407407407407,
      "distractor_irrelevant_rate": 0.037037037037037035
    },
    "slot_recall_probe_turns": {
      "num_questions": 26,
      "remember_correct_rate": 0.19230769230769232,
      "not_remember_rate": 0.7307692307692307,
      "distractor_irrelevant_rate": 0.07692307692307693
    }
  },
  "no_use@E_release@E_test@L": {
    "whole_recall_key_turns": {
      "num_questions": 3,
      "remember_correct_rate": 0.3333333333333333,
      "not_remember_rate": 0.6666666666666666,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 3,
      "remember_correct_rate": 0.6666666666666666,
      "not_remember_rate": 0.3333333333333333,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 8,
      "remember_correct_rate": 0.125,
      "not_remember_rate": 0.75,
      "distractor_irrelevant_rate": 0.125
    },
    "slot_recall_probe_turns": {
      "num_questions": 9,
      "remember_correct_rate": 0.3333333333333333,
      "not_remember_rate": 0.3333333333333333,
      "distractor_irrelevant_rate": 0.3333333333333333
    }
  }
}
```

### gpt-5.4-mini + LangMem

Immediate suppression / persistence / recovery:
```json
{
  "no_use@E_test@E": {
    "whole_recall_key_turns": {
      "num_questions": 6,
      "remember_correct_rate": 0.6666666666666666,
      "not_remember_rate": 0.3333333333333333,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 6,
      "remember_correct_rate": 0.8333333333333334,
      "not_remember_rate": 0.16666666666666666,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 27,
      "remember_correct_rate": 0.8148148148148148,
      "not_remember_rate": 0.07407407407407407,
      "distractor_irrelevant_rate": 0.1111111111111111
    },
    "slot_recall_probe_turns": {
      "num_questions": 26,
      "remember_correct_rate": 0.8076923076923077,
      "not_remember_rate": 0.15384615384615385,
      "distractor_irrelevant_rate": 0.038461538461538464
    }
  },
  "no_use@I_test@I": {
    "whole_recall_key_turns": {
      "num_questions": 6,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 6,
      "remember_correct_rate": 0.8333333333333334,
      "not_remember_rate": 0.16666666666666666,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 27,
      "remember_correct_rate": 0.8888888888888888,
      "not_remember_rate": 0.07407407407407407,
      "distractor_irrelevant_rate": 0.037037037037037035
    },
    "slot_recall_probe_turns": {
      "num_questions": 26,
      "remember_correct_rate": 0.8076923076923077,
      "not_remember_rate": 0.19230769230769232,
      "distractor_irrelevant_rate": 0.0
    }
  },
  "no_use@L_test@L": {
    "whole_recall_key_turns": {
      "num_questions": 3,
      "remember_correct_rate": 0.6666666666666666,
      "not_remember_rate": 0.3333333333333333,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 3,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 8,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_probe_turns": {
      "num_questions": 9,
      "remember_correct_rate": 0.7777777777777778,
      "not_remember_rate": 0.2222222222222222,
      "distractor_irrelevant_rate": 0.0
    }
  },
  "no_use@E_test@I": {
    "whole_recall_key_turns": {
      "num_questions": 3,
      "remember_correct_rate": 0.6666666666666666,
      "not_remember_rate": 0.3333333333333333,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 3,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 8,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_probe_turns": {
      "num_questions": 9,
      "remember_correct_rate": 0.8888888888888888,
      "not_remember_rate": 0.1111111111111111,
      "distractor_irrelevant_rate": 0.0
    }
  },
  "no_use@E_test@L": {
    "whole_recall_key_turns": {
      "num_questions": 3,
      "remember_correct_rate": 0.6666666666666666,
      "not_remember_rate": 0.3333333333333333,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 3,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 8,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_probe_turns": {
      "num_questions": 9,
      "remember_correct_rate": 0.7777777777777778,
      "not_remember_rate": 0.2222222222222222,
      "distractor_irrelevant_rate": 0.0
    }
  },
  "no_use@E_release@E_test@E": {
    "whole_recall_key_turns": {
      "num_questions": 3,
      "remember_correct_rate": 0.6666666666666666,
      "not_remember_rate": 0.3333333333333333,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 3,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 8,
      "remember_correct_rate": 0.875,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.125
    },
    "slot_recall_probe_turns": {
      "num_questions": 9,
      "remember_correct_rate": 0.8888888888888888,
      "not_remember_rate": 0.1111111111111111,
      "distractor_irrelevant_rate": 0.0
    }
  },
  "no_use@E_release@E_test@I": {
    "whole_recall_key_turns": {
      "num_questions": 3,
      "remember_correct_rate": 0.6666666666666666,
      "not_remember_rate": 0.3333333333333333,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 3,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 8,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_probe_turns": {
      "num_questions": 9,
      "remember_correct_rate": 0.8888888888888888,
      "not_remember_rate": 0.1111111111111111,
      "distractor_irrelevant_rate": 0.0
    }
  },
  "no_use@E_release@E_test@L": {
    "whole_recall_key_turns": {
      "num_questions": 3,
      "remember_correct_rate": 0.6666666666666666,
      "not_remember_rate": 0.3333333333333333,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 3,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 8,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_probe_turns": {
      "num_questions": 9,
      "remember_correct_rate": 0.7777777777777778,
      "not_remember_rate": 0.2222222222222222,
      "distractor_irrelevant_rate": 0.0
    }
  }
}
```

### gpt-4o + LangMem

Immediate suppression / persistence / recovery:
```json
{
  "no_use@E_test@E": {
    "whole_recall_key_turns": {
      "num_questions": 3,
      "remember_correct_rate": 0.6666666666666666,
      "not_remember_rate": 0.3333333333333333,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 3,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 8,
      "remember_correct_rate": 0.875,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.125
    },
    "slot_recall_probe_turns": {
      "num_questions": 9,
      "remember_correct_rate": 0.7777777777777778,
      "not_remember_rate": 0.1111111111111111,
      "distractor_irrelevant_rate": 0.1111111111111111
    }
  },
  "no_use@I_test@I": {
    "whole_recall_key_turns": {
      "num_questions": 3,
      "remember_correct_rate": 0.3333333333333333,
      "not_remember_rate": 0.6666666666666666,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 3,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 8,
      "remember_correct_rate": 0.875,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.125
    },
    "slot_recall_probe_turns": {
      "num_questions": 9,
      "remember_correct_rate": 0.6666666666666666,
      "not_remember_rate": 0.2222222222222222,
      "distractor_irrelevant_rate": 0.1111111111111111
    }
  },
  "no_use@L_test@L": {
    "whole_recall_key_turns": {
      "num_questions": 3,
      "remember_correct_rate": 0.3333333333333333,
      "not_remember_rate": 0.6666666666666666,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 3,
      "remember_correct_rate": 0.6666666666666666,
      "not_remember_rate": 0.3333333333333333,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 8,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_probe_turns": {
      "num_questions": 9,
      "remember_correct_rate": 0.8888888888888888,
      "not_remember_rate": 0.1111111111111111,
      "distractor_irrelevant_rate": 0.0
    }
  },
  "no_use@E_test@I": {
    "whole_recall_key_turns": {
      "num_questions": 3,
      "remember_correct_rate": 0.6666666666666666,
      "not_remember_rate": 0.3333333333333333,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 3,
      "remember_correct_rate": 0.6666666666666666,
      "not_remember_rate": 0.3333333333333333,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 8,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_probe_turns": {
      "num_questions": 9,
      "remember_correct_rate": 0.6666666666666666,
      "not_remember_rate": 0.2222222222222222,
      "distractor_irrelevant_rate": 0.1111111111111111
    }
  },
  "no_use@E_test@L": {
    "whole_recall_key_turns": {
      "num_questions": 3,
      "remember_correct_rate": 0.3333333333333333,
      "not_remember_rate": 0.6666666666666666,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 3,
      "remember_correct_rate": 0.6666666666666666,
      "not_remember_rate": 0.3333333333333333,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 8,
      "remember_correct_rate": 0.875,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.125
    },
    "slot_recall_probe_turns": {
      "num_questions": 9,
      "remember_correct_rate": 0.6666666666666666,
      "not_remember_rate": 0.1111111111111111,
      "distractor_irrelevant_rate": 0.2222222222222222
    }
  },
  "no_use@E_release@E_test@E": {
    "whole_recall_key_turns": {
      "num_questions": 3,
      "remember_correct_rate": 0.6666666666666666,
      "not_remember_rate": 0.3333333333333333,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 3,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 8,
      "remember_correct_rate": 0.75,
      "not_remember_rate": 0.125,
      "distractor_irrelevant_rate": 0.125
    },
    "slot_recall_probe_turns": {
      "num_questions": 9,
      "remember_correct_rate": 0.7777777777777778,
      "not_remember_rate": 0.2222222222222222,
      "distractor_irrelevant_rate": 0.0
    }
  },
  "no_use@E_release@E_test@I": {
    "whole_recall_key_turns": {
      "num_questions": 3,
      "remember_correct_rate": 0.6666666666666666,
      "not_remember_rate": 0.3333333333333333,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 3,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 8,
      "remember_correct_rate": 0.625,
      "not_remember_rate": 0.125,
      "distractor_irrelevant_rate": 0.25
    },
    "slot_recall_probe_turns": {
      "num_questions": 9,
      "remember_correct_rate": 0.7777777777777778,
      "not_remember_rate": 0.2222222222222222,
      "distractor_irrelevant_rate": 0.0
    }
  },
  "no_use@E_release@E_test@L": {
    "whole_recall_key_turns": {
      "num_questions": 3,
      "remember_correct_rate": 0.6666666666666666,
      "not_remember_rate": 0.3333333333333333,
      "distractor_irrelevant_rate": 0.0
    },
    "whole_recall_probe_turns": {
      "num_questions": 3,
      "remember_correct_rate": 0.6666666666666666,
      "not_remember_rate": 0.3333333333333333,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_key_turns": {
      "num_questions": 8,
      "remember_correct_rate": 1.0,
      "not_remember_rate": 0.0,
      "distractor_irrelevant_rate": 0.0
    },
    "slot_recall_probe_turns": {
      "num_questions": 9,
      "remember_correct_rate": 0.7777777777777778,
      "not_remember_rate": 0.2222222222222222,
      "distractor_irrelevant_rate": 0.0
    }
  }
}
```

## Slot-Type Difficulty

The slot-type section is being generated separately with LLM post-processing over the plain results and will be appended in the full report.
