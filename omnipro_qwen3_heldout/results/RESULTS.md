# QueryStream + Qwen3-VL-8B: OmniPro Online

Held-out samples: 750. Video-only; none/helpful subset; seed 42.

|Task|N|Time F1 (%)|Joint F1 (%)|
|---|---:|---:|---:|
|cumulative_counting|82|21.16|7.16|
|dedup_counting|227|22.76|9.15|
|event_narration|77|16.69|4.26|
|explicit_target_grounding|10|6.45|0.00|
|instant_event_alert|31|8.96|8.96|
|realtime_state_monitor|114|18.63|6.97|
|semantic_condition_alert|31|17.91|17.91|
|sequential_step_instruction|119|37.36|9.15|
|snapshot_counting|59|9.27|2.65|

**Macro joint F1: 7.36%**

Adapted implementation: official QDP mathematics + reproduction Qwen3 wrapper with audited position slicing. See PROTOCOL.md for complete configuration and differences.
