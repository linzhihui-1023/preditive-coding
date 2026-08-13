# Learned Recurrent-error Validation

Frozen backbone, feedback decoders, and existing Predify parameters. Only the new recurrent transition was trained on drives 0005/0013/0014/0036. Validation uses drives 0011/0039 and the unchanged 40 clean / 80 blur / 40 recovery protocol. Frozen Test drives 0051/0056 were not read.

| Condition | Disturbance normalized L2 | Recovery first 10 | Recovery last 10 |
| --- | ---: | ---: | ---: |
| Current stateful | 0.784296992 | 0.427156845 | 0.001205088 |
| Learned recurrent-error | 0.501548966 | 0.280066796 | 0.004235435 |
| Learned recurrent-error zeroed | 0.521438669 | 0.301188880 | 0.045429500 |

Learned vs current disturbance improvement: 36.051143%.
Learned vs zeroed disturbance improvement: 3.814390%.
Conclusion: recurrent_transition_and_dynamic_error_both_help.

## Per Drive

| Drive | Current | Learned | Zeroed |
| --- | ---: | ---: | ---: |
| 0011 | 0.756414548 | 0.522033436 | 0.534274023 |
| 0039 | 0.812179436 | 0.481064495 | 0.508603316 |

## Per Layer

| Layer | Current | Learned | Zeroed |
| ---: | ---: | ---: | ---: |
| 1 | 0.522014509 | 0.548431637 | 0.549759817 |
| 2 | 0.809568361 | 0.702847445 | 0.723749197 |
| 3 | 0.778951880 | 0.548871507 | 0.558689311 |
| 4 | 0.823902837 | 0.419169117 | 0.420849670 |
| 5 | 0.987047375 | 0.288425122 | 0.354145353 |
