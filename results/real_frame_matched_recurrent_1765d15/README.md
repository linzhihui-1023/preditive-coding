# Prediction-error-driven Recurrent Validation

Frozen backbone, feedback decoders, and existing Predify parameters. Only the new recurrent transition was trained on drives 0005/0013/0014/0036. Validation uses drives 0011/0039 and the unchanged 40 clean / 80 blur / 40 recovery protocol. Frozen Test drives 0051/0056 were not read.

| Condition | Next-frame MSE | Disturbance normalized L2 | Recovery first 10 | Recovery last 10 |
| --- | ---: | ---: | ---: | ---: |
| Current stateful | 2.671453703 | 0.784296992 | 0.427156845 | 0.001205088 |
| Observation-driven recurrent | 2.344646957 | 0.510357280 | 0.341280196 | 0.042804741 |
| Error-driven recurrent | 2.341551072 | 0.522036018 | 0.391024056 | 0.060610952 |

Error-driven vs current disturbance improvement: 33.438988%.
Error-driven vs observation disturbance improvement: -2.288345%.
Conclusion: benefit_mainly_from_recurrent_temporal_modeling.

Sanity check: error-zeroed disturbance normalized L2 = 0.000000000; it is not used as the core performance control.

## Per Drive

| Drive | Current | Observation | Error |
| --- | ---: | ---: | ---: |
| 0011 | 0.756414548 | 0.515648767 | 0.527544671 |
| 0039 | 0.812179436 | 0.505065794 | 0.516527365 |

## Per Layer

| Layer | Current | Observation | Error |
| ---: | ---: | ---: | ---: |
| 1 | 0.522014509 | 0.330025678 | 0.385264914 |
| 2 | 0.809568361 | 0.666513532 | 0.626145863 |
| 3 | 0.778951880 | 0.748977268 | 0.727296677 |
| 4 | 0.823902837 | 0.803001711 | 0.859442558 |
| 5 | 0.987047375 | 0.003268213 | 0.012030079 |
