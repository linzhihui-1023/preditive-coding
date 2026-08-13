# Prediction-error-driven Recurrent V2 Validation

Frozen backbone and non-recurrent Predify body. The recurrent transition, temporal predictor, and dedicated signed-error encoder were trained on drives 0005/0013/0014/0036. Validation uses drives 0011/0039 and the unchanged 40 clean / 80 blur / 40 recovery protocol. Frozen Test drives 0051/0056 were not read.

| Condition | Next-frame MSE | Disturbance normalized L2 | Recovery first 10 | Recovery last 10 |
| --- | ---: | ---: | ---: | ---: |
| Current stateful | 2.671453703 | 0.784296992 | 0.427156845 | 0.001205088 |
| Observation-driven recurrent | 2.355917884 | 0.660923713 | 0.514935964 | 0.189150613 |
| Error-driven recurrent v2 | 1.805955697 | 0.660516654 | 0.524020957 | 0.175822271 |

Error-driven vs current disturbance improvement: 15.782330%.
Error-driven vs observation disturbance improvement: 0.061589%.
Conclusion: benefit_mainly_from_recurrent_temporal_modeling.

GO/NO-GO: NO-GO for independent prediction-error value. Error-driven v2 is
better than current-stateful, but it is effectively tied with the matched
observation-driven recurrent control on the primary disturbance metric.

## Per Drive

| Drive | Current | Observation | Error |
| --- | ---: | ---: | ---: |
| 0011 | 0.756414548 | 0.675982305 | 0.671194616 |
| 0039 | 0.812179436 | 0.645865122 | 0.649838692 |

## Per Layer

| Layer | Current | Observation | Error |
| ---: | ---: | ---: | ---: |
| 1 | 0.522014509 | 0.367012720 | 0.339891071 |
| 2 | 0.809568361 | 0.695076514 | 0.689531979 |
| 3 | 0.778951880 | 0.769609230 | 0.735309200 |
| 4 | 0.823902837 | 0.853442215 | 0.850364266 |
| 5 | 0.987047375 | 0.619477886 | 0.687486755 |
