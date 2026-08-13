# Prediction-error-driven Recurrent Validation

Frozen backbone, feedback decoders, and existing Predify parameters. Only the new recurrent transition was trained on drives 0005/0013/0014/0036. Validation uses drives 0011/0039 and the unchanged 40 clean / 80 blur / 40 recovery protocol. Frozen Test drives 0051/0056 were not read.

| Condition | Disturbance normalized L2 | Recovery first 10 | Recovery last 10 |
| --- | ---: | ---: | ---: |
| Current stateful | 0.784296992 | 0.427156845 | 0.001205088 |
| Learned error-driven | 0.522036018 | 0.391024056 | 0.060610952 |
| Learned error-driven zeroed | 0.000000000 | 0.000000000 | 0.000000000 |

Learned vs current disturbance improvement: 33.438988%.
Learned vs zeroed disturbance improvement: undefined (zeroed disturbance is exactly zero).
Conclusion: error_zeroed_is_input_blind_prediction_error_mechanism_not_established.

## Per Drive

| Drive | Current | Learned | Zeroed |
| --- | ---: | ---: | ---: |
| 0011 | 0.756414548 | 0.527544671 | 0.000000000 |
| 0039 | 0.812179436 | 0.516527365 | 0.000000000 |

## Per Layer

| Layer | Current | Learned | Zeroed |
| ---: | ---: | ---: | ---: |
| 1 | 0.522014509 | 0.385264914 | 0.000000000 |
| 2 | 0.809568361 | 0.626145863 | 0.000000000 |
| 3 | 0.778951880 | 0.727296677 | 0.000000000 |
| 4 | 0.823902837 | 0.859442558 | 0.000000000 |
| 5 | 0.987047375 | 0.012030079 | 0.000000000 |
