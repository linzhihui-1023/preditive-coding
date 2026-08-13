# Learned Recurrent-error Frozen Test

Frozen c9fec52 structure and epoch-1 checkpoint on Test drives 0051/0056. The protocol is unchanged: 40 clean, 80 persistent Gaussian-blur, and 40 clean recovery frames. No training, tuning, or checkpoint update occurred.

| Condition | Disturbance normalized L2 | Recovery first 10 | Recovery last 10 |
| --- | ---: | ---: | ---: |
| current_stateful | 0.819635281 | 0.446208276 | 0.001090620 |
| learned_recurrent_error | 0.536746322 | 0.305349457 | 0.001978208 |
| learned_recurrent_error_zeroed | 0.548573083 | 0.310442773 | 0.040968571 |

Learned vs current: 34.514005%.
Learned vs zeroed: 2.155914%.
Decision: PASS_MAIN_MECHANISM.

## Per Drive

| Drive | Current | Learned | Zeroed | Learned vs current | Learned vs zeroed |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0051 | 0.834337897 | 0.524248475 | 0.527247155 | 37.165928% | 0.568743% |
| 0056 | 0.804932665 | 0.549244169 | 0.569899012 | 31.765203% | 3.624299% |

## Per Layer

| Layer | Current | Learned | Zeroed |
| ---: | ---: | ---: | ---: |
| 1 | 0.596170468 | 0.618264148 | 0.620883156 |
| 2 | 0.875425224 | 0.762378494 | 0.791565973 |
| 3 | 0.805920135 | 0.562883748 | 0.564215742 |
| 4 | 0.835098554 | 0.442382258 | 0.427868754 |
| 5 | 0.985562024 | 0.297822963 | 0.338331792 |
