# Recurrent-error Cross-corruption Validation

Frozen c9fec52 epoch-1 checkpoint on Val drives 0011/0039. Each corruption uses 40 clean, 80 persistent disturbed, and 40 recovery frames. No training, tuning, or checkpoint selection occurred.

## gaussian_noise

| Condition | Disturbance normalized L2 | Recovery first 10 | Recovery last 10 |
| --- | ---: | ---: | ---: |
| current_stateful | 0.650661257 | 0.392622591 | 0.001167595 |
| learned_recurrent_error | 0.490607851 | 0.316237350 | 0.007056333 |
| learned_recurrent_error_zeroed | 0.487297715 | 0.307258255 | 0.036697759 |

Learned vs current: 24.598576%.
Learned vs zeroed: -0.679284%.

| Drive | Learned vs current | Learned vs zeroed |
| --- | ---: | ---: |
| 0011 | 20.729315% | -1.875611% |
| 0039 | 27.903761% | 0.419022% |

| Layer | Current | Learned | Zeroed |
| ---: | ---: | ---: | ---: |
| 1 | 0.501351438 | 0.543043448 | 0.537632121 |
| 2 | 0.690264312 | 0.666891314 | 0.655046186 |
| 3 | 0.644353668 | 0.554452222 | 0.526470893 |
| 4 | 0.668879399 | 0.421117659 | 0.413002680 |
| 5 | 0.748457470 | 0.267534613 | 0.304336695 |

## brightness_shift

| Condition | Disturbance normalized L2 | Recovery first 10 | Recovery last 10 |
| --- | ---: | ---: | ---: |
| current_stateful | 0.272876398 | 0.156972760 | 0.000737066 |
| learned_recurrent_error | 0.184116804 | 0.102595143 | 0.000840261 |
| learned_recurrent_error_zeroed | 0.197603509 | 0.114548865 | 0.020621419 |

Learned vs current: 32.527399%.
Learned vs zeroed: 6.825134%.

| Drive | Learned vs current | Learned vs zeroed |
| --- | ---: | ---: |
| 0011 | 28.913155% | 5.344403% |
| 0039 | 35.859688% | 8.291103% |

| Layer | Current | Learned | Zeroed |
| ---: | ---: | ---: | ---: |
| 1 | 0.269836823 | 0.272410645 | 0.272025146 |
| 2 | 0.250077559 | 0.231147184 | 0.237122130 |
| 3 | 0.242613433 | 0.178593842 | 0.184399914 |
| 4 | 0.284969827 | 0.142576447 | 0.152715387 |
| 5 | 0.316884348 | 0.095855902 | 0.141754966 |

## Overall Macro Mean

| Condition | Disturbance normalized L2 | Recovery first 10 | Recovery last 10 |
| --- | ---: | ---: | ---: |
| current_stateful | 0.461768828 | 0.274797676 | 0.000952331 |
| learned_recurrent_error | 0.337362328 | 0.209416246 | 0.003948297 |
| learned_recurrent_error_zeroed | 0.342450612 | 0.210903560 | 0.028659589 |

Learned vs current: 26.941295%.
Learned vs zeroed: 1.485845%.
