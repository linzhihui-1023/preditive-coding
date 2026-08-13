# Error-memory Recurrent Validation

VGG, original Predify, and feedback decoders are frozen. Only the observation/error recurrent transition and signed-error encoder train. Validation uses only the three matched formal conditions on drives 0011/0039 and the unchanged 40 clean / 80 corruption / 40 recovery protocol. Frozen Test drives 0051/0056 were not read.

## gaussian_blur

| Condition | Next-frame MSE | Disturbance normalized L2 | Recovery first 10 | Recovery last 10 |
| --- | ---: | ---: | ---: | ---: |
| temporal_only | 2.334490107 | 0.517564676 | 0.351396176 | 0.049391711 |
| instant_error | 1.907473086 | 0.618940690 | 0.455200213 | 0.128084182 |
| error_memory | 2.025569703 | 0.579650802 | 0.411636176 | 0.093996459 |

Conclusion: benefit_mainly_from_temporal_recurrence.

| Drive | temporal_only | instant_error | error_memory |
| --- | ---: | ---: | ---: |
| 0011 | 0.522200405 | 0.628115711 | 0.587847014 |
| 0039 | 0.512928946 | 0.609765669 | 0.571454590 |

## brightness_overexposure

| Condition | Next-frame MSE | Disturbance normalized L2 | Recovery first 10 | Recovery last 10 |
| --- | ---: | ---: | ---: | ---: |
| temporal_only | 3.242816785 | 0.184214546 | 0.127207593 | 0.021436488 |
| instant_error | 2.767941671 | 0.218418550 | 0.159805887 | 0.048202725 |
| error_memory | 2.956473686 | 0.199469868 | 0.143912379 | 0.048761494 |

Conclusion: benefit_mainly_from_temporal_recurrence.

| Drive | temporal_only | instant_error | error_memory |
| --- | ---: | ---: | ---: |
| 0011 | 0.185761141 | 0.219222981 | 0.200687166 |
| 0039 | 0.182667951 | 0.217614119 | 0.198252570 |

Overall conclusion: benefit_mainly_from_temporal_recurrence.
