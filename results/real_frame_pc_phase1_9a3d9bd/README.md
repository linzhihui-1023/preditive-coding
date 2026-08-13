# Real-frame Predictive Coding Phase 1

Frozen inference on KITTI Val drives 0011 and 0039. The trajectory is 40 clean frames, 80 frames of persistent Gaussian blur, then 40 clean recovery frames. Frozen Test drives 0051/0056 were not read.

| Condition | Disturbance normalized L2 | Recovery normalized L2 | Recovery last 10 |
| --- | ---: | ---: | ---: |
| feedforward | 0.890284632 | 0.000000000 | 0.000000000 |
| representation_memory_only | 0.803649702 | 0.135519479 | 0.000850533 |
| pc_no_error | 0.784450073 | 0.126158581 | 0.001216764 |
| pc_dynamic_error | 0.784296992 | 0.126198921 | 0.001205088 |

## Adjacent Contributions

A -> B representation memory: 9.731150%.
B -> C top-down feedback: 2.389054%.
C -> D dynamic error: 0.019514%.
The A -> C improvement is mainly from representation_memory.

## NO-GO

PC-dynamic-error changed disturbance distance relative to PC-no-error by 0.019514% (GO threshold: at least 5.0% with both drives improving).

Dynamic recovery distance changed from 0.427156845 in the first 10 recovery frames to 0.001205088 in the last 10.
