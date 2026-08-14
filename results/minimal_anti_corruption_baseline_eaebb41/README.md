# Minimal Anti-corruption Baseline

Evaluator-only comparison on Val drives 0011/0039. All models use the same frames, corruption implementation, and paired clean/corrupted reference protocol. Frozen Test drives 0051/0056 were not read.

| Model | Blur disturbance | Brightness disturbance | Recovery first10 | Recovery last10 |
| --- | ---: | ---: | ---: | ---: |
| frozen_vgg16 | 0.890284630 | 0.271695565 | 0.000000000 | 0.000000000 |
| original_predify | 0.849276881 | 0.237481374 | 0.000000000 | 0.000000000 |
| temporal_only | 0.517564677 | 0.184214546 | 0.239301884 | 0.035414099 |

## gaussian_blur

Predify vs VGG improvement: 4.606139%.
Temporal-only vs Original Predify improvement: 39.058193%.
Temporal-only vs VGG improvement: 41.865258%.

| Drive | Frozen VGG16 | Original Predify | Temporal-only |
| --- | ---: | ---: | ---: |
| 0011 | 0.886308037 | 0.829467372 | 0.522200407 |
| 0039 | 0.894261223 | 0.869086390 | 0.512928947 |

## brightness_overexposure

Predify vs VGG improvement: 12.592841%.
Temporal-only vs Original Predify improvement: 22.429897%.
Temporal-only vs VGG improvement: 32.198177%.

| Drive | Frozen VGG16 | Original Predify | Temporal-only |
| --- | ---: | ---: | ---: |
| 0011 | 0.270142815 | 0.235929674 | 0.185761141 |
| 0039 | 0.273248316 | 0.239033073 | 0.182667951 |

