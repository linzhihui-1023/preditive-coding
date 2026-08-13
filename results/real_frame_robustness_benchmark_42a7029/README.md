# KITTI Real-frame Robustness Benchmark

Val drives 0011/0039 only. Every model is compared with its own paired clean trajectory under a 40-clean/80-corruption/40-recovery protocol. Values are normalized representation deviations, not mCE.

Original Predify uses the legacy pvgg PCoder path and independent per-frame t=0..10 internal inference. It is not current-stateful.

| Model | gaussian_blur | gaussian_noise | brightness | motion_blur | contrast | fog | jpeg_compression | mean_normalized_representation_deviation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen_vgg16 | 0.757654998 | 0.653358877 | 0.194903249 | 0.737504105 | 0.365978110 | 0.226278462 | 0.557365998 | 0.499006257 |
| original_predify | 0.716148451 | 0.535334964 | 0.168492414 | 0.667429700 | 0.334295628 | 0.201913237 | 0.464284787 | 0.441128454 |
| current_stateful | 0.665488582 | 0.525299476 | 0.191541171 | 0.600955211 | 0.354961933 | 0.214037226 | 0.370445343 | 0.417532706 |
| learned_recurrent_error_zeroed | 0.437785831 | 0.386890857 | 0.139895083 | 0.410381381 | 0.273875571 | 0.158903629 | 0.238413200 | 0.292306507 |
| learned_recurrent_error | 0.420092229 | 0.387541873 | 0.130466308 | 0.387373548 | 0.259958758 | 0.148263839 | 0.220550219 | 0.279178111 |

Learned recurrent error improved over current-stateful by 33.1362% overall
and at every corruption/severity combination. It improved over the same-
checkpoint zeroed control by 4.4913% overall and for six of seven corruptions.
Gaussian noise was the exception: learned was 0.1683% worse than zeroed on the
three-severity mean, including 0.3367% and 0.6793% regressions at severities 2
and 3. The recurrent transition has no corruption-level failure against
current-stateful here, but the dynamic-error-input benefit does not generalize
to i.i.d. Gaussian noise.

## gaussian_blur

### Severity 1

| Model | Disturbance | Recovery first 10 | Recovery last 10 |
| --- | ---: | ---: | ---: |
| frozen_vgg16 | 0.569243084 | 0.000000000 | 0.000000000 |
| original_predify | 0.530167655 | 0.000000000 | 0.000000000 |
| current_stateful | 0.493510638 | 0.275019372 | 0.000672482 |
| learned_recurrent_error_zeroed | 0.319115519 | 0.188002378 | 0.036063400 |
| learned_recurrent_error | 0.305577681 | 0.172905899 | 0.003390001 |

### Severity 2

| Model | Disturbance | Recovery first 10 | Recovery last 10 |
| --- | ---: | ---: | ---: |
| frozen_vgg16 | 0.813437277 | 0.000000000 | 0.000000000 |
| original_predify | 0.769000819 | 0.000000000 | 0.000000000 |
| current_stateful | 0.718658849 | 0.389748556 | 0.001041848 |
| learned_recurrent_error_zeroed | 0.472803305 | 0.275159321 | 0.040122936 |
| learned_recurrent_error | 0.453150039 | 0.252604754 | 0.004158981 |

### Severity 3

| Model | Disturbance | Recovery first 10 | Recovery last 10 |
| --- | ---: | ---: | ---: |
| frozen_vgg16 | 0.890284632 | 0.000000000 | 0.000000000 |
| original_predify | 0.849276880 | 0.000000000 | 0.000000000 |
| current_stateful | 0.784296260 | 0.427153717 | 0.001206349 |
| learned_recurrent_error_zeroed | 0.521438669 | 0.301188880 | 0.045429500 |
| learned_recurrent_error | 0.501548966 | 0.280066796 | 0.004235435 |

## gaussian_noise

### Severity 1

| Model | Disturbance | Recovery first 10 | Recovery last 10 |
| --- | ---: | ---: | ---: |
| frozen_vgg16 | 0.473228072 | 0.000000000 | 0.000000000 |
| original_predify | 0.380596731 | 0.000000000 | 0.000000000 |
| current_stateful | 0.381348763 | 0.219153090 | 0.000547726 |
| learned_recurrent_error_zeroed | 0.274256466 | 0.174291107 | 0.026083872 |
| learned_recurrent_error | 0.271555436 | 0.171647445 | 0.005439039 |

### Severity 2

| Model | Disturbance | Recovery first 10 | Recovery last 10 |
| --- | ---: | ---: | ---: |
| frozen_vgg16 | 0.672465651 | 0.000000000 | 0.000000000 |
| original_predify | 0.552102110 | 0.000000000 | 0.000000000 |
| current_stateful | 0.543892211 | 0.322489673 | 0.000896405 |
| learned_recurrent_error_zeroed | 0.399118390 | 0.254228515 | 0.032553493 |
| learned_recurrent_error | 0.400462333 | 0.257522785 | 0.006280074 |

### Severity 3

| Model | Disturbance | Recovery first 10 | Recovery last 10 |
| --- | ---: | ---: | ---: |
| frozen_vgg16 | 0.814382907 | 0.000000000 | 0.000000000 |
| original_predify | 0.673306051 | 0.000000000 | 0.000000000 |
| current_stateful | 0.650657453 | 0.392616085 | 0.001170747 |
| learned_recurrent_error_zeroed | 0.487297715 | 0.307258255 | 0.036697759 |
| learned_recurrent_error | 0.490607851 | 0.316237350 | 0.007056333 |

## brightness

### Severity 1

| Model | Disturbance | Recovery first 10 | Recovery last 10 |
| --- | ---: | ---: | ---: |
| frozen_vgg16 | 0.113109696 | 0.000000000 | 0.000000000 |
| original_predify | 0.096360886 | 0.000000000 | 0.000000000 |
| current_stateful | 0.105876292 | 0.062893472 | 0.000321555 |
| learned_recurrent_error_zeroed | 0.078531274 | 0.048592816 | 0.014232495 |
| learned_recurrent_error | 0.073748691 | 0.042143748 | 0.000374516 |

### Severity 2

| Model | Disturbance | Recovery first 10 | Recovery last 10 |
| --- | ---: | ---: | ---: |
| frozen_vgg16 | 0.199904488 | 0.000000000 | 0.000000000 |
| original_predify | 0.171634983 | 0.000000000 | 0.000000000 |
| current_stateful | 0.195872549 | 0.113378013 | 0.000535347 |
| learned_recurrent_error_zeroed | 0.143550466 | 0.086106139 | 0.020425502 |
| learned_recurrent_error | 0.133533430 | 0.075024306 | 0.000648191 |

### Severity 3

| Model | Disturbance | Recovery first 10 | Recovery last 10 |
| --- | ---: | ---: | ---: |
| frozen_vgg16 | 0.271695565 | 0.000000000 | 0.000000000 |
| original_predify | 0.237481373 | 0.000000000 | 0.000000000 |
| current_stateful | 0.272874671 | 0.156972332 | 0.000737018 |
| learned_recurrent_error_zeroed | 0.197603509 | 0.114548865 | 0.020621419 |
| learned_recurrent_error | 0.184116804 | 0.102595143 | 0.000840261 |

## motion_blur

### Severity 1

| Model | Disturbance | Recovery first 10 | Recovery last 10 |
| --- | ---: | ---: | ---: |
| frozen_vgg16 | 0.596016479 | 0.000000000 | 0.000000000 |
| original_predify | 0.525160117 | 0.000000000 | 0.000000000 |
| current_stateful | 0.480350066 | 0.267155640 | 0.000584695 |
| learned_recurrent_error_zeroed | 0.327373455 | 0.192111809 | 0.032355532 |
| learned_recurrent_error | 0.308138564 | 0.175622543 | 0.001316025 |

### Severity 2

| Model | Disturbance | Recovery first 10 | Recovery last 10 |
| --- | ---: | ---: | ---: |
| frozen_vgg16 | 0.772692897 | 0.000000000 | 0.000000000 |
| original_predify | 0.700799202 | 0.000000000 | 0.000000000 |
| current_stateful | 0.627495168 | 0.352124825 | 0.000844083 |
| learned_recurrent_error_zeroed | 0.428656316 | 0.249602649 | 0.030631138 |
| learned_recurrent_error | 0.404066392 | 0.230042559 | 0.005146653 |

### Severity 3

| Model | Disturbance | Recovery first 10 | Recovery last 10 |
| --- | ---: | ---: | ---: |
| frozen_vgg16 | 0.843802937 | 0.000000000 | 0.000000000 |
| original_predify | 0.776329780 | 0.000000000 | 0.000000000 |
| current_stateful | 0.695020400 | 0.391137560 | 0.001038146 |
| learned_recurrent_error_zeroed | 0.475114372 | 0.274505445 | 0.036178621 |
| learned_recurrent_error | 0.449915687 | 0.256216236 | 0.005290571 |

## contrast

### Severity 1

| Model | Disturbance | Recovery first 10 | Recovery last 10 |
| --- | ---: | ---: | ---: |
| frozen_vgg16 | 0.194174273 | 0.000000000 | 0.000000000 |
| original_predify | 0.171883400 | 0.000000000 | 0.000000000 |
| current_stateful | 0.194257228 | 0.120713698 | 0.000564428 |
| learned_recurrent_error_zeroed | 0.150303805 | 0.090831742 | 0.016286650 |
| learned_recurrent_error | 0.139718020 | 0.084209483 | 0.000854217 |

### Severity 2

| Model | Disturbance | Recovery first 10 | Recovery last 10 |
| --- | ---: | ---: | ---: |
| frozen_vgg16 | 0.366692208 | 0.000000000 | 0.000000000 |
| original_predify | 0.329228855 | 0.000000000 | 0.000000000 |
| current_stateful | 0.355438064 | 0.214563275 | 0.000977520 |
| learned_recurrent_error_zeroed | 0.276393622 | 0.167320324 | 0.025435983 |
| learned_recurrent_error | 0.261327283 | 0.157281826 | 0.001446006 |

### Severity 3

| Model | Disturbance | Recovery first 10 | Recovery last 10 |
| --- | ---: | ---: | ---: |
| frozen_vgg16 | 0.537067849 | 0.000000000 | 0.000000000 |
| original_predify | 0.501774628 | 0.000000000 | 0.000000000 |
| current_stateful | 0.515190508 | 0.303390802 | 0.001319940 |
| learned_recurrent_error_zeroed | 0.394929287 | 0.241945929 | 0.034060577 |
| learned_recurrent_error | 0.378830972 | 0.228612847 | 0.005535258 |

## fog

### Severity 1

| Model | Disturbance | Recovery first 10 | Recovery last 10 |
| --- | ---: | ---: | ---: |
| frozen_vgg16 | 0.128737450 | 0.000000000 | 0.000000000 |
| original_predify | 0.112944172 | 0.000000000 | 0.000000000 |
| current_stateful | 0.121780212 | 0.080904274 | 0.000395415 |
| learned_recurrent_error_zeroed | 0.092548134 | 0.059530821 | 0.012870492 |
| learned_recurrent_error | 0.086061244 | 0.054984465 | 0.000456534 |

### Severity 2

| Model | Disturbance | Recovery first 10 | Recovery last 10 |
| --- | ---: | ---: | ---: |
| frozen_vgg16 | 0.230665404 | 0.000000000 | 0.000000000 |
| original_predify | 0.204087605 | 0.000000000 | 0.000000000 |
| current_stateful | 0.216918380 | 0.139802440 | 0.000633127 |
| learned_recurrent_error_zeroed | 0.162223777 | 0.100081945 | 0.013844965 |
| learned_recurrent_error | 0.151081209 | 0.093624012 | 0.000780093 |

### Severity 3

| Model | Disturbance | Recovery first 10 | Recovery last 10 |
| --- | ---: | ---: | ---: |
| frozen_vgg16 | 0.319432531 | 0.000000000 | 0.000000000 |
| original_predify | 0.288707934 | 0.000000000 | 0.000000000 |
| current_stateful | 0.303413084 | 0.190835905 | 0.000842638 |
| learned_recurrent_error_zeroed | 0.221938974 | 0.135454289 | 0.022989961 |
| learned_recurrent_error | 0.207649064 | 0.126777691 | 0.001058040 |

## jpeg_compression

### Severity 1

| Model | Disturbance | Recovery first 10 | Recovery last 10 |
| --- | ---: | ---: | ---: |
| frozen_vgg16 | 0.414989341 | 0.000000000 | 0.000000000 |
| original_predify | 0.329148706 | 0.000000000 | 0.000000000 |
| current_stateful | 0.253058540 | 0.137908242 | 0.000391237 |
| learned_recurrent_error_zeroed | 0.166345223 | 0.098022159 | 0.024543779 |
| learned_recurrent_error | 0.150988509 | 0.081624957 | 0.000595904 |

### Severity 2

| Model | Disturbance | Recovery first 10 | Recovery last 10 |
| --- | ---: | ---: | ---: |
| frozen_vgg16 | 0.549973676 | 0.000000000 | 0.000000000 |
| original_predify | 0.455875538 | 0.000000000 | 0.000000000 |
| current_stateful | 0.356153526 | 0.193322160 | 0.000553150 |
| learned_recurrent_error_zeroed | 0.233628739 | 0.133277993 | 0.030998933 |
| learned_recurrent_error | 0.213938607 | 0.115132867 | 0.000757442 |

### Severity 3

| Model | Disturbance | Recovery first 10 | Recovery last 10 |
| --- | ---: | ---: | ---: |
| frozen_vgg16 | 0.707134976 | 0.000000000 | 0.000000000 |
| original_predify | 0.607830116 | 0.000000000 | 0.000000000 |
| current_stateful | 0.502123962 | 0.276882368 | 0.001001999 |
| learned_recurrent_error_zeroed | 0.315265639 | 0.172060370 | 0.020790946 |
| learned_recurrent_error | 0.296723541 | 0.160975489 | 0.001095410 |
