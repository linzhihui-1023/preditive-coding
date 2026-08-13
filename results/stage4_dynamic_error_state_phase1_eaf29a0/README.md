# Stage-4 Dynamic Prediction Error State: Phase 1

This directory contains the lightweight auditable outputs from the frozen-
network diagnostic executed at revision
`eaf29a04fc69b6d4e2e71da7dc044cca027a7278`. The offline separability ranking
was corrected and re-audited at revision
`307139167ba22f9e1197ae915a6dad6d76859417`; no network replay was performed.

## State Chain

```text
e_t^4 = Fhat_(t|t-1)^4 - F_t^4
epsilon_t^4 = 0.207 e_t^4 + 0.793 epsilon_(t-1)^4
```

The fixed epoch-7 `c887e94` Stage-4 aligned temporal-difference predictor was
not modified or trained. Each trajectory reset at its start, updated the state
once after observing the target, detached it, and performed no same-frame
iterations. The predictor did not consume `epsilon_t`, and no optimizer or
online adaptation existed.

## Protocol

- Drives: existing Val drives 0011 and 0039 only. Frozen Test 0051/0056 was
  not read.
- Corruptions: Gaussian blur, per-frame deterministic i.i.d. Gaussian noise,
  and fixed RGB bias as a limited domain-shift proxy.
- Conditions: long severity plateaus (`persistent`) versus temporally shuffled
  severity (`shuffled`).
- Each pair used the same severity multiset, occupancy, raw frames, checkpoint,
  and corruption operator. Four replicates counterbalanced each absolute
  disturbance frame across severities `0.25/0.5/0.75/1.0`.
- Each of 48 independently reset trajectories contained 40 clean baseline, 80
  disturbance, and 40 recovery transitions.
- Primary unit: nonoverlapping 8-transition disturbance window; 480 windows.
- Fixed scores: `RMS(e_t)`, scalar `EMA(RMS(e_t))`, and
  `RMS(epsilon_t)`. A matched tensor EMA was retained only as an equivalence
  audit.

## Result

| Score | Higher-is-persistent AUROC | Direction-independent separability AUROC |
| --- | ---: | ---: |
| Instantaneous `RMS(e_t)` | 0.35102431 | 0.64897569 |
| Scalar `EMA(RMS(e_t))` | 0.34848958 | 0.65151042 |
| Dynamic `RMS(epsilon_t)` | 0.50489583 | 0.50489583 |

Dynamic-state separability was `0.14407986` below instantaneous error and
`0.14661458` below scalar EMA. Its persistent-minus-shuffled score contrast
was only `+0.00137091`, compared with `-0.10242821` for instantaneous error
and `-0.09760925` for scalar EMA. The negative control directions are retained
rather than silently flipped; direction-independent AUROC reports their
ability to separate temporal organizations.

| Corruption | Instant | Scalar EMA | Dynamic state |
| --- | ---: | ---: | ---: |
| Gaussian blur | 0.85328125 | 0.86250000 | 0.51312500 |
| i.i.d. Gaussian noise | 0.71359375 | 0.73234375 | 0.50687500 |
| RGB-bias domain proxy | 0.58812500 | 0.59609375 | 0.50343750 |

Drive-level aggregate dynamic AUROC was `0.50277778` on 0011 and `0.51784722`
on 0039. The dynamic formula error and maximum difference from the matched
tensor EMA were both exactly `0.0`. With `K=1`, this dynamic state is the same
tensor EMA, as expected.

**Decision: no-go.** This fixed signed tensor state does not represent
persistent prediction failure better than instantaneous error or a scalar
error-magnitude EMA. Phase 2 selective online adaptation is not started.

## Audit

The audit verified 7680 unique frame rows, 480 windows, 48 trajectories, exact
phase and severity counts, absolute-frame counterbalancing, Stage-4 error
shape semantics, update indices `1--160`, zero scalar-EMA recurrence error,
zero dynamic formula error, zero dynamic/tensor-EMA difference, checkpoint
identity, absence of frozen Test drives, no parameter update, and clean logs.

Checkpoint SHA-256:
`8a29c40fc71f0d54a2d21306a9444407d25865a01262713d05f522b495ab7754`.

Tracked artifact SHA-256 values:

- `summary.json`: `9452c1594215d0249730c7fe99eb42c37495992461215dcb18af7f2cca54a200`
- `summary_before_separability_fix.json`: `0fc71c58eefd17b8de72c68fb968a610a90f758652fc44f48cc84f5c594dffdc`
- `per_frame.csv`: `c2ac7e894bdfbe2e378c00f61777f22876d7c902ebf9e6e20b03e4ce6ca2d5e8`
- `windows.csv`: `a98a213cd0b2d4c7c71377d73da1ce2222323de19d028a6599fdef42e0e9ea99`
- `manifest.txt`: `e5dc4057dc0781cd264f4a059a7909773a0da352976a084011a7ba17b857fcf8`

The frozen checkpoint and execution log remain under
`/tmp/predify-storage/experiments/stage4_dynamic_error_state_phase1_eaf29a0*`.
