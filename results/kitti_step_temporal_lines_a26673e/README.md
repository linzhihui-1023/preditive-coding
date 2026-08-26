# KITTI-STEP Temporal Lines

## Line A

| Path | mIoU | wIoU | mVC8 | mVC16 |
|---|---:|---:|---:|---:|
| static | 0.6552125562 | 0.8575019578 | 0.9224884819 | 0.9208394622 |
| persistence | 0.6043412404 | 0.8264099567 | 0.9209609841 | 0.9239870233 |
| predictor | 0.6090962178 | 0.8313474658 | 0.9225980296 | 0.9252708025 |

Predictor temporal signal: **GO**. Temporal semantic capability: **NO-GO**.

## Line B

| Phase | Static | Observation-only | Full Error |
|---|---:|---:|---:|
| clean | 0.5739135738 | 0.5442824103 | 0.5784776712 |
| transition | 0.5142144932 | 0.5610336481 | 0.5623331686 |
| persistent | 0.5166237873 | 0.5296264462 | 0.5310166825 |

Persistent Full minus Observation-only: `0.0013902363`. Recovery: Observation-only `0.2269629485`, Full Error `0.2512296882`. Persistent mVC8/mVC16 Full Error `0.9062362205` / `0.9055611045`.

Decision: **PARTIAL-GO**.
