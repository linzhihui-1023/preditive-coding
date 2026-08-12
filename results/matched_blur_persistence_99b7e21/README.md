# Matched-blur persistence diagnostic

This directory contains the lightweight outputs from the inference-only
matched-persistence experiment at revision
`99b7e210f7922c747a08d57b9340162f359527ad`. It used the frozen strict
Temporal Error checkpoint from revision
`3ffbff0156dc9435fe3b060f4c9999703729ea8a`; no optimizer was created and no
network parameter changed.

Persistent and shuffled conditions used the same 11x11 Gaussian blur, the
same four sigma values (`0.75`, `1.5`, `2.25`, `3.0`), 20 frames at each
value, and 100% disturbed-frame blur occupancy. Only temporal ordering
differed. Four replicates counterbalanced sigma so every absolute video frame
saw every strength once per condition.

Drive 0005 selected higher `cos(e_t,e_(t-1))`. With that metric and direction
frozen, drive 0011 reached AUROC `0.9725` over nonoverlapping eight-frame
windows. This passes the user-defined go/no-go threshold, but the windows
reuse two video drives and are diagnostic units rather than independent drive
samples.

`per_frame.csv` stores all 2,400 causal records. `windows.csv` stores the 160
primary analysis units. `summary.json` records the protocol, exact marginal
invariants, checkpoint provenance, AUROCs, and decision.
