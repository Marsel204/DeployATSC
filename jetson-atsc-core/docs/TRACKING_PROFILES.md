# Observation tracking profiles

`baseline` is the default. `low_score` is an opt-in tracking profile with measured
benefit on one fixed night-camera recording. It has **not been field validated**.
The owner should judge it on representative footage before using its measurements.
No default preset changes.

## Enable and revert

Add this top-level field to the JSON sent to `POST /api/observation/start`, alongside
the existing system/acquisition calibrations and other run settings:

```json
{"tracking_profile": "low_score"}
```

The same field can be added to a **copy** of a start-payload preset used by
`scripts/launch_demo.py --preset <copy.json>`. Keep the original preset and its
geometry unchanged. There is no dashboard selector. Unknown profile names return
HTTP 422. To revert, start a new run with `"tracking_profile": "baseline"`, or omit
the field. A running session keeps the choice it started with.

Only Camera 2 (acquisition) receives the selected profile. Camera 1 still-image
detection and the tracking exception's plain-predict fallback keep their configured
confidence, 0.28 in observation sessions. Injected/demo detectors without the
profile capability still work; evidence marks the profile as unapplied to an
injected component, with unknown effective thresholds rather than a fabricated hash.

## What changes

Baseline preserves the existing `model.track` call: `tracker="bytetrack.yaml"`
and `conf=conf_threshold` (0.28 in observation sessions). `low_score` passes
`conf=0.10` and [this ByteTrack YAML](../configs/trackers/low_score.yaml).
That admits detections to ByteTrack's second association stage. Detections below
0.25 can sustain existing tracks; the high/birth thresholds remain 0.25. The
newly admitted 0.25–0.28 band can also satisfy those unchanged birth thresholds.

The YAML retains `track_high_thresh=0.25`, `new_track_thresh=0.25`,
`track_low_thresh=0.10`, `track_buffer=30` **frames**, `match_thresh=0.8`, and
`fuse_score=true`. NMS stays class-aware at IoU 0.5. There are no new API dials.
The buffer's duration therefore still depends on source FPS. Association timeout
stays 1.0 s under the default RunPolicy; velocity gaps, epoch resets, row accounting,
signal phases and prediction/storage behavior are unchanged. Missing time remains
unknown; no stopped/red time is imputed and no identity stitching is added.

## Evidence and limits (30–31 October 2026)

These are CPU-replica results using cached real detections and the product's
velocity/VehicleWaits code. The replica was checked against live ByteTrack output.
Rows are assigned to halves by first observation and retain their full histories;
these totals are vehicle-seconds, not clip duration or independently labelled truth.
The profile was chosen after seeing the sweep results and confirmed on windows
not used to pick it; this was a post-hoc choice, not a pre-registered profile.

| Window | Unknown seconds B → low_score | Observed stopped seconds B → low_score | Valid-time fraction B → low_score |
|---|---:|---:|---:|
| JH15 first half (selection context) | 77.8 → 33.0 | 291.6 → 345.5 | 0.872 → 0.950 |
| JH15 second half (confirmatory) | 47.9 → 32.5 | 119.9 → 147.3 | 0.842 → 0.898 |
| JH05 (FPS robustness) | 72.8 → 51.2 | 142.2 → 176.2 | 0.766 → 0.848 |
| P15 (handheld safety) | 72.9 → 76.9 | 10.8 → 11.0 | 0.8142 → 0.8141 |

The confirmatory criteria passed: unknown seconds fell at least 20% on the two
night-camera decision windows without reducing valid-time fraction or stopped
seconds; zero persistent detection chains were lost on JH15/second, JH05, V3 and
V6; jump counts stayed within baseline +2 on every window. Indicative elapsed
tracker medians stayed within baseline +2 ms with one BLAS thread.
Independent review by Sonnet 5.5 re-derived these confirmatory numbers from the
raw replay rows. Across JH15/all, unknown seconds fell 125.7 → 65.5, observed
stopped seconds rose 411.5 → 492.9, and valid-time fraction rose 0.862 → 0.933.

The profile reduces dropouts and unknown time but does **not** reduce rows or ID
churn: JH15/all had 156 → 155 rows, JH15/second rows/min stayed 99.5, and interior
breaks/min rose 135.3 → 145.4. Handheld evidence is
neutral-to-unknown: P15 unknown time rose 5.5%; P05 persistent-chain coverage fell
0.879 → 0.867 (two lost chains). No global default change is justified. V3/V6 are
short daytime smoke clips without physical calibration. JH05 is a lower-FPS view,
not independent footage. Jackson Hole is one sparse night scene, and neither it
nor handheld Pasteur is the owner's intersection. Detection-chain coverage is a
proxy, not labelled vehicle recall.

Sonnet's independent blind audit covered all 24 new links on the fixed clips
(JH15 16, JH05 4, V6 4; V3 none). No wrong link was found: strict labels were
19 same / 0 different / 5 unsure (20.8% unsure); the adopted same-object rule
gave 20 / 0 / 4 (16.7% unsure, precision 20/20 among decidable links). The
one-label difference counts two boxes on the same traffic light as the same
object, as required for a false-merge audit. G6 is satisfied in substance, with
4–5 links still uncertain depending on that convention. L1 also lost 32 baseline
continuations versus 24 new links, consistent with unchanged rows and increased
interior breaks. This audit is not labelled vehicle recall or field validation.

Longer buffers, longer row timeouts, higher birth thresholds and other trackers
were deliberately excluded.

Stage B-2's memory-aborted G2 retained only a matching 673/1,200-frame prefix
(6,385 instances). The board subsequently stopped without a recorded shutdown,
OOM or panic. That prefix was not a full-clip pass. Stage B-3 completed the
following validation on 31 October, on boot
`a09f04b1-f75d-4c7d-a748-e333085af3cc`, without the owner's demo running.
Saved A1/live and CPU-replay references from the earlier boot were read-only;
cross-boot fidelity comparisons were explicitly allowed for this stage.

| Live product gate | Frames | Instances | Maximum box error px | ID agreement | Exact class/confidence agreement |
|---|---:|---:|---:|---:|---:|
| G2 default profile vs saved A1 live P15 | 1,200 | 10,383 | 0 | 10,383/10,383 numeric IDs | 10,383/10,383 each |
| G3a low_score P15 vs offline L1 | 1,200 | 11,079 | 0 | 11,073/11,079 partition (99.945843%) | 11,079/11,079 each |
| G3b low_score JH15 vs offline L1 | 1,610 | 16,241 | 0 | 16,241/16,241 partition | 16,241/16,241 each |

All three meet the registered criteria: boxes within 1e-3 px after Ultralytics
clipping, identical numeric IDs for G2, and ID-partition agreement >=99.5% for
G3. G2 omitted the profile argument. JH15 used frame step 2 and resize 960×540
through the registered calibration path. Timestamps and source sequences agree;
per-instance stable flags have zero mismatches. G3a's six matching unstable-ID
instances remain in the denominator and are excluded from the ID numerator;
every seven-column vehicle row, including those six, matches exactly.

There is a retained G3a harness-summary discrepancy on **empty frames**:
frame-level `stable` booleans differ on 74 frames for saved-vs-product,
80 for same-boot raw-vs-product, and 6 for saved-vs-raw. The product harness uses
`all([])` and the original raw unit masks unconfirmed tracks; these empty
summaries do not describe any vehicle instance. A conservative same-boot check
regenerated the direct Ultralytics L1/P15 reference using the original raw-unit
call/conversion. Comparison against it also passed: 11,079 instances, 0px,
11,073/11,079 partition, exact class/confidence and zero per-instance stable
differences. Both comparisons and all empty-summary differences are retained;
no acceptance tolerance or comparator predicate was changed.

G5a and G5b ran the real `ObservationSession` to P15 EOF using the preset's
unchanged geometry, `step_mode=false`, and isolated temporary storage. G5a
requested `low_score`; G5b omitted the profile field. Both processed 1,200
acquisition frames, finalized, released resources, built valid ZIP exports and
reopened them successfully. The vehicle tables contained **407 low_score /
403 baseline rows**; exports were **86,040 / 83,637 bytes**. Their manifests
stayed immutable and agreed with execution provenance on name, requested choice,
applied roles `['acquisition']`, tracker YAML path/SHA256 and effective thresholds.
Low-score acquisition conf was 0.10; system/predict conf stayed 0.28. Default
acquisition/system conf was 0.28. System inference calls were zero in both runs.
G5 did not separately measure box/ID/class/confidence fidelity.

| Session telemetry (indicative only) | Inference mean ms | Inference median ms | Steady playback FPS | Playback factor |
|---|---:|---:|---:|---:|
| low_score | 43.930906 | 43.340558 | 14.31 | 0.95 |
| default baseline | 43.317405 | 42.762156 | 14.38 | 0.96 |

These figures use the sessions' own timings: mean/median over all 1,200
acquisition calls, including cold start and velocity update; steady FPS/factor
are medians of active recent-five-second telemetry at source times 10–70s
(61 low_score / 62 baseline samples). Late-frame share is not exposed.
Clocks were **unpinned**, with CPU `schedutil` and GPU `performance` governors.
The runner applied `nice -n 10` both to supervisors and their worker commands,
giving workers effective niceness **19** through additive inheritance. Runtime
numbers are indicative at that priority, from one serial run per profile on
one boot; they do not establish profile overhead or real-time qualification.
All six GPU units, including the raw-reference diagnostic, completed under
four minutes without a memory/RSS abort or new relevant kernel error.

Full **product-suite parity passed** as the final validation step:
**638 passed / the same 75 inherited failures / 1 skipped**, in **388.965863s**.
All 28 added tests passed; failing identities exactly equal the saved baseline,
with zero new or resolved failures. All 70 product test files ran: 66 nonbrowser,
then four browser files as a separate final pass. The historical 616-pass Stage
4c command additionally included six external review-only tests; those six
were not included here. Product-only baseline passes are 610, plus 28 added
cases gives 638. This is parity with an inherited failing suite, not a green
suite. Quiet-start and the 20-minute/memory watchdog conditions were met.

Still unverified: **Stage C authoritative board timing under pinned clocks and
independent field footage**. Both remain with the owner.

## Evidence binding

New run manifests add `tracking_profile`: requested/effective name, application
status, applied roles, tracker YAML path/SHA256, and effective thresholds including
tracking and plain-detection confidence. `execution.detectors.<role>.tracking_profile`
records each component's binding; `execution.tracking_profile` also carries the
run-level requested choice and applied roles. Repository YAML paths are relative to the core;
the installed default is identified as `ultralytics/cfg/trackers/bytetrack.yaml`
alongside the existing package version. Existing source/model hashes remain.
Run creation freezes this metadata; reset, sampling and export do not rewrite the
manifest. Historical runs lacking the field remain readable and are not relabelled.
