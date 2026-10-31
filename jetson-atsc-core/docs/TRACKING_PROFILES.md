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

Stage B-2 attempted G2 using the feature-branch product `DetectorTracker`, with
the profile argument omitted (default `baseline`), on the full P15 frame path
against the saved A1 live output. The memory guard aborted after 75.769491 s:
peak sampled unit RSS was 1,483,497,472 bytes and minimum MemAvailable was
1,179,385,856 bytes, below the required 1.2 GB. An interrupted temporary output
retained a recoverable prefix of 673 of 1,200 frames: all 6,385 instances had
zero box error, identical numeric IDs and stable flags, and exact class and
confidence agreement (6,385/6,385 each; ID agreement 1.0). This prefix does not
pass the full-clip G2 gate; complete-clip agreement remains unmeasured.

G3a (product `low_score` on P15 versus the saved offline L1 replay) and G3b
(the same on JH15, frame step 2, resized to 960×540 through the harness registry)
were not run. The session was interrupted during the recovery wait; on resumption
the board had restarted, preventing further runs under the saved-trace protocol. Instances
compared, box error, ID agreement, and class/confidence agreement for both G3
units are unmeasured. G2/G3 product-profile live fidelity therefore remains
unverified.

Other validation still unverified:

- G5 replay-session smoke and export through the real ObservationSession path.
- Full-suite parity (targeted parity passed; the full run was incomplete).
- Stage C board timing.
- Field footage.

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
