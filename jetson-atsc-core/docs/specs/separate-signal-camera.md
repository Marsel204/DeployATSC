# Two-camera ANFIS observation experiment and dashboard redesign

Status: revised specification for review. This supersedes the earlier dedicated-signal-camera draft; application implementation has not started.

## Confirmed experiment

The system has two cameras with different jobs:

| Camera | Purpose | Processing and calibration |
| --- | --- | --- |
| 1 — System camera | Recommend the upcoming green duration | Lens correction, perspective/homography calibration, road containment and metric ruler, YOLO detection, PKJI traffic inputs, ANFIS inference |
| 2 — Acquisition camera | Measure what actually happens | Independent YOLO tracking, persistent vehicle IDs, red-light waiting/stopped intervals, stop-line discharge, actual signal phases and green duration; the traffic-light box belongs here |

The operator clicks **Predict next green** shortly before green begins. This captures camera 1's current valid traffic snapshot and records an ANFIS recommendation. The experiment records recommendations alongside the existing signal timing; it does not send green commands to the real controller. No third camera is required.

Camera 2 retains and expands its existing acquisition/audit role. It observes the same approach/movement as the recommendation, from its own position. Camera 1 and camera 2 may have different lens characteristics, resolutions and frame rates. Their calibrations and tracking identities remain separate.

## Repository findings and changes required

- The current dashboard and replay manager accept one video and run road detection and yellow monitoring on the same frame. They need two independent sources and camera-specific streams/state.
- `edge_daemon.py` already provides system and acquisition inputs. Acquisition currently counts discharge through a tripwire; it does not maintain per-vehicle waiting-time records.
- The daemon currently calls serial actuation when yellow triggers. Observation mode must bypass that actuation path entirely, including controller heartbeats/commands; hardware control code is not started by the observation workflow.
- `OpticalPhaseMonitor` detects yellow only. Actual red waiting intervals and green duration require explicit red/yellow/green/unknown observations. Absence of yellow cannot be treated as evidence of green.
- `DetectorTracker` can fall back to per-frame array indices when persistent tracking fails. These are not valid vehicle identities for delay measurement. Acquisition records need stable tracker identity plus loss/reacquisition quality handling.
- Existing cycle/ledger records store ANFIS duration and served counts, but not actual green, prediction timing, per-vehicle waits or measurement completeness.
- Calibration snapshots are raw while inference is rectified; each camera's drawing preview must match its processing coordinates.
- Preserve the user's edited `configs/custom_simpang5_jamika.json` and road geometry. This project is not AEK-enrolled; the existing development workflow applies.

## Dashboard redesign

Retain a dark operational dashboard with clearer spacing, typography and accessible contrast. Use cyan for camera 1 and amber for camera 2, always accompanied by text labels. Each camera has its source, calibration tools, readiness and health beside its own image. Desktop shows the views side by side; mobile stacks system first, acquisition second, without horizontal overflow.

| Area | Contents |
| --- | --- |
| Session toolbar | Experiment/run ID, Observation mode, profile, Start/Pause/Resume/Step/Stop, export |
| Panel 1: System camera | Independent source, annotated road view, road polygon/perspective/ruler tools, V_w/Q/L, snapshot freshness, prominent Predict next green button and pending recommendation |
| Panel 2: Acquisition camera | Independent source, vehicle ID and red-wait overlays, traffic-light box, waiting-area and stop-line tools, signal phase/quality, per-vehicle table and source/frame-age status |
| Cycle comparison | Cycle ID, prediction time/lead, ANFIS green, actual green, signed difference, actual served count, observed mean/median/P95 wait and incomplete-track counts |
| Experiment summary | Valid cycles, observed delay distributions, measurement coverage, chosen LOS method/status and downloadable evidence |

Use **ANFIS recommendation** and **Observed green** labels. Show actual duration as Running while green is active and finalize only when green ends. The timing difference is `recommended_green_sec - actual_green_sec`; it is a comparison, not prediction accuracy against an independently verified optimum. ANFIS output is not proof that a duration is optimal.

Calibration actions are camera-specific. Camera 1 owns the original road tools. Camera 2 owns its light box, waiting/measurement region and stop-line crossing direction. If metric velocity is used to classify stops, camera 2 needs its own ground-plane calibration; camera 1's homography cannot be applied to camera 2. A pixel-motion alternative must have a declared calibrated threshold and units. Saving calibration does not start or restart playback.

Panel 2 shows a per-vehicle ledger: experiment vehicle ID, class, tracking state, red wait, total observed stopped time, first/last seen, crossed stop line, cycles experienced and quality/completeness. Placeholder values must not masquerade as measurements. Signal Unknown, Lost tracking, Partial observation and Late prediction have explicit text states.

For local testing, allow an explicit Reuse system video option on camera 2 and mark it Shared-source test. Production uses separately configured sources. The downloaded Jamika clip supports layout/smoke checks; it does not establish two-camera synchronization or experiment validity.

The companion `separate-signal-camera-preview.html` is a standalone layout preview, not a replacement for the running app.

## Manual prediction workflow

1. Open both sources and load their independent calibration. Start one observation session; acquisition tracking continues throughout red, green and inter-cycle transitions.
2. During red, the operator clicks Predict next green. Capture the latest valid camera-1 snapshot with its frame/source timestamp, click timestamp, inference start/completion, V_w/Q/L, model version and calibration version.
3. Save one recommendation for the upcoming observed green. A request ID makes double clicks/retries idempotent; disable repeat capture while a recommendation is pending. If replacement is later supported, require an explicit action and retain the superseded record.
4. On camera 2's confirmed green onset for the selected movement, associate the pending recommendation with that cycle. Measure the actual green until the confirmed green-to-yellow transition; unknown/ambiguous boundaries mark the duration incomplete.
5. Store click lead and completed-prediction lead relative to observed green onset. If inference finishes after green begins, mark Late and exclude it from valid pre-green comparisons. Never use later camera-1 frames to reconstruct an earlier prediction.
6. Finalize cycle results independently from vehicle journeys. Vehicles still queued across cycles retain their cumulative records. A new recommendation becomes available for the next red/upcoming green.

The prediction button depends on a fresh, valid camera-1 snapshot and the intended target movement. In confirmed green it cannot create a record labeled pre-green. Unknown signal state may permit a clearly marked unaligned manual capture for diagnostics, but cannot establish a valid pre-green result without later boundary evidence. Pending records that cannot be matched are retained as unmatched/cancelled, never silently assigned to another run.

Operator clicks are the prediction trigger; periodic and yellow-triggered automatic recommendations are not part of this experiment. No requirement for controller schedule integration remains. This interface does not promise automatic pre-green timing for later autonomous deployment.

## Acquisition measurement rules

Track only vehicles in the calibrated target movement/measurement region. IDs are unique within `(experiment, acquisition camera, tracker epoch, local track ID)` and remain stable while the tracker can reliably associate the vehicle. Do not match IDs across cameras; cycle and movement/time alignment is sufficient for this experiment.

For vehicle i, accumulate **red waiting time** only over valid observed intervals where it is within the measurement region, confirmed stopped under the declared motion threshold and the selected movement's signal is red. Integrate timestamp durations, including clipping intervals at signal transitions. This differs from time merely present during red while moving, which may be exported separately as red exposure.

Also record **total observed stopped time**, including waiting that continues after green. The red-wait counter stops at green even if the vehicle remains queued; the total-stopped counter continues until movement resumes. Use configurable persistence/hysteresis to avoid jitter creating many stops. Never reset a vehicle's cumulative wait just because a new signal cycle starts.

Use the actual elapsed time, not nominal FPS or number of displayed frames. Pause and replay-speed changes do not add waiting time. Process new acquisition frames exactly once. Unknown signal intervals and long tracking gaps do not contribute assumed red wait. They contribute explicit unknown-duration/quality fields.

Brief occlusion may retain identity according to a declared association timeout, but unobserved stop state must not be invented. Longer losses or tracker resets create partial/censored records rather than a zero-wait completed vehicle. Missing stable IDs disable vehicle-level timing for the affected frames; detector-only array indices must never be exported as stable experiment IDs.

Finalize a vehicle's observed journey at a valid stop-line crossing or configured measurement exit, with the stop-line direction preventing repeated counts. Keep vehicles arriving already stopped, those visible before recording starts, and those still present at recording end flagged as partial observations. Report the completed and partial populations separately to expose survival/tracking bias. Attribute wait intervals to their observed cycles; count a discharged vehicle once in its crossing cycle even if it waited through several reds.

Actual signal monitoring is independently calibrated on camera 2. Support confirmed Red/Yellow/Green/Unknown with debounced transitions and confidence/evidence timestamps. The head ROI must identify the selected movement; use lamp-specific regions where needed to separate colors or nearby heads. Ambiguous, conflicting, occluded or flashing evidence is Unknown; it cannot create a completed actual-green interval.

## Delay, LOS and what this experiment can conclude

Export red waiting and total observed stopped time as direct measurements. These are not automatically HCM control delay. FHWA describes signal control delay as including deceleration, queuing and acceleration, and uses control delay for signalized LOS: [Traffic Signal Timing Manual, section 3.4](https://ops.fhwa.dot.gov/publications/fhwahop08024/chapter3.htm).

The LOS framework, edition, analysis period, movement scope and required inputs must be selected explicitly in the experiment policy. PKJI PCU weights in existing analytics do not by themselves establish an LOS calculation. Do not silently select HCM versus an Indonesian methodology or assign a grade from red wait alone.

If camera 2 covers the necessary entry/exit observations and a reviewed free-flow reference exists, calculate a documented travel/control-delay estimate with its assumptions. Otherwise display LOS unavailable — control-delay method/coverage required while still exporting the valid waiting metrics. Show the actual measured movement's LOS; do not extrapolate a single approach to a whole intersection. Include entering vehicles with zero delay in applicable denominators, not just stopped or discharged vehicles, and disclose incomplete coverage.

This confirmed observation-only experiment compares recommended versus actual timing and describes observed delay/LOS under actual timing. It cannot measure a causal improvement from applying ANFIS because the recommendations are not applied. Any later recommendation-based delay/LOS forecast must be labeled a simulation/model estimate and requires a separately specified, validated counterfactual method. A real treatment comparison is outside this request.

## Sources, synchronization and failure behavior

Keep existing system/acquisition production source roles. Replay accepts two independent video files; production uses supported camera/device/RTSP sources. Each role owns its lens settings, processed dimensions, ROI, tracker and measurement state. Validate geometry against that role's processed frame; source/resolution/lens changes invalidate affected calibration bindings only.

Legacy profiles remain readable. Existing lamp coordinates stay associated with their original source until explicitly recalibrated for acquisition. Preserve road homography, model weights, PKJI factors and ANFIS bounds. Add an explicit observation experiment mode; legacy control mode must not be selected by accident through compatibility defaults.

Use one experiment time base with per-source timestamp mapping and adjustable replay offset (`acquisition time = system time + offset`). Consume all newly due acquisition frames needed for phase/vehicle measurements, within bounded processing queues. Never equate frame indices between different-FPS sources or advance timers from repeated latest-frame reads. In production, document clock mapping/freshness and account for signal/acquisition latency.

Pause freezes both video-time clocks; Step advances one system frame and corresponding acquisition frames. Stop/source replacement releases both captures and state. If acquisition fails, mark measurement unavailable/incomplete; system diagnostics can continue but new valid cycle comparisons cannot. If camera 1 is stale, disable prediction. If thermal relief suspends acquisition tracking, mark affected measurements degraded/incomplete instead of reporting zero waits or complete LOS.

No serial controller or field hardware is opened by observation mode. Production validation uses mocked hardware and two independent inputs; physical actuation remains outside the user's selected experiment.

## Evidence and exports

Retain a versioned run manifest with experiment/movement IDs, source identities, source offsets, calibrations, model/settings versions, stop/phase thresholds, LOS policy and observation mode.

Export linked prediction records, phase transition/interval records, vehicle records and per-cycle summaries. Each has run/camera/cycle IDs, source timestamps, missing/partial flags and the relevant evidence references. Vehicle waiting is not reduced to the existing single cycle CSV row; keep a dedicated vehicle ledger. Repeated red episodes and censored records remain inspectable.

Cycle comparisons include recommendation, measured actual duration, signed timing difference, prediction lead/completion, input snapshot, served vehicles and wait summaries with sample sizes and eligibility rules. Group results by matching movement/run and never combine unrelated camera tracks. Existing ledgers must be preserved or migrated with a declared schema, not append incompatible columns under an old CSV header.

## Acceptance criteria

1. Camera 1 provides its own rectified road geometry, YOLO observations and valid V_w/Q/L to ANFIS. Camera 2 independently displays stable vehicle IDs and the traffic-light box; neither panel uses the other's calibration.
2. An operator click records one immutable recommendation from a contemporaneous system snapshot. Double click/retry yields one record; inference completing after green is flagged Late.
3. Observation mode produces no serial/actuation calls, including startup and heartbeat paths. Actual timing always comes from observations, never from the recommended duration.
4. A deterministic signal sequence with green from t=30 to t=55 produces an observed green of 25 seconds within the documented frame/phase-debounce tolerance and records evidence boundaries separately from confirmation latency. Missing a boundary produces an incomplete interval.
5. A vehicle stopped from t=10 to t=34 with red ending at t=30 produces 20 seconds of red wait and 24 seconds of total observed stopped time, within documented frame/motion-state tolerance. A moving vehicle present during red does not accrue stopped red wait.
6. Vehicles spanning two reds retain their identity and accumulate both valid wait episodes. Stop-line crossing counts once; camera-local IDs cannot collide in the experiment ledger.
7. Different FPS, offsets, pause/resume and Step preserve deterministic time-based measurements. Repeating a frame or pausing adds no wait. Occlusion, lost IDs, ambiguous phase, early EOF and thermal suspension produce quality/censoring evidence rather than fabricated measurements.
8. Camera-2-only red/green observations determine phase and actual green. Yellow absence alone does not imply green. Non-target movement traffic and neighboring signal heads cannot contaminate measurements.
9. An unconfigured/unsupported LOS method or insufficient measurement coverage displays unavailable with a reason. Red wait alone never produces an official control-delay LOS grade or claimed ANFIS improvement.
10. Desktop panels are side by side; at 390px they stack without overflow. Source, ROI and error controls are keyboard accessible and camera-specific. Saving does not unexpectedly start/restart replay.
11. The user's saved Jamika road profile and existing evidence survive migration. Exports can reconstruct predictions, actual phase intervals and completed/partial vehicle waits; sample sizes and run identity are visible.
12. Benchmark two-camera inference/tracking and capture backlog on this Jetson. Shared YOLO weights may be reused, but each camera has isolated tracker state. Missing a runtime performance target must be resolved with measured baseline data; dropped/late frames are reported.

## Implementation and verification surface

Expected changes include dashboard markup/state, FastAPI request/stream/telemetry contracts, dual-source replay management, acquisition tracking and delay accumulation, multi-color signal monitoring, manual prediction records, common data types and versioned export sinks. Production daemon routing needs an explicit observation-mode path that bypasses hardware startup and actuation. Reuse system/acquisition inputs; do not add a third signal input.

Tests should use short synthetic two-camera videos with controlled vehicle motion, red/green boundaries, FPS/offset differences and occlusions. Add requirements-focused tests for timestamps, tracker isolation, wait accumulation, late/double predictions, observation-mode absence of hardware calls, exported linkage, LOS eligibility, and independent calibration; preserve existing relevant analytics/ANFIS regressions. Use temporary configurations so tests never overwrite the user's profile. Existing UI lifecycle tests reference absent sample datasets; provision deterministic fixtures for new tests.

Browser validation covers both panels, manual prediction, calibration tools, stable ID/wait overlays, responsive drawing and independent errors. A longer paired field recording must validate tracking continuity, clock offsets, phase labels and vehicle-timing measurements against manual ground truth before experiment results are treated as reliable. The 37-second downloaded video is a smoke-test asset, not a sufficient dataset for LOS or impact claims.

Remaining method decisions for planning: stop-state threshold/hysteresis and gap policy, experiment movement/measurement coverage, and the specific LOS methodology/reference. The camera roles, manual trigger and recommendation-only operating mode are confirmed.
