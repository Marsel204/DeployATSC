# Two-camera observation experiment

The dashboard now separates the **system camera** from the **acquisition camera**. The system camera measures V_w/Q/L using its own road geometry and YOLO, run once on a single still frame per explicit capture. The acquisition camera tracks vehicle identities and waiting intervals, reads the target movement's signal and measures actual green. **Analyze frame & predict** records an ANFIS recommendation; it never commands a cabinet.

## Start the dashboard

From `jetson-atsc-core`:

```bash
.venv/bin/python scripts/launch_ui.py --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000`. Observation is loopback-only; LAN is disabled pending a provisioned gateway and the conditional access checks. The observation CLI can preset live inputs without opening them until the operator starts a calibrated run:

```bash
.venv/bin/python edge_daemon.py --operation observation --source-mode live \
  --source_sys /dev/video0 --source_acq /dev/video1
```

The ordinary daemon entry point defaults to observation. Historical cabinet operation requires the explicit `--operation legacy-control` selection and is outside this experiment. Observation startup, processing and stop do not instantiate `SignalInterface`, watchdogs or serial connections. Legacy runtime start/control/stream endpoints return 409 in observation mode. Sample/upload/configuration helpers used by this dashboard remain available.

## Set up both cameras

1. Select independent videos, upload local files, or choose Live cameras in experiment settings and enter device/RTSP sources. Both views must observe the same movement. Preview each source before calibration.
2. Camera 1: draw the road polygon, then four anchors in **Far-Left → Far-Right → Near-Right → Near-Left** order. Enter physical road dimensions. An optional two-point ruler calibrates the metric scale. The measured polygon's projected physical area determines the occupancy denominator.
3. Camera 2: draw the waiting/measurement area, its own four perspective anchors and metric scale, the traffic-light head box, and a two-point stop line. Flip the stop-line arrow to match vehicle travel. Camera 1's homography is never applied to camera 2.
4. If adjacent lamps/heads or colored backgrounds interfere, draw optional red/yellow/green lamp regions inside the head box. Darkness, conflicting colors or missing observations remain Unknown; no yellow does not mean green.
5. Save each role's profile. Saving does not start/restart playback. Choose lens correction independently, including explicit None for footage that should be processed without lens remapping. The drawing image and runtime use the same transform. Changing source, image dimensions or lens invalidates that binding.

Existing road geometry can be loaded for review using Camera 1's **Existing road geometry** selector. Saving creates a separate observation profile under `data/observation_profiles/`; it does not overwrite `configs/custom_simpang5_jamika.json`. Old road-source light boxes are not silently assigned to acquisition.

For a single-file demonstration, explicitly enable **Reuse system video for testing**. Both views then use independent calibration/tracker state on the same clip, clearly labeled Shared test. This is unavailable for live sources and is not evidence of real two-camera synchronization. The downloaded Jamika segment is available in Samples, but its 37 seconds may not contain both boundaries of a complete green cycle.

## Run and capture a recommendation

Start observation, optionally paused for inspection. Camera 2 runs YOLO/tracking continuously across red and green. Camera 1 only decodes a lightweight preview: it performs **no** background YOLO during start, playback, steps, pauses or previews. Click **Analyze frame & predict** during red shortly before the expected green. The server freezes that click's latest fresh Camera 1 frame (sequence, mapped/source timestamp, origin), click time and measured phase, then runs exactly one still-image YOLO forward on that frame and ANFIS on a background worker; Camera 2 keeps advancing meanwhile. The record keeps the frame identity, camera-analysis completion/latency/lead, ANFIS completion and the total completion lead (green onset minus ANFIS completion). Until the first click Camera 1 metrics are unavailable; afterwards the dashboard shows the last analysed frame and its age, not a live measurement.

A still image has no speed or stopped-state evidence. V_w counts and footprint occupancy L are extracted normally. Q is an explicitly labelled **single-frame spatial queue proxy** (`single_frame_spatial_proxy`: farthest visible approach-corridor vehicle from the stop line) with `complete=false`, `measured=false` and reasons `single_frame_no_motion_evidence`, `stopped_state_unknown`; the stopped-vehicle count is stored as unknown (null). Such captures can be timing-eligible but stay diagnostic, never qualified; historical records keep their original meaning.

The capture is idempotent: retries use the same request ID and never create another recommendation or another Camera 1 inference (including retries of archived captures). A new click while one capture is pending or still being analysed is rejected; there is no queue. Cancelling, stopping, replacing or a storage failure during the Camera 1 call keeps the model owned until the native call returns, and its late result is never published. One recommendation can be pending for the upcoming cycle; it can be explicitly cancelled while unmatched. It becomes a valid comparison only when inference finishes before a subsequently observed complete green interval. Green beginning during inference makes the result Late. Capturing with Unknown phase is explicitly diagnostic/unaligned rather than a valid before-green result.

Actual green is measured from its evidence onset to the observed green-to-yellow evidence boundary, after debounce confirmation. Confirmation time is recorded separately. A missing boundary, unexpected direct green-to-red transition or unknown gap produces an incomplete cycle. Actual timing never comes from ANFIS output.

Replay pause freezes video time. Step advances one system frame and all acquisition frames newly due at that experiment time; different FPS are aligned by timestamp, not equal frame numbers. Acquisition source time is system/experiment time plus the configured offset. Live Pause suspends observation while camera time continues; Resume discards the paused backlog, resets tracking identity epochs and records a gap. Live sources cannot Step/rewind.

A pending recommendation becomes Unmatched when a live pause, acquisition outage, signal observation gap or ambiguous signal breaks observation continuity. Its captured inputs and any completed inference remain in the evidence, but it cannot be assigned to a later green. Capture a new recommendation after observation resumes. A file-replay pause preserves the pending recommendation because its source time is frozen.

Stop closes both sources and the observation worker and preserves completed/partial evidence. Source failures and EOF are visible; a failed required source cannot leave a partially active startup. In live mode, stale processed road snapshots cannot predict, and capture freshness is distinct from processing lag.

## Interpret vehicle records

IDs are namespaced by run, acquisition role and tracker epoch. They stay stable while the tracker reliably associates the vehicle; this is not cross-camera re-identification. Missing tracker IDs and velocity warmup are unknown observations, not stationary vehicles with invented IDs.

- **Red wait:** observed stopped time in the selected movement's waiting region while its signal is confirmed red.
- **Total stopped:** observed stopped time in that region, including a vehicle that remains queued after green.
- **Cycles:** the intervals/cycles experienced by that vehicle; its wait is not reset at each red.
- **Stop line:** a directional crossing counted once in its discharge cycle.
- **Quality:** missing phase, tracking gaps, partial arrival/departure, recording end, or other measurement limitations.

Stop/resume hysteresis, persistence, association timeout, gap limits, phase confirmation and color area ratio are configurable pilot settings recorded in the run manifest. Validate them against manually annotated footage; they are not universal traffic-study constants. Do not fill occlusion or lost-track intervals with assumed red/stopped time. Phase ambiguity stops red-wait credit; valid total stopped observations remain separate.

The summary includes observed zero-wait tracks and exposes partial/in-progress records. The displayed observed mean/median/P95 includes incomplete lower bounds; complete journeys are reported separately. These sample populations must not be substituted for an unobserved full entering population.

When a lost track returns after the association timeout, its replacement segment remains a partial arrival even if it is moving and subsequently crosses the stop line. It is retained in observed-track summaries and excluded from complete-journey statistics.

## Export and research scope

**Export evidence** downloads a ZIP containing `manifest.json`, `evidence.json`, and prediction/phase/vehicle/comparison CSVs. Detailed vehicle intervals remain in the JSON. Working run artifacts also live under `data/experiments/<run_id>/`, with schema-2 checkpoint/archive storage (described below). The manifest records sources, source offsets, camera profiles/revisions, measurement policy, model hashes, package/provider versions and software-source hashes. Existing historical CSV schemas are preserved.

Cycle difference is **recommended green − actual green**. It is a timing comparison, not error against an independently established optimum. ANFIS recommendations are not applied in this experiment, so the run describes observed timing/delay rather than a causal treatment improvement.

LOS intentionally remains **ungraded** until the study selects its framework/edition, movement/analysis period, control-delay measurement and sufficient coverage. Red wait alone does not establish control delay or a whole-intersection LOS. No counterfactual ANFIS delay/LOS is fabricated.

## Validation and current performance limits

Automated checks cover independent geometry, persistent IDs, exact time-based waits, debounced phase evidence, manual/double/late predictions, replay offsets/FPS, live clock/queue behavior, cleanup, API/export linkage and browser interactions at 1440px and 390px. A paired synthetic experiment reproduces green t=30–55 (25 s), stop t=10–34 (24 s), and red through t=30 (20 s red wait).

In the **14 October 2026 historical smoke**, the actual `Final.pt` and ANFIS ONNX models were exercised on the downloaded 1080p Jamika clip in two independent replay pipelines. Both produced road/track evidence and an ANFIS recommendation. This setup currently runs below real time: the final cold-start plus 11-pair smoke took 11.6 s; ten post-startup pairs took 3.2 s. Last measured inference was about 103 ms for system and 69 ms for acquisition, and process-wide peak resident memory was about 2.13 GiB. These are local smoke observations, not a validated throughput target. The installed PyTorch build warns about this Orin GPU architecture; a CUDA-available flag does not establish compatible GPU execution. Runtime telemetry reports the actual predictor device, per-role inference time, drops, and live processing age.

Before field measurements, use a compatible tested inference stack, benchmark both real inputs, verify clock/source offset and calibrations, and compare signal/vehicle annotations to manual ground truth. Replay timing remains source-time based even when processing is slower than playback. See [implementation validation](OBSERVATION_VALIDATION.md) for test results and the pre-existing asset-dependent checks that cannot pass in this checkout.

## Phase A measurement qualification (26 October 2026)

New runs emit schema 2 manifests and evidence. JSONL/CSV collection names and
existing numeric column meanings remain; new fields describe input quality,
qualification and snapshot revisions. Existing run files are not rewritten.
Old records without qualification remain unknown; they are never promoted by a
reader. Schema 2 metadata alone does not establish transactional storage; the
implemented Phase B protocol is identified separately below.

Observation enables strict queue extraction. Stable identity, valid velocity and
persistence are required for queue evidence. `metrics.quality.Q.state` distinguishes
`measured`, `measured_zero`, `partial` (a lower bound), and `unavailable`. Unavailable
Q is JSON null, and inference is not run with a substituted zero. An observed empty
corridor can establish measured zero. Counts (`V_w`) and footprint occupancy (`L`)
remain independently available. Shared legacy extractor callers retain their
numeric default behavior and receive `legacy_unverified` queue quality.

Prediction `eligible` retains timing/backend semantics: confirmed measurement RED
at capture, positive inference-completion lead, and successful backend. Comparison
`eligible` also requires complete observed green. `qualified` additionally requires
complete input-quality evidence under the persisted `qualification.policy`.
New captures use `motion-phase-scale-v2`, which also requires explicit scale evidence
and reviewed geometry bindings for both cameras. Existing `motion-phase-v1` captures
keep their original motion/phase meaning; missing-policy records remain unknown.
Old/default profiles stay diagnostic until reviewed measurement evidence is supplied.
Model-domain status remains unknown; qualification does not establish field accuracy.
`qualification.state` and `qualification.reasons` explain diagnostics. Missing
quality stays unknown. Display phase and measurement phase, candidate onset and
last transition/confirmation evidence are frozen separately with each capture.

All prediction representations in a snapshot use one `prediction_revision` view;
`snapshot_revision` also includes engine observations and asynchronous completion.
These are telemetry revisions, not fsync acknowledgments. `prediction_busy` is a
capture-admission indicator, not proof that native inference resources released.

Directional crossings during UNKNOWN retain time/direction/point evidence with
null cycle attribution. Known stopped time still accrues during unknown phase,
without red-wait or guessed-cycle credit. Summary unknown durations are sums of
vehicle-seconds; motion and phase unknown can overlap. Qualified-comparison fraction
uses timing-eligible comparisons as its denominator; partial fraction uses all
observed track records. Zero denominators yield null. Observed wait means include
partial lower bounds, with complete-journey populations reported separately.

Both launchers default to `127.0.0.1`; observation rejects reload and explicitly
uses one worker. Local mode/route isolation is implemented as described below.
Conditional LAN authentication remains disabled. No controller actuation is added.

### Lifecycle and ownership (local observation)

Observation launchers keep one loopback worker and select observation mode on the actual
application. Legacy session start/control/telemetry and legacy video streaming return
409 in that mode; sample videos, uploads, configuration and calibration helpers remain
available. Direct application imports retain the explicit legacy compatibility mode.
Requests require an allowed Host and, when supplied, an allowed Origin. CORS has no
credentialed wildcard. LAN gateway authentication and source policy remain unimplemented;
the observation launcher rejects non-loopback bindings.

Control responses retain `state` and add `requested_state`, `applied_state`, `control_seq`
and `applied_control_seq`. Controls are queued in order. A blocked packet may finish in
its existing epoch; only the worker applies tracking resets and phase/wait boundaries.
Live resume drops its remaining local batch and capture backlog before reset. Replay
pause preserves its source-time backlog; Step finishes one system-frame boundary,
including all acquisition packets due for that frame, without crediting wall-clock pause.

A replacement reserves ownership and validates static policy, calibration, lane-metrics
configuration, source, model availability and writable storage before ending the current
observation or legacy replay. It allocates replacement models only after resource handoff.
Resource acquisition failure then reports the previous run ended and the replacement failed;
it cannot restore the previous live run. Concurrent starts and unreleased owners produce
conflicts. One process-wide owner covers both runtimes and every manager: a start conflicts
with any unreleased owner other than the caller's own previous session, and a failed
startup that still owns a resource stays registered until its release is confirmed.

A source preview serves the observation owner's clean cached frame for that owner's source,
or conflicts. Canonical device/file spellings identify ownership. Legacy replay's annotated
JPEG is not a calibration frame: its source conflicts until released. Both snapshot HTTP
entrypoints use this rule and recheck ownership under a preview lease before opening.
A new capture has a reachable preview owner from acquisition through read and successful
release. Release failure retains that owner, its capture and its error; a daemon retries
with backoff from 0.02 to 0.5 seconds. Stop wakes/waits for cleanup; manager shutdown reports
`preview_release_pending` while release is unconfirmed. A blocked native read/release is
never cancelled. Different sources may preview concurrently; the same source conflicts
while its preview is pending. No runtime start may reserve while any preview is owned, and
no new preview capture opens during a start reservation.

Telemetry includes `lifecycle` and `last_start`. `stop_requested`, `measurement_sealed`,
`resources_released` and `finalized` are separate facts. Capture readers release their own
captures after read exit; inference and read calls cannot be forcibly cancelled here.
A stopped admission slot does not prove the prediction future has exited. Models remain
owned until native tracking/prediction calls exit; capture release does not gate them.
Each capture and detector release is attempted even if another raised. A failed release
is recorded in `release_error`, keeps `resources_released`/`finalized` false and is retried
on a paced loop (a later `stop()` wakes the retry). The ASGI lifespan closes both managers with the cooperative
`ATSC_SHUTDOWN_TIMEOUT_SEC` budget (default 5 seconds), and records its last-known result
in `app.state.shutdown_report`. Pending native work retains its owner after the deadline;
this deadline does not guarantee interpreter termination.

Failed replay/live pair constructors propagate every acquired partial leaf through the
failed session, including a leaf whose assignment never completed. Acquisition errors
remain the original errors when cleanup also fails. Release retries remain independent
and idempotent. Startup disposal clears exited locals throughout exception causes, contexts
and grouped errors, with cycle protection; traceback locations and pending resource owners
remain available. These rules do not add hard cancellation of native work.

`finalized` means resource cleanup plus an acknowledged final generation of the
typed archive described below (`durable_finalized`). A final write publishes no premature
finalized claim; a failure reports `unclean`. (Before 27 October 2026 this paragraph
described the earlier legacy snapshot store; those runs remain readable as `legacy`.)

### Phase B evidence storage (27 October 2026)

New run manifests keep `schema_version: 2` and identify the layout separately with
`storage_protocol: typed-archive-v1`. Phase A also wrote schema 2, so schema alone
is never used to infer this protocol. Runs without this marker, including v1 and
plain v2 `legacy_snapshot` evidence, remain readable and are never rewritten.

For new runs, `archive.jsonl` contains typed sealed interval fragments and finalized
records. `evidence.json` is an authoritative checkpoint, not the full JSON export.
It pins `committed_bytes`, `generation`, `snapshot_revision`, bounded retained state
and incremental aggregates. Commit appends and fsyncs archive bytes, fsyncs a temporary
checkpoint, atomically replaces the checkpoint and fsyncs the run directory. An
uncommitted tail is ignored by readers and removed at the next append. Retrying an
uncertain replace confirms the same generation without appending its batch twice.
Readers pin one checkpoint and never combine files from different generations.
`manifest.json` is written last, after the empty archive and generation 0 are durable,
so a run is discoverable only when coherent. A failed or interrupted creation never
lists a run or blocks later starts; a listed run with missing/corrupt committed bytes
still blocks new admission until repaired.

A commit in flight never suspends measurement admission or captures. Acknowledgment
evicts only the rows and interval prefixes it prepared, which can no longer change;
evidence admitted during the commit stays live for the next generation. A routine
worker commit does not wait behind another commit. Only an actual storage error
suspends admission. Pending captures (`predicting`/`awaiting_green`) can be cancelled;
terminal captures are immutable.

Native track identity `(tracker_epoch, local_id)` is retired only by the tracker epoch.
A track's first row uses `run:acquisition:epoch:local_id`; each later reacquisition of
the same identity in the same epoch is `...:segment-N` (N = rows already seen) and is
left-censored. A crossed identity that reappears is ignored, not counted again. This
holds after rows are evicted: the live writer keeps a derived on-disk lookup of evicted
identities and archived capture IDs (unlinked temporary files beside the runs, rebuilt
from archive records when needed), so memory stays bounded and a new capture never
scans the archive. The archive/checkpoint remain the only authority.

Use `ExperimentStore.open(folder).read()` or `GET /api/observation/evidence?run_id=...`
for full JSON. ZIP exports contain full reconstructed JSON and CSV, including all
archived intervals, cycles, vehicles, predictions, phase events and gaps. Existing
column meanings are unchanged; schema-2 additions include measurement qualification,
unknown-duration/crossing evidence and durability/lifecycle metadata. CSV headers
are generated within each new ZIP; an existing run's CSV header/file is never appended
to or rewritten. ZIP construction and explicit historical reads currently use memory
proportional to exported history; streaming export is not implemented.

`GET /api/observation/history/{vehicles|comparisons|predictions|phase_events|observation_gaps}`
accepts `run_id`, `offset`, `limit` (1–200) and an optional `generation`. It returns
rows, exact total, next offset and pinned generation/revision. A continuation for a
changed generation returns 409 so the caller can restart pagination. The request
lookup endpoint is `GET /api/observation/predictions/{request_id}?run_id=...`; retrying
an existing capture ID is idempotent, including after eviction or restart. A saved
run cannot accept a new capture. `GET /api/observation/runs` lists every saved run with
`finalized`, `unclean`, `abandoned`, `unavailable` or (old snapshot layout) `legacy`
status; a recovered run keeps `recovered_from`. Pages of one pinned generation reuse a
single explicit rebuild. The dashboard's **Saved runs** viewer browses and exports listed runs, including after a restart; see the workflow below.

Routine telemetry contains bounded live windows and exact additive aggregates.
Its `history` metadata declares totals, retained counts and history endpoints.
Per-vehicle history metadata identifies cycle/interval fragments outside the live
window. The configurable `retention_budget` (default 64, API 1–4096) controls display
windows and commit work; it is an engineering memory budget, not a traffic-study
threshold. Active/unresolved populations and the earliest revisable hysteresis or
phase-boundary suffix remain resident. Exact historical median/P95 are explicitly
unavailable in live telemetry after samples are archived. The dashboard's explicit
quantile action calls `/api/observation/quantiles?run_id=...`, computes exact values
once for a durable generation and labels cached values from older generations.
It never substitutes an approximate historical statistic.

The dashboard distinguishes snapshot and durable revisions, storage suspension,
resource release and durable finalization. Disk failure seals new measurements and
captures, retains one pending generation and reports the last acknowledgment.
Diagnostic images/health may continue. A failed capture write returns an error;
it does not report durable capture success. Retry the retained evidence with
`POST /api/observation/recover?run_id=...` or the dashboard retry action. Repair stops
and finalizes that interrupted run before an explicitly requested replacement.
No measurements resume inside the interrupted run. Native release or final IO may
outlast a cooperative stop timeout. The live owner cannot claim durable finalization
until acknowledgment, even when recovery can already read a complete newly replaced
checkpoint after an uncertain error.

On restart, a committed open new-protocol run is reported as unclean; sealed but
unfinalized evidence is abandoned. Recovery terminalizes unresolved rows with
`unclean_recovery`, preserves diagnostic evidence, and commits an interrupted final
generation. It never resumes cameras/models or reclassifies old qualification.
Active owners are excluded from another manager's recovery. Missing/corrupt committed
bytes are an explicit read/recovery failure. Legacy folders remain read-only.

Deterministic tests use temporary folders and fake inference/captures. The
20/200/2,000-cycle retention probes measure retained objects, visited records,
checkpoint bytes, latency and RSS at equal active populations. They are software
scaling evidence; their default calibration scale is unverified, and they make no
field-accuracy, official LOS or deployment claim. Browser provisioning and the Phase E software checks are recorded in OBSERVATION_VALIDATION.md. Paired-input field qualification remains separate work.


## Calibration and provenance (Phase C)

Choose **Measured width and length** OR **Measured ruler** in each calibration
dialog, supply a measurement reference, and confirm the source/lens/geometry review.
A ruler requires two endpoints and its measured distance. Width/length defaults and
old rulers are not silently claimed as measured. An unverified profile remains
usable for diagnostic runs. Saving creates a separate content revision and preserves
the original Jamika configuration. Rebinding from a saved reviewed profile records
`parent_revision` when available. Typed API fields are `scale_evidence` (method,
reference) and `binding` (reviewed, geometry_digest). Saved profiles can be submitted
to calibration/start again. Actual lens file bytes are checked against the saved
`rectification_digest` at save and run preflight; a mismatch requires explicit review
and a new binding, even when image dimensions are unchanged.

Captures persist `inference_diagnostics`: raw model output (null on backend failure),
bounded and rounded outputs, clipping, bound touch, rounding and fallback flags.
For example, raw 10 is a bound touch, raw -5 is clipped, and raw 10.004 rounds to 10
without touching the bound before rounding. Fallback 30 is invalid backend evidence.
A valid saturated output remains timing-eligible; it does not prove a training-domain
match or mismatch. No research candidate is selected automatically.

The manifest records configured model-file hashes (`model_hashes_scope`), component
classes, initial execution facts, file-content media hashes and initial clock facts.
An uninitialized predictor has a null effective device/settings. Current and final
`execution` evidence records reported model paths/hashes, actual ONNX providers,
effective detector device/settings/precision when available, separately from requested
settings. Unknown values remain null. Captures freeze their execution and clock
provenance; final evidence preserves it after models are released. Runtime camera
stats include the last source/mapped timestamp, origin and sequence. Replay clocks
report presentation time or per-frame validated-FPS fallback and cumulative fallback
count. Acquisition mapping subtracts `offset_sec`; live mapping uses host monotonic
receive time minus the pair origin and offset, not an unverified camera hardware clock.

Remote URL/pipeline inputs are registered locally through
`POST /api/observation/sources`. Public profiles, telemetry, manifests and preview
requests use random `source_…` handles; credentials and hashes of credential-bearing
strings are never source identities. Browser preview uses POST bodies rather than
source query strings. GET preview remains compatible with local paths/devices and
public handles. Handles resolve only in the current process: after restart, re-enter
the private source locally and review/rebind it. Credentials are not written into a
profile to restore them automatically. Validation/backend errors are redacted, and
legacy exports redact a public copy without rewriting historical evidence. Local
loopback operation remains available; the optional LAN gateway is still disabled.


## Saved-run workflow (Phase E, resumed 28 October 2026)

Each Start allocates a new `run_id`; request IDs are idempotent within that run, so the same request ID can occur in two independent runs. Stop requests are asynchronous. A stopped display alone does not prove release/final fsync: check the durable revision and terminal status. Restart lists committed runs from disk without opening cameras/models. Browsing never recovers, replaces or resumes a session. Interrupted-run recovery remains explicit through the existing API/admission workflow and cannot fabricate missing observations.

In **Saved runs · durable evidence**, choose a run, then select **Vehicles**, **Predictions** or **Cycle comparisons**. Selection loads up to 100 rows; Previous/Next carry the pinned generation. **Reload selected evidence** restarts at page one. A 409 clears the page and quantiles and visibly requests a restart rather than combining generations. The viewed run identity, saved status, generation and revision stay above that table. All current-run metrics, streams and controls remain in the workspace above; its summary is not a summary of the selected saved run.

**Compute selected run exact quantiles** explicitly reconstructs that run's durable vehicle population and reports its median/P95, sample count and generation, including partial lower bounds. Different page/quantile generations are labelled. Legacy generation/revision is unknown, not inferred. No full history or quantile request is repeated by the 500 ms telemetry poll. Pages have bounded responses; the existing historical reconstruction and ZIP export use memory/time proportional to full history. This UI does not introduce a new retention design.

**Export selected run** pins the requested run independently of the current session. The ZIP includes `manifest.json`, full reconstructed `evidence.json`, `predictions.csv`, `comparisons.csv`, `vehicles.csv`, `phase_events.csv` and `observation_gaps.csv`. JSON retains nested clocks, execution/quality evidence and detailed vehicle intervals. Follow request ID → prediction/cycle ID → comparison; vehicle crossing cycle links served counts, and namespaced vehicle IDs identify run/role/epoch. CSV nested values are JSON encoded. Separate requests may observe different generations while a run is still changing; each page, exact-quantile result and export declares its own generation.

Detailed interval JSON can exceed Python's default CSV field limit of 128 KiB. CSV consumers must allow the exported field size; the app retains the complete evidence rather than truncating those arrays. Full JSON is also available in the same ZIP.

New captures keep v2 qualification with both-camera scale/binding evidence; preserved v1 policy means motion/phase-only, and absent legacy qualification means unknown. Old saved evidence and original calibration files remain immutable on read/export; public legacy views redact source secrets without rewriting disk bytes. This is app runtime storage, separate from research definitions, datasets, training and weights.

### Local software checks

From `jetson-atsc-core`, with the development dependencies installed and system Chromium at `/usr/bin/chromium`:

```bash
CUDA_VISIBLE_DEVICES='' YOLO_OFFLINE=true YOLO_AUTOINSTALL=false \
  .venv/bin/python -m pytest -q tests/test_phase_e_workflow.py \
  tests/test_observation_pipeline.py tests/test_observation_dashboard_browser.py \
  tests/test_phase_c_review_corrections.py tests/test_archive_index_failures.py
```

These tests use generated media, temporary runs/profiles and injected inference. Browser requests exercise the shipped source through an isolated real API at 1440px and 390px. The repeated-run API check restarts the manager and verifies read/export/quantile access without changing evidence bytes. The paired timing oracle retains its no-actuation spies. See the dated validation report for exact results, the combined gate's explicit legacy-daemon adapters, and unavailable asset-dependent checks. A passing synthetic run does not establish field accuracy, a qualified GPU stack, or real-time paired-input throughput.

Preview overlays/JPEGs use one bounded background encoder: pending previews coalesce per camera; measurement frames and full-resolution Camera 1 capture candidates do not. While running, streams serve the latest completed preview. Paused/stepped JPEG reads wait up to 250 ms for their pending preview. Unchanged MJPEG frames are resent at most once per second.
Telemetry adds `preview_encoded_frames`, `preview_coalesced_frames`, and `preview_errors`. Execution provenance is sampled on the first three acquisition frames, after controls, then every 75 frames or five source seconds, and before final release; sampling is excluded from inference latency.
