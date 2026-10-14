# Two-camera observation experiment

The dashboard now separates the **system camera** from the **acquisition camera**. The system camera measures V_w/Q/L using its own road geometry and YOLO. The acquisition camera tracks vehicle identities and waiting intervals, reads the target movement's signal and measures actual green. **Predict next green** records an ANFIS recommendation; it never commands a cabinet.

## Start the dashboard

From `jetson-atsc-core`:

```bash
.venv/bin/python scripts/launch_ui.py --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`, or the device's LAN address on port 8000. The observation CLI can preset live inputs without opening them until the operator starts a calibrated run:

```bash
.venv/bin/python edge_daemon.py --operation observation --source-mode live \
  --source_sys /dev/video0 --source_acq /dev/video1
```

The ordinary daemon entry point defaults to observation. Historical cabinet operation requires the explicit `--operation legacy-control` selection and is outside this experiment. Observation startup, processing and stop do not instantiate `SignalInterface`, watchdogs or serial connections. Existing legacy replay endpoints remain available for compatibility.

## Set up both cameras

1. Select independent videos, upload local files, or choose Live cameras in experiment settings and enter device/RTSP sources. Both views must observe the same movement. Preview each source before calibration.
2. Camera 1: draw the road polygon, then four anchors in **Far-Left → Far-Right → Near-Right → Near-Left** order. Enter physical road dimensions. An optional two-point ruler calibrates the metric scale. The measured polygon's projected physical area determines the occupancy denominator.
3. Camera 2: draw the waiting/measurement area, its own four perspective anchors and metric scale, the traffic-light head box, and a two-point stop line. Flip the stop-line arrow to match vehicle travel. Camera 1's homography is never applied to camera 2.
4. If adjacent lamps/heads or colored backgrounds interfere, draw optional red/yellow/green lamp regions inside the head box. Darkness, conflicting colors or missing observations remain Unknown; no yellow does not mean green.
5. Save each role's profile. Saving does not start/restart playback. Choose lens correction independently, including explicit None for footage that should be processed without lens remapping. The drawing image and runtime use the same transform. Changing source, image dimensions or lens invalidates that binding.

Existing road geometry can be loaded for review using Camera 1's **Existing road geometry** selector. Saving creates a separate observation profile under `data/observation_profiles/`; it does not overwrite `configs/custom_simpang5_jamika.json`. Old road-source light boxes are not silently assigned to acquisition.

For a single-file demonstration, explicitly enable **Reuse system video for testing**. Both views then use independent calibration/tracker state on the same clip, clearly labeled Shared test. This is unavailable for live sources and is not evidence of real two-camera synchronization. The downloaded Jamika segment is available in Samples, but its 37 seconds may not contain both boundaries of a complete green cycle.

## Run and capture a recommendation

Start observation, optionally paused for inspection. Camera 2 measures continuously across red and green; camera 1 maintains the latest processed road snapshot. Click **Predict next green** during red shortly before the expected green. The current V_w/Q/L snapshot, server capture time, inference completion and recommendation are retained.

The capture is idempotent: retries use the same request ID and cannot create multiple recommendations. One recommendation can be pending for the upcoming cycle; it can be explicitly cancelled while unmatched. It becomes a valid comparison only when inference finishes before a subsequently observed complete green interval. Green beginning during inference makes the result Late. Capturing with Unknown phase is explicitly diagnostic/unaligned rather than a valid before-green result.

Actual green is measured from its evidence onset to the observed green-to-yellow evidence boundary, after debounce confirmation. Confirmation time is recorded separately. A missing boundary, unexpected direct green-to-red transition or unknown gap produces an incomplete cycle. Actual timing never comes from ANFIS output.

Replay pause freezes video time. Step advances one system frame and all acquisition frames newly due at that experiment time; different FPS are aligned by timestamp, not equal frame numbers. Acquisition source time is system/experiment time plus the configured offset. Live Pause suspends observation while camera time continues; Resume discards the paused backlog, resets tracking identity epochs and records a gap. Live sources cannot Step/rewind.

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

## Export and research scope

**Export evidence** downloads a ZIP containing `manifest.json`, `evidence.json`, and prediction/phase/vehicle/comparison CSVs. Detailed vehicle intervals remain in the JSON. Working run artifacts also live under `data/experiments/<run_id>/`, with versioned schemas and JSONL collections. The manifest records sources, source offsets, camera profiles/revisions, measurement policy, model hashes, package/provider versions and software-source hashes. Existing historical CSV schemas are preserved.

Cycle difference is **recommended green − actual green**. It is a timing comparison, not error against an independently established optimum. ANFIS recommendations are not applied in this experiment, so the run describes observed timing/delay rather than a causal treatment improvement.

LOS intentionally remains **ungraded** until the study selects its framework/edition, movement/analysis period, control-delay measurement and sufficient coverage. Red wait alone does not establish control delay or a whole-intersection LOS. No counterfactual ANFIS delay/LOS is fabricated.

## Validation and current performance limits

Automated checks cover independent geometry, persistent IDs, exact time-based waits, debounced phase evidence, manual/double/late predictions, replay offsets/FPS, live clock/queue behavior, cleanup, API/export linkage and browser interactions at 1440px and 390px. A paired synthetic experiment reproduces green t=30–55 (25 s), stop t=10–34 (24 s), and red through t=30 (20 s red wait).

The actual `Final.pt` and ANFIS ONNX models were exercised on the downloaded 1080p Jamika clip in two independent replay pipelines. Both produced road/track evidence and an ANFIS recommendation. This setup currently runs below real time: the final cold-start plus 11-pair smoke took 11.6 s; ten post-startup pairs took 3.2 s. Last measured inference was about 103 ms for system and 69 ms for acquisition, and process-wide peak resident memory was about 2.13 GiB. These are local smoke observations, not a validated throughput target. The installed PyTorch build warns about this Orin GPU architecture; a CUDA-available flag does not establish compatible GPU execution. Runtime telemetry reports the actual predictor device, per-role inference time, drops, and live processing age.

Before field measurements, use a compatible tested inference stack, benchmark both real inputs, verify clock/source offset and calibrations, and compare signal/vehicle annotations to manual ground truth. Replay timing remains source-time based even when processing is slower than playback. See [implementation validation](OBSERVATION_VALIDATION.md) for test results and the pre-existing asset-dependent checks that cannot pass in this checkout.
