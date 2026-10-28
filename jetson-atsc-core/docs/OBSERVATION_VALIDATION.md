# Two-camera observation validation

This checkout is not AEK-enrolled. The latest PR-preparation results are recorded first; the earlier Phase E and 14 October results remain dated history. Verification does not establish field accuracy or authorize controller actuation. The final TensorRT integration and accumulated PR diff are self-reviewed; earlier independent reviews apply to their recorded snapshots rather than constituting independent approval of the current commit.

## PR preparation and TensorRT integration — 29 October 2026

The final raw full suite completed with **531 passed, 75 failed and one skipped**. The set of failing test identities exactly matches the earlier 75 missing-asset failures, with no new failure identities. The full suite is **not green**. The missing assets include `models/yolo11s.pt`, stress-test telemetry traces, comparative media and benchmark videos. No asset-dependent assertion was removed, converted to a skip or treated as passing.

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' YOLO_AUTOINSTALL=False \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider
```

A separate focused selection passed **25 checks** covering engine export integrity, runtime/weights/engine binding, native factory selection, the single-frame engine guard, Camera 1 latest-frame capture despite preview throttling, lazy full-resolution clean previews, FPS accounting, callback retention, and the unchanged paired Camera 2 timing/wait/export oracle. These checks overlap with the full suite and must not be added to it. The full suite also exercises the shipped dashboard with real Chromium against isolated fake-inference sessions. JavaScript syntax and staged whitespace checks pass.

Native, sequential Jetson benchmarks on the same 640-pixel FP16 TensorRT engine measured an integrated processing ceiling of approximately **17 FPS** using a 30 FPS source. The application retains full per-frame Camera 2 tracking and measurement; only annotated display resolution and Camera 1 display refresh are reduced. The local demo uses a **15 FPS** excerpt for playback headroom. This is a functional/performance observation on this board at its existing 15 W setting, not GPU-stack or field qualification. TensorRT permits FP16 kernels while this engine's input tensors are FP32; provenance records the input-precision scope separately from the engine's FP16-enabled build.

The reproducible board-local builder, selection and rollback instructions, prior dataset comparison and precision limitations are in [TensorRT observation setup](TENSORRT_OBSERVATION.md). Engine binaries, downloaded video/datasets and saved experiment runs are not part of the source change. The final HTTP playback-check attempt was interrupted by the tool/server restart; the recorded headless benchmarks and software checks are the completed evidence. Raw logs, JUnit and the failure-set comparison are retained locally under `reviews/tensorrt-integration-2026-10-29/` outside the application repository.

## Historical Phase E software verification — 28 October 2026

| Completed selection | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| Phase E workflow/pipeline/browser, Phase C closure and checked-copy index cases | 33 | 0 | 0 |
| Final combined deterministic gate, including six current-run response cases (overlapping selection) | 410 | 0 | 0 |
| Last raw full `pytest -q tests`, before the final UI correction, without external adapters | 485 | 75 | 1 |

Thirteen real Chromium browser cases pass. The original seven use the shipped source at 1440px/390px: saved pages/quantiles/export, generation conflict, stale responses, text-safe records, legacy labels, independent calibration, reviewed scale save/load, active controls/streams, and repeated-run isolation. Six additional cases reject delayed successes and errors after switching current runs, returning to live view, or issuing newer requests. These are included in the selections above, not additional counts. The 11 Phase C correction cases and seven index checked-copy cases pass. JavaScript syntax and Git whitespace checks also pass. The current warning is the Starlette/httpx test-client deprecation.

The final review independently verified the Phase B index and Phase C corrections, and found one remaining current-run display race. Current history and quantile requests now check both initiating run and request sequence; rebinding or returning to live view invalidates outstanding responses and clears their labels. The original browser reproduction failed before the correction and passed afterward. This narrow final correction was self-reviewed by the primary agent. The final gate ran against unchanged source/test bytes; its commands, JUnit and preserved screenshots are in the sibling `final-ui-closure-*` artifacts. The earlier raw full suite was not repeated for this UI-only correction, and its missing-asset failures remain a limitation.

The two-run API test starts actual app sessions using generated clips and fake inference, uses `same-click` in each independent run, waits for finalization, restarts the manager, browses durable pages, computes exact quantiles and checks JSON/CSV exports. Hashes prove these reads do not rewrite evidence. Frozen copies of both generation-3 runs are preserved with the local verification artifacts. The paired-video oracle still asserts **25 s green / 20 s red wait / 24 s stopped**, recommendation/cycle/vehicle/clock linkage and no cabinet interaction. Adverse-quality and UNKNOWN-crossing exports retain diagnostic/null attribution rather than inventing measurements or served counts.

Every raw failure was examined: 25 adversarial benchmark cases, 21 comparative-media cases, 22 telemetry-trace cases and seven calibration checks fail on missing assets. The failure set is exactly the same 75 asset cases from the interrupted full run. Expected paths are checked absent, and those test files match baseline HEAD. Examples include `models/yolo11s.pt`, `data/telemetry_traces/trace_stress_*.jsonl`, benchmark clips under `data/sample_videos/stress_test/` and `multi_angle_test/`, and comparison videos/snapshots under `data/comparative_media/`. The detection-class-mapping skip also requires `yolo11s.pt`. No missing asset was fabricated, downloaded or converted to a new skip. Their downstream assertions remain unverified; **the full suite is not green**.

All resumed pytest commands used CPU/offline settings: `CUDA_VISIBLE_DEVICES=''`, `YOLO_OFFLINE=true`, `YOLO_AUTOINSTALL=false`, `PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=2`. Playwright was already installed at resume; system Chromium is `/usr/bin/chromium`. The combined gate retains the previously declared external `phase_a_daemon_fakes` plugin scoped **only** to `test_edge_daemon_pipeline.py`. The raw full suite loads no such plugin. Commands, environment, exit codes, logs, JUnit, source hashes and per-case failure audits are under the sibling `reviews/architecture-2026-10-26/phase-e-resume-*` artifacts. See `implementation-phase-e-workflow.md` there for the cumulative/incremental frozen binding and the earlier interrupted results. Selections overlap and must not be summed.

The research tree, datasets, training, weights, existing evidence and Jamika configuration are unchanged. This validates local software behavior with synthetic/injected inputs. It does not qualify paired physical cameras, a GPU stack, field accuracy, real-time throughput, LOS or causal improvement. Conditional LAN remains disabled pending its separate provisioned-gateway gate.

## Historical feature acceptance — 14 October 2026

The focused feature suite passed **55 checks**, including two native Chromium browser tests:

```bash
.venv/bin/python -m pytest tests/test_observation_*.py \
  tests/test_camera_tracker_isolation.py tests/test_manual_predictions.py \
  tests/test_signal_phase_observer.py tests/test_vehicle_waits.py -q
```

Coverage includes independent camera calibration and tracking, stable namespaced IDs, velocity validity, stopped-state hysteresis, multi-cycle waits, missing observations, directional stop-line crossing at an area boundary, distinct-frame signal confirmation, manual/idempotent/late/cancelled predictions, rejection of failed ANFIS fallback, different replay FPS and offsets, live monotonic completion time and paused-backlog discard, source cleanup, typed API contracts, evidence export and immutable export snapshots during interval revisions.

The complete paired synthetic-video test uses 25 FPS system and 30 FPS acquisition inputs. Injected known motion and phase evidence reconstruct **25 seconds actual green**, **20 seconds red wait**, **24 seconds total stopped**, a 35-second recommendation, a +10-second timing difference and one served vehicle, within the declared frame tolerance. Hardware-construction/watchdog/dispatch spies raise if the observation path attempts any cabinet interaction. Detector injection checks interval and integration behavior; it does not establish YOLO field accuracy.

Browser tests draw and save separate camera profiles, verify that signal tools belong to acquisition only, predict manually, block duplicate capture, restore an active run after reload, step/stop/export, cancel a draft and exercise keyboard focus/Escape. Desktop width is 1440px and mobile width is 390px. The running port-8000 app was also checked with native Chromium: no JavaScript errors, source preview visible, acquisition setup available, prediction disabled before setup, and no mobile horizontal overflow.

## Historical regressions and missing baseline assets — 14 October 2026

The selected legacy regressions passed **28 checks with one skip**:

```bash
.venv/bin/python -m pytest tests/test_phase_monitor.py tests/test_types.py \
  tests/test_decision_engine.py tests/test_anfis_inference.py \
  tests/test_homography_metrics.py tests/test_optics_rectifier.py \
  tests/test_detection_class_mapping.py tests/test_sinks.py \
  tests/test_roi_calibration.py tests/test_multipoint_polygon_and_ruler.py \
  tests/test_ui_api.py tests/test_edge_daemon_pipeline.py -q
```

The skipped detection check requires absent `models/yolo11s.pt`. The ruler regression exposed a valid homography with H33=0; the implementation now uses an invertible DLT homography rather than rejecting that projection. API tests use isolated calibration/media fixtures and retain their assertions. External LLM credential discovery is disabled in tests so verification does not contact a paid audit service.

The combined feature and selected regression checks were rerun before publication: **83 passed, one skipped**, with a Starlette test-client deprecation warning and the Torch architecture warning described below. Publishing review was self-review of the final diff; no independent review or remote CI approval is claimed.

The earlier broad suite run, excluding the new static dashboard and browser checks while their final UI was staged, reported **161 passed, 76 failed and one skipped**. Its failures were investigated:

| Existing suite | Failures | Finding |
| --- | ---: | --- |
| Adversarial challenger | 25 | Missing benchmark media/telemetry/comparative artifacts |
| Comparative media | 21 | Missing comparison artifacts |
| Stress telemetry | 22 | Missing trace JSONLs |
| Calibration profiles | 7 | Expected YOLO model absent |
| Legacy daemon integration | 1 | Actuation polling timeout in that run; isolated rerun passed, and the later selected regression run also passed |

Those 75 asset-dependent checks remain unavailable in this checkout. Their assertions were not removed or weakened. The broad suite is **not claimed green**. The generated clips test geometry/timing contracts and do not replace missing annotated field benchmark datasets.

## Historical native model smoke and performance — 14 October 2026

Actual `models/Final.pt` and `models/anfis_sugeno.onnx` were exercised on the downloaded 1920×1080, 25 FPS Jamika clip. Each role had its own native detector/tracker and calibration. This was explicitly shared-file diagnostic replay, not two physical synchronized cameras. Final measured smoke run: `run_4d41fca3512d4a41`.

| Measurement | Local result |
| --- | ---: |
| Processed source frames | 11 system / 11 acquisition |
| Model startup plus first pair | 8.416 s |
| Total run including startup, capture and stop | 11.602 s |
| Ten subsequent pairs, including processing/capture/stop | 3.186 s |
| Last system / acquisition inference | 102.85 / 69.13 ms |
| Mean system / acquisition inference including cold first call | 481.45 / 105.37 ms |
| Process-wide peak resident memory | 2179.76 MiB |
| Stable acquisition track records | 24 |
| File replay backlog at stop | False; no deliberate frame dropping |

The selected neural device was `cuda:0`, but installed Torch `2.13.0+cu132` warns that its published CUDA architectures do not support this Orin's compute capability 8.7. Native execution succeeded in this smoke; this warning and measured throughput prevent claiming a qualified GPU stack or real-time operation. Ten post-startup pairs in 3.186 seconds is about 3.1 pairs/sec, below the input's 25 FPS. File replay preserves presentation-time measurements while running slowly. Runtime telemetry reports inference device/latency, source drops and processed-frame age; stale live road snapshots cannot produce a recommendation.

The native input snapshot was V_w=6.6 PCU, Q=27.21 m and L=16.25%, using the calibrated polygon's physical area. ANFIS returned 14.42 seconds. The selected signal region did not establish a confirmed phase, so this recommendation was diagnostic, unmatched and ineligible for a valid cycle comparison. No measured green or field delay accuracy is inferred from it.

No actual dual-live-camera throughput, live drop rate or calibrated signal/vehicle ground-truth accuracy was measured. Live source lifecycle/clock/gap behavior was checked through injected adapters. Field qualification needs two real inputs, source alignment, independent calibration, annotated observations and a compatible benchmarked inference stack.

## Historical preserved state and local evidence — 14 October 2026

The user's pre-existing `configs/custom_simpang5_jamika.json` is byte-identical to the build-start snapshot. SHA-256:

```text
ce82b0ab0c4e81c3a8b466dd44ecb741bdc191aea0252c55c96ac7d008017194
```

Local evidence is under `.venv/`: `observation-feature-tests.log`, `observation-regressions.log`, `observation-full-suite.log`, `legacy-dual-recheck.log`, `native-smoke-final.json`, `native-smoke-final.log`, `native-smoke-runs/`, and `observation-final-desktop.png` / `observation-final-mobile.png`. These environment artifacts are not committed application datasets. New operator profiles and experiment runs use their own directories, separate from the user's legacy calibration and ledgers.

The app runs in observation mode. LOS remains ungraded because its methodology/edition and required control-delay coverage have not been selected. Recommendations are recorded alongside existing signal timing; the experiment does not measure a causal ANFIS treatment effect. See the [operator guide](TWO_CAMERA_OBSERVATION.md) for setup and record interpretation.
