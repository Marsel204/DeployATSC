# Two-camera observation validation

Validated locally on 14 October 2026 in the Jetson workspace. This checkout is not AEK-enrolled. No controller actuation, merge or deployment was performed. The subsequent publishing request covers a feature branch and draft PR.

## Feature acceptance

The focused feature suite passed **55 checks**, including two native Chromium browser tests:

```bash
.venv/bin/python -m pytest tests/test_observation_*.py \
  tests/test_camera_tracker_isolation.py tests/test_manual_predictions.py \
  tests/test_signal_phase_observer.py tests/test_vehicle_waits.py -q
```

Coverage includes independent camera calibration and tracking, stable namespaced IDs, velocity validity, stopped-state hysteresis, multi-cycle waits, missing observations, directional stop-line crossing at an area boundary, distinct-frame signal confirmation, manual/idempotent/late/cancelled predictions, rejection of failed ANFIS fallback, different replay FPS and offsets, live monotonic completion time and paused-backlog discard, source cleanup, typed API contracts, evidence export and immutable export snapshots during interval revisions.

The complete paired synthetic-video test uses 25 FPS system and 30 FPS acquisition inputs. Injected known motion and phase evidence reconstruct **25 seconds actual green**, **20 seconds red wait**, **24 seconds total stopped**, a 35-second recommendation, a +10-second timing difference and one served vehicle, within the declared frame tolerance. Hardware-construction/watchdog/dispatch spies raise if the observation path attempts any cabinet interaction. Detector injection checks interval and integration behavior; it does not establish YOLO field accuracy.

Browser tests draw and save separate camera profiles, verify that signal tools belong to acquisition only, predict manually, block duplicate capture, restore an active run after reload, step/stop/export, cancel a draft and exercise keyboard focus/Escape. Desktop width is 1440px and mobile width is 390px. The running port-8000 app was also checked with native Chromium: no JavaScript errors, source preview visible, acquisition setup available, prediction disabled before setup, and no mobile horizontal overflow.

## Existing regressions and missing baseline assets

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

## Native model smoke and performance

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

## Preserved state and local evidence

The user's pre-existing `configs/custom_simpang5_jamika.json` is byte-identical to the build-start snapshot. SHA-256:

```text
ce82b0ab0c4e81c3a8b466dd44ecb741bdc191aea0252c55c96ac7d008017194
```

Local evidence is under `.venv/`: `observation-feature-tests.log`, `observation-regressions.log`, `observation-full-suite.log`, `legacy-dual-recheck.log`, `native-smoke-final.json`, `native-smoke-final.log`, `native-smoke-runs/`, and `observation-final-desktop.png` / `observation-final-mobile.png`. These environment artifacts are not committed application datasets. New operator profiles and experiment runs use their own directories, separate from the user's legacy calibration and ledgers.

The app runs in observation mode. LOS remains ungraded because its methodology/edition and required control-delay coverage have not been selected. Recommendations are recorded alongside existing signal timing; the experiment does not measure a causal ANFIS treatment effect. See the [operator guide](TWO_CAMERA_OBSERVATION.md) for setup and record interpretation.
