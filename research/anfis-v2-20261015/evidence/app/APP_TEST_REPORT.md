# New candidate: application integration test

**Passed application integration on 15 October 2026.** The new checkpoint was
exported with the app's existing exporter and tested in a separate copy of the
app. The original application source and deployed ONNX model remain unchanged.
This test was performed after the completed SUMO experiment.

## What ran

- The actual `ANFISInferenceEngine` and `ATSCDecisionEngine` processed 5,163
  mixed test inputs: 120 new SUMO group observations, 1,293 SUMO decision inputs,
  691 cached Bellevue app snapshots, 56 cached FLUID queue-adapter inputs,
  3,000 synthetic boundary probes, and three explicit boundary cases.
- ONNX versus native PyTorch maximum absolute difference was **0.000214 seconds**.
  All bounded app decisions agreed within the declared rounding tolerance, with
  **zero inference fallbacks**. Median/p95 direct model inference was
  0.026/0.032 ms on this CPU; this excludes video detection/tracking.
- The isolated FastAPI app served its dashboard HTML, accepted calibration and
  start requests, processed scheduled prediction requests, returned telemetry,
  and exported native evidence. Expected requests during green/yellow or with
  an existing pending prediction were correctly rejected with HTTP 409.
- Both camera roles replayed the same prepared 400-second Bellevue recording,
  using the real detector, independent trackers, metric extraction, optical
  signal observation, new ANFIS, prediction/cycle matching, and evidence store.
  No mocked detector or inference model was used. Cabinet-hardware entry points
  were guarded to fail the test if invoked; none were invoked.
- All **1,000 system frames and 1,000 acquisition frames** completed in
  **438.6 seconds (7 minutes 19 seconds)** on CPU, with no runtime or storage
  errors. Final sampled video time is 399.6 seconds at 2.5 fps.

## Observed recommendations

Five prediction requests matched observed green starts. Four matched complete
green intervals; the last matched partial interval is excluded from scoring.

| Complete cycle | Recorded green (s) | New recommendation (s) | Difference (s) |
| --- | ---: | ---: | ---: |
| 2 | 24.8 | 17.35 | -7.45 |
| 3 | 9.2 | 10.00 | +0.80 |
| 4 | 12.0 | 10.00 | -2.00 |
| 5 | 24.0 | 10.00 | -14.00 |

Recorded-timing agreement MAE is **6.06 seconds over four complete comparisons**.
Recorded timings are not optimal-control labels, so this is not an optimal-green
accuracy score. The earlier app run had an early collection gap and only three
eligible comparisons; its headline MAE is not a paired comparison with this run.
On the three common complete cycles, both runs recommended 10 seconds.

On all 691 reused Bellevue snapshots, the old model's output range was
10–10.74 seconds (645 at minimum) and the candidate's was 10–55.81 seconds
(532 at minimum). This demonstrates different inference behavior on the same
inputs; greater output variation does not establish better control.

## What this does and does not establish

The candidate loads and runs correctly through the app's video/API/decision path.
Native export parity, valid output bounds, no-fallback operation, cycle matching
and evidence export passed. Browser clicks and layout were not automated.

Vehicle measurement remains limited: **80 acquisition track records, one complete
wait and 79 partial waits**. The complete wait was 3.2 seconds. The 4.59-second
all-record observed mean includes incomplete lower bounds and cannot establish
mean traffic delay. LOS remains ungraded. These counts match the earlier app run;
changing ANFIS did not fix detection, tracking, or coverage.

The original prepared source and calibration were reused: one prerecorded source
for both roles, a partial approach with approximate 10.5×15 m geometry and signal
color preprocessing. This does not validate independent cameras, the trained
45 m observation region in a field installation, or the full incoming queue.
FLUID inputs in the parity test are cached controller-adapter development inputs,
not a new FLUID video/delay experiment.

Replay cannot measure how vehicles respond to a changed recommendation. The
earlier SUMO comparison supplies the separate traffic-response result. Also,
the app retains 0.01-second output rounding, while that SUMO comparison applied
integer seconds; they are not identical actuation-quantization experiments.

Keep the model as a development candidate; no deployment was performed.

## Artifacts

- `candidate_anfis.onnx`: separate exported development model.
- `export_manifest.json`: checkpoint/ONNX/source provenance and development flag.
- `app_result.json`: final computed application results.
- `parity.json`, `native_reference.npz`, `reference_inputs.json`, and
  `runtime_decisions.json`: numerical and decision-path evidence.
- `evidence.zip`: native application manifest, complete evidence and CSV exports;
  all ZIP entry CRCs passed, and the manifest identifies the new ONNX hash.
- `final_telemetry.json`, `timeseries.jsonl`, and `request_log.jsonl`: replay records.
- `isolation_manifest.json`: original app hashes; all remained unchanged.
- `run_app_test.py`: application/API/video verification harness. The app copy's
  34 Python source files match the original application source.

Checkpoint SHA256:
`41aa2836e92f587250f1d58ea5e8d262dea4e546afbfee67f98233907f2e657c`

Candidate ONNX SHA256:
`ffccea91aadc348ce4e633f4c9ab028e95332754970a83bde4c51d0aa1f3790d`
