# Three-input ANFIS development candidate and reproducible backtest

This bundle preserves the 15 October 2026 local development candidate so a fresh
checkout can evaluate it without repeating SUMO generation or training. It is an
**experimental candidate**, not an approved replacement for the runtime model.
The app continues to load `jetson-atsc-core/models/anfis_sugeno.onnx` by default.

## After pulling this branch

From the repository root:

```bash
python3 research/anfis-v2-20261015/verify_bundle.py
python3.11 -m venv /tmp/anfis-backtest-venv
/tmp/anfis-backtest-venv/bin/python -m pip install \
  -r research/anfis-v2-20261015/requirements-backtest.txt
/tmp/anfis-backtest-venv/bin/python \
  research/anfis-v2-20261015/rerun_backtest.py \
  --output /tmp/anfis-backtest-my-new-run
```

Choose a new output directory each time; existing directories are refused. The
runner uses only files in this checkout, checks the bundle hashes, reuses frozen
20 January calibration, compares both ONNX policies, and asserts that all seven
controller-output CSVs exactly reproduce the committed evidence. It also checks
all 2,849 historical vehicle/scenario rows, all 136 scored test vehicle identities,
and unchanged-green scheduling. CPU inference suffices; SUMO, PyTorch, a GPU,
original videos, and the external Orchestrator repository are not required.

The bundled prepared arrivals and signals are sufficient for this controller
backtest. They do not rerun trajectory extraction from raw FLUID CSVs. To also
verify the nine original CSV hashes, provide the original `derived_data` directory:

```bash
/tmp/anfis-backtest-venv/bin/python \
  research/anfis-v2-20261015/rerun_backtest.py \
  --output /tmp/anfis-backtest-with-raw-check \
  --raw-data-root /path/to/fluid_FIDRT/derived_data
```

The new output manifest explicitly records whether raw sources were checked.
`inputs/run_backtest.py` is the unchanged historical adapter source retained for
provenance; the portable runner extracts its reviewed function definitions without
executing that historical file's machine-specific driver. Original manifests and
reports retain their historical absolute paths as provenance, not dependencies.

## What is included

- `model/candidate_anfis.onnx`: app-compatible frozen candidate; absolute green
  duration in seconds, clipped to 10–120. Input order is exactly **V_w, Q, L**:
  weighted vehicle count (PCU), queue length (m), fixed-area occupancy (%).
- `model/anfis_best_model.pt`: native training checkpoint for preservation;
  `training_summary.json` and `export_manifest.json` identify its source and export.
  ONNX is the self-contained inference artifact. Native training code is in the
  separate TrafficDataExtractor/Orchestrator development pipeline, not this bundle.
- `training/dataset.json`: all 120 complete generated groups, three matched-demand
  contexts, 12 candidate greens (10, 20, …, 120 seconds), 84/18/18 family splits.
  This is the bounded development dataset, not the exhaustive thesis dataset.
- `inputs/`: prepared FLUID vehicle/signal data, frozen calibration, historical
  adapter, historical ONNX, historical vehicle results, protocol and audit.
- `evidence/backtest/`: all 136 scored reused test vehicles, all 333 modeled vehicles
  including calibration/background, all eleven scenarios, controller decisions,
  reports, plots, provenance and verification checks.
- `evidence/app/`: recorded app/API and ONNX parity test evidence. Camera replay
  and controller-only backtesting measure different things.
- `presentation/`: speaking notes and English captions for the professor recording.
  The approximately 103.5 MB MP4 and ZIP are intentionally outside Git; their
  identity is retained in `evidence/recording_verification.json`. A checkout alone
  does not download or recreate the recorded video.
- `checksums.json`: SHA-256 for every delivered bundle file except itself.

FLUID attribution and the CC BY 4.0 source-data license are in
[DATA_CREDITS.txt](DATA_CREDITS.txt). Derived prepared inputs and analyses are
identified as modifications there.

## Read the results accurately

Start with [the backtest report](evidence/backtest/RESULTS.md). On the better
validated 26 May recording (109 scored vehicles), modeled mean corridor delay
falls from **47.98 to 41.67 s/vehicle (13.2%)**. The indicative delay band remains
**D** relative to original timing. The 25 May baseline fails its validation screen;
its large apparent gain is exploratory. The pooled 136-vehicle result therefore
combines different baseline quality. The reused recordings are development data.

This is a 45 m E/W corridor point-queue estimate. N/S movements, turns,
pedestrians, lane interactions, spillback, and complete intersection control delay
are unmeasured. It does not establish verified whole-intersection LOS or field
benefit. Some vehicles worsen; unfinished vehicles remain included and censored.
The synthetic SUMO panel also has mixed scenario-level wins. No automatic model
promotion follows from these results.

## Evaluate the candidate through the app inference class

Using the existing app Python environment, from the repository root, this checks
all 5,163 saved native-reference inputs through the app's ONNX inference engine:

```bash
jetson-atsc-core/.venv/bin/python research/anfis-v2-20261015/check_app_inference.py
```

For an individual inference instance, from `jetson-atsc-core/`:

```python
from src.control.anfis_inference import ANFISInferenceEngine
engine = ANFISInferenceEngine(
    '../research/anfis-v2-20261015/model/candidate_anfis.onnx',
    prefer_cuda=False,
)
print(engine.evaluate_green_time(10.0, 20.0, 15.0))
assert engine.last_error is None
```

This explicitly selects the candidate for that engine instance. It leaves the
application's default checkpoint and hardware configuration unchanged. Rolling
back consists of using the default engine/model path.
