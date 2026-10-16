# Repository bundle verification — 16 October 2026

Review mode: **self-review**. This repository has no `.aek/project.json`; AEK
approval and publishing enforcement do not apply.

## Bundle checks passed

- Portable controller backtest ran from a relocated bundle with `/tmp` as its
  working directory. It required no original machine paths, videos, SUMO,
  external training repositories, or raw trajectory CSVs.
- All seven controller-output CSVs reproduced the recorded evidence byte for byte,
  including all 2,849 historical vehicle/scenario comparisons, all 136 scored
  test vehicles, sensitivity summaries, background audit and cycle decisions.
- Existing output directories were refused without modification.
- A changed prepared-data file was rejected before creating an output directory.
- The dataset and native checkpoint hashes match the ONNX export manifest. The
  frozen historical ONNX matches the repository's current default model.
- Candidate app inference passed all 5,163 frozen native-reference inputs with
  maximum raw output difference 0.000213623 seconds and zero inference fallbacks.
- The historical adapter source and calibration are copied unchanged. Only the
  outer runner's path handling, hash checking, output location and reproduction
  checks were adapted for a self-contained checkout.
- Changes are limited to this research bundle and a repository README link. There
  are no changes under `jetson-atsc-core/`, including its default model.

## Full app-suite limitation: publishing gate not passed

Command, from `jetson-atsc-core/`:

```bash
.venv/bin/python -m pytest tests -q
```

Result: **179 passed, 75 failed, 1 skipped** in 91.40 seconds. The checkout lacks
legacy stress telemetry traces, source/comparison videos and snapshots, and the
legacy `models/yolo11s.pt` asset referenced by calibration profiles. These are
outside the delivered bundle. The failing test groups are
`test_adversarial_challenger_2.py`, `test_calibration_profiles.py`,
`test_comparative_media.py`, and `test_stress_telemetry_traces.py`.

A focused reproduction reports the missing `models/yolo11s.pt` immediately.
Missing `data/telemetry_traces/trace_stress_night_arterial.jsonl` and legacy video
fixtures also produce explicit file-not-found failures in the full run. No app
source or tests differ from the remote main branch used for this bundle. These
are pre-existing fixture gaps, not evidence that the research runner failed.
Nevertheless, the complete repository test suite is **not green**. No tests
were weakened, skipped or rewritten to claim a passing suite.

The bundle is prepared locally. Publishing requires an explicit exception to the
ship workflow's full-suite gate or restoration of the missing legacy fixtures.

## Interpretation retained

The better-validated 26 May development recording shows 13.2% lower modeled
corridor delay; indicative LOS remains D against recorded timing. The 25 May
baseline screen fails. Reused data, incomplete movement coverage, right censoring
and absent field validation remain disclosed in the result report. This bundle
does not promote a model or establish whole-intersection LOS improvement.
