# Identical-input 22-second demo comparison

Both candidates reproduced every earlier v6 owner and all 154 reference-pair flags. Physical tracking remains incomplete; neither candidate is adopted.

Exactly 660 source frames at 30fps, 540x960 source pixels, source times 0–21.9666667s, unchanged ROI. Both use the same custom YOLO11m TensorRT S10 raw cache at confidence floor 0.10. No detector inference or native tracker was rerun. Fresh actual ObservationEngine CPU replay, feature extraction and timing were performed. Cached native timing is disclosed separately, and engine timing excludes decode/rendering/features.

| Candidate | Same pairs /94 | Wrong links / evaluable different pairs | Detected | Recovered | Loss | In ROI at end |
|---|---:|---:|---:|---:|---:|---:|
| ByteTrack + v6 | 80 | 0 / 11 of 18 | 34 | 12 | 37 | 4 |
| BoT-SORT + v6 | 81 | 1 / 11 of 18 | 38 | 9 | 28 | 4 |

Application IDs are local to each run. The row counts are not unique physical cars; fragmentation, admission and physical rejection affect them. Only 11/18 different-labelled reference pairs have both application owners in each arm; the other seven do not prove correct separation. The original references are correlated and biased toward ByteTrack failures. These are development diagnostics, not precision/recall, HOTA, IDF1 or whole-clip success rates.

## Physical audit

- Original V2 endpoints f0→437 now have distinct owners: ByteTrack V2→V62, BoT-SORT V2→V56. This prevents that exact history transfer; it does not prove a complete physical trajectory.
- Original V5 endpoints f0→517 have no later application owner in either candidate. The returned raw red-car box is unassigned, not a certified recovery.
- Original V12 endpoints f50→172 now have distinct owners: ByteTrack V12→V65, BoT-SORT V13→V57. Local numbering differs from the original run.
- The frozen merged-box f144→145 pair still has V9 in both candidates. Its per-car duration/departure is unknown and status is Loss.
- The f460→541 different-car pair still has V36 in BoT-SORT. ByteTrack has no owner at the later endpoint; no success is credited for that missing owner.
- The earlier legacy V69 photographed endpoints f547→582 still share ByteTrack V77 and BoT-SORT V67. Thus ByteTrack having zero wrong links on the 18 reference pairs does not mean zero known physical errors overall.
- Ten apparently successful histories were inspected separately. VISUAL_INSPECTION.md records confidence limits, including occlusion/box ambiguity and the extra BoT-SORT V8 trajectory sheet. These are self-reviewed snapshots, without new independent labels.

## Timing and interpretation

- ByteTrack + v6: fresh engine median/p95 26.59/28.21ms; fresh features 18.39/20.97ms. Reused earlier native update median/p95 8.74/10.44ms. Supported/unassigned car-ROI samples 12299/649.
- BoT-SORT + v6: fresh engine median/p95 26.58/28.00ms; fresh features 18.82/21.20ms. Reused earlier native update median/p95 8.86/10.47ms. Supported/unassigned car-ROI samples 12409/580.

These are single-thread CPU, unpinned-clock, one-run measurements. Do not sum component p95s or infer live throughput. The candidates have similar engine cost; the observed one-pair continuity difference is not reliable evidence of superiority. For waiting timers, incorrect identity transfers still invalidate a per-car timer. V6 also retains its previously failed Semarang acceptance; this queue replay does not retest or waive it. Physical scale is unverified and signal phase UNKNOWN.

Source timestamps, observed-sample ROI duration, explicit unknowns, loss/recovery timestamps, separate unassigned candidates and clip-end presence are preserved in each detailed report. Confirmed departures require the frozen 0.30s outside-evidence rule, with an immediately preceding inside sample.

Files: report.html, comparison-22s.mp4, comparison.csv, COMPARISON.json, AUDIT_CHECKS.json, VISUAL_INSPECTION.md, VERIFICATION.json, and per-arm bytetrack/ and botsort/ reports, annotated replays, requested CSV, detailed timings and unassigned-candidate CSV.
