# Vehicle ROI timing checkpoint — 1 November 2026

This draft checkpoint preserves the moving-queue identity repair, experimental
v3–v6 resolvers, their regression fixtures, and the identical-input 22-second
ByteTrack/BoT-SORT demo. It does not establish a physical-car timing accuracy
percentage. Native identity mode remains the default; when persistent identity
is selected, its association default remains
`joint_image_motion_appearance_v2`. Experimental `delayed_trajectory_v6` is
explicitly selected and still fails adoption requirements.

## What is included

The moving-queue repair separates the latest geometry from older appearance,
predicts image motion without treating unverified scale as metric evidence,
requires stronger changed-native recovery evidence, and makes retirement
idempotent. The later experimental resolver isolates tentative confirmation to
its own stable native segment. Its complete-prefix fixtures retain competitors
that narrow endpoint fixtures previously omitted.

The stage2d test now reads a checked-in, byte-identical copy of the original demo
preset. It no longer needs a preset from the parent workspace's `reviews/`
directory. The source-path strings inside the preset are provenance only in
this fixture replay; no source video is opened. No assertion or acceptance
threshold was changed.

The frozen report, physical audit, requested per-car CSVs, detailed CSVs,
unassigned candidates, provenance, and verification are included in
[evidence/roi-timing-2026-11-01](evidence/roi-timing-2026-11-01/README.md).
The full local interactive/video bundle remains outside Git; its absence from
a fresh clone is documented rather than implied reproducibility.

## Identical-input demo

Both candidates used exactly 660 frames at 30 fps, 540 × 960 pixels, original
ROI, and source times 0–21.9666667 seconds. Both consumed the same custom
YOLO11m TensorRT detector cache at confidence floor 0.10. Fresh actual
ObservationEngine/v6 and feature replays reproduced every earlier owner and all
154 reference-pair flags. Detector inference and native tracking were reused,
not rerun. Source SHA-256 is
`849a35eb11023009b2e650c8742b7b151311a768e57d4839652226450f3632d3`;
raw detector-stream SHA-256 is
`18b77ac987908d052c8a40cbd9cbbdfa23b6bd54b0e0327f0217e88bbb8b324d`.

| Diagnostic | ByteTrack + v6 | BoT-SORT + v6 |
|---|---:|---:|
| Retained selected same-car pairs | 80/94 (85.1%) | 81/94 (86.2%) |
| Wrong links among different-car pairs with both owners | 0/11 | 1/11 |
| Application-ID rows | 87 | 79 |
| Detected / recovered / loss / in ROI at end | 34 / 12 / 37 / 4 | 38 / 9 / 28 / 4 |
| Fresh engine median / p95 | 26.59 / 28.21 ms | 26.58 / 28.00 ms |
| Fresh feature median / p95 | 18.39 / 20.97 ms | 18.82 / 21.20 ms |

There are 18 different-car reference pairs; seven lack both application owners
and cannot be credited as correct separation. Selected pairs are correlated,
repeat vehicles, and favor earlier ByteTrack failure cases. These values are
neither unique-car counts nor detector recall, IDF1, HOTA, physical tracking
accuracy, or per-car ROI duration accuracy. No reliable tracker superiority is
established. Runtime is one unpinned CPU run of separate components; component
p95s cannot be added to claim end-to-end latency or live throughput.

The original V2 and V12 wrong-body endpoints now have separate owners. Original
V5's later red-car endpoint has no owner, which is missing coverage. Both
candidates still accept the frozen merged-box f144→145 support as local V9 and
the earlier wrong-body f547→582 link as ByteTrack V77 / BoT-SORT V67. BoT-SORT
also links different bodies at f460→541 as V36. Ten successful-looking histories
were self-reviewed; occlusion and changing boxes leave several uncertain.
IDs are local to each run. No new independent physical labels are claimed.

## Meaning of the ROI timer

Observed duration is supported in-ROI samples divided by 30; missing, held, and
unassigned samples add no observed time. Tentative observations are included
only if that identity is subsequently admitted. Known mixed identities have
unknown duration/departure. A departure needs an immediately preceding inside
sample and at least 0.30 seconds of consecutive outside evidence. Last-seen
samples, stop-line crossings, and the clip cutoff are not departures. Cars still
inside at cutoff are reported separately.

A car need not keep the same ID after leaving the ROI for its completed visit
to be measured correctly. Within the visit, fragmentation, wrong-body links,
and missed observations still affect the result. Observed sample duration is
not complete physical residence through an unseen interval. Physical scale is
unverified and signal phase UNKNOWN; no calibrated red-light waiting accuracy
is evaluated here.

## Current software verification

- Complete nonbrowser suite, excluding the three browser modules: **806 passed,
  76 failed, one existing skip** across 883 cases (379.23 seconds). All 75 inherited
  failure identities match the earlier verified baseline. The additional failure
  is the known Semarang motorcycle-coverage acceptance test.
- A source/tests-only temporary copy without the parent review directory runs the
  four experimental modules with v6 explicitly selected: **93 passed, eight
  failed**, no skips. The failures exactly match the seven previously recorded
  v6-targeted assertions plus that same Semarang acceptance check. All 14 narrow
  stage2d checks pass; its two acceptance checks split one pass / one failure.
- The broad run preceded the fixture-path portability edit. Production source is
  byte-for-byte unchanged from the evaluated commit; the exact original preset
  was copied, all 37 stage2d assertions are unchanged, and the final affected
  harness was exercised by the isolated run.
- The three existing browser modules pass **7/7** using their unmodified
  Chromium harness. The separately blocked T3 preview does not prevent these
  tests. Combined default-config result: **813 passed, 76 failed, one skipped**.
- All 48 files in the demo's source freeze and all 86 recorded original bundle
  artifacts match their hashes. All 21 copied evidence files match their originals.
  No inference, replay, relabelling, or new runtime benchmark was performed for
  this PR checkpoint. No host security settings were changed.

Exact commands, identities, source-tree/preset hashes, and scope are recorded in
[CHECKPOINT_VERIFICATION.json](evidence/roi-timing-2026-11-01/CHECKPOINT_VERIFICATION.json).

V6 remains rejected: the frozen Semarang acceptance loses all 18 required
motorcycle observations after a partial-box contraction and two stationary-car
samples. Earlier v6-targeted assertions also remain failed. Known failures must
remain visible; this draft is an experimental checkpoint, not a release gate.
Earlier results in the follow-up documents describe their own commits and do
not certify this snapshot. Review is by the implementing agent; no independent
review, merge, or deployment is claimed.

## Proposed target and next experiments

Use a metric tied to this product before seeking a result above 90%. A proposed
primary target is **more than 90% of scorable, complete physical-car ROI visits
produce a correctly attributed total duration within ±1.0 second of independent
manual timing**. The one-second tolerance is a proposal, not an agreed product
requirement or measured result. Count missing, unknown, wrongly merged, and
unreconstructed split outputs as failures. Use one outcome per physical visit;
multiple fragments cannot improve the denominator. Predeclare ground-truth
ambiguity and boundary-censoring rules, and report already-present and
still-present visits separately. Also report coverage, median/p95 absolute
error, wrong-body links, missed/false cars, sample size, and uncertainty.

1. **Label physical ROI visits and locate the loss stage.** On the existing
   development clip, label entry/exit and supported identity through occlusion.
   Trace each bad timer through raw detector boxes, native returns, logical
   admission, and ROI membership. Separate a missing box from a returned box
   rejected by a tracker or the application. Reserve previously uninspected
   clips/camera periods for the final timing result; the 22-second demo is
   already development data. Keep Medan and Semarang as regression inputs.

2. **Run a controlled recall/association sweep from the existing cache.** R1
   already lowered the raw detector floor from the actual legacy 0.28 to 0.10,
   adding 4,387 boxes. Filtering the new stream at 0.28 reproduces all 24,210
   ordered legacy boxes exactly. The old `S30` label is a documented provenance
   error; its real floor was 0.28. Compare detector floors 0.10, 0.15, 0.20, and
   0.28 on that same cache with the native high/new thresholds fixed first.
   Then vary only the low-score association threshold while keeping track
   creation conservative. Keep tracker and logical-association factors
   separate, including the default v2 control. Score timing, coverage,
   duplicate births, wrong-body links, and latency. Boxes below 0.10 are absent
   from this cache; testing 0.05 needs a new capture. NMS or resolution changes
   also need new detector output.

3. **Improve detection only where the labels show missing raw boxes.** Test a
   higher detector input resolution compatible with the TensorRT export and
   measure deployed cost. It cannot restore detail absent from the source.
   If crowded/partial vehicles are the repeated cause, fine-tune with reviewed
   labels from this camera and held-out time segments. Test resolution,
   confidence, NMS, or weights one factor at a time. More low-score boxes can
   also be duplicates, merged boxes, or background; raw count is not recall.

4. **Fix supported-box rejection and remaining wrong links.** Diagnose the
   v6 partial-box veto that loses Semarang's motorcycle and car observations;
   distinguish contraction of one object from expansion across multiple cars.
   Preserve the merged-box safety fixtures. If identity confusion remains after
   usable detections are present, test a vehicle-trained ReID encoder on the
   same fixed boxes, with pinned preprocessing and weights. The earlier person
   encoders did not establish an adoption benefit; the vehicle-encoder arms
   were not completed. FastReID provides vehicle-domain configurations and
   weights, but their transfer to this camera and device is unproven.

5. **Accept only the measured improvement.** Evaluate once on held-out physical
   ROI visits, recheck stationary coverage and timer/export/source-boundary/A-B
   behavior, and measure complete pipeline latency/memory on the target Jetson.
   Keep observed and any separately justified hidden-time estimates separate.
   Do not meet the timing target by extending stale IDs, joining different
   cars, omitting failed visits, or counting unseen frames as observed time.

Increasing usable detector recall is a plausible next lever, not an established
root cause or a promise of >90%. The current evidence demonstrates both missing
coverage and incorrect associations, and the per-car timing baseline has not
yet been measured against independent physical visits.

## Primary references

- [ByteTrack paper](https://arxiv.org/abs/2110.06864): associates low-score boxes
  with existing tracklets rather than discarding all low-confidence detections.
- [Ultralytics tracking documentation](https://docs.ultralytics.com/modes/track/):
  separates detector filtering, high/low matching thresholds, new-track
  thresholds, buffers, and optional ReID. Pin the locally installed tracker
  implementation/version; current online defaults are not this experiment's
  frozen configuration.
- [Ultralytics model-evaluation guidance](https://docs.ultralytics.com/guides/model-evaluation-insights/):
  discusses confidence/precision/recall tradeoffs, image size, and held-out
  evaluation. Its generic dataset examples do not predict this clip's result.
- [FastReID model zoo](https://github.com/JDAI-CV/fast-reid/blob/master/MODEL_ZOO.md):
  includes VeRi and VehicleID models. Training-domain benchmarks do not establish
  vehicle identity accuracy or acceptable runtime on this deployment.
