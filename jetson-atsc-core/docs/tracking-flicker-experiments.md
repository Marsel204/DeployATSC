# Experimental identity resolvers: local checkpoint

Default association remains `joint_image_motion_appearance_v2`. The opt-in
`delayed_trajectory_v3` through `delayed_trajectory_v6` versions are preserved for
reproducible investigation. **None is adopted as a replacement. V6 is rejected.**
This commit records the bounded experiment and its regressions; it is not a
claim of production readiness, a green full suite, or solved physical tracking.

V6 confines nonadmitted confirmation evidence to one stable `(epoch, native)`
segment. An ineligible tentative cannot enter another segment's association
matrix, steal its detection, create false ambiguity, or inherit its age and
observations. Missing/unstable aliases cannot confirm an anonymous tentative.
The preceding v3–v5 implementations are dependencies retained for comparison.
Association provenance follows the explicitly selected version; native mode and
the v2 association default are unchanged.

Complete-prefix fixtures retain all competitors. V6 separates the frozen
different-vehicle pairs at queue frames 52→57, 75→81 and 141→185, including
provisional owners, and retains the same-SUV 90→91 link. Narrow regressions pass
14/14. All twelve earlier-arm streams across queue, Medan and Semarang reproduce
byte for byte, including admission/generation fields and waiting hashes.

Adoption fails: Semarang loses all 18 required motorcycle observations after a
full-rider/bike detection contracts to the lower bike, and stationary native28
car support falls 186→184. The symmetric shape veto still blocks a real merged
two-car expansion, so simply loosening it is not a complete solution. The
inherited same-native queue f144→145 wrong/merged-support link remains accepted.
The frozen v6-targeted selection has **157 passed and seven failed** assertions;
the separate coverage selection has **one pass and one failure**. Six existing
failures involve manually seeded entries without segment metadata; another
expects the prohibited tentative lineage. These explanations do not erase the
failures. No frozen assertions were weakened, skipped or xfailed.

Run the new narrow regressions with:

```sh
.venv/bin/python -B -m pytest -q tests/test_vehicle_identity_stage2d.py -k 'not acceptance'
```

Run the unchanged adoption requirements explicitly with:

```sh
.venv/bin/python -B -m pytest -q tests/test_vehicle_identity_stage2d.py -k acceptance
```

The second command intentionally remains red at this checkpoint; the default
complete pytest collection also includes that failed acceptance test. This is a
user-authorized local experimental commit, not a release. Historical full-suite
results do not certify this snapshot. Keep the known failures visible.

The five-block 50-vehicle/200-frame normal CPU workload reports median paired
p95 delta +0.241ms for v6, within the incremental +1ms gate. V6 median block p95
is 25.288ms, above the earlier 5ms total aspiration. Separate instrumented real
exceptional paths are not a paired live latency guarantee.

Local evidence is outside Git under
`reviews/tracking-flicker-2026-11-01/stage2d/v6/` and `diagnostic-v6/` in the
parent workspace. It includes exact source/harness freezes, corrected assignment
ownership, tests, gate decisions, raw replays, per-car CSV, photographed anchors
and the 660-frame annotated diagnostic. The 339 rendered delta endpoints have
no fresh independent labels; selected links are not a physical success rate.
The report-build log's pre-print hash mismatch is disclosed separately; original
evidence is preserved. Physical scale remains unverified and phase UNKNOWN.

Next investigation changes detector weights independently of tracker and
application association. A model comparison must save boxes before tracking,
normalize class names, keep the same source frames/ROI, and compare trackers on
identical cached detections. Fewer logical IDs alone do not establish success.
