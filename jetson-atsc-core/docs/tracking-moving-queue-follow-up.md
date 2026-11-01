# Moving queue tracking follow-up (1 November 2026)

**Physical tracking verdict: FAIL.** This repair blocks the original V2/V5/V12
wrong native-ID transitions, but other transfers, merged boxes and unresolved
losses remain. A stable application or native ID does not certify one physical
vehicle. No physical success rate is claimed.

The experiment evaluates the first 660 frames of
<https://www.youtube.com/shorts/IyER_OHm0OM> at 30 fps, 540×960 pixels, with the
original ROI, TensorRT weights and detector/ByteTrack settings fixed. The last
sample is 21.9666667 seconds. Physical scale is unverified and phase is UNKNOWN;
these are image-ROI sample timings, not calibrated red waits or full residence.

## Demonstrated causes and repair

- The appearance-trusted box also supplied geometry. At frame 364 / 12.133s,
  V2's geometry reference was 8.5 seconds old but used a recent last-seen clock.
  Geometry now comes from the latest associated observation; reliable appearance
  can update on supported sparse returns.
- Image prediction depended on metric-speed validity and a noisy single-step
  extrapolation. At frame 77 / 2.567s, V12 chose raw413 at cost .21636 while its
  original raw232 SUV was still detected and gated out. A bounded image history
  now predicts motion independently of physical scale, with tighter residuals.
- At frame 517 / 17.233s, V5 chose appearance-unreliable red raw1803 at cost .33451
  over the still-visible white SUV raw9 at .49204. The .15752 difference passed
  the old .12 ambiguity margin. Changed native references now require reliable,
  closer appearance and spatial support. Native references remain hints.
- Completed crossed tracks were retired every frame, repeatedly clearing their
  motion evidence while still eligible for association. Retirement is idempotent;
  sparse observations retain prior image motion, and known moving vehicles cannot
  reconnect across long blind gaps using stale coordinates.

Association provenance is `joint_image_motion_appearance_v2`; the opt-in API mode
remains `persistent_v1`. The ten-second retention and five-second estimate cap,
confirmed/estimated/provisional accounting, export schema and A/B calculations
are preserved. Image motion does not validate physical scale or metric waits.

## Evidence and limits

The actual baseline was reproduced exactly. The final actual-engine TensorRT run
`run_53f78f61a3b44b82` has identical detector/feature evidence on all 660 frames and
identical assignments to the fixed engine on the frozen baseline detector stream.
Raw YOLO boxes before ByteTrack were also captured. Two diagnostic runs that
accidentally selected an injected PyTorch backend were rejected and retained as
failed attempts, excluded from the comparison.

The fixed original V5 follows only native9 and has a sampled supported ROI exit at
17.533s. Baseline V12 becomes fixed V13 and loses support after 2.467s; later
fragments do not establish complete tracking. V2's later raw816 transfer is blocked,
but its earlier same-native merged-box support remains unreliable. Endpoint
self-review of all 20 apparent recoveries also rejects V10, V38, V55 and V69.
For example, V38 changes from a dark/red car at 15.333s to a light vehicle at
18.033s. Simple crop color, partial boxes and stable native labels cannot resolve
all crowded physical identities. No exhaustive physical annotations are claimed.

| Software outcome | Original audited baseline | Fixed audited report |
|---|---:|---:|
| Application car-ID rows | 81 | 87 |
| Detected | 22 | 34 |
| Loss and Recovered | 15 | 16 |
| Loss | 44 | 37 |
| Never-admitted raw-car candidates (separate) | 9 | 10 |
| Admitted in-ROI car samples / same returned samples | 10,630 / 12,829 | 11,334 / 12,829 |

Counts can include fragments and wrong associations; they are not unique-car
counts, recall or physical tracking accuracy. The fixed audit sampled more tracks
than the original audit, so the number of rejected rows is not a measured error
rate comparison. Rejected per-car duration/departure is `-`. Missing samples never
count as observed ROI time; an exit needs direct inside→outside evidence and a
consecutive .30s outside run. Stop-line crossings and cutoff are not ROI exits.

The earlier frozen stationary inputs were rechecked: Medan's van raw37→116 still
recovers V28; Semarang's orange truck retains V13 with transient raw95/103 left
unresolved. Whole-scene admitted ROI coverage decreases 6,342→6,106 (-3.72%) and
2,266→2,146 (-5.30%). Application IDs increase 33→37 in Medan and remain 18 in
Semarang. These are disclosed conservative-association tradeoffs.

Local review bundle (outside Git):
`reviews/vehicle-tracking-2026-11-01/verified-engine/` in the parent workspace.
It contains the frozen spec and capture, candidate costs, pre/post-ByteTrack frame
pairs, audit sheets, CSVs, interactive report, annotated replay and investigation.
Original artifacts remain in `reviews/youtube-stationary-2026-11-01/`.
Software checks, source/harness hashes and paired CPU stress results are retained
under the follow-up's `validation/` directory. Review is self-review by Codex Sol
with max reasoning; no independent review or AEK enforcement is claimed.

Repository publication is local-only at the user's request. PR #6 was already
merged; no replacement PR, push, merge or deployment is part of this follow-up.

## Verification and runtime

The ten focused regressions all fail against the original registry for the
expected association/motion assertions. They cover three recorded transitions,
stale geometry, unreliable changed-native returns, image-only prediction,
unverified-scale invariance, sparse motion and repeated retirement.

The full nonbrowser suite completes with **693 passed, 75 inherited failures and
one existing skip**; failed identities exactly match the previously verified
commit. The initial run was aborted by a memory guard and is not a passing run.
The accepted rerun preserved memory/time guards and used the current one-minute
startup load rather than the previous five-minute window, which still included
the aborted work. Exact commands, hashes and JUnit results are in the review bundle.

Paired original 200-frame synthetic CPU stress (50 visible vehicles, ten warmup
frames excluded, one thread, no neural inference): baseline median **19.64ms**,
p95 **21.22ms**; fixed median **24.50ms**, p95 **25.82ms**. The p95 increase is
**4.60ms**. This exceeds the proposed **5ms total feature/identity/accounting p95**
target and the historical 19.62ms result. CPU clocks were unpinned; fixed ran after
baseline in the same process. Allocator-dependent RSS deltas are not a clean
incremental-memory estimate. Real-time acceptance remains unmet.

The separate full browser group passes **20 tests**. Combined software result:
**713 passed, 75 inherited failures with identical failure identities, one existing
skip; no new failure identities.** This is baseline parity, not a green suite.
