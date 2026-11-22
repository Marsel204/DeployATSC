# Native fragment continuation (experimental, off by default)

With **Native tracker IDs**, a vehicle that ByteTrack loses for a few frames
comes back under a new native ID and is normally booked as a second ledger row.
`RunPolicy.fragment_continuation` lets the new native ID continue the lost row
instead. It is opt-in and does nothing in **Persistent vehicle IDs** mode.

## Enable and revert

- Dashboard: *Setup → Detection thresholds, identity and A/B comparison →
  Native tracker IDs →* tick **Continue a lost vehicle with a new native ID
  (experimental)** and/or set **ROI exit persistence (s)** (default `0`), then
  start the run. While a run is current, the page restores both controls from
  that run's live telemetry (`policy`). The saved-run browser loads evidence
  rows only, not the settings form; a saved run's setting is in its manifest
  policy and export.
- API: `POST /api/observation/start` with
  `"policy": {"fragment_continuation": true, "exit_persistence_sec": 0.3, ...}`.
- Python: `RunPolicy(fragment_continuation=True, exit_persistence_sec=0.3)`.

Untick the box / set `exit_persistence_sec` to `0` (or omit the fields) to
restore immediate exit completion and one row per native ID. Both values are
saved in the run manifest's policy and ignored in **Persistent vehicle IDs**
mode (`PersistentVehicleWaits` uses `IdentityPolicy.exit_persistence_sec`
instead and never double-applies `RunPolicy.exit_persistence_sec`).

## Rule

A native ID that this run has never seen (not active, not handed off, not seen
outside the region in the last association timeout, not in the archive index)
continues a row whose last observation was at most `max_gap_sec` earlier, when:

0. the lost ID has at least three recent box sightings, each at most
   `max_gap_sec` after the previous one, spanning at least 0.1 s (a one- or
   two-frame fragment, such as an edge sliver, is never continued). The
   sightings need not be on consecutive frames: frames 0, 2 and 4 qualify;
1. on position alone, the newborn and the lost row are each other's only
   candidate within the motion gate (0.35 box heights, plus 0.3 per second of
   gap when at least three samples show image motion; without motion only a
   gap of up to 0.1 s is bridged). Whenever the bottom edge retreats upward
   while the box shrinks (`vy < 0` and `dh_dt < 0`), motion qualification uses
   the top-edge fit `(vx, min(0, vy_top))` against the maximum validated height
   in recent history so box shrinkage never lowers the speed threshold or
   promotes a slow/stationary vehicle to moving; when height loss dominates
   (`vy_top > dh_dt`), vertical prediction also falls back to `min(0, vy_top)`,
   preserving horizontal velocity without lifting the prediction upward. When a
   stationary vehicle shows an unambiguous bottom-edge
   occlusion tail (two consecutive earlier sightings supporting a taller baseline
   `h_base`, height shrinking to at most `0.75 * h_base` with net negative
   height slope `< -0.35 * h_base/s` from the bottom edge moving upward while
   its top edge and horizontal center stay within `0.15 * h_base` and top speed
   stays under `0.35 * h_base/s`), reappearance up to 0.35 s (capped by
   `max_gap_sec`) is anchored on the unoccluded top-center within `0.25 * h_base`
   (`rule: "mutually_unique_uncontested_occlusion_tail_gate"`), with the
   newborn's bottom edge within `0.15 * h_base` of the lost vehicle's
   clipped-to-baseline vertical band and height between `last_height / 1.5` and
   `1.2 * h_base`. Plain stationary and short (< 3 sightings) lost tracks stay
   ineligible to link past 0.1 s (blocking queue pull-up replacements), while
   remaining ambiguity vetoes against extended occlusion-tail links throughout
   the 0.35 s window. A newborn counts as a candidate of every lost row whose
   gate holds either its emitted anchor or its raw detection's (`source_bbox`),
   but it links only through its emitted anchor: the raw box widens ambiguity,
   never acceptance.
   A supplied `source_bbox` that is unreadable, non-finite or degenerate fails
   closed: that newborn can still make others ambiguous but never links;
2. the class matches and the box heights agree (at most 1.5x for normal links;
   bounded to `[last_height / 1.5, 1.2 * peak_height]` for occlusion-tail links);
3. no other box of the same frame, of any vehicle class, size or ID state,
   has its anchor in the gate (for occlusion-tail links, also no competitor
   footpoint within the static gate of the last footpoint or within 0.25 peak
   heights of the pre-occlusion peak footpoint), or covers at least half of the
   smaller box with footpoints (or top-centers for occlusion-tail links) within
   half a box height of the newborn's (the f66 case in the 0SgQn95Q07U replay).
   The overlap test runs for both the newborn's emitted box and its raw
   detection, against each other box (review probe: a newborn's Kalman box
   covered under half of a withheld detection its raw box covered most of).
   "Other boxes" are:
   - every other observation's emitted (Kalman-smoothed) box, **and** its raw
     post-NMS detection (`source_bbox`) when the tracker emitted it from one.
     Smoothing can move an emitted box out of the gate while its detection is
     in it (review probe: ID 1 lost at x=100, 14 px gate, newborn at x=112;
     an existing track's detection at x=86.1 was emitted at x=85.47), so
     either one vetoes;
   - the frame's actual post-NMS detections that ByteTrack withheld
     (discarded low-score boxes and new tracks awaiting confirmation).
   Raw sources and withheld boxes are identified by the tracker's own
   detection index, so the newborn's own detection is never its competitor
   however much the tracker smoothed its box. They only veto links; they
   never become ledger rows.
4. that same-frame evidence is complete. `DetectorTracker.track` returns
   per-frame lists carrying it (`TrackedFrame.withheld`, plus `source_bbox`
   on every emitted vehicle). When it is unavailable (tracker error and
   detection-only fallback, a detector without the provenance hook, a frame
   from an older tracker epoch, an image whose tracker updates cannot be
   attributed to it, an emitted vehicle without an attributed `source_bbox`,
   non-finite vehicle coordinates, or any unreadable/malformed competitor box
   or withheld entry)
   no link is made in that frame (`continuation_unavailable_frames`, counted
   when a newborn and a lost row were both present).
   Direct callers pass the withheld boxes themselves:
   `VehicleWaits.observe(..., unassigned=...)` and
   `ObservationEngine.observe_acquisition(..., unassigned_boxes=...)`. `()`
   asserts that the observations are every detection of the frame, and the
   default (`None`) declares the evidence unavailable. A direct caller's
   `VehicleObservation.source_bbox=None` means `bbox` is itself the raw
   detection; a custom detector that sets `TrackedFrame.withheld` must also
   set `source_bbox` on each vehicle, or the session treats the frame as
   unavailable.

A row is continued at most 8 times. A frame with more than 4,096
newborn x lost pairs skips continuation (`continuation_skipped_frames`).
The provenance hook is an `on_predict_batch_start` callback: it runs after
Ultralytics' `on_predict_start` created the trackers and before
`on_predict_postprocess_end` updates them, and records each update's image,
input detections and output indices without a second inference. Capture is
opt-in (`DetectorTracker.enable_withheld_evidence`): a session enables it on
the Camera 2 detector only when continuation is on and identity is native, so
default-off and persistent-identity runs pay nothing and keep the historical
`observe_acquisition` call. Updates are attributed to images by identity, not
position: Ultralytics 8.3.x skips the update for an image without
detections (that image has nothing withheld); an image with detections and no
update, or with more than one update, has no evidence. `reset_tracking`
removes the installed tracker callbacks before dropping the predictor, since
8.3.x would otherwise append a second registration and update twice per frame.
When `track` fails (for example a tracker exception), the detection-only
`predict` fallback runs with the installed tracker callbacks detached and
restored afterwards: 8.3.x would otherwise run `tracker.update` in predict
mode, returning tracker IDs after a one-shot failure or raising again after a
repeated one. Fallback frames have no stable IDs and no evidence; the next
`track` continues the same trackers. When the tracker was never built (for
example `BYTETracker.__init__` raised on the first frame, or left an empty
`trackers` placeholder) the callbacks stay detached and the placeholder is
removed, on the model's and the predictor's callback lists alike, so the
retry registers exactly one pair on 8.3.0 and 8.4.x. If trackers are rebuilt
after the epoch has emitted stable IDs (8.3.0's `BYTETracker.__init__`
restarts the ID counter), the detector starts a new epoch first, so
`(epoch, local_id)` never names two vehicles; a first-frame failure before
any stable ID keeps epoch 0.

A handed-off native ID seen again contests the link before any row of that
frame is completed, so a return on the frame where the continuing ID crosses
the stop line flags the link whatever the observation order. Sightings
outside the region older than `association_gap_sec` are expired before
matching, so they never block a newborn on the frame they lapse.

## Accounting

- The interval between the old ID's last frame and the new ID's first frame
  is unknown coverage, even on adjacent frames. It gives no wait, no stop-line
  crossing and no motion history. The stop state restarts on the new ID.
- A crossing is counted only from the new ID's own consecutive frames.
- **Bottom-edge occlusion crossing resilience** (active when
  `fragment_continuation=True`): when a stationary vehicle's box shrinks upward
  from the bottom edge (`height <= 0.85 * h_base` against two consecutive
  earlier sightings or a preserved stationary pre-clip baseline, bottom-edge
  share `>= 0.70`, pinned top edge and horizontal center within `0.15 * h_base`,
  top speed `< 0.35 * h_base/s`, with history spanning at least 0.1 s up to 32
  samples at high frame rates) and restoring `h_base` places its unclipped rear
  back behind the stop line (`side((x, y1 + h_base)) < 0`), the premature
  `SERVED` crossing is held as a pending crossing rather than completing the
  row. A qualifying stationary clip also saves the pre-clip baseline behind the
  line, and behind-line sightings keep the pending anchor until two consecutive
  behind-line observations geometrically recover (`height >= 0.95 * h_base`),
  preserving the pre-clip baseline while the box top and center stay within
  `0.15 * h_base` so sustained behind-line clips or partial/95% recoveries
  followed by another clip cannot degrade the baseline; genuine departures
  (including full-height departures after recovery, or clipped departures whose
  unclipped rear `side((x, y1 + h_base)) >= 0` clears the line) confirm
  immediately across consecutive in-ROI, `EXIT_PENDING` outside, and same-ID
  re-entry frames. Frame gaps, handoffs, confirmed exits, and invalidations
  clear any pending crossing and stationary clip baseline.
- **Opt-in ROI exit hysteresis (`RunPolicy.exit_persistence_sec`, default `0.0`,
  bounds `[0, 5]`)**: when non-zero, an active native vehicle stepping outside
  the ROI without a stop-line crossing enters `EXIT_PENDING` rather than
  immediately completing as `EXITED`. Genuine stop-line crossings on exit or on
  consecutive `EXIT_PENDING` outside frames still complete immediately as
  `SERVED` (using the same `max_gap_sec` continuity check as exit timing). While
  `EXIT_PENDING`, no outside intervals or wait are fabricated,
  `state["exit_candidate"]` protects revisable prefix retention, and tracked
  outside sightings refresh association-gap expiry while preserving
  `row["last_seen"]` as the last in-ROI timestamp. If the vehicle returns inside
  the ROI before `exit_persistence_sec` of consecutive outside observations
  elapses, the row stays continuous (no `:segment-*` split), crossing continuity
  is preserved across consecutive same-ID transitions, and the outside dip is
  recorded honestly as `unknown_sec` (`observation_gap`). Once consecutive
  outside sightings span at least `exit_persistence_sec`, the row completes as
  `EXITED` at the first outside timestamp (`completed_at = exit_candidate`). If
  the track disappears before confirmation, normal `tracking_lost` /
  `right_censored` handling applies. Native exit timer reads and resets are
  strictly gated on `RunPolicy.exit_persistence_sec > 0` so they never collide
  with `PersistentVehicleWaits`'s own `exit_candidate`. Tracks in `EXIT_PENDING`
  cannot continue onto newborns inside the ROI, and use their latest outside
  timestamp and box to veto nearby extended occlusion-tail links.
- No departure is inferred from a vanishing vehicle.
- Rows carry `native_ids`, `identity_links` and the `identity_continued` flag.
  If a handed-off ID is seen again while the row is unfinished, the row gets
  `identity_link_contested` and the link gets `contested_at`. The returning ID
  starts its own row (its own next segment), marked left-censored.
- Each raw ID keeps its own row count and crossing history in memory and in
  the archive index, so eviction timing, acknowledgment order and index
  rebuilds give the same rows.

Ledger rows are not physical vehicle counts. On the frozen 0SgQn95Q07U
post-NMS boxes (both 220-frame ROIs) the rule makes two links (#191 -> #213,
#670 -> #692), which frame review supports. The third pairing the frames
support, #368 -> #534 at f132, is declined: ByteTrack withheld a 0.343
motorcycle box on the same motorcycle, and the rule cannot tell a duplicate
box from a neighbour. On the 660-frame junction clip (Ultralytics 8.4.189),
bottom-edge crossing resilience prevents premature `SERVED` completions on
`#232` (f134, later crosses cleanly at f373), `#407` (f135), and `#300` (f142),
unlocking five verified links: `#626 -> #648` (moving), plus `#291 -> #598`,
`#746 -> #966`, `#407 -> #733`, and `#300 -> #755` (stationary bottom-edge
occlusion tails); on Ultralytics 8.3.0, two links are made (`#235 -> #241`
moving and `#164 -> #221` occlusion tail). Enabling `exit_persistence_sec=0.3`
on the junction clip eliminates border-wobble `:segment-*` splits on `#1429`,
`#1468`, `#1484`, `#1924`, and collapses `#350` from 22 fragmented rows to 3.
None of this validates physical identity in general.

## Tracker profile evaluation (ByteTrack / BoT-SORT)

Sixteen tracker parameterizations (ByteTrack `track_buffer` 15–90,
`match_thresh` 0.80–0.90, `fuse_score` on/off, `track_high_thresh` 0.25–0.35,
`low_score` floor 0.10, and BoT-SORT with `gmc_method=none` and `sparseOptFlow`)
were benchmarked across frozen post-NMS detections on `0SgQn95Q07U` (left
approach) and `PKVSKsmBNZA` (crowded right approach) on Ultralytics 8.4.189 and
8.3.0:
- Relaxing ByteTrack (`match_thresh=0.85/0.90`, `fuse_score=False`, or
  `track_buffer=45/60/90`) substantially increased cross-vehicle ID jumps
  (`>1.0x` box height) and class flips on `PKVSKsmBNZA` (`24 -> 47..86` jumps,
  `25 -> 33..56` class flips) or increased ledger row counts.
- BoT-SORT with `gmc_method=sparseOptFlow` was ~4x slower (~28 ms median vs
  ~6.8 ms) for stationary CCTV with no continuity benefit; with
  `gmc_method=none` it dropped a genuine crossing on `0SgQn95Q07U` (15 -> 14),
  re-attached across a 27-frame gap in dense flow, and raised `PKVSKsmBNZA`
  class flips (25 -> 29).
- Tighter settings (`track_buffer=15`, tighter BoT-SORT) reduced jumps at the
  cost of more ID fragmentation.

Accordingly, `baseline` (`bytetrack.yaml`) and `low_score` remain the only
registered tracking profiles; no unsafe tracker profile was added.

## Known limits

- An ROI that reaches the image border limits witnessed entries: vehicles
  entering there are first seen already inside the region.
- Prolonged ROI departures exceeding `exit_persistence_sec` (e.g. a vehicle
  re-entering after >0.30 s outside) intentionally start a new visit segment.
- ByteTrack's internal body switches (one native ID moving to another vehicle)
  remain upstream of `VehicleWaits`; continuation neither causes nor repairs
  them.
