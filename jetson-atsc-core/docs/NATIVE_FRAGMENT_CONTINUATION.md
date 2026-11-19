# Native fragment continuation (experimental, off by default)

With **Native tracker IDs**, a vehicle that ByteTrack loses for a few frames
comes back under a new native ID and is normally booked as a second ledger row.
`RunPolicy.fragment_continuation` lets the new native ID continue the lost row
instead. It is opt-in and does nothing in **Persistent vehicle IDs** mode.

## Enable and revert

- Dashboard: *Setup → Detection thresholds, identity and A/B comparison →
  Native tracker IDs →* tick **Continue a lost vehicle with a new native ID
  (experimental)**, then start the run. While a run is current, the page
  restores the box from that run's live telemetry (`policy`). The saved-run
  browser loads evidence rows only, not the settings form; a saved run's
  setting is in its manifest policy and export.
- API: `POST /api/observation/start` with
  `"policy": {"fragment_continuation": true, ...}`.
- Python: `RunPolicy(fragment_continuation=True)`.

Untick the box (or send `false` / omit the field) to restore one row per
native ID. The value is saved in the run manifest's policy.

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
   gap of up to 0.1 s is bridged);
2. the class matches and the box heights differ by at most 1.5x;
3. no other box of the same frame, of any vehicle class, size or ID state,
   has its footpoint in the gate, or covers at least half of the smaller box
   with footpoints within half a box height of the newborn's (the f66 case in
   the 0SgQn95Q07U replay). "Other boxes" are:
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
   attributed to it, an emitted vehicle without an attributed `source_bbox`)
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
`track` continues the same trackers.

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
box from a neighbour. On the 660-frame junction clip the one link
(#626 -> #648) is unchanged. None of this validates physical identity in
general.
