# Native fragment continuation (experimental, off by default)

With **Native tracker IDs**, a vehicle that ByteTrack loses for a few frames
comes back under a new native ID and is normally booked as a second ledger row.
`RunPolicy.fragment_continuation` lets the new native ID continue the lost row
instead. It is opt-in and does nothing in **Persistent vehicle IDs** mode.

## Enable and revert

- Dashboard: *Setup → Detection thresholds, identity and A/B comparison →
  Native tracker IDs →* tick **Continue a lost vehicle with a new native ID
  (experimental)**, then start the run. A saved run reloads with the box set
  as it was recorded.
- API: `POST /api/observation/start` with
  `"policy": {"fragment_continuation": true, ...}`.
- Python: `RunPolicy(fragment_continuation=True)`.

Untick the box (or send `false` / omit the field) to restore one row per
native ID. The value is saved in the run manifest's policy.

## Rule

A native ID that this run has never seen (not active, not handed off, not seen
outside the region in the last association timeout, not in the archive index)
continues a row whose last observation was at most `max_gap_sec` earlier, when:

0. the lost ID was seen with a box in at least its last three frames before
   the gap (a one- or two-frame fragment, such as an edge sliver, is never
   continued);
1. on position alone, the newborn and the lost row are each other's only
   candidate within the motion gate (0.35 box heights, plus 0.3 per second of
   gap when at least three samples show image motion; without motion only a
   gap of up to 0.1 s is bridged);
2. the class matches and the box heights differ by at most 1.5x;
3. no other observed box, of any class, size or ID state, has its footpoint
   in the gate, or covers at least half of the smaller box with footpoints
   within half a box height of the newborn's (the f66 case in the
   0SgQn95Q07U replay).

A row is continued at most 8 times. A frame with more than 4,096
newborn x lost pairs skips continuation (`continuation_skipped_frames`).

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

Ledger rows are not physical vehicle counts. Frame review of the 0SgQn95Q07U
replay supports the three links the rule makes there; it is not a validation
of physical identity in general.
