# Persistent vehicle IDs and red-stop timers

This optional observation mode gives the application ownership of a vehicle's
identity and waiting history. Native detector/tracker IDs remain evidence. A
supported returning detection can reconnect to the same logical vehicle even
when the native tracker assigns another ID.

In **Experiment settings**, select **Persistent vehicle IDs** before starting a
new run. The initial settings retain an occluded identity for ten source seconds
and allow estimates only for an entire supported gap of at most five seconds.
The run manifest freezes the requested/effective settings and algorithm versions.
Settings cannot be changed during the run.

The live and saved vehicle tables show confirmed, accepted estimated and pending
provisional red-stop time separately. A provisional amount can be withdrawn when
recovery contradicts it or the gap exceeds the cap. Previously confirmed time is
preserved. Missing acquisition frames, unknown signal intervals and source
discontinuities cannot become confirmed stopped time. File-replay pause does not
advance the timer.

Identity uses joint geometry, size and visible crop appearance. It abstains when
multiple assignments compete. Confirmed departure ends a visit; a replacement
gets a new logical ID. Logical motion history survives supported native-ID changes
and warms up again after a real gap. Raw IDs are available in row/interval evidence.

## A/B schedule comparison

To compare B waiting with A's recommended green:

1. Select persistent vehicle IDs and enable **Compare B waits using A's recommended green**.
2. Configure separate A/System and B/Acquisition sources. Open Camera 1 calibration
   and draw A's signal head (and individual lamp regions if needed). Save its
   geometry and scale reference. Camera 2 still needs B's signal and stop line.
3. Name A and B explicitly. Confirm the A-then-B relationship and the unchanged
   transition/clearance assumption only when it matches the experiment.
4. Start observation. Camera 1 automatically analyzes one image for each confirmed
   A-green episode; B tracking continues independently. Manual capture is disabled
   in this mode so two workers cannot use Camera 1's model concurrently.

The comparison pairs the A episode with fresh B-red evidence and waits for the
needed phase boundaries. Each vehicle's savings are an intersection of its own
red-stop intervals with the potentially shortened red interval. Late arrivals get
their own result. Missing boundaries, uncertain pairing and an unsupported later
release produce an unavailable result. Late inference is labeled retrospective.
The action deadline is A's proposed green end, before clearance, rather than B's
later release. A stale A feed cannot start analysis or reset B's vehicle timers.

These are schedule estimates under a declared assumption. They do not simulate
physical queue discharge and never control the traffic signal.

## API and saved evidence

Add this fragment to the existing calibrated start payload:

```json
{
  "vehicle_identity": {
    "mode": "persistent_v1",
    "occlusion_retention_sec": 10.0,
    "max_estimated_gap_sec": 5.0
  },
  "schedule_comparison": {
    "enabled": true,
    "a_movement_id": "approach-A",
    "b_movement_id": "approach-1",
    "unchanged_clearance": true
  }
}
```

Omit `schedule_comparison` for persistent timers alone. B must match the existing
`policy.movement_id`. Omitting both options uses the original native mode.

`red_wait_sec` stays confirmed-only. New fields include
`estimated_red_wait_sec`, `pending_estimated_red_wait_sec`, `display_red_wait_sec`,
`unresolved_coverage_sec` and `identity_scheme`. Estimated coverage is unobserved:
it overlaps the legacy unknown-motion duration rather than being additional time.
Best-estimate summaries exclude pending time and disclose partial/uncertain
canonical visits. Unresolved splits are not proof of exact physical uniqueness.

Exports contain the decomposition and source evidence. A/B runs also include
`schedule_episodes.csv` and `vehicle_savings.csv`, using optional typed records in
the existing archive protocol. Historical APIs accept both table names. Exact
quantiles accept `metric=confirmed` (the default) or `metric=best_estimate`;
the latter excludes pending estimates. Saved-run reading never resumes tracking.
Explicit recovery cancels unfinished pending estimates and keeps confirmed time.

## Validation and rollback

This is an experimental, opt-in mode. Deterministic and synthetic two-camera
tests verify accounting and episode binding. Uncalibrated stationary footage can
test identity continuity; it cannot certify physical per-vehicle red-wait accuracy.
The initial 50-vehicle CPU stress result exceeds the proposed five-millisecond
additional-cost target. Real-time and calibrated field acceptance remain pending.

Use a new native-mode run to roll back. Existing persistent runs remain readable;
their fields are not reinterpreted and old native runs are not migrated. Model
weights, native tracking defaults and observation-only signal operation remain
unchanged.
