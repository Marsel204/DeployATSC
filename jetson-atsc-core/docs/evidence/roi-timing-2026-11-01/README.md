# Frozen 22-second ROI tracking evidence

These files are byte-for-byte copies from the completed local demo on
1 November 2026, evaluated at product commit
`65990137637df478f810ec0f9e37db41f7481cea`. They are historical experiment
evidence; `VERIFICATION.json` does not describe the later PR test run.
`SNAPSHOT_MANIFEST.json` hashes every copied file. The scoped `.gitattributes`
preserves the copied CSVs' original CRLF and cell whitespace, excluding only
those frozen CSV data files from whitespace lint. No CSV content is normalized. The separately authored
`CHECKPOINT_VERIFICATION.json` records the current software checks and is not a
historical copy.

The main and detailed CSVs are included for both candidates. Unknown per-car
durations/departures remain `-`; unassigned candidates remain separate. Neither
candidate is adopted, and the selected pair scores are not physical-car timing
accuracy. See [the checkpoint](../../roi-timing-checkpoint.md) for interpretation,
current software verification, and the proposed next experiments.

The source video, detector weights, cached raw/native streams, frame traces,
interactive pages, crops, and annotated videos remain in the parent workspace
under `reviews/tracking-flicker-2026-11-01/`. They are not bundled here, so this
snapshot alone cannot rerun inference or reproduce the replay. Absolute paths in
the copied protocol identify those original local inputs. The completed demo is
available through the existing LAN gateway at
`http://192.168.0.107:8000/tracking-report/demo-22s/` while that gateway is running;
GitHub readers outside that LAN cannot use it. `FINAL_MANIFEST.json` retains the
original bundle hashes, including the omitted videos.
