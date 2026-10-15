# Professor demonstration

Repository edition: the video is distributed separately, not checked into Git.
The timestamps and speaking notes below describe that existing recording.

Play `recordings/ANFIS_full_experiment.mp4`. It contains the full fresh browser recording followed by a readable vehicle-ledger appendix; neither recording is cut or sped up. It includes reviewed setup, the complete 400-second application replay (all 2,000 role-frames processed), the fresh deterministic backtest, all 136 scored vehicle results, sensitivities and limitations. The appendix repeats the same results with every row fully within the frame; the first table pass has some partially clipped bottom rows. Pause on any appendix page to inspect the individual vehicles. Some setup controls are collapsed through a browser-only recording stylesheet after startup so the feeds and three inputs remain visible. The application source and deployed model are unchanged.

The recording is silent, with optional English captions embedded in the MP4 and supplied as SRT. MP4 chapters support seeking. Use the short speaking notes below while presenting.

The original recording scripts retain machine-specific paths. This repository includes a portable controller-backtest runner in the parent directory; it does not include or reproduce the app video recording.

## Chapters

- **00:01** Introduction
- **00:11** Application setup
- **00:26** Full application replay
- **09:27** Application evidence and measurement limits
- **09:51** Fresh vehicle backtest
- **10:02** Backtest summary
- **10:20** All 136 vehicle results
- **11:33** All eleven sensitivity scenarios
- **11:47** Conclusion and limitations
- **12:03** Readable vehicle ledger appendix

## Speaking notes

1. “This model uses exactly three inputs: weighted count, queue length and occupancy. The frozen candidate has 27 Sugeno rules. This demonstration reuses its weights; training is not repeated.”
2. “The first experiment is the actual application on a local PC CPU. We review geometry and signal regions, then replay the prepared Bellevue video through two independent tracker roles. Both roles share one recording. The recommendation is observational; no traffic signal is controlled.”
3. “We retain the app's measurement limitations. Partial waiting-time records cannot establish full control delay, so the app correctly leaves LOS ungraded.”
4. “The second experiment uses FLUID trajectory arrivals in the same frozen E/W point-queue adapter as the previous analysis. It compares original timing, the old deployed model and the new candidate for the same 136 eligible vehicles. It is separate from the Bellevue video.”
5. “On the better-validated 26 May case, modeled mean delay changes from 47.98 to 41.67 seconds per vehicle, or 13.2% lower than the modeled recorded schedule. The old model produced 56.20 seconds. Every one of its 109 scored vehicles is served.”
6. “Pause the vehicle pages to inspect individual winners and losers. Mean improvement does not mean every driver benefits. We retain the failed 25 May baseline check, its unfinished journeys and all eleven sensitivity scenarios.”
7. “The indicative delay band improves E to D versus the old model on 26 May, but remains D versus recorded timings. Full-intersection LOS and field benefit remain unproven because competing movements and complete control-delay coverage are missing.”

## Evidence and credits

In this repository, `../evidence/app/` contains the app test summaries and inference reference. `../evidence/backtest/` contains every scored vehicle, background/eligibility audit, scenario records and model/source identities. `../evidence/recording_verification.json` records the original video codec, resolution, chapter count, hashes and complete decoding check. The full native app recording evidence remains in the separately distributed presentation ZIP.

Application video: City of Bellevue, 116th Avenue recording, retained as the project's prepared 400-second, 2.5 fps source with signal-color preprocessing. Original acquisition/source documentation remains in the existing experiment folder. Source project: https://github.com/City-of-Bellevue/TrafficVideoDataset . The geometry is approximate and covers a partial approach.

Vehicle references: Chen et al., FLUID, https://figshare.com/articles/dataset/29974954 , derived-data file 64179082, CC BY 4.0. These previously inspected recordings are reused development data. No fresh-confirmation or field-causal claim is made.

LOS bands: FHWA 2008 Traffic Signal Timing Manual, Table 3-3, https://ops.fhwa.dot.gov/publications/fhwahop08024/chapter3.htm . Applied here only as indicative mean corridor-delay bands.

The `_layout-check` directory is an initial framing check that was stopped to make the input metrics readable. It is excluded from the presentation package and does not contribute performance results. The final recording starts a fresh complete application run.
