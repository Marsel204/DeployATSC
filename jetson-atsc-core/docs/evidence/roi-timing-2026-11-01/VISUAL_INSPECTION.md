# Selected visual inspection of both 22-second replays

Implementing-agent self-review of source crops, not an independent physical-car census. Ten apparently successful histories were selected mechanically: the three longest recovered and two longest detected rows in each candidate. The frozen 94/18/42 reference labels were not changed. All other physical identities remain unvalidated.

The original V2 and V12 strips show visibly different bodies at their historical endpoints. Both candidates now assign separate owners to those endpoints. The original V5 later red-car box has no application owner in either candidate; this is not a correct recovery. The f460→541 strip still shows BoT-SORT V36 on two different bodies, while ByteTrack lacks a later owner. Both keep V9 through the already labelled merged/occluded support at f144→145. Exact rows touched by rejected evidence retain Loss and unknown per-car duration/departure.

| Candidate/local ID | Sampled frames | Reading |
|---|---|---|
| ByteTrack V35 | 117,450,452,551 | Dark SUV snapshots look consistent across the short gap; early bus/foreground occlusion limits identity confidence. |
| ByteTrack V46 | 133,174,178,423 | Distinctive dark car and continuous rear light bar look consistent across sampled endpoints. Does not certify the whole trajectory. |
| ByteTrack V65 | 169,174,176,440 | Heavy overlap with another dark vehicle makes the detector's physical support ambiguous. Recovery remains software-only and uncertified. |
| ByteTrack V5 | 0,175,351,526 | White-car snapshots look consistent, with substantial partial/foreground occlusion. This local V5 is not a claim that the original run's V5 red-car recovery succeeded. |
| ByteTrack V1 | 0,172,344,515 | Sampled white SUV looks consistent through observed outside-ROI support. |
| BoT-SORT V32 | 117,117,119,551 | Dark SUV snapshots look consistent, but an initial bus occludes substantial body area. |
| BoT-SORT V13 | 50,74,76,407 | White SUV snapshots appear compatible; late overlap prevents certification of all intervening support. |
| BoT-SORT V8 | 27,27,44,391 | Four snapshots alone are insufficient: framing and apparent shape change substantially. An additional trajectory sheet examines 27,74,90,117,133,141,152,160,175,250,391. Similar white vehicles, changing framing and overlap still prevent a confident physical-identity verdict; no new different-car label is invented. |
| BoT-SORT V5 | 0,175,351,526 | Same limits as ByteTrack V5: compatible white-body snapshots with occlusion. |
| BoT-SORT V1 | 0,172,344,515 | Sampled white SUV looks consistent through observed outside-ROI support. |

The additional BoT-SORT V8 native alias changes 87→777 between f141 and f152. Keeping its logical ID is an application reassociation, not proof that both native segments follow one physical car. The sheet is linked for inspection.

Selected apparently good snapshots are not an exhaustive check and do not establish physical success rates. A zero observed wrong-link count on the selected reference pairs does not waive the other frozen wrong/merged evidence or the failed Semarang acceptance requirement.

The first report-adapter attempt stopped while mapping an early, still-tentative legacy V10 sample. It was corrected to use the report's existing logical-or-tentative ownership rule; neither replay, native output nor production code changed. The first render process exited 143 after encoding all three 660-frame videos. The videos were retained and decoded independently with single-thread FFmpeg checks; no partial video is accepted as the final replay.
