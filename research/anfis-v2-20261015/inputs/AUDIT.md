# Interpretation audit — 15 October 2026

The existing numerical results are unchanged. They describe the specified east/west replacement policy in a simplified queue model. They do not identify whether the complete traffic-control system would improve or worsen intersection-wide delay.

## 1. FLUID is useful reference data, not an optimal-control oracle

FLUID is the strongest reference source used in this project so far for reconstructing movements and signal states. It supplies metric trajectories, intersection entry times, maps, video and manually annotated signal states. Its authors compare trajectory processing against DataFromSky, manual counts and an instrumented vehicle on the first five minutes of FIDRT 20250526. The trajectories are processed model outputs, with smoothing and interpolation, rather than independent manual truth for every vehicle.

The paper describes a four-phase signal at FIDRT. It does not establish that the recorded signal timings minimize delay. A high-quality trajectory-processing method is not evidence of an optimal signal-control method. There is no supported basis for explaining this experiment's adverse result by saying that FLUID already uses the most efficient controller.

Source: [FLUID Scientific Data paper](https://www.nature.com/articles/s41597-026-07110-9), especially Data Records and Technical Validation. This is a statement about the evidence examined, not proof that the field controller was unoptimized.

## 2. The experiment does not isolate ANFIS quality

The experiment changes only the shared east/west through green, keeps its intervening red intervals at recorded lengths, and scores only east/west through vehicles. Other movements are not dynamically controlled or scored. This adapter was an experimental choice; it is not an implemented whole-intersection control policy.

Shortening green while keeping red fixed reduces the share of time available to the measured movements. For 20250526, the recorded greens are all 26 seconds and the median red interval is 114 seconds. ANFIS's median requested green is 14.015 seconds. Using these representative durations, the programmed green share falls from 26/(26+114) = 18.6% to 14.015/(14.015+114) = 10.9%. This illustration excludes the three-second yellow-service allowance; it is not a separately simulated outcome or a measured saturation-flow estimate.

This is a confirmed mechanism by which the chosen policy reduces east/west service. Its individual contribution to the observed delay difference has not been isolated by ablation. Shorter cycles also shift arrival alignment and could affect unmeasured movements. Consequently, the experiment supports an adverse conditional result for this adapter, not a general judgment that ANFIS is an inferior controller.

The eleven sensitivity scenarios retain this policy structure. Their consistently adverse results therefore do not establish robustness to a different phase-allocation policy.

## 3. A model/input compatibility concern is now documented

The original checkpoint is `/home/marsel/Projects/ANFISProductionSystem/run_s1000_skewed/anfis_best_model.pt`. It names features `V_w`, `Q`, `L` and target `G_opt`.

| Feature | Saved normalization minimum–maximum | Held-out primary input minimum–maximum |
|---|---:|---:|
| Weighted count Vw | 1.60–77.60 | 0–9.20 |
| Queue Q | 8.21–195.64 | 0–45.00 |
| Occupancy L | 4.74–44.50 | 0–25.55 |

Across 34 direction/opportunity inputs, 4 Vw, 4 Q and 6 L values fall below their saved normalization minimum and are clipped to zero by the deployed normalization. The largest tested Vw normalizes to 0.100 and the largest Q to 0.196. The experiment's approximate visible-buffer capacities are 9.75 PCU east and 8.27 PCU west; extra demand remains in the virtual queue while visible count and queue tail saturate.

These facts flag a possible mismatch between training feature definitions and the deployed 45 m observation region. They do not reveal the training distribution or prove that every low-range prediction is wrong. The training dataset and the rule used to construct `G_opt` were not present in the checkpoint directory inspected. Without those definitions, the model's labels cannot be treated as independently established delay-minimizing timings.

The checkpoint reports MAE 5.02 s and R² 0.901 against its test targets. These are stored training evaluation metrics, not independently reproduced here and not evidence of reduced traffic delay.

## 4. Export corruption is not supported by the diagnostic

The deployed ONNX model and the original checkpoint, evaluated through the repository's export architecture, agree to a maximum absolute difference of 0.00002861 seconds over all 34 held-out primary direction/opportunity inputs plus three diagnostic probes. This checks numerical export parity on these inputs; it does not validate the feature definitions or target construction.

The runtime interprets the output as an absolute green duration. The root README instead calls it a green extension. The checkpoint's `G_opt` name alone cannot settle the original target definition. This documentation conflict should be resolved against the training data and target-generation code before testing an alternative interpretation. The completed backtest consistently used the current runtime's absolute-duration interpretation.

## 5. Other limits still apply

One held-out recording failed the baseline validation screen (20250525 MAE 6.61 s). The other passed (20250526 MAE 3.24 s). Reconstructing the recorded schedule reasonably well does not independently validate behavior under a new schedule.

The experiment uses aggregate FIFO queues, an inferred queue-tail/count relationship, a 45 m observation region and a selected completed-route cohort. It does not model lanes, spatial spillback, motorcycles, opposing demand or pedestrian delay. Two ANFIS vehicles remain unserved at the recording horizon. The camera detector and tracker are bypassed, so their previously observed errors cannot explain this controller-only result.

## Defensible conclusion and next comparison

Keep FLUID as reference data. The current experiment found worse estimated east/west delay under its specified adapter. It has not established that the trained ANFIS itself is the cause, that the complete system fails, or that the recorded timings are optimal.

Before another benefit claim: recover the training feature and `G_opt` definitions; verify that the app computes the same quantities over a compatible observation region; specify the complete multi-phase scheduling policy; and compare recorded timing, a demand-calibrated fixed-time controller, a simple demand-responsive controller and ANFIS on identical arrivals, geometry, service and clearance constraints. Evaluate all controlled movements and report unfinished queues. Policy and feature changes must be labeled as a revised experiment, with fresh hold-out data where feasible; do not tune against these outcomes and present the same recordings as untouched confirmation.
