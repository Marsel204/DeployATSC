# Two-hour ANFIS development run

Status: **candidate_and_comparison_complete**

Exactly three inputs: V_w, Q, L. This is a provisional development experiment, not the exhaustive thesis study.

Teacher: 10,20,…,120-second candidates; three equally weighted matched-demand contexts; unchanged 600-second horizon and 0.1-second simulation step.

Only complete twenty-family waves enter the dataset, preserving 70/15/15 parent splits and five-regime coverage. Partial/failed groups remain excluded and recorded.

Native/reference parity is checked for every context; fresh history is replayed for every candidate; the selected action is independently verified stepwise.

Selected generation workers: 3
Complete groups retained for training: 120

## Frozen candidate

Checkpoint SHA256: `41aa2836e92f587250f1d58ea5e8d262dea4e546afbfee67f98233907f2e657c`

Normal hybrid training, selected using validation only. The checkpoint is frozen before held-out label evaluation.
Coarse curves cannot give exact teacher regret for unevaluated integer model actions; no interpolation is reported as measured regret.

```json
{
  "test": {
    "bound_hit_fraction": 0.2222222222222222,
    "mae_sec": 22.41352293226454,
    "r2": 0.042991950540717694,
    "raw_mae_sec": 22.476712120903862,
    "rmse_sec": 29.573620390105543,
    "teacher_regret_note": "coarse teacher did not evaluate every model action; use direct SUMO comparison",
    "teacher_regret_vehicle_seconds": null
  },
  "validation": {
    "bound_hit_fraction": 0.2777777777777778,
    "mae_sec": 18.042521158854168,
    "r2": 0.2976822251194067,
    "raw_mae_sec": 18.10638173421224,
    "rmse_sec": 26.055847374141823,
    "teacher_regret_note": "coarse teacher did not evaluate every model action; use direct SUMO comparison",
    "teacher_regret_vehicle_seconds": null
  }
}
```

## Fresh development comparison

| Controller | Mean cost, vehicle-seconds | Running + pending |
| --- | ---: | ---: |
| candidate_anfis | 76296.661 | 2073 |
| deployed_anfis | 79087.785 | 2220 |
| fixed_30 | 81992.825 | 2318 |
| responsive_default | 77389.150 | 2147 |
| supplied_anfis | 104049.135 | 3010 |

Completed paired episodes: 10. Fixed/responsive parameters are predeclared, not calibrated on these outcomes.

No final benefit claim, field validation, automatic deployment, or change to historical checkpoints. CPU training is deliberate for the tiny 126-parameter model; the cloud's CPU/RAM accelerates generation.
