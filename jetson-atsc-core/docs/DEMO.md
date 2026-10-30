# Feature demonstration preset

## Real traffic footage (prepared in this workspace)

```bash
cd jetson-atsc-core
.venv/bin/python scripts/launch_demo.py --preset data/demo_real/preset.json --device cuda:0 --port 8001
```

Open **http://localhost:8001**. This mode uses the actual `Final.pt` YOLO detector and CPU ANFIS, with no scripted detections. Camera 2 runs YOLO + ByteTrack continuously; Camera 1 only decodes a preview and runs YOLO once, on the single current frame, each time **Analyze frame & predict** is clicked. The preset and both camera profiles are loaded automatically. Click **Analyze frame & predict**, then **Resume**. Pause, step, stop, browse the saved evidence and export normally. The signal box and label are hidden from the video preview.

Use `--device cpu` when CUDA is unavailable. The GPU invocation above was smoke-tested with the local video; this is a demo execution check, not GPU-stack qualification.

The current locally prepared footage is the user-selected [Pasteur junction video by Abdul Muhyi TV](https://www.youtube.com/watch?v=YCd_bH9uM0M). The demo uses a contiguous 5:20–6:40 excerpt, resized to 960×540 and uniformly sampled at **15 FPS**. No objects, tracks or signal colors are generated. Media and local preset files are not included in Git. Exact source identity, hashes and processing notes are stored locally in `data/demo_real/REAL_SOURCE.json`. The previous 5 and 10 FPS clips and preset/source notes remain available, as do the Jackson Hole preset/source notes; existing saved runs are retained. Without these local assets, use the generated demonstration below or supply your own calibrated replay sources through the regular app.

The runtime line shows Camera 2's recent active playback FPS, source FPS, playback speed relative to real time, and steady inference latency. Paused playback reports zero FPS. The first cold inference is recorded separately, so it does not inflate steady inference latency. The historical `mean_inference_ms` field remains a lifetime average for compatibility. Clean calibration JPEGs are encoded on request, rather than for every playback frame. Tracking still processes every Camera 2 frame; changing the source FPS changes the duration represented by frame-count-based tracking thresholds, so this demonstration does not establish equivalent field measurements at different frame rates.

The observation backend prefers a locally verified TensorRT FP16 engine; see [setup and rollback](TENSORRT_OBSERVATION.md). Annotated streams are at most 640 pixels wide, and Camera 1 preview refresh is capped at 10 Hz during playback. Camera 1 still caches every decoded frame for the capture button, Camera 2 still processes and publishes every measurement frame, and clean calibration previews retain full source resolution. The engine's measured processing ceiling is distinct from a clip's playback FPS; use the displayed playback factor to detect slower-than-real-time processing.

This is a shared-recording demonstration: both camera roles reuse the same real clip. This handheld recording moves, so a fixed signal region can lose the lamp and produce UNKNOWN phases; the excerpt does not establish a complete measured green interval. Geometry is approximate and scale remains unverified; this is not a synchronized two-camera field experiment. Real-demo profiles and runs are isolated under `data/demo_real/`. Normal app settings and the earlier generated demo are untouched.

## Optional generated scene

```bash
cd jetson-atsc-core
.venv/bin/python scripts/launch_demo.py --port 8001
```

Open **http://localhost:8001**. The demo opens paused at 3 seconds with two separate, pre-calibrated video feeds. A completed example is already available under **Saved runs**.

1. Click **Analyze frame & predict**, then **Resume**. Red changes to green at video time 12 s; green ends at 20 s. The cycle table then shows the actual duration, recommendation and difference.
2. Try **Pause**, **Step 1 frame**, and **Resume**. Watch the three vehicles queue, accrue waiting time, cross the stop line and leave.
3. **Stop**, then **Start observation** to create another independent run. Each generated clip contains three 22-second cycles.
4. In **Saved runs**, select the completed example or your stopped run. Browse vehicles, predictions and cycles, compute exact quantiles, and export its JSON/CSV ZIP.
5. Both **DEMO** profiles can be loaded and opened in the calibration editor. The synthetic scene uses nominal virtual dimensions; it is not a real camera survey.

The demo uses **generated videos and scripted vehicle tracks**, clearly labelled on the dashboard. It exercises the normal observation engine, manual prediction, the actual deployed CPU ANFIS model, calibration, storage, history and exports. It does **not** demonstrate real YOLO detection, camera accuracy, hardware actuation, validated traffic measurements or LOS grading.

Demo media, profiles and runs are isolated under `data/demo/`; the reproducible start payload is `data/demo/preset.json`. Existing experiment evidence, research, weights and camera configuration are unchanged. The normal launcher remains `scripts/launch_ui.py`. The demo detector is injected only by `launch_demo.py`; ordinary app operation is unaffected. To regenerate only the assets/preset without starting a server, use `--prepare-only`.

## Real-time setup on Jetson

Before launching, pin clocks with `sudo jetson_clocks`. Revert with
`sudo jetson_clocks --restore`; clock settings reset at reboot. This stage never
changes clocks automatically. Schedutil under-clocks the bursty pipeline; an
uncapped OpenBLAS pool can otherwise prop up clocks by spinning.

The launchers apply `ATSC_BLAS_THREADS=auto` (default) **before numerical imports**:
cap OpenBLAS at one thread only when every online CPU policy is pinned (equal
min/max frequency or performance governor). Unknown/unpinned clocks leave BLAS
alone. `ATSC_BLAS_THREADS=default` disables tuning; a positive integer requests
that cap. An existing `OPENBLAS_NUM_THREADS` always wins. Restart after changes.
Telemetry/manifest host facts record this decision; `performance_advice` explains
unpinned clocks or uncapped BLAS, with the first advice beside Camera 2 runtime.

Start payload `persist_mode` defaults to `inline`. Opt-in `background` coalesces
routine commits while running, keeping disk IO off the measurement thread.
Paused/stepped commits and HTTP persists retain their existing behavior. Stop
waits for the worker before final persistence. A crash can lose measurements
since the last durable commit; the existing interrupted-run recovery still applies.

One-board measurements (Stage 3, pinned clocks, optimized tree): default BLAS →
one thread improved playback 0.949× → 0.986×, late frames 12.5% → 7.0–7.1%,
track p50/p95/p99 41.8/56.8/77.1 → 37.5/40.4/45.6 ms, CPU 0.352 → 0.054
CPU-s/frame (−85%), device power 8.39 → 7.76 W; tracker outputs were bit-identical.
Without pinned clocks the cap worsened playback by 1.3% as clocks fell to ~850 MHz.
These are one-board results, not performance guarantees for other clips/hardware.
