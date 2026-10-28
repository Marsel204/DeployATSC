# TensorRT observation backend

Observation mode supports a board-local FP16, batch-one, 640×640 TensorRT engine. Camera 2 tracks one frame per call; Camera 1 detects one frozen frame only after a capture click. Separate detector instances keep their trackers and execution contexts independent. Legacy batched control does not automatically select this engine.

## Build and select

On the target CUDA Jetson, with its matching PyTorch, TensorRT and Ultralytics environment:

```bash
cd jetson-atsc-core
.venv/bin/python scripts/export_observation_engine.py
.venv/bin/python scripts/launch_ui.py --backend auto --port 8000
```

The builder copies `models/Final.pt` into a temporary directory before export. It publishes `models/Final_fp16.engine` and `models/Final_fp16.engine.json` together with hashes of the actual engine and source weights, the TensorRT version, GPU compute capability, precision, image size and batch size. The weights stay unchanged. Engine binaries and bindings are local build outputs, excluded from Git. Build time can be substantial; the tested engine took about 17 minutes on this board.

`auto` is the default. It validates the binding and runtime before selecting the engine, otherwise selecting PyTorch with a recorded reason. CPU demo launches use PyTorch. `--backend tensorrt` requires a valid engine and fails static preflight if it is unavailable or incompatible; `--backend pytorch` forces the original model. `ATSC_DETECTOR_BACKEND` sets the same default for observation sessions launched elsewhere, including the daemon's observation entrypoint. A runtime inference failure is reported; the app does not silently change models midway through a run.

```bash
# Explicit accelerated demo, using locally prepared media and profiles:
.venv/bin/python scripts/launch_demo.py --preset data/demo_real/preset.json --device cuda:0 --backend tensorrt --port 8001

# Roll back without changing weights or rewriting saved runs:
.venv/bin/python scripts/launch_demo.py --preset data/demo_real/preset.json --device cuda:0 --backend pytorch --port 8001
```

Run manifests include both the selected engine hash and original weights hash. Execution provenance records the effective backend artifact, device, precision and inference settings; telemetry also exposes the selection reason. Rebuild after changing weights, TensorRT or GPU compatibility. Do not reuse this batch-one engine for the old daemon's two-frame batches.

TensorRT's FP16 builder setting permits FP16 kernels; it does not mean every layer or the input tensor must use FP16. This tested engine accepts FP32 input. Execution provenance labels the legacy `effective_precision` field with `precision_scope: input_tensor` and records `engine_build_precision: fp16_enabled` separately, rather than misreporting FP32 input as an unaccelerated engine.

## Performance and interpretation

The earlier controlled tracking benchmark measured 60.3 ms/frame with PyTorch FP32 and 35.9 ms/frame with TensorRT FP16. In the integrated two-role pipeline, the measured 30-FPS-source processing ceiling was about 17 FPS at the board's existing 15 W power setting. This includes decoding, tracking, measurements, previews and persistence; it is lower than detector-only capacity. The local demo therefore uses a 15 FPS excerpt for playback headroom. Power and thermal settings were not changed.

The earlier 1,000-image, non-held-out dataset comparison measured mAP50–95 of 0.7201 for FP32 and 0.7130 for TensorRT FP16. This is evidence of a small numerical/detection difference, not held-out field accuracy. Track identities and traffic metrics can also change; do not combine runs from different backends as though their measurements were identical. The deterministic Camera 2 timing/wait oracle tests verify software semantics, not native-model accuracy.

Preview work is output-only: annotated streams are reduced to at most 640 pixels wide, Camera 1 preview refresh is capped during playback, and clean JPEGs are generated on request. The latest Camera 1 click candidate still updates on every decoded frame, and no Camera 2 tracking or measurement frames are deliberately skipped. Measurement timestamps remain on the source-time axis.
