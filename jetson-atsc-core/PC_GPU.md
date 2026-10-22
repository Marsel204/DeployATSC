# NVIDIA PC runtime

This branch supports a Linux x86_64 PC with an NVIDIA GPU, separately from the
Jetson/JetPack setup in `requirements-edge.txt`. The PC dependency set uses the
official CUDA 13.0 wheels for PyTorch 2.12.0 and torchvision 0.27.0. See the
[official PyTorch package matrix](https://pytorch.org/get-started/previous-versions/).
An NVIDIA driver compatible with CUDA 13.0 is required. A separate system CUDA
toolkit installation is not required for these prebuilt wheels.

From `jetson-atsc-core/`, using Python 3.11:

```bash
python3.11 -m venv .venv
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install -r requirements-pc.txt
.venv/bin/python -m pip check
```

To replace just the torch packages in an existing app environment:

```bash
.venv/bin/python -m pip install \
  torch==2.12.0 torchvision==0.27.0 \
  --index-url https://download.pytorch.org/whl/cu130
```

Verify the environment before launching:

```bash
.venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), (torch.__version__, torch.version.cuda)
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_name(0))
x = torch.ones((64, 64), device="cuda:0")
assert (x @ x).sum().item() == 64 ** 3
print("CUDA computation passed")
PY
.venv/bin/python scripts/launch_ui.py --host 127.0.0.1 --port 8000
```

Restart an already running app after installing packages. Its existing Python
process retains imported packages and model objects until restarted.

The detector defaults to `device="auto"`: it selects `cuda:0` when PyTorch can
use CUDA, otherwise `cpu`. Both YOLO tracking and the prediction fallback receive
the selected device. An explicit `device="cpu"` or `device="cuda:1"` remains
available to callers. Explicit CUDA requests are not silently changed to CPU.

After starting an observation, inspect `/api/observation/telemetry`:
`runtime.system.neural_device` and `runtime.acquisition.neural_device` should
both be `cuda:0`. `nvidia-smi` should also show the app's Python process holding
GPU memory. These observations verify actual YOLO execution rather than GPU
discovery alone. The small ANFIS model continues to use ONNX CPU inference in the
observation app; this setup accelerates its expensive YOLO detector path.

The prerecorded source used for verification contains only 2.5 frames/second.
The app paces replay to source time, so full replay throughput is bounded by that
rate even when uncapped detector inference is much faster. CUDA acceleration
does not fix calibration, track fragmentation, or incomplete wait measurements.

For unit tests:

```bash
.venv/bin/python -m pip install -r requirements-dev-observation.txt
.venv/bin/python -m pytest tests/test_detector_device.py \
  tests/test_camera_tracker_isolation.py tests/test_detection_class_mapping.py -q
```

Keep PC changes on `codex/pc-gpu-acceleration`. This setup does not replace the
default ANFIS model or change cabinet hardware settings.

## Hardware verification — 22 October 2026

Verified on an RTX 5060 Ti (8 GB), driver 615.78.08, Python 3.11.16. CUDA tensor
computation passed; PyTorch reports CUDA 13.0 and compute capability 12.0, with
`sm_120` included in its supported architectures. `pip check` reports no broken
requirements, and the installed environment satisfies `requirements-pc.txt`.

The real observation API processed 21 frames per camera in a diagnostic stepped
replay. Both predictors reported `cuda:0`, the app process held 558 MiB of GPU
memory, and there were no reported runtime or storage errors. The run was stopped
after verification; the dashboard remains available for another observation.
Its startup-inclusive average FPS is not a steady-state throughput measurement.

With the same `Final.pt` weights, image at video time 30 seconds, input size 640,
FP32 inference, and 20 timed calls after five warm-up calls:

| YOLO tracking path | Median | p95 |
| --- | ---: | ---: |
| CPU | 241.16 ms | 265.61 ms |
| RTX 5060 Ti | 9.79 ms | 12.09 ms |

This is approximately 24.6× lower median detector latency on this sample. CUDA
was synchronized around each timed call, and the inference backend's parameter
device was checked. This repeated-frame microbenchmark excludes decoding,
rendering, storage, and replay pacing; it is not a 102 FPS full-app claim.

The focused device, tracker-isolation, class-mapping, and observation-session checks
passed: **15 passed, 1 skipped**. The skip requires the absent legacy
`models/yolo11s.pt`. Before the fix, the new device tests reproduced five expected
assertion failures; all eight new device cases now pass.

The exact branch test set (all tracked `tests/test_*.py` files) returned
**187 passed, 75 failed, 1 skipped**. All 75 failures require absent legacy traces,
videos, snapshots or `yolo11s.pt`. The complete working-tree run also included the
intentionally red, untracked research regression scaffold from the preceding
review, returning 192 passed, 76 failed, 1 skipped; that scaffold is not included
in this branch. No existing tests were weakened or skipped to obtain these
results. The full suite is not green. Self-review found no additional issue in
the device selection/forwarding change. AEK harness enforcement is absent in this
repository.

Recorded hardware results and the detector source hash are in
[`docs/pc-gpu-verification.json`](docs/pc-gpu-verification.json).
