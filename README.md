# DeployATSC: Adaptive Traffic Signal Control Production Edge System

[![Platform: Jetson Orin Nano](https://img.shields.io/badge/Platform-NVIDIA%20Jetson%20Orin%20Nano-76B900?logo=nvidia)](https://developer.nvidia.com/embedded/jetson-orin)
[![JetPack: 6.x](https://img.shields.io/badge/JetPack-6.x-green)](https://developer.nvidia.com/embedded/jetpack)
[![Inference: TensorRT / PyTorch](https://img.shields.io/badge/Inference-TensorRT%20%7C%20YOLO11-blue)](https://github.com/ultralytics/ultralytics)
[![ANFIS: Sugeno 126--Param](https://img.shields.io/badge/ANFIS-Sugeno%20126--Param-orange)]()
[![Tests: 192 Passing](https://img.shields.io/badge/Tests-192%20Passing-brightgreen)]()

Production-grade, end-to-end edge deployment runtime for real-time **Adaptive Traffic Signal Control (ATSC)** on **NVIDIA Jetson** edge devices.

The system continuously coordinates computer vision perception (YOLO11 + ByteTrack), metric homography (PKJI 2014 passenger car equivalents and metric queue lengths), a 126-parameter Sugeno ANFIS fuzzy inference engine, hardware RS-485 cabinet actuation, and an interactive Web Calibration & Replay Dashboard.

---

## 🚀 Key Highlights & Capabilities

- **Real-Time Edge Perception**: Batched YOLO11 detection with ByteTrack persistent multi-object tracking, custom camera lens rectification (Kannala-Brandt fisheye & rectilinear models), and optical phase monitoring.
- **Metric Homography & Geometry Engine**: 
  - Arbitrary $N \ge 3$ point polygon approach corridor containment.
  - Interactive metric ruler scale calibration matching physical ground markings (e.g. 2.0m lane dash).
  - Ground-projected vehicle footprint positioning and true metric queue length ($L$) extraction.
- **Sugeno ANFIS Decision Core**: Pure functional 126-parameter Sugeno fuzzy inference engine generating mathematically bounded green duration extensions ($\Delta g$).
- **Fail-Safe RS-485 Actuation**: Asynchronous serial communication (`/dev/ttyTHS1`) with hardware watchdog heartbeat and automatic fallback to fixed-time controller operation upon telemetry loss.
- **Interactive Web Dashboard**: FastAPI and HTML5 Canvas GUI for one-click calibration, video replay (supports local MP4 uploads, YouTube URLs, and pre-loaded test videos), and live PKJI metrics monitoring ($V_w, Q, L$).

---

## 📂 Repository Architecture

```text
DeployATSC/
├── jetson-atsc-core/               # Core edge deployment codebase
│   ├── configs/                    # Camera profiles, lens calibrations, and intersection ROIs
│   ├── docs/images/                # Illustrated step-by-step calibration diagrams
│   ├── models/                     # YOLO11 and Sugeno ANFIS model weights (.pt / .onnx / .engine)
│   ├── scripts/
│   │   ├── launch_ui.py            # Entrypoint for Web Control & Replay Dashboard
│   │   └── run_edge_daemon.py      # Headless production daemon for field Jetson devices
│   ├── src/
│   │   ├── actuation/              # RS-485 serial controller & safety interlocks
│   │   ├── analytics/              # Inverse Perspective Mapping (IPM) & PKJI 2014 metrics
│   │   ├── decision/               # Sugeno ANFIS fuzzy inference engine
│   │   ├── perception/             # YOLO11 detector, ByteTrack tracker, & optical phase monitor
│   │   ├── storage/                # Thread-safe ring buffer video recorder & CSV audit ledger
│   │   └── ui/                     # FastAPI backend & web dashboard application
│   ├── tests/                      # 192 automated unit, integration, and adversarial stress tests
│   ├── requirements-edge.txt       # Production dependencies
│   └── README.md                   # In-depth module documentation & calibration manual
└── README.md                       # Repository overview & quickstart
```

---

## ⚡ Quick Start

### 1. Prerequisites & Virtual Environment

Ensure Python 3.10+ is installed (standard on JetPack 5/6):

```bash
cd jetson-atsc-core
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-edge.txt
```

### 2. Launch the Web Control & Calibration Dashboard

```bash
python scripts/launch_ui.py --host 0.0.0.0 --port 8000
```

Open **`http://localhost:8000`** in your browser:
1. Select a video source (**Pre-loaded Samples**, **YouTube Link**, or **Local MP4 Upload**).
2. Follow the 4-step interactive calibration modal:
   - 🟡 **Traffic Light Box**: Bounding box over signal lamp.
   - 🔷 **Road Corridor**: Click $\ge 3$ points covering the approach area.
   - 🟨 **Perspective Anchors**: Click 4 points in **Far-to-Near** order (`Far-Left` $\to$ `Far-Right` $\to$ `Near-Right` $\to$ `Near-Left`).
   - 📏 **Sample Ruler**: Click 2 points along a known distance reference (e.g. 2.0m lane dash).
3. Click **INITIALIZE & PLAY** to run real-time detection, homography, and Sugeno ANFIS control.

### 3. Run Automated Tests

Execute the comprehensive test suite (192 tests):

```bash
cd jetson-atsc-core
pytest tests/
```

---

## 📖 Detailed Documentation

For full mathematical derivations, PKJI 2014 formulation, RS-485 serial protocol specs, and calibration step-by-step visuals, refer to the [**`jetson-atsc-core/README.md`**](jetson-atsc-core/README.md).

---

## 📄 License

MIT License. Designed for academic research and intelligent transportation systems deployment.
