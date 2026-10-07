# Adaptive Traffic Signal Controller (ATSC) Production Edge Runtime

Production-grade edge runtime for real-time Adaptive Traffic Signal Control (ATSC) deployed on the **NVIDIA Jetson Orin Nano (8GB, JetPack 6.x)**.

Coordinates computer vision perception (YOLO11 + ByteTrack), metric homography (PKJI 2014), a 126-parameter Sugeno ANFIS control engine, hardware RS-485 cabinet actuation, and non-blocking asynchronous audit sinks.

---

## 1. System Architecture

```mermaid
graph TD
    subgraph Perception Layer
        C1[Kamera Sistem: Approach] --> LR[Lens Rectifier Kannala-Brandt]
        C2[Kamera Akuisisi: Discharge] --> DT[DetectorTracker: Batched YOLO11 + ByteTrack]
        LR --> PM[Optical Yellow Phase Monitor]
        LR --> DT
        DT --> HE[Homography Engine IPM]
        HE --> TE[PKJI 2014 Metrics Extractor]
    end

    subgraph Pure Decision Layer
        TE -->|TrafficSnapshot| DE[ATSCDecisionEngine]
        DE -->|ONNX Inference| ANFIS[126-Param Sugeno ANFIS Engine]
        ANFIS -->|ActuationDecision| DE
    end

    subgraph Actuation & Observers
        DE --> SI[Signal Interface RS-485 /dev/ttyTHS1]
        DE --> OBS[CycleSink Event Dispatcher]
        OBS --> S1[LedgerSink CSV field_experiment_ledger.csv]
        OBS --> S2[RecorderSink Ring-Buffered MP4 Ground-Truth]
        OBS --> S3[LLMAuditSink Non-blocking API QA Critic]
    end
```

---

## 2. Directory Layout

```text
.
├── configs/
│   ├── brica_fisheye_calib.npz        # Kannala-Brandt K & D distortion matrices
│   ├── intersection_roi.json          # Homography, lamp ROI, physical area, mode config
│   └── hardware_config.json           # Serial/RS-485 ports, watchdog thresholds, fallbacks
├── models/
│   ├── Final.pt / yolo11s_fp16.engine # Batched YOLO11 vehicle detector
│   └── anfis_sugeno.onnx              # 126-parameter Sugeno ANFIS green-time engine [10s, 120s]
├── src/
│   ├── common/                        # Strongly-typed domain dataclasses (TrafficSnapshot, CycleEvent)
│   ├── perception/                    # Video ingest, lens unwarping, YOLO11+ByteTrack, phase monitor
│   ├── analytics/                     # Homography IPM engine & PKJI 2014 traffic metrics
│   ├── control/                       # ANFIS ONNX inference, pure decision engine, RS-485 interface
│   ├── monitoring/                    # Tegrastats hardware telemetry & dynamic thermal fallback
│   ├── storage/                       # Async MP4 video recorder
│   ├── audit/                         # Async LLM cycle auditor & heuristic fallback
│   └── sinks/                         # Pluggable cycle sinks (CSV ledger, video, LLM)
├── scripts/
│   ├── export_models.py               # TensorRT FP16 engine builder
│   └── replay_debug.py                # Deterministic single-threaded offline replay debugger
├── tests/                             # Comprehensive test suite (19 unit/integration tests)
├── edge_daemon.py                     # Central multi-threaded orchestrator daemon
├── requirements-edge.txt              # JetPack & edge dependencies
└── field_experiment_ledger.csv        # Empirical experiment CSV ledger
```

---

## 3. Installation & Setup

### Requirements
- Ubuntu 22.04 LTS (JetPack 6.x on Jetson Orin Nano, or Linux x86_64)
- Python 3.10+
- PyTorch, ONNXRuntime, OpenCV, Ultralytics, PySerial

```bash
# Clone repository
git clone https://github.com/Marsel204/DeploySkripsi.git
cd DeploySkripsi

# Install dependencies
pip install -r requirements-edge.txt
```

---

## 4. Usage

### 1. Launching Production Edge Daemon (Live Cameras)
```bash
python3 edge_daemon.py \
    --mode DUAL_CAM \
    --source_sys /dev/video0 \
    --source_acq /dev/video1 \
    --headless
```

### 2. Running Field Simulation (Offline MP4 Video with Loop)
```bash
python3 edge_daemon.py \
    --mode DUAL_CAM \
    --source_sys /path/to/approach.mp4 \
    --source_acq /path/to/discharge.mp4 \
    --loop \
    --mock_hardware
```

### 3. Interactive Deterministic Replay Debugger
Inspect perception, IPM tracking, and ANFIS decisions frame-by-frame:
```bash
python3 scripts/replay_debug.py \
    --source /path/to/video.mp4 \
    --step
```
**Controls**:
- `[SPACE]` : Step forward 1 frame
- `[c]` : Run continuously until next Yellow Phase trigger
- `[p]` : Print detailed JSON state dump to console
- `[s]` : Save snapshot image with HUD overlay
- `[q]` : Exit

### 4. Running the Test Suite
```bash
pytest tests/ -v
```

---

## 5. Web Dashboard & Interactive Calibration Tutorial

The system features an interactive real-time calibration dashboard accessible from any browser:

```bash
# Launch ATSC Web Dashboard on port 8000
python3 scripts/launch_ui.py --host 0.0.0.0 --port 8000
```
Open **`http://localhost:8000`** in your browser.

---

### Step-by-Step Interactive Calibration Tutorial

To calibrate a new camera angle or intersection, click **🎯 DRAW YELLOW BOXES & CORRIDOR** on the dashboard.

![Complete 4-Step Calibration Overview](docs/images/calibration_step_by_step.jpg)

#### 🟡 Step 1: Traffic Light Optical Box (Tool 1)
* **Goal**: Isolate the optical traffic signal face (yellow lens) for phase trigger detection.
* **How**: Click & drag a bounding rectangle over the traffic signal housing.
* *Note*: If the camera view has no optical signal visible, the system automatically runs on the dynamic periodic fallback timer.

![Step 1: Traffic Light Optical Box](docs/images/tutorial_step1_lamp.jpg)

#### 🔷 Step 2: Roadway Containment Area (Tool 2: "Points & Connect")
* **Goal**: Define the exact physical roadway zone where vehicles should be counted.
* **How**:
  1. Click sequential points ($P_1, P_2, \dots, P_N$) following curved curbs, sidewalks, and lane edges from the stop line all the way up into the distance.
  2. Click **✓ Close Area** (or click near $P_1$) to seal the cyan polygon.
* **Behavior**: Any vehicle whose tires are inside this cyan boundary is included in traffic calculations (Occupancy $L$, Volume $V_w$, and Queue $Q$). Vehicles on sidewalks or opposing lanes are excluded.

![Step 2: Roadway Containment Area](docs/images/tutorial_step2_polygon.jpg)

#### 🟨 Step 3: 4-Point Perspective Anchor Grid (Tool 3)
* **Goal**: Teach the system the camera's tilt angle and 3D planar perspective foreshortening.
* **CRITICAL — Click in this exact Far-to-Near sequence**:
  1. **Click #1**: **Far-Left** (deep in the background, left lane divider)
  2. **Click #2**: **Far-Right** (deep in the background, right curb/lane edge)
  3. **Click #3**: **Near-Right** (foreground near camera, right curb/lane edge)
  4. **Click #4**: **Near-Left** (foreground near camera, left lane divider)

![Step 3: 4 Perspective Anchors](docs/images/tutorial_step3_anchors.jpg)

> [!WARNING]
> **Avoid the 90° Axis Rotation Mistake**:
> * **CORRECT**: Clicking Far-Left $\to$ Far-Right $\to$ Near-Right $\to$ Near-Left produces **horizontal distance bars** across the lanes (`10m`, `20m`, `30m`).
> * **WRONG**: Clicking down the side of the road produces **vertical lines** with distance markers along the bottom edge, inverting the lateral ($X$) and longitudinal ($Y$) axes.

![Correct vs Wrong Clicking Order](docs/images/calibration_anchor_order.jpg)

#### 📏 Step 4: Sample Distance Ruler (Tool 4)
* **Goal**: Automatically tune and verify physical ground metric scaling.
* **How**:
  1. Click **Point 1** and **Point 2** across a known ground landmark (e.g. the start and end of a standard $2.0\text{m}$ white dashed lane line).
  2. Set **Length (m)** to `2.0` (or your sample's real length).
  3. The system automatically computes the scaling factor $S$ and applies it to the homography matrix $H$, guaranteeing $< 2\%$ metric error across the entire approach.
4. Click **💾 SAVE & APPLY TO REPLAY** to immediately re-launch replay with the verified calibration.

![Step 4: Sample Distance Ruler](docs/images/tutorial_step4_ruler.jpg)

---

### How Vehicles are Counted Towards Model Metrics ($L$, $V_w$, $Q$)

![How Model Occupancy is Calculated](docs/images/occupancy_calculation_diagram.jpg)

* **Perspective Correction Applies Everywhere**: The homography matrix $H$ derived from the 4 anchor points is a continuous geometric projective transform. It calculates exact metric distances for all vehicles across the entire road plane—including vehicles 30 meters beyond the anchor box.
* **Occupancy Ratio ($L$)**:
  $$\text{Occupied Area } (\text{m}^2) = \sum (\text{Footprints of ALL vehicles in the Cyan Road Area})$$
  $$L = \left( \frac{\text{Occupied Area}}{\text{Physical Road Area}} \right) \times 100\%$$
* Any vehicle inside the **Cyan Polygon** (even if outside the small yellow anchor box) contributes $100\%$ to $L$, $V_w$, and $Q$.

