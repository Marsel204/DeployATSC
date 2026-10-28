#!/usr/bin/env python3
"""Isolated feature demos: native real-footage presets or labelled generated scenes."""
from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.perception.detector_tracker import DetectorTracker, TrackedVehicle
from src.ui.observation_session import ObservationManager, ObservationSession

FPS = 10
SIZE = (480, 320)
CYCLE = 22


def phase_at(timestamp):
    t = timestamp % CYCLE
    return 'RED' if t < 12 else 'GREEN' if t < 20 else 'YELLOW'


def scene_tracks(timestamp, epoch=0):
    t = timestamp % CYCLE
    cycle = int(timestamp // CYCLE)
    tracks = []
    for index, (x, target, kind, width, height) in enumerate([
        (160, 210, 'car', 28, 34), (235, 166, 'motorcycle', 18, 24),
        (310, 122, 'truck', 38, 45),
    ]):
        release = 12 + index * .8
        bottom = target - 40 * max(0, 2 - t) + 50 * max(0, t - release)
        if bottom > 310:
            continue
        moving = t < 2 or t > release
        vehicle = TrackedVehicle(cycle * 10 + index + 1, index, kind,
            [x-width/2, bottom-height, x+width/2, bottom], 1., timestamp, epoch=epoch)
        vehicle.velocity_valid = True
        vehicle.velocity_mps = 8. if moving else 0.
        tracks.append(vehicle)
    return tracks


class DemoDetector:
    """Known scene trajectories, deliberately NOT a real detector/tracker."""
    def __init__(self, *args, **kwargs):
        self.epoch = 0
        self.device = 'scripted-demo-no-YOLO'

    def track(self, frames, timestamp=None):
        return [scene_tracks(timestamp or 0., self.epoch) for _ in frames]

    def detect_once(self, frame, timestamp=None):
        # Camera 1 still image: scripted boxes only; no identity or motion evidence.
        vehicles = scene_tracks(timestamp or 0., self.epoch)
        for vehicle in vehicles:
            vehicle.stable_id = vehicle.velocity_valid = False
            vehicle.velocity_mps = 0.
        return vehicles

    def update_velocities(self, tracks, project, timestamp=None):
        for vehicle in tracks:
            vehicle.ground_pos = project(*vehicle.bottom_center)

    def reset_tracking(self):
        self.epoch += 1


def draw_scene(role, timestamp):
    frame = np.full((SIZE[1], SIZE[0], 3), (45, 65, 45), np.uint8)
    cv2.rectangle(frame, (70, 45), (390, 319), (65, 65, 65), -1)
    for x in (120, 200, 275, 355):
        for y in range(55, 320, 44):
            cv2.line(frame, (x, y), (x, y+22), (180, 180, 180), 2)
    cv2.line(frame, (75, 245), (385, 245), (240, 240, 240), 4)
    colors = {'car': (210, 145, 65), 'motorcycle': (190, 80, 180), 'truck': (90, 170, 225)}
    for vehicle in scene_tracks(timestamp):
        x1, y1, x2, y2 = map(int, vehicle.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colors[vehicle.class_name], -1)
        cv2.rectangle(frame, (x1+3, y1+4), (x2-3, y1+10), (45, 50, 60), -1)
    cv2.rectangle(frame, (414, 48), (466, 178), (15, 15, 15), -1)
    phase = phase_at(timestamp)
    for name, y, color in [('RED', 76, (0, 0, 255)), ('YELLOW', 112, (0, 255, 255)), ('GREEN', 148, (0, 255, 0))]:
        cv2.circle(frame, (440, y), 12, color if name == phase else (40, 40, 40), -1)
    cv2.putText(frame, f'DEMO {role.upper()} - scripted scene', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .48, (245, 245, 245), 1)
    cv2.putText(frame, f'{timestamp:04.1f}s  {phase}  cycle {int(timestamp//CYCLE)+1}', (12, 306), cv2.FONT_HERSHEY_SIMPLEX, .48, (245, 245, 245), 1)
    return frame


def make_preset(directory, duration=66):
    directory = Path(directory).resolve()
    media = directory / 'data' / 'sample_videos'
    media.mkdir(parents=True, exist_ok=True)
    payload = {'source_mode': 'replay', 'step_mode': True,
               'policy': {'movement_id': 'DEMO-synthetic-traffic', 'debounce_frames': 1}}
    for role in ('system', 'acquisition'):
        path = media / f'DEMO_{role}.mp4'
        if not path.exists():
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), FPS, SIZE)
            if not writer.isOpened():
                raise RuntimeError('Cannot create the demo video')
            try:
                for index in range(int(duration * FPS)):
                    writer.write(draw_scene(role, index / FPS))
            finally:
                writer.release()
        payload[role] = {'name': f'DEMO_{role}_synthetic', 'role': role, 'source': str(path),
            'image_size': list(SIZE), 'anchors': [[70, 50], [390, 50], [390, 305], [70, 305]],
            'polygon': [[70, 50], [390, 50], [390, 305], [70, 305]],
            'road_width_m': 8, 'road_length_m': 40,
            'scale_evidence': {'method': 'measured_dimensions',
                'reference': 'DEMO ONLY: virtual 8 x 40 metre scene by construction; not a physical survey'},
            'binding': {'reviewed': True}}
    payload['acquisition'].update(signal_roi={'xmin': 416, 'ymin': 52, 'xmax': 464, 'ymax': 174},
        lamp_rois={phase: {'xmin': 426, 'ymin': y-14, 'xmax': 454, 'ymax': y+14}
                   for phase, y in [('RED', 76), ('YELLOW', 112), ('GREEN', 148)]},
        stop_line=[[75, 245], [385, 245]], crossing_direction=1)
    (directory / 'preset.json').write_text(json.dumps(payload, indent=2)+'\n')
    return payload


def make_manager(directory, *, scripted=True, device='cpu'):
    directory = Path(directory)
    return ObservationManager(directory / 'runs', directory / 'profiles',
        partial(ObservationSession, detector_factory=DemoDetector if scripted else partial(DetectorTracker, device=device)))


def step_to(session, timestamp):
    while session.engine.time < timestamp:
        status = session.control('step')
        deadline = time.monotonic() + 10
        while True:
            with session.lock:
                done = session.applied_control_seq >= status['control_seq'] and not session.steps and not session.packets
            if done:
                break
            if session.error or session.pair.ended or time.monotonic() > deadline:
                raise RuntimeError(session.error or 'Demo replay failed to advance')
            time.sleep(.002)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--preset', type=Path, help='Use an existing real-footage start payload with native YOLO tracking')
    parser.add_argument('--device', choices=['cpu', 'cuda:0'], default='cpu', help='Native detector device for a real-footage preset')
    parser.add_argument('--backend', choices=['auto','pytorch','tensorrt'], default=os.environ.get('ATSC_DETECTOR_BACKEND','auto'))
    args = parser.parse_args()
    os.environ['ATSC_DETECTOR_BACKEND']=args.backend
    scripted = args.preset is None
    directory = ROOT / 'data' / 'demo' if scripted else args.preset.resolve().parent
    payload = make_preset(directory) if scripted else json.loads(args.preset.read_text())
    if not scripted:
        # Only this demo process is configured; normal app settings stay intact.
        if args.device == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        import torch
        torch.set_num_threads(2)
    manager = make_manager(directory, scripted=scripted, device=args.device)
    for role in ('system', 'acquisition'):
        manager.save_calibration(payload[role])
    if args.prepare_only:
        print(directory / 'preset.json')
        return

    # Demo assets, profiles, uploads and evidence never use the real run folders.
    import src.ui.app as app_module
    import src.ui.observation_api as api
    import src.ui.session_manager as legacy
    api.observations = manager
    app_module.CORE_ROOT = legacy.CORE_ROOT = directory
    html = (ROOT / 'src/ui/static/index.html').read_text()
    banner = '<div class="notice" style="margin:16px" role="note"><strong>DEMO ONLY</strong> · Generated video and scripted vehicle tracks; real CPU ANFIS and app storage. No real detector or field measurements. Click <strong>Analyze frame &amp; predict</strong> (Camera 1 analyses that one frame), then <strong>Resume</strong>: green starts around 12s and ends at 20s. Use Stop/Start to repeat; Saved runs has a completed example.</div>'
    if not scripted:
        banner = '<div class="notice" style="margin:16px" role="note"><strong>REAL FOOTAGE DEMO</strong> · Actual traffic video and ANFIS. Camera 2 runs YOLO + ByteTrack continuously; Camera 1 runs YOLO once per <strong>Analyze frame &amp; predict</strong> click on that single frame. One recording is reused for the two camera roles. Preset geometry is approximate, for demonstration only. Try Analyze frame &amp; predict, Resume, Pause, Step, Stop and saved-run export.</div>'
    movement = 'DEMO-synthetic-traffic' if scripted else 'DEMO-real-footage'
    html = html.replace('<body>', '<body>'+banner, 1).replace('value="approach-1"', f'value="{movement}"')
    html = html.replace('id="phaseDebounce" type="number" value="3"', 'id="phaseDebounce" type="number" value="1"')
    html = html.replace('<option value="no">Play immediately</option><option value="yes">Paused for inspection</option>',
        '<option value="no">Play immediately</option><option value="yes" selected>Paused for inspection</option>')
    if scripted:
        html = html.replace('Perspective correction · YOLO · ANFIS', 'Synthetic traffic · scripted tracks · real ANFIS')
    page = directory / 'dashboard.html'
    page.write_text(html)
    os.environ['ATSC_UI_PATH'] = str(page)
    if scripted and not manager.runs():
        example = manager.start(payload)
        step_to(example, 3)
        example.predict('demo-example')
        example.engine.predictions.await_idle()
        step_to(example, 22)
        example.stop()
    current = manager.start(payload)
    step_to(current, 3 if scripted else 1)
    print('DEMO READY: paused; '+('generated video/tracks' if scripted else 'real video, native YOLO: Camera 2 continuous, Camera 1 on click')+', real CPU ANFIS.')
    from scripts.launch_ui import serve_observation
    try:
        serve_observation(port=args.port, source_sys=payload['system']['source'], source_acq=payload['acquisition']['source'])
    finally:
        manager.shutdown()


if __name__ == '__main__':
    main()
