import numpy as np
import pytest
from src.experiment.contracts import RunPolicy
from src.perception.signal_phase_observer import SignalPhaseObserver

COLORS = {"RED": (0,0,255), "YELLOW": (0,255,255), "GREEN": (0,255,0), "UNKNOWN": (0,0,0)}


def observe(observer, phase, timestamp, sequence):
    image = np.full((60,80,3), COLORS[phase], np.uint8)
    return observer.observe(image, timestamp, sequence)


def test_actual_green_evidence_excludes_confirmation_latency():
    monitor = SignalPhaseObserver({"xmin":0,"ymin":0,"xmax":80,"ymax":60}, RunPolicy(max_gap_sec=30))
    for t, phase in [(0,"RED"),(.1,"RED"),(.2,"RED"),(30,"GREEN"),(30.1,"GREEN"),(30.2,"GREEN"),(55,"YELLOW"),(55.1,"YELLOW"),(55.2,"YELLOW")]:
        observe(monitor,phase,t,int(t*100)+1)
    cycle = monitor.cycles[0]
    assert cycle["actual_green_sec"] == pytest.approx(25)
    assert cycle["green_confirmed_at"] == 30.2
    assert cycle["complete"]


def test_repeat_frame_and_yellow_absence_do_not_imply_green():
    monitor = SignalPhaseObserver({"xmin":0,"ymin":0,"xmax":80,"ymax":60}, RunPolicy())
    for _ in range(5):observe(monitor,"GREEN",0,1)
    assert monitor.phase == "UNKNOWN"
    observe(monitor,"UNKNOWN",.1,2)
    assert monitor.phase == "UNKNOWN"
    assert not monitor.cycles


def test_unknown_breaks_interval_and_conflicting_colors_are_unknown():
    monitor = SignalPhaseObserver({"xmin":0,"ymin":0,"xmax":80,"ymax":60}, RunPolicy(debounce_frames=1))
    observe(monitor,"RED",0,1);observe(monitor,"GREEN",.1,2)
    observe(monitor,"UNKNOWN",.2,3)
    assert not monitor.cycles[0]["complete"]
    assert monitor.cycles[0]["yellow_confirmed_at"] is None
    assert monitor.cycles[0]["end_phase"] == "UNKNOWN"
    image = np.zeros((60,80,3),np.uint8)
    image[:,:40]=(0,0,255);image[:,40:]=(0,255,0)
    monitor.observe(image,.3,4)
    assert monitor.phase == "UNKNOWN"
