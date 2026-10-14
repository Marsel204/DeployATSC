from collections import deque
import threading
import numpy as np
from src.perception.frame_grabber import FrameGrabber


def test_live_packet_sequence_is_consumed_once():
    source=FrameGrabber.__new__(FrameGrabber)
    source._lock=threading.Lock()
    source._observation_queue=deque([{"sequence":1,"timestamp":10.,"frame":np.zeros((20,30,3),np.uint8)}])
    first=source.read_observation()
    assert first["sequence"]==1
    assert source.read_observation() is None
