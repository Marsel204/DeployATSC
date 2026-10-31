from src.perception.detector_tracker import TrackedVehicle
from src.ui.observation_session import ObservationSession
from test_observation_sessions import FakeDetector, FakeANFIS, run_payload, wait_applied


def test_real_session_uses_logical_motion_and_same_timer(video_factory, tmp_path):
    class Detector(FakeDetector):
        def track(self, frames, timestamp=None):
            return [[TrackedVehicle(4 if timestamp < .5 else 228, 1, 'car',
                [40, 70, 80, 120], .95, timestamp, stable_id=True, epoch=self.epoch)]]

        def update_velocities(self, *args, **kwargs):
            raise AssertionError('Persistent mode must not use native-ID velocity history')

    data = run_payload(video_factory)
    data['vehicle_identity'] = {'mode': 'persistent_v1'}
    session = ObservationSession(data, tmp_path/'runs', detector_factory=Detector, anfis_factory=FakeANFIS)
    try:
        for _ in range(30):
            control = session.control('step')
            wait_applied(session, control['control_seq'])
        session.control('play')
        session.thread.join(timeout=5)
        assert not session.thread.is_alive()
        rows = session.engine.snapshot(evidence=True)['vehicles']
        assert len(rows) == 1 and rows[0]['red_wait_sec'] > .5
        assert rows[0]['native_reference']['local_id'] == 228
        assert session.runtime['system']['frames'] == 0
        assert session.runtime['acquisition']['frames'] >= 30
    finally:
        session.stop()
