import json,sys
import edge_daemon
from types import SimpleNamespace
from scripts import launch_ui


def test_observation_routes_before_hardware_construction(tmp_path,monkeypatch):
    config=tmp_path/"camera.json";config.write_text(json.dumps({"cameras":{}}))
    called=[]
    def forbidden(*args,**kwargs):raise AssertionError("Hardware/legacy controller was constructed")
    monkeypatch.setattr(edge_daemon,"EdgeDaemon",forbidden)
    monkeypatch.setattr(edge_daemon.SignalInterface,"from_config",forbidden)
    monkeypatch.setattr(launch_ui,"serve_observation",lambda **kwargs:called.append(kwargs))
    monkeypatch.setattr(sys,"argv",["edge_daemon.py","--operation","observation","--config",str(config),"--source_sys",str(config),"--source_acq",str(config)])
    edge_daemon.main()
    assert len(called)==1 and called[0]["source_mode"]=="replay"
    assert called[0]["source_sys"]==str(config)


def test_default_entrypoint_is_observation_and_control_requires_explicit_choice(tmp_path,monkeypatch):
    config=tmp_path/"config.json";config.write_text('{}')
    events=[]
    monkeypatch.setattr(launch_ui,"serve_observation",lambda **kwargs:events.append("observation"))
    class FakeController:
        def __init__(self,args):events.append("legacy-control")
        def start(self):pass
    monkeypatch.setattr(edge_daemon,"EdgeDaemon",FakeController)
    monkeypatch.setattr(sys,"argv",["edge_daemon.py","--config",str(config)])
    edge_daemon.main();assert events==["observation"]
    monkeypatch.setattr(sys,"argv",["edge_daemon.py","--operation","legacy-control","--config",str(config)])
    edge_daemon.main();assert events==["observation","legacy-control"]


def test_legacy_mock_flag_is_applied_before_serial_connection(tmp_path,monkeypatch):
    camera=tmp_path/"camera.json";camera.write_text('{}')
    hardware=tmp_path/"hardware.json";hardware.write_text(json.dumps({"serial_interface":{"mock_mode":False}}))
    for name in ("LensRectifier","ANFISInferenceEngine"):
        monkeypatch.setattr(edge_daemon,name,lambda *args,**kwargs:object())
    for name in ("OpticalPhaseMonitor","HomographyEngine","TrafficMetricsExtractor"):
        monkeypatch.setattr(getattr(edge_daemon,name),"from_config",lambda *args,**kwargs:object())
    class ReachedHardware(Exception):pass
    def check(settings):
        assert settings["serial_interface"]["mock_mode"] is True
        raise ReachedHardware
    monkeypatch.setattr(edge_daemon.SignalInterface,"from_config",check)
    args=SimpleNamespace(config=str(camera),hardware_config=str(hardware),mode="DUAL_CAM",
                         source_sys=None,source_acq=None,ledger=str(tmp_path/'ledger.csv'),mock_hardware=True)
    import pytest
    with pytest.raises(ReachedHardware):edge_daemon.EdgeDaemon(args)


def test_launchers_default_loopback_and_observation_rejects_reload(monkeypatch):
    import pytest
    monkeypatch.setattr(sys,'argv',['launch_ui.py'])
    assert launch_ui.parse_args().host=='127.0.0.1'
    monkeypatch.setattr(sys,'argv',['edge_daemon.py'])
    assert edge_daemon.parse_arguments().host=='127.0.0.1'
    calls=[]
    monkeypatch.setattr(launch_ui.uvicorn,'run',lambda *args,**kwargs:calls.append(kwargs))
    launch_ui.serve_observation()
    assert calls[0]['host']=='127.0.0.1' and calls[0]['workers']==1 and not calls[0]['reload']
    with pytest.raises(ValueError,match='reload'):launch_ui.serve_observation(reload=True)
    assert len(calls)==1
