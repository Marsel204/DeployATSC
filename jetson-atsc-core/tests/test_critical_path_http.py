import asyncio
from dataclasses import dataclass
from datetime import datetime,timezone
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
import numpy as np
import pytest
import src.ui.observation_api as api
from src.ui.observation_session import ObservationSession
from test_observation_sessions import FakeDetector,FakeANFIS,run_payload,wait_applied


def body(response):return response.body


def test_route_returns_response_without_changing_json(monkeypatch):
    @dataclass
    class Record:value:int=3
    for payload in [{'state':'idle'},{'state':'running','vehicles':[{'value':np.float64(.5)}],'predictions':[]},{'state':'paused','date':datetime(2026,10,29,tzinfo=timezone.utc),'path':Path('local'),'record':Record()}, {'state':'ended','error':None}, {'state':'error','error':'safe'}]:
        manager=SimpleNamespace(starting=False,last_start=None,current=SimpleNamespace(telemetry=lambda *a:payload))
        monkeypatch.setattr(api,'observations',manager)
        old=JSONResponse(jsonable_encoder({**payload,'start_pending':False,'last_start':None}))
        response=api.telemetry(0,100)
        assert isinstance(response,JSONResponse)
        assert json.loads(body(response))==json.loads(body(old))
    # FastAPI's previous encoder rejects np.int64; preserve failure rather than inventing data conversion.
    payload={'bad':np.int64(1)}
    with pytest.raises(ValueError):jsonable_encoder(payload)
    with pytest.raises(ValueError):api.telemetry(0,100)


def test_real_telemetry_route_states_equal_old_body(video_factory,tmp_path,monkeypatch):
    class Manual(ObservationSession):
        def _worker(self):self.ready.set()
    session=Manual(run_payload(video_factory),tmp_path/'runs',detector_factory=FakeDetector,anfis_factory=FakeANFIS)
    session.thread.join();session._process_batch();session.initializing=False
    for role in ('system','acquisition'):assert session.jpeg(role)
    monkeypatch.setattr(api,'observations',SimpleNamespace(current=session,starting=False,last_start=None))
    app=FastAPI();app.include_router(api.router)
    try:
        with TestClient(app) as client:
            for state in ('paused','running','ended','error'):
                # Fixed owner state and clocks make old/new JSON comparable, without running extra measurements.
                session.state=session.applied_state=state
                import time
                with monkeypatch.context() as clock:
                    clock.setattr(time,'monotonic',lambda:session.created_monotonic+1)
                    expected=jsonable_encoder({**session.telemetry(),'start_pending':False,'last_start':None})
                    assert client.get('/api/observation/telemetry').json()==expected
    finally:session.stop()


def test_mjpeg_first_changed_keepalive_paused_and_limit(monkeypatch):
    clock=[0.];sent=[];first=bytes(bytearray(b'jpeg-a'));second=bytes(bytearray(b'jpeg-b'))
    run=SimpleNamespace(state='paused',jpeg=lambda role:first if clock[0]<.2 else second)
    monkeypatch.setattr(api,'observations',SimpleNamespace(require=lambda:run))
    monkeypatch.setattr(api.time,'monotonic',lambda:clock[0])
    async def advance(delay):clock[0]+=delay
    monkeypatch.setattr(api.asyncio,'sleep',advance)
    async def consume():
        response=api.stream('system',3)
        async for part in response.body_iterator:sent.append((clock[0],part))
    asyncio.run(consume())
    assert len(sent)==3 and sent[0][0]==0
    assert b'jpeg-a' in sent[0][1] and b'jpeg-b' in sent[1][1]
    assert .2<=sent[1][0]<.25 and sent[2][0]-sent[1][0]>=1


def test_mjpeg_terminal_still_delivers_last_frame(monkeypatch):
    monkeypatch.setattr(api,'observations',SimpleNamespace(require=lambda:SimpleNamespace(state='ended',jpeg=lambda role:b'final')))
    async def consume():return [part async for part in api.stream('acquisition',5).body_iterator]
    parts=asyncio.run(consume());assert len(parts)==1 and b'final' in parts[0]
