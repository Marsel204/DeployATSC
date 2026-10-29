from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Literal

from fastapi.encoders import jsonable_encoder
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from src.common.source_identity import public_data, public_source, private_source, safe_text
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.experiment.calibration import legacy_system_payload
from src.experiment.contracts import RunPolicy
from src.experiment.predictions import PredictionConflict
from src.ui.observation_session import ObservationManager, RuntimeConflict
from src.ui.session_manager import manager as legacy_manager

class TelemetryResponse(JSONResponse):
    def render(self,content):
        return json.dumps(content,ensure_ascii=False,allow_nan=False,separators=(",",":"),default=jsonable_encoder).encode("utf-8")


class PublicErrorsRoute(APIRoute):
    def get_route_handler(self):
        handler=super().get_route_handler()
        async def public_handler(request:Request):
            try:return await handler(request)
            except RequestValidationError as error:
                # Pydantic includes rejected input verbatim by default.
                return JSONResponse(status_code=422,content={"detail":public_data([
                    {key:row[key] for key in ("loc","msg","type") if key in row} for row in error.errors()])})
        return public_handler


router=APIRouter(route_class=PublicErrorsRoute)
observations=ObservationManager()
launch_settings={}


class StrictModel(BaseModel):
    model_config=ConfigDict(extra="forbid")


class Roi(StrictModel):
    xmin:int=Field(ge=0)
    ymin:int=Field(ge=0)
    xmax:int=Field(gt=0)
    ymax:int=Field(gt=0)


class Ruler(StrictModel):
    p1:tuple[float,float]
    p2:tuple[float,float]
    length_m:float=Field(gt=0)


class ScaleEvidence(StrictModel):
    method:Literal["measured_dimensions","ruler"]
    reference:str=Field(min_length=1,max_length=1000)


class GeometryBinding(StrictModel):
    reviewed:bool=False
    geometry_digest:str|None=Field(default=None,pattern=r"^[a-f0-9]{64}$")


class CameraSettings(StrictModel):
    role:Literal["system","acquisition"]
    source:str=Field(min_length=1,max_length=2048)
    image_size:tuple[int,int]
    anchors:list[tuple[float,float]]=Field(min_length=4,max_length=4)
    polygon:list[tuple[float,float]]|None=Field(default=None,min_length=3)
    road_width_m:float=Field(default=8,gt=0)
    road_length_m:float=Field(default=40,gt=0)
    rectification_path:str|None=None
    rectification_digest:str|None=None
    signal_roi:Roi|None=None
    lamp_rois:dict[Literal["RED","YELLOW","GREEN"],Roi]=Field(default_factory=dict)
    stop_line:list[tuple[float,float]]|None=Field(default=None,min_length=2,max_length=2)
    crossing_direction:Literal[-1,1]=1
    ruler:Ruler|None=None
    scale_evidence:ScaleEvidence|None=None
    binding:GeometryBinding|None=None
    parent_revision:str|None=Field(default=None,pattern=r"^[a-f0-9]{16}$")
    lane_metrics:dict[str,Any]=Field(default_factory=dict)
    name:str=Field(default="profile",max_length=100)

    @model_validator(mode="after")
    def explicit_dimensions(self):
        if self.scale_evidence and self.scale_evidence.method=="measured_dimensions" and not {
                "road_width_m","road_length_m"}.issubset(self.model_fields_set):
            raise ValueError("Measured dimensions must explicitly supply width and length")
        return self


class StartObservation(StrictModel):
    system:CameraSettings
    acquisition:CameraSettings
    source_mode:Literal["replay","live"]="replay"
    step_mode:bool=False
    shared_test:bool=False
    retention_budget:int=Field(default=64,ge=1,le=4096)
    policy:dict[str,Any]=Field(default_factory=dict)

    @field_validator("policy")
    @classmethod
    def valid_policy(cls,value):
        try:RunPolicy(**value)
        except (TypeError,ValueError) as error:raise ValueError(safe_text(error)) from error
        return value


class ControlObservation(StrictModel):
    run_id:str
    action:Literal["pause","play","step","stop"]


class CapturePrediction(StrictModel):
    run_id:str
    request_id:str=Field(min_length=1,max_length=128)


class LegacyProfile(StrictModel):
    config_path:str
    source:str
    image_size:tuple[int,int]


def translate(error):
    if isinstance(error,(PredictionConflict,RuntimeConflict)):raise HTTPException(409,safe_text(error))
    if isinstance(error,(ValueError,TypeError,KeyError,FileNotFoundError)):raise HTTPException(422,safe_text(error))
    raise HTTPException(500,safe_text(error))


@router.get("/api/observation/bootstrap")
def bootstrap():
    return public_data({"operation":"observation",**launch_settings})


@router.get("/api/observation/profiles")
def profiles():
    return {"profiles":observations.profiles()}


@router.post("/api/observation/calibration")
def save_calibration(settings:CameraSettings):
    try:return observations.save_calibration(settings.model_dump(mode="json",exclude_none=True))
    except Exception as error:translate(error)


@router.post("/api/observation/legacy_profile")
def legacy_profile(settings:LegacyProfile):
    try:
        payload=legacy_system_payload(json.loads(Path(settings.config_path).read_text()),settings.source,list(settings.image_size))
        return {"calibration":public_data(payload),"requires_review":True}
    except Exception as error:translate(error)


class SourceInput(StrictModel):
    source:str=Field(min_length=1,max_length=2048)


class PreviewInput(SourceInput):
    role:Literal["system","acquisition"]
    rectification_path:str|None=None
    source_mode:Literal["replay","live"]="replay"


@router.post("/api/observation/sources")
def register_source(settings:SourceInput):
    return {"source":public_source(settings.source)}


@router.post("/api/observation/snapshot")
def private_snapshot(settings:PreviewInput):
    try:return Response(observations.preview(settings.role,public_source(settings.source),settings.rectification_path,settings.source_mode),media_type="image/jpeg")
    except Exception as error:translate(error)


@router.get("/api/observation/snapshot")
def snapshot(role:Literal["system","acquisition"],source:str,
             rectification_path:str|None=None,source_mode:Literal["replay","live"]="replay"):
    if private_source(source):raise HTTPException(422,"Register this source using POST and preview its public handle")
    try:return Response(observations.preview(role,source,rectification_path,source_mode),media_type="image/jpeg")
    except Exception as error:translate(error)


@router.post("/api/observation/start")
def start(settings:StartObservation):
    try:
        # Only an explicitly requested new observation replaces a legacy replay.
        run=observations.start(settings.model_dump(mode="json",exclude_none=True),before_handoff=legacy_manager.stop_session)
        return {"run_id":run.run_id,"state":run.state,"operation":"observation","handoff":observations.last_start}
    except Exception as error:translate(error)


@router.post("/api/observation/control")
def control(settings:ControlObservation):
    try:return observations.require(settings.run_id).control(settings.action)
    except Exception as error:translate(error)


@router.post("/api/observation/predict")
def predict(settings:CapturePrediction):
    try:
        if observations.current is not None and observations.current.run_id==settings.run_id:
            return observations.current.predict(settings.request_id)
        record=observations.evidence_store(settings.run_id).lookup(settings.request_id)
        if record is None:raise PredictionConflict("Historical run has no such capture; start a new run")
        return public_data(record)
    except Exception as error:translate(error)


@router.post("/api/observation/predictions/{request_id}/cancel")
def cancel_prediction(request_id:str,run_id:str):
    try:
        return observations.require(run_id).cancel(request_id)
    except Exception as error:translate(error)


@router.get("/api/observation/telemetry")
def telemetry(offset:int=Query(0,ge=0),limit:int=Query(100,ge=1,le=200)):
    handoff={"start_pending":observations.starting,"last_start":observations.last_start}
    if observations.current is None:return TelemetryResponse({"state":"idle","operation":"observation",**handoff})
    return TelemetryResponse({**observations.current.telemetry(offset,limit),**handoff})


@router.get("/api/observation/export")
def export_run(run_id:str|None=None):
    try:
        run=observations.current
        identity=run_id or observations.require().run_id
        content=run.export() if run is not None and run.run_id==identity else observations.evidence_store(identity).export()
        return Response(content,media_type="application/zip",
                        headers={"Content-Disposition":f'attachment; filename="{identity}.zip"'})
    except Exception as error:translate(error)


@router.get("/api/observation/predictions/{request_id}")
def historical_prediction(request_id:str,run_id:str):
    try:
        record=observations.evidence_store(run_id).lookup(request_id)
        if record is None:raise HTTPException(404,"Prediction was not found")
        return public_data(record)
    except HTTPException:raise
    except Exception as error:translate(error)


@router.get("/api/observation/evidence")
def full_evidence(run_id:str):
    try:return public_data(observations.evidence_store(run_id).read())
    except Exception as error:translate(error)


@router.get("/api/observation/history/{kind}")
def history(kind:Literal["vehicles","predictions","comparisons","phase_events","observation_gaps"],run_id:str,
            offset:int=Query(0,ge=0),limit:int=Query(100,ge=1,le=200),generation:int|None=None):
    try:
        store=observations.evidence_store(run_id);pinned=store.pin()
        if generation is not None and generation!=pinned["generation"]:
            raise HTTPException(409,"Durable generation changed; restart pagination")
        return public_data(store.page(kind,offset,limit,pinned))
    except HTTPException:raise
    except Exception as error:translate(error)


@router.get("/api/observation/quantiles")
def quantiles(run_id:str):
    try:return observations.evidence_store(run_id).quantiles()
    except Exception as error:translate(error)


@router.get("/api/observation/runs")
def evidence_runs():
    return {"runs":observations.runs()}


@router.post("/api/observation/recover")
def recover_evidence(run_id:str):
    try:return observations.recover_interrupted(run_id)
    except Exception as error:translate(error)


@router.get("/stream/observation/{role}")
def stream(role:Literal["system","acquisition"],limit:int|None=Query(None,ge=1,le=100)):
    try:run=observations.require()
    except Exception as error:translate(error)
    async def frames():
        count=0;previous=None;last_sent=0.
        while limit is None or count<limit:
            jpeg=run.jpeg(role) if run.state=="running" else await asyncio.to_thread(run.jpeg,role)
            now=time.monotonic()
            if jpeg and (jpeg is not previous or now-last_sent>=1):
                previous=jpeg;last_sent=now
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+jpeg+b"\r\n"
                count+=1
            if run.state in ("ended","stopped","error"):break
            await asyncio.sleep(.04)
    return StreamingResponse(frames(),media_type="multipart/x-mixed-replace; boundary=frame")
