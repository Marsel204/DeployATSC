from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.experiment.calibration import legacy_system_payload
from src.experiment.contracts import RunPolicy
from src.experiment.predictions import PredictionConflict
from src.ui.observation_session import ObservationManager
from src.ui.session_manager import manager as legacy_manager

router=APIRouter()
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
    lane_metrics:dict[str,Any]=Field(default_factory=dict)
    name:str=Field(default="profile",max_length=100)


class StartObservation(StrictModel):
    system:CameraSettings
    acquisition:CameraSettings
    source_mode:Literal["replay","live"]="replay"
    step_mode:bool=False
    shared_test:bool=False
    policy:dict[str,Any]=Field(default_factory=dict)

    @field_validator("policy")
    @classmethod
    def valid_policy(cls,value):
        try:RunPolicy(**value)
        except (TypeError,ValueError) as error:raise ValueError(str(error)) from error
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
    if isinstance(error,PredictionConflict):raise HTTPException(409,str(error))
    if isinstance(error,(ValueError,TypeError,KeyError,FileNotFoundError)):raise HTTPException(422,str(error))
    raise HTTPException(500,str(error))


@router.get("/api/observation/bootstrap")
def bootstrap():
    return {"operation":"observation",**launch_settings}


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
        return {"calibration":payload,"requires_review":True}
    except Exception as error:translate(error)


@router.get("/api/observation/snapshot")
def snapshot(role:Literal["system","acquisition"],source:str,
             rectification_path:str|None=None,source_mode:Literal["replay","live"]="replay"):
    try:return Response(observations.preview(role,source,rectification_path,source_mode),media_type="image/jpeg")
    except Exception as error:translate(error)


@router.post("/api/observation/start")
def start(settings:StartObservation):
    try:
        # Only an explicitly requested new observation replaces a legacy replay.
        legacy_manager.stop_session()
        run=observations.start(settings.model_dump(mode="json",exclude_none=True))
        return {"run_id":run.run_id,"state":run.state,"operation":"observation"}
    except Exception as error:translate(error)


@router.post("/api/observation/control")
def control(settings:ControlObservation):
    try:return observations.require(settings.run_id).control(settings.action)
    except Exception as error:translate(error)


@router.post("/api/observation/predict")
def predict(settings:CapturePrediction):
    try:return observations.require(settings.run_id).predict(settings.request_id)
    except Exception as error:translate(error)


@router.post("/api/observation/predictions/{request_id}/cancel")
def cancel_prediction(request_id:str,run_id:str):
    try:
        run=observations.require(run_id)
        result=run.engine.predictions.cancel(request_id)
        run.persist();return result
    except Exception as error:translate(error)


@router.get("/api/observation/telemetry")
def telemetry(offset:int=Query(0,ge=0),limit:int=Query(100,ge=1,le=200)):
    if observations.current is None:return {"state":"idle","operation":"observation"}
    return observations.current.telemetry(offset,limit)


@router.get("/api/observation/export")
def export_run():
    try:
        run=observations.require()
        return Response(run.export(),media_type="application/zip",
                        headers={"Content-Disposition":f'attachment; filename="{run.run_id}.zip"'})
    except Exception as error:translate(error)


@router.get("/stream/observation/{role}")
def stream(role:Literal["system","acquisition"],limit:int|None=Query(None,ge=1,le=100)):
    try:run=observations.require()
    except Exception as error:translate(error)
    async def frames():
        count=0
        while limit is None or count<limit:
            jpeg=run.jpeg(role)
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+jpeg+b"\r\n"
                count+=1
            if run.state in ("ended","stopped","error"):break
            await asyncio.sleep(.04)
    return StreamingResponse(frames(),media_type="multipart/x-mixed-replace; boundary=frame")
