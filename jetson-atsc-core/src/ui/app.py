"""FastAPI Backend Application for ATSC Interactive Web Dashboard."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

from src.ui.runtime_ownership import RuntimeConflict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.ui.session_manager import manager

logger = logging.getLogger("ui_app")

CORE_ROOT = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

@asynccontextmanager
async def lifespan(application):
    from src.ui import observation_api
    try:timeout=float(os.environ.get("ATSC_SHUTDOWN_TIMEOUT_SEC","5"))
    except ValueError as error:raise ValueError("Invalid shutdown deadline") from error
    if not math.isfinite(timeout) or timeout<0:raise ValueError("Invalid shutdown deadline")
    # Reopening an application lifespan allows a new serving interval.
    if hasattr(observation_api.observations,"closing"):observation_api.observations.closing=False
    if hasattr(manager,"closing"):manager.closing=False
    try:yield
    finally:
        deadline=time.monotonic()+max(0,timeout)
        async def close(owner):
            try:return await asyncio.to_thread(lambda:owner.shutdown(max(0,deadline-time.monotonic())))
            except Exception as error:return {"resources_released":False,"finalized":False,"error":str(error)}
        legacy,observation=await asyncio.gather(close(manager),close(observation_api.observations))
        application.state.shutdown_report={"legacy":legacy,"observation":observation}


app = FastAPI(
    lifespan=lifespan,
    title="ATSC Edge Control & Replay Dashboard",
    description="Adaptive Traffic Signal Controller Interactive Perception & ANFIS Dashboard",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000","http://127.0.0.1:8000","http://[::1]:8000"],
    allow_credentials=False,
    allow_methods=["GET","POST","OPTIONS"],
    allow_headers=["Content-Type"],
)


app.state.operation="legacy"  # Direct app imports retain the legacy compatibility entrypoint.
app.state.allowed_hosts={"localhost","127.0.0.1","::1","testserver"}
app.state.allowed_origins={"http://localhost:8000","http://127.0.0.1:8000","http://[::1]:8000"}


@app.middleware("http")
async def http_boundaries(request,call_next):
    try:
        authority=urlsplit("//"+request.headers.get("host",""))
        authority.port  # Validate numeric port syntax as well as the hostname.
        host=authority.hostname if not (authority.username or authority.password or authority.path or authority.query or authority.fragment) else None
    except ValueError:host=None
    if host not in app.state.allowed_hosts:
        return JSONResponse({"detail":"Untrusted Host"},status_code=400)
    origin=request.headers.get("origin")
    same_origin=f"{request.url.scheme}://{request.headers.get('host','')}"
    if origin is not None and origin not in app.state.allowed_origins and origin!=same_origin:
        return JSONResponse({"detail":"Untrusted Origin"},status_code=403)
    path=request.url.path
    if app.state.operation=="observation" and (path.startswith("/api/session/") or path=="/stream/video_feed"):
        return JSONResponse({"detail":"Legacy runtime routes are disabled in observation mode"},status_code=409)
    return await call_next(request)


class StartSessionRequest(BaseModel):
    video_path: str
    config_path: str
    trigger_interval_sec: float = 15.0
    step_mode: bool = False


class ControlSessionRequest(BaseModel):
    action: str  # "play", "pause", "step", "stop"


class YouTubeDownloadRequest(BaseModel):
    url: str
    duration_sec: int = 30


class LampRoiModel(BaseModel):
    ymin: int
    xmin: int
    ymax: int
    xmax: int


class GroundDimensionsModel(BaseModel):
    width: float = 8.0
    length: float = 40.0


class SaveCustomConfigRequest(BaseModel):
    name: str
    lamp_roi_pixel: LampRoiModel
    src_points_pixel: list[list[float]]
    ground_dimensions_m: GroundDimensionsModel = Field(default_factory=GroundDimensionsModel)
    description: Optional[str] = "User-Drawn Custom Calibration"
    polygon_points_pixel: Optional[list[list[float]]] = None
    sample_ruler: Optional[Dict[str, Any]] = None


@app.get("/api/video/snapshot")
def get_video_snapshot(video_path: str):
    try:
        jpeg_bytes = manager.get_video_snapshot(video_path)
        return Response(content=jpeg_bytes, media_type="image/jpeg")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeConflict as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/configs/save_custom")
def save_custom_config(req: SaveCustomConfigRequest):
    try:
        result = manager.validate_and_save_custom_config(req.model_dump(exclude_none=True))
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = Path(os.environ.get("ATSC_UI_PATH", STATIC_DIR / "index.html"))
    if not index_path.exists():
        return HTMLResponse("<h2>ATSC Dashboard Loading...</h2>", status_code=200)
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/configs")
async def get_configs():
    configs = manager.list_configs()
    return {"configs": configs}


@app.get("/api/sample_videos")
async def get_sample_videos():
    videos = manager.list_videos()
    return {"videos": videos}


@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename.endswith((".mp4", ".avi", ".mkv", ".mov")):
        raise HTTPException(status_code=400, detail="Only MP4/video files are supported.")

    uploads_dir = CORE_ROOT / "data" / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    dest_path = uploads_dir / f"upload_{int(time.time())}_{file.filename}"
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "status": "uploaded",
        "video_path": str(dest_path),
        "filename": dest_path.name,
        "size_mb": round(dest_path.stat().st_size / (1024 * 1024), 2),
    }


@app.post("/api/download_youtube")
async def download_youtube(req: YouTubeDownloadRequest):
    try:
        final_path = manager.download_youtube_video(req.url, req.duration_sec)
        return {
            "status": "downloaded",
            "video_path": final_path,
            "filename": Path(final_path).name,
        }
    except Exception as e:
        logger.error(f"YouTube download failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to acquire YouTube video: {str(e)}")


@app.post("/api/session/start")
async def start_session(req: StartSessionRequest):
    if not os.path.exists(req.video_path):
        raise HTTPException(status_code=404, detail=f"Video not found: {req.video_path}")
    if not os.path.exists(req.config_path):
        raise HTTPException(status_code=404, detail=f"Config not found: {req.config_path}")

    try:
        session = manager.start_session(
            video_path=req.video_path,
            config_path=req.config_path,
            trigger_interval_sec=req.trigger_interval_sec,
            step_mode=req.step_mode,
        )
        return {
            "status": "initialized",
            "session_id": session.session_id,
            "total_frames": session.total_frames,
            "fps": session.native_fps,
        }
    except RuntimeConflict as e:
        raise HTTPException(status_code=409,detail=str(e))
    except Exception as e:
        logger.error(f"Session start failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/session/control")
async def control_session(req: ControlSessionRequest):
    session = manager.active_session
    if not session:
        return {"status": "no_active_session"}

    action = req.action.lower()
    if action == "play":
        session.play()
    elif action == "pause":
        session.pause()
    elif action == "step":
        session.step()
    elif action == "stop":
        manager.stop_session()
        return {"status": "stopped"}
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {req.action}")

    return {"status": session.state}


@app.get("/api/session/telemetry")
async def get_telemetry():
    session = manager.active_session
    if not session:
        return {
            "status": "idle",
            "frame_idx": 0,
            "total_frames": 0,
            "fps": 0.0,
            "metrics": {"V_w": 0.0, "Q": 0.0, "L": 0.0, "n_mc": 0, "n_lv": 0, "n_hv": 0},
            "decision": None,
            "history_decisions": [],
        }
    return session.get_telemetry()


@app.get("/stream/video_feed")
def video_feed(limit: Optional[int] = None):
    def frame_generator():
        count = 0
        try:
            while True:
                if limit is not None and count >= limit:
                    break
                session = manager.active_session
                if not session or session.state == "stopped":
                    time.sleep(0.05)
                    continue

                frame_bytes = session.get_latest_frame()
                if frame_bytes:
                    count += 1
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                    )
                time.sleep(0.033)
        except GeneratorExit:
            return

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# Mount static assets if present
from src.ui.observation_api import router as observation_router
app.include_router(observation_router)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
