#!/usr/bin/env python3
"""Launch script for ATSC Web Control & Replay Dashboard."""

from __future__ import annotations

# Direct script invocation needs the project root before the first application import.
__import__("sys").path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
from src.common.runtime_tuning import apply_blas_policy
apply_blas_policy()

import argparse
import os
import os
import sys
import uvicorn

CORE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CORE_ROOT not in sys.path:
    sys.path.insert(0, CORE_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch ATSC Web Control & Replay Dashboard")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve dashboard on (default: 8000)")
    parser.add_argument("--backend", choices=["auto","pytorch","tensorrt"], default=os.environ.get("ATSC_DETECTOR_BACKEND","auto"))
    parser.add_argument("--reload", action="store_true", help="Unsupported in observation mode")
    parser.add_argument("--source-mode", choices=["replay", "live"], default="replay")
    parser.add_argument("--source-sys", default=None)
    parser.add_argument("--source-acq", default=None)
    parser.add_argument("--config", default=None, help="Existing road profile for operator review")
    return parser.parse_args()


def serve_observation(host="127.0.0.1", port=8000, source_mode="replay", source_sys=None,
                      source_acq=None, config=None, reload=False) -> None:
    if host not in ("127.0.0.1","localhost","::1"):
        raise ValueError("Observation remains loopback-only until gateway deployment is provisioned")
    if reload:
        raise ValueError("Observation reload is disabled; run one worker without reload")
    from src.ui.app import app
    app.state.operation="observation"
    app.state.allowed_origins={f"http://{name}:{port}" for name in ("localhost","127.0.0.1","[::1]")}
    from src.ui.observation_api import launch_settings
    launch_settings.clear()
    launch_settings.update(source_mode=source_mode, source_sys=source_sys,
                           source_acq=source_acq, config_path=config)
    print("\n==================================================================")
    print("      🚀 ATSC JETSON EDGE CONTROLLER & REPLAY DASHBOARD           ")
    print("==================================================================")
    print(f" Web UI URL : http://localhost:{port}")
    print(f" Binding    : {host}:{port}; observation only, no controller actuation")
    print(" Inference  : Camera 2 continuous YOLO/tracking; Camera 1 YOLO only on 'Analyze frame & predict' (one frame per click)")
    print(" Press Ctrl+C to terminate.")
    print("==================================================================\n")

    uvicorn.run(
        "src.ui.app:app",
        host=host,
        port=port,
        reload=False,
        workers=1,
        log_level="info",
    )


def main() -> None:
    args = parse_args()
    os.environ["ATSC_DETECTOR_BACKEND"]=args.backend
    serve_observation(host=args.host,port=args.port,source_mode=args.source_mode,
                      source_sys=args.source_sys,source_acq=args.source_acq,
                      config=args.config,reload=args.reload)


if __name__ == "__main__":
    main()
