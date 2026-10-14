#!/usr/bin/env python3
"""Launch script for ATSC Web Control & Replay Dashboard."""

from __future__ import annotations

import argparse
import os
import sys
import uvicorn

CORE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CORE_ROOT not in sys.path:
    sys.path.insert(0, CORE_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch ATSC Web Control & Replay Dashboard")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve dashboard on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--source-mode", choices=["replay", "live"], default="replay")
    parser.add_argument("--source-sys", default=None)
    parser.add_argument("--source-acq", default=None)
    parser.add_argument("--config", default=None, help="Existing road profile for operator review")
    return parser.parse_args()


def serve_observation(host="0.0.0.0", port=8000, source_mode="replay", source_sys=None,
                      source_acq=None, config=None, reload=False) -> None:
    from src.ui.observation_api import launch_settings
    launch_settings.clear()
    launch_settings.update(source_mode=source_mode, source_sys=source_sys,
                           source_acq=source_acq, config_path=config)
    print("\n==================================================================")
    print("      🚀 ATSC JETSON EDGE CONTROLLER & REPLAY DASHBOARD           ")
    print("==================================================================")
    print(f" Web UI URL : http://localhost:{port} (or http://<jetson-ip>:{port})")
    print(f" Binding    : {host}:{port}; observation only, no controller actuation")
    print(" Press Ctrl+C to terminate.")
    print("==================================================================\n")

    uvicorn.run(
        "src.ui.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


def main() -> None:
    args = parse_args()
    serve_observation(host=args.host,port=args.port,source_mode=args.source_mode,
                      source_sys=args.source_sys,source_acq=args.source_acq,
                      config=args.config,reload=args.reload)


if __name__ == "__main__":
    main()
