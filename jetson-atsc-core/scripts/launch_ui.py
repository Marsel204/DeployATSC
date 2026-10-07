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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("\n==================================================================")
    print("      🚀 ATSC JETSON EDGE CONTROLLER & REPLAY DASHBOARD           ")
    print("==================================================================")
    print(f" Web UI URL : http://localhost:{args.port} (or http://<jetson-ip>:{args.port})")
    print(f" Binding    : {args.host}:{args.port}")
    print(" Press Ctrl+C to terminate.")
    print("==================================================================\n")

    uvicorn.run(
        "src.ui.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
