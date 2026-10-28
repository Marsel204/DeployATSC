#!/usr/bin/env python3
"""Build the board-local FP16 engine and its verified source/runtime binding."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

ROOT=Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_engine(weights, destination, *, exporter, runtime_info):
    weights=Path(weights).resolve();destination=Path(destination).resolve()
    if not weights.is_file():raise ValueError('Source weights do not exist')
    if destination.suffix!='.engine':raise ValueError('Output must be an .engine file')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.observation-export-',dir=destination.parent) as directory:
        directory=Path(directory)
        staged=directory/'Final.pt';shutil.copy2(weights,staged)
        source_hash=sha256(staged)
        exported=Path(exporter(staged))
        if not exported.is_file() or not exported.stat().st_size:raise ValueError('Exporter produced no engine')
        pending=directory/'publish.engine';shutil.copyfile(exported,pending)
        metadata={**runtime_info,'source_sha256':source_hash,'engine_sha256':sha256(pending),
                  'batch':1,'imgsz':640,'precision':'fp16'}
        manifest=directory/'publish.engine.json'
        manifest.write_text(json.dumps(metadata,indent=2)+'\n')
        if sha256(weights)!=source_hash:raise ValueError('Source weights changed during export; retry the build')
        # Readers validate both hashes: a crash between these replacements safely falls back.
        os.replace(pending,destination)
        os.replace(manifest,destination.with_suffix('.engine.json'))
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',type=Path,default=ROOT/'models/Final.pt')
    parser.add_argument('--output',type=Path,default=ROOT/'models/Final_fp16.engine')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--workspace',type=float,default=2.,help='TensorRT builder workspace in GiB')
    args=parser.parse_args()
    import torch
    import tensorrt
    from ultralytics import YOLO
    if not torch.cuda.is_available():raise RuntimeError('A CUDA device is required to build the engine')
    runtime={'tensorrt_version':tensorrt.__version__,
             'compute_capability':list(torch.cuda.get_device_capability(args.device))}
    def export(path):
        return YOLO(str(path)).export(format='engine',half=True,imgsz=640,batch=1,
            device=args.device,workspace=args.workspace,simplify=False)
    metadata=build_engine(args.weights,args.output,exporter=export,runtime_info=runtime)
    print(json.dumps({'engine':str(args.output),**metadata},indent=2))


if __name__=='__main__':main()
