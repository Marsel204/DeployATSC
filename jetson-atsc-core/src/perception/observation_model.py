"""Select a verified single-frame observation engine; retain explicit PyTorch rollback."""
from functools import lru_cache, partial
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

from src.perception.detector_tracker import DetectorTracker


@lru_cache(maxsize=32)
def _digest(path, modified, size):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(path):
    stat=path.stat()
    return _digest(str(path),stat.st_mtime_ns,stat.st_size)


def runtime_error(metadata, device):
    try:
        if importlib.metadata.version('tensorrt')!=metadata['tensorrt_version']:
            return 'different TensorRT runtime; rebuild the engine'
        import torch
        if not torch.cuda.is_available():return 'CUDA is unavailable'
        if list(torch.cuda.get_device_capability(device))!=metadata['compute_capability']:
            return 'different GPU compute capability; rebuild the engine'
    except (ImportError, RuntimeError, ValueError, importlib.metadata.PackageNotFoundError) as error:
        return 'TensorRT runtime unavailable: '+str(error)
    return None


def select_model(root, device='cuda:0', backend=None):
    root=Path(root)
    backend=backend or os.environ.get('ATSC_DETECTOR_BACKEND','auto')
    if backend not in ('auto','pytorch','tensorrt'):
        raise ValueError('Detector backend must be auto, pytorch or tensorrt')
    weights=root/'Final.pt';engine=root/'Final_fp16.engine'
    choice={'requested':backend,'backend':'pytorch','path':str(weights),'reason':'PyTorch selected'}
    if backend=='pytorch':return choice
    reason=None
    try:
        if str(device)=='cpu':reason='TensorRT requires a CUDA device'
        elif not engine.is_file():reason='TensorRT engine is not installed'
        else:
            metadata=json.loads(engine.with_suffix('.engine.json').read_text())
            if metadata['source_sha256']!=digest(weights):reason='source weights hash differs; rebuild the engine'
            elif metadata['engine_sha256']!=digest(engine):reason='TensorRT engine hash differs'
            elif (metadata['batch'],metadata['imgsz'],metadata['precision'])!=(1,640,'fp16'):
                reason='TensorRT engine must be FP16, batch 1, 640 pixels'
            else:reason=runtime_error(metadata,device)
    except (OSError, ValueError, KeyError, TypeError) as error:
        reason='Invalid TensorRT engine metadata: '+str(error)
    if reason:
        if backend=='tensorrt':raise ValueError(reason)
        choice['reason']=reason+'; using PyTorch'
        return choice
    return {**choice,'backend':'tensorrt','path':str(engine),'reason':'Verified TensorRT FP16 engine',
            'engine_build_precision':'fp16_enabled'}


def model_for_factory(root, factory=None):
    native=factory is None or factory is DetectorTracker
    device='cuda:0'
    if isinstance(factory,partial) and factory.func is DetectorTracker:
        native=True;device=factory.keywords.get('device',device)
    if native:return select_model(root,device)
    return {'requested':'injected','backend':'injected','path':str(Path(root)/'Final.pt'),
            'reason':'Caller-provided inference component'}
