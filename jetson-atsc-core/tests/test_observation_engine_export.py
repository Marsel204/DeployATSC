import hashlib
import json

import pytest

from scripts.export_observation_engine import build_engine


def test_build_publishes_verified_engine_without_modifying_source(tmp_path):
    weights=tmp_path/'Final.pt';weights.write_bytes(b'original weights')
    output=tmp_path/'models/Final_fp16.engine'
    def export(staged):
        assert staged!=weights and staged.read_bytes()==weights.read_bytes()
        engine=staged.with_suffix('.engine');engine.write_bytes(b'exported fp16 engine')
        return engine
    metadata=build_engine(weights,output,exporter=export,
        runtime_info={'tensorrt_version':'test','compute_capability':[8,7]})
    assert weights.read_bytes()==b'original weights'
    assert metadata==json.loads(output.with_suffix('.engine.json').read_text())
    assert metadata['source_sha256']==hashlib.sha256(weights.read_bytes()).hexdigest()
    assert metadata['engine_sha256']==hashlib.sha256(output.read_bytes()).hexdigest()
    assert metadata['batch']==1 and metadata['imgsz']==640 and metadata['precision']=='fp16'
    assert not list(output.parent.glob('.observation-export-*'))


def test_failed_export_preserves_existing_artifacts(tmp_path):
    weights=tmp_path/'Final.pt';weights.write_bytes(b'weights')
    output=tmp_path/'Final_fp16.engine';output.write_bytes(b'existing engine')
    manifest=output.with_suffix('.engine.json');manifest.write_bytes(b'existing binding')
    def failed(staged):raise RuntimeError('build failed')
    with pytest.raises(RuntimeError,match='build failed'):
        build_engine(weights,output,exporter=failed,runtime_info={})
    assert output.read_bytes()==b'existing engine'
    assert manifest.read_bytes()==b'existing binding'
    assert weights.read_bytes()==b'weights'
