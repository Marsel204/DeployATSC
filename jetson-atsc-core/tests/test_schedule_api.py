import io
import json
import zipfile

from test_observation_api import client
from test_observation_sessions import run_payload


def test_schedule_requires_explicit_persistent_mode_a_phase_and_assumption(client, video_factory):
    data=run_payload(video_factory)
    data['schedule_comparison']={'enabled':True,'unchanged_clearance':True}
    assert client.post('/api/observation/start',json=data).status_code==422
    data['vehicle_identity']={'mode':'persistent_v1'}
    data['system'].pop('signal_roi',None)
    assert client.post('/api/observation/start',json=data).status_code==422


def test_schedule_manifest_and_automatic_capture_ownership(client, video_factory):
    data=run_payload(video_factory)
    data['vehicle_identity']={'mode':'persistent_v1'}
    data['schedule_comparison']={'enabled':True,'unchanged_clearance':True}
    response=client.post('/api/observation/start',json=data)
    assert response.status_code==200,response.text
    run_id=response.json()['run_id']
    assert client.post('/api/observation/predict',json={'run_id':run_id,'request_id':'manual'}).status_code==409
    with zipfile.ZipFile(io.BytesIO(client.get('/api/observation/export',params={'run_id':run_id}).content)) as bundle:
        assert json.loads(bundle.read('manifest.json'))['schedule_comparison']['phase_source']=='camera_roi'
        assert 'vehicle_savings.csv' in bundle.namelist()
