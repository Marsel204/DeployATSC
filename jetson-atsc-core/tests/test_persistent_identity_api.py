import io
import json
import zipfile

import pytest

from test_observation_api import client
from test_observation_sessions import run_payload


@pytest.mark.parametrize('settings', [{'mode': 'invalid'}, {'capacity': True},
    {'mode': 'persistent_v1', 'max_estimated_gap_sec': 20}])
def test_identity_api_rejects_invalid_settings_without_starting(client, video_factory, settings):
    data = run_payload(video_factory)
    data['vehicle_identity'] = settings
    assert client.post('/api/observation/start', json=data).status_code == 422


def test_persistent_api_policy_is_frozen_in_manifest_and_old_requests_still_work(client, video_factory):
    data = run_payload(video_factory)
    data['vehicle_identity'] = {'mode': 'persistent_v1', 'max_estimated_gap_sec': 3}
    response = client.post('/api/observation/start', json=data)
    assert response.status_code == 200, response.text
    run_id = response.json()['run_id']
    telemetry = client.get('/api/observation/telemetry').json()
    assert telemetry['vehicle_identity']['max_estimated_gap_sec'] == 3
    export = client.get('/api/observation/export', params={'run_id': run_id})
    with zipfile.ZipFile(io.BytesIO(export.content)) as bundle:
        assert json.loads(bundle.read('manifest.json'))['vehicle_identity']['max_estimated_gap_sec'] == 3
    assert client.get('/api/observation/quantiles', params={'run_id': run_id, 'metric': 'other'}).status_code == 422
