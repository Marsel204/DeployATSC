"""Saved evidence browsing is independent of the current measurement owner."""
from functools import partial
from pathlib import Path
import csv
import hashlib
import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient
from playwright.sync_api import sync_playwright, expect

from src.experiment.export import ExperimentStore
from src.ui.app import app
import src.ui.observation_api as api
from src.ui.observation_session import ObservationManager, ObservationSession
from test_observation_sessions import FakeANFIS, run_payload
from test_observation_pipeline import KnownVehicleDetector
from test_observation_dashboard_browser import browser_server  # isolated real localhost API

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


def test_two_real_runs_same_request_restart_browse_quantiles_and_linked_export(tmp_path, video_factory, monkeypatch):
    class InsideDetector(KnownVehicleDetector):
        def track(self, frames, timestamp=None):
            result=super().track(frames,timestamp)
            for row in result[0]:
                row.bbox=[150,100,170,120];row.bottom_center=(160.,120.)
            return result
    factory = partial(ObservationSession, detector_factory=InsideDetector, anfis_factory=FakeANFIS)
    manager = ObservationManager(tmp_path/'runs', tmp_path/'profiles', factory)
    monkeypatch.setattr(api, 'observations', manager)
    identities = []
    with TestClient(app) as client:
        for _ in range(2):
            response = client.post('/api/observation/start', json=run_payload(video_factory))
            assert response.status_code == 200, response.text
            identity = response.json()['run_id']; identities.append(identity)
            capture = client.post('/api/observation/predict', json={'run_id': identity, 'request_id': 'same-click'})
            assert capture.status_code == 200, capture.text
            manager.current.engine.predictions.await_idle()
            assert client.post('/api/observation/control', json={'run_id': identity, 'action': 'stop'}).status_code == 200
            # HTTP stop is requested/applied asynchronously; freeze only after final commit.
            manager.current.stop()
            assert manager.current.finalized
        assert len(set(identities)) == 2
        before = fingerprint(tmp_path/'runs')
        restarted = ObservationManager(tmp_path/'runs', tmp_path/'profiles', factory)
        monkeypatch.setattr(api, 'observations', restarted)
        assert {r['run_id'] for r in client.get('/api/observation/runs').json()['runs']} == set(identities)
        for identity in identities:
            params = {'run_id': identity}
            evidence = client.get('/api/observation/evidence', params=params).json()
            assert evidence['run_id'] == identity
            assert evidence['predictions'][0]['request_id'] == 'same-click'
            for kind in ('vehicles', 'predictions', 'comparisons'):
                page = client.get('/api/observation/history/'+kind, params=params).json()
                assert page['rows'] == evidence[kind]
                assert page['generation'] == evidence['durability']['generation']
            assert evidence['vehicles'] and all(v['vehicle_id'].startswith(identity+':') for v in evidence['vehicles'])
            quantiles = client.get('/api/observation/quantiles', params=params).json()
            assert quantiles['sample_count'] == len(evidence['vehicles'])
            assert quantiles['median_red_wait_sec'] == evidence['summary']['observed_median_red_wait_sec']
            with zipfile.ZipFile(io.BytesIO(client.get('/api/observation/export', params=params).content)) as bundle:
                exported = json.loads(bundle.read('evidence.json'))
                assert exported == evidence
                assert json.loads(bundle.read('manifest.json'))['run_id'] == identity
                predictions = list(csv.DictReader(io.StringIO(bundle.read('predictions.csv').decode())))
                vehicles = list(csv.DictReader(io.StringIO(bundle.read('vehicles.csv').decode())))
                assert predictions[0]['request_id'] == 'same-click'
                assert {v['vehicle_id'] for v in vehicles} == {v['vehicle_id'] for v in evidence['vehicles']}
        assert restarted.current is None
        assert fingerprint(tmp_path/'runs') == before


@pytest.fixture
def saved_browser_server(tmp_path, browser_server):
    # Same durable protocol as the app; 101 rows exercise actual second pages.
    # Seed only before browser access, never substitute the history API.
    for identity, wait in [('saved_alpha', 20), ('saved_beta', 40)]:
        store = ExperimentStore(tmp_path/'runs', identity, {'operation': 'observation'})
        store.persist({'run_id': identity, 'snapshot_revision': 1, 'closed': True,
            'lifecycle': {'durable_finalized': True},
            'vehicles': [{'vehicle_id': f'{identity}:acquisition:0:{i}', 'local_id': i,
                'tracker_epoch': 0, 'class_name': '<img src=x onerror=alert(1)>',
                'red_wait_sec': wait, 'total_stopped_sec': wait+4, 'tracking_state': 'STOPPED',
                'cycles': [i+1], 'quality_flags': [], 'completed': True} for i in range(101)],
            'predictions': [{'request_id': 'same-click' if i==0 else f'{identity}-click-{i}',
                'cycle_id': i+1, 'recommended_green_sec': wait+15, 'status': 'matched',
                'eligible': True, 'qualified': False,
                'qualification': {'policy': 'motion-phase-v1', 'state': 'diagnostic'}} for i in range(101)],
            'comparisons': [{'cycle_id': i+1, 'actual_green_sec': wait+5,
                'recommended_green_sec': wait+15, 'difference_sec': 10,
                'prediction': {'request_id': 'same-click' if i==0 else f'{identity}-click-{i}', 'status': 'matched'},
                'eligible': True} for i in range(101)]})
    legacy = tmp_path/'runs'/'saved_legacy'; legacy.mkdir()
    (legacy/'manifest.json').write_text(json.dumps({'schema_version':1, 'run_id':'saved_legacy'}))
    (legacy/'evidence.json').write_text(json.dumps({'run_id':'saved_legacy', 'vehicles':[],
        'predictions':[{'request_id':'legacy-click', 'status':'unmatched'}], 'comparisons':[]}))
    yield browser_server


@pytest.mark.parametrize('width', [1440, 390])
def test_saved_browser_pages_quantiles_export_and_active_isolation(saved_browser_server, tmp_path, width):
    before = fingerprint(tmp_path/'runs')
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path='/usr/bin/chromium', headless=True, args=['--no-sandbox'])
        page = browser.new_page(viewport={'width':width, 'height':1000})
        errors=[]; calls=[]
        page.on('pageerror', lambda e:errors.append(str(e)))
        page.on('request', lambda r:calls.append((r.method, r.url)))
        page.goto(saved_browser_server, wait_until='domcontentloaded')
        expect(page.locator('#savedRuns option')).to_have_count(3)
        page.locator('#savedRuns').select_option('saved_alpha')
        page.locator('#browseSaved').click()
        expect(page.locator('#viewedRunStatus')).to_contain_text('saved_alpha')
        expect(page.locator('#savedEvidence tbody tr')).to_have_count(100)
        expect(page.locator('#savedEvidence')).to_contain_text('20.0 s')
        assert page.locator('#savedEvidence img').count() == 0  # untrusted strings are text
        page.locator('#nextSaved').click()
        expect(page.locator('#savedPage')).to_contain_text('101–101 of 101')
        page.locator('#previousSaved').click()
        expect(page.locator('#savedPage')).to_contain_text('1–100 of 101')
        for kind in ('predictions', 'comparisons'):
            page.locator('#savedKind').select_option(kind)
            expect(page.locator('#savedEvidence')).to_contain_text('same-click')
            expect(page.locator('#savedEvidence')).to_contain_text('motion-phase-v1' if kind=='predictions' else 'qualification unknown')
            page.locator('#nextSaved').click()
            expect(page.locator('#savedPage')).to_contain_text('101–101 of 101')
            page.locator('#previousSaved').click()
        page.locator('#savedQuantiles').click()
        expect(page.locator('#savedQuantileStatus')).to_contain_text('20.0 / 20.0 s')
        with page.expect_download() as downloaded:page.locator('#exportSaved').click()
        with zipfile.ZipFile(downloaded.value.path()) as bundle:
            evidence=json.loads(bundle.read('evidence.json'))
            assert evidence['run_id']=='saved_alpha' and len(evidence['vehicles'])==101
            assert evidence['predictions'][0]['request_id']==evidence['comparisons'][0]['prediction']['request_id']
        page.locator('#savedRuns').select_option('saved_beta')
        expect(page.locator('#viewedRunStatus')).to_contain_text('saved_beta')
        expect(page.locator('#savedQuantileStatus')).not_to_contain_text('20.0 /')
        page.locator('#savedKind').select_option('vehicles')
        expect(page.locator('#savedEvidence')).to_contain_text('40.0 s')
        assert '20.0 s' not in page.locator('#savedEvidence').inner_text()
        page.locator('#savedQuantiles').click()
        expect(page.locator('#savedQuantileStatus')).to_contain_text('40.0 / 40.0 s')
        # Let multiple live telemetry polls occur; historical reads must remain explicit.
        count = sum('/history/' in url or '/quantiles?' in url for _,url in calls)
        page.wait_for_timeout(1100)
        assert sum('/history/' in url or '/quantiles?' in url for _,url in calls)==count
        expect(page.locator('#meanWait')).to_have_text('—')
        expect(page.locator('#metricQ')).to_have_text('—')
        page.reload(wait_until='domcontentloaded')
        expect(page.locator('#savedRuns option')).to_have_count(3)
        page.locator('#savedRuns').select_option('saved_legacy');page.locator('#browseSaved').click()
        page.locator('#savedKind').select_option('predictions')
        expect(page.locator('#viewedRunStatus')).to_contain_text('legacy')
        expect(page.locator('#savedEvidence')).to_contain_text('qualification unknown')
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path=str(ROOT/f'.venv/phase-e-resume-saved-{width}.png'), full_page=True)
        assert not errors
        assert not any(method=='POST' and ('/start' in url or '/control' in url or '/recover' in url) for method,url in calls)
        browser.close()
    assert fingerprint(tmp_path/'runs') == before


def test_saved_browser_generation_conflict_and_late_response(saved_browser_server):
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
        page=browser.new_page();page.goto(saved_browser_server,wait_until='domcontentloaded')
        expect(page.locator('#savedRuns option')).to_have_count(3)
        page.locator('#savedRuns').select_option('saved_alpha');page.locator('#browseSaved').click()
        expect(page.locator('#nextSaved')).to_be_enabled()
        def conflict(route):
            assert 'generation=1' in route.request.url
            route.fulfill(status=409,json={'detail':'Durable generation changed; restart pagination'})
        page.route('**/history/vehicles?*offset=100*',conflict)
        page.locator('#nextSaved').click()
        expect(page.locator('#viewedRunStatus')).to_contain_text('restart pagination')
        expect(page.locator('#nextSaved')).to_be_disabled()
        expect(page.locator('#savedEvidence')).not_to_contain_text('20.0 s')
        page.locator('#browseSaved').click()
        expect(page.locator('#savedPage')).to_contain_text('1–100 of 101')
        held=[]
        page.route('**/history/predictions?*',lambda route:held.append(route))
        page.locator('#savedKind').select_option('predictions')
        page.wait_for_timeout(150)
        assert held
        page.locator('#savedRuns').select_option('saved_beta')
        page.locator('#savedKind').select_option('vehicles')
        expect(page.locator('#savedEvidence')).to_contain_text('40.0 s')
        for route in held:route.fulfill(json={'rows':[{'request_id':'STALE'}], 'offset':0,'total':1,'next_offset':None,'generation':1})
        expect(page.locator('#savedEvidence')).not_to_contain_text('STALE')
        expect(page.locator('#viewedRunStatus')).to_contain_text('saved_beta')
        browser.close()


def test_saved_view_has_separate_identity_paging_and_comparison_controls():
    from html.parser import HTMLParser
    class IDs(HTMLParser):
        def __init__(self):super().__init__();self.ids=set()
        def handle_starttag(self, tag, attrs):
            self.ids.update(value for key,value in attrs if key=='id')
    parser=IDs();parser.feed((ROOT/'src/ui/static/index.html').read_text())
    assert {'browseSaved','viewedRunStatus','savedEvidence','savedKind','previousSaved',
            'nextSaved','savedQuantiles','savedQuantileStatus'} <= parser.ids


def test_real_browser_repeated_runs_keep_active_controls_and_streams(saved_browser_server, video_factory):
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1440,'height':1100})
        identities=[]
        for _ in range(2):
            response=page.request.post(saved_browser_server+'/api/observation/start',data=run_payload(video_factory))
            assert response.ok,response.text()
            identity=response.json()['run_id'];identities.append(identity)
            response=page.request.post(saved_browser_server+'/api/observation/predict',data={'run_id':identity,'request_id':'same-click'})
            assert response.ok,response.text()
            if len(identities)==1:
                assert page.request.post(saved_browser_server+'/api/observation/control',data={'run_id':identity,'action':'stop'}).ok
        assert identities[0]!=identities[1]
        page.goto(saved_browser_server,wait_until='domcontentloaded')
        expect(page.locator('#sessionStatus')).to_have_text('PAUSED')
        expect(page.locator('#runLabel')).to_have_text(identities[1])
        feeds={role:page.locator('#'+role+'Feed').get_attribute('src') for role in ('system','acquisition')}
        calls=[];page.on('request',lambda r:calls.append((r.method,r.url)))
        for identity in identities:
            page.locator('#savedRuns').select_option(identity);page.locator('#browseSaved').click()
            page.locator('#savedKind').select_option('predictions')
            expect(page.locator('#savedEvidence')).to_contain_text('same-click')
            expect(page.locator('#viewedRunStatus')).to_contain_text(identity)
            page.locator('#savedQuantiles').click()
            expect(page.locator('#savedQuantileStatus')).to_contain_text(identity)
            with page.expect_download() as download:page.locator('#exportSaved').click()
            with zipfile.ZipFile(download.value.path()) as bundle:
                assert json.loads(bundle.read('manifest.json'))['run_id']==identity
                assert json.loads(bundle.read('evidence.json'))['predictions'][0]['request_id']=='same-click'
            expect(page.locator('#runLabel')).to_have_text(identities[1])
            expect(page.locator('#btnStep')).to_be_enabled()
            expect(page.locator('#btnStop')).to_be_enabled()
            assert feeds=={role:page.locator('#'+role+'Feed').get_attribute('src') for role in feeds}
        assert not any(method=='POST' for method,_ in calls)
        page.locator('#btnStep').click();page.locator('#btnStop').click()
        expect(page.locator('#sessionStatus')).to_have_text('STOPPED')
        browser.close()


def test_browser_explicit_scale_review_save_and_load(browser_server):
    from test_observation_dashboard_browser import click_point
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1440,'height':1100})
        page.goto(browser_server,wait_until='domcontentloaded')
        page.wait_for_selector('#systemFeed:not([hidden])')
        page.locator('#systemCalibrate').click()
        page.locator('#toolPolygon').click()
        for point in [(20,50),(290,50),(300,220),(10,220)]:click_point(page,*point)
        page.locator('#toolAnchors').click()
        for point in [(20,50),(290,50),(300,220),(10,220)]:click_point(page,*point)
        page.locator('#scaleMethod').select_option('measured_dimensions')
        page.locator('#groundWidth').fill('8');page.locator('#groundLength').fill('40')
        page.locator('#scaleReference').fill('Synthetic browser fixture, not a field survey')
        page.locator('#saveCalibration').click()
        expect(page.locator('#calibrationError')).to_contain_text('confirm')
        page.locator('#bindingReviewed').check();page.locator('#saveCalibration').click()
        expect(page.locator('#systemReadiness')).to_have_text('Calibration saved')
        profile=page.locator('#systemProfile').input_value()
        page.reload(wait_until='domcontentloaded')
        expect(page.locator('#systemProfile option')).to_have_count(2)
        page.locator('#systemProfile').select_option(profile)
        expect(page.locator('#systemReadiness')).to_have_text('Calibration saved')
        page.locator('#systemCalibrate').click()
        expect(page.locator('#scaleMethod')).to_have_value('measured_dimensions')
        expect(page.locator('#groundWidth')).to_have_value('8')
        expect(page.locator('#groundLength')).to_have_value('40')
        expect(page.locator('#bindingReviewed')).not_to_be_checked()
        page.keyboard.press('Escape');browser.close()
