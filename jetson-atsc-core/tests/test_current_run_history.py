"""Current-run response ownership regressions against shipped JS and real APIs."""
from pathlib import Path
import os,shutil,socket,subprocess,sys,time,urllib.request
ROOT=Path(__file__).resolve().parents[1]
import pytest
from playwright.sync_api import sync_playwright, expect
from test_observation_sessions import run_payload


@pytest.mark.parametrize('transition', ['new_run', 'live', 'superseded'])
@pytest.mark.parametrize('response_kind', ['success', 'error'])
def test_current_history_and_quantiles_reject_obsolete_responses(tracked_server, video_factory, transition, response_kind):
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
        page=browser.new_page()
        first=page.request.post(tracked_server+'/api/observation/start',data=run_payload(video_factory)).json()['run_id']
        page.goto(tracked_server,wait_until='domcontentloaded')
        expect(page.locator('#runLabel')).to_have_text(first)
        page.locator('#requestQuantiles').click()
        expect(page.locator('#quantileStatus')).to_contain_text('Exact durable')
        held=[]
        def hold(route):
            response=route.fetch()
            held.append((route,response.json()))
        page.route('**/history/vehicles?*',hold)
        page.route('**/quantiles?*',hold)
        def request_pair():
            page.locator('#historyVehicles').click();page.locator('#requestQuantiles').click()
            deadline=time.monotonic()+5
            while len(held)%2 or not held:
                assert time.monotonic()<deadline, 'Responses did not reach the barrier'
                page.wait_for_timeout(20)
        request_pair()
        assert len(held)==2
        second=first
        if transition=='new_run':
            page.locator('#btnStop').click();expect(page.locator('#sessionStatus')).to_have_text('STOPPED')
            page.locator('.settings summary').click();page.locator('#startPaused').select_option('yes');page.locator('#btnStart').click()
            expect(page.locator('#sessionStatus')).to_have_text('PAUSED')
            expect(page.locator('#runLabel')).not_to_have_text(first)
            second=page.locator('#runLabel').inner_text()
        elif transition=='live':
            page.locator('#liveVehicles').click()
        else:
            # A new durable generation distinguishes latest results from an older
            # success for this same run, even when both contain the same vehicle.
            capture=page.request.post(tracked_server+'/api/observation/predict',data={'run_id':first,'request_id':'new-generation'})
            assert capture.status==200,capture.text()
            page.locator('#historyVehicles').click();page.locator('#requestQuantiles').click()
            deadline=time.monotonic()+5
            while len(held)<4:
                assert time.monotonic()<deadline
                page.wait_for_timeout(20)
            for route,data in held[2:]:route.fulfill(json=data)
            latest_generation=held[2][1]['generation']
            expect(page.locator('#vehiclePage')).to_contain_text('durable records')
            expect(page.locator('#quantileStatus')).to_contain_text('Exact durable')
        if transition!='superseded':
            expect(page.locator('#vehiclePage')).to_contain_text('Live window')
            expect(page.locator('#quantileStatus')).to_have_text('No durable quantiles requested.')
        for route,data in held[:2]:
            if response_kind=='success':route.fulfill(json=data)
            else:route.fulfill(status=409,json={'detail':'Obsolete request error'})
        page.wait_for_timeout(700)  # let response continuations and the next poll run
        assert page.locator('#vehicleLedger td[title]').first.get_attribute('title').startswith(second+':')
        assert 'Obsolete request error' not in page.locator('#message').inner_text()
        if transition=='superseded':
            expect(page.locator('#vehiclePage')).to_contain_text(f'generation {latest_generation}')
            expect(page.locator('#quantileStatus')).to_contain_text(f'generation {latest_generation}')
        else:
            expect(page.locator('#vehiclePage')).to_contain_text('Live window')
            expect(page.locator('#quantileStatus')).to_have_text('No durable quantiles requested.')
        page.locator('#btnStop').click()
        browser.close()


@pytest.fixture
def tracked_server(tmp_path,video_factory):
    media=tmp_path/"data/sample_videos";media.mkdir(parents=True)
    shutil.copy2(video_factory("road.mp4",seconds=5),media/"road.mp4")
    shutil.copy2(video_factory("signal.mp4",seconds=5,color=(0,0,255)),media/"signal.mp4")
    listener=socket.socket();listener.bind(('127.0.0.1',0));port=listener.getsockname()[1];listener.close()
    server=tmp_path/"server.py"
    server.write_text(f'''
import sys
from pathlib import Path
from functools import partial
sys.path.insert(0,{str(ROOT)!r})
sys.path.insert(0,{str(ROOT/'tests')!r})
from src.ui.app import app
import src.ui.app as app_module
import src.ui.session_manager as legacy
import src.ui.observation_api as api
from src.ui.observation_session import ObservationManager,ObservationSession
from test_observation_sessions import FakeANFIS
from test_observation_pipeline import KnownVehicleDetector
class FakeDetector(KnownVehicleDetector):
    def track(self,frames,timestamp=None):
        result=super().track(frames,timestamp)
        for row in result[0]:row.bbox=[150,100,170,120];row.bottom_center=(160.,120.)
        return result
legacy.CORE_ROOT=Path({str(tmp_path)!r})
app_module.CORE_ROOT=Path({str(tmp_path)!r})
api.observations=ObservationManager(Path({str(tmp_path/'runs')!r}),Path({str(tmp_path/'profiles')!r}),partial(ObservationSession,detector_factory=FakeDetector,anfis_factory=FakeANFIS))
import uvicorn
uvicorn.run(app,host='127.0.0.1',port={port},log_level='error')
''')
    env=os.environ.copy()
    env['ATSC_UI_PATH']=str(ROOT/'src/ui/static/index.html')
    log=(tmp_path/"server.log").open('w')
    process=subprocess.Popen([sys.executable,str(server)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    base=f'http://127.0.0.1:{port}'
    try:
        for _ in range(100):
            try:
                with urllib.request.urlopen(base,timeout=.5):break
            except OSError:
                if process.poll() is not None:raise AssertionError((tmp_path/"server.log").read_text())
                time.sleep(.05)
        else:raise AssertionError('Browser API did not start')
        yield base
    finally:
        process.terminate();process.wait(timeout=5);log.close()
