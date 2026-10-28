"""Actual browser actions against an isolated API with controlled detections."""
from pathlib import Path
import os,shutil,socket,subprocess,sys,time,urllib.request
import pytest
from playwright.sync_api import sync_playwright,expect

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def browser_server(tmp_path,video_factory,request):
    detector=getattr(request,'param','FakeDetector')
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
from test_observation_sessions import FakeDetector,FakeANFIS
import time as _time
from src.perception.detector_tracker import TrackedVehicle
class GatedDetector(FakeDetector):
    """Camera 1 still-image call blocks until the test creates the gate file."""
    def detect_once(self,frame,timestamp=None):
        gate=Path({str(tmp_path/'camera1.gate')!r})
        while not gate.exists():_time.sleep(.02)
        return [TrackedVehicle(0,1,'car',[150,100,170,130],.9,timestamp,stable_id=False)]
class FailingDetector(FakeDetector):
    def detect_once(self,frame,timestamp=None):raise RuntimeError('camera 1 inference failed in test')
legacy.CORE_ROOT=Path({str(tmp_path)!r})
app_module.CORE_ROOT=Path({str(tmp_path)!r})
api.observations=ObservationManager(Path({str(tmp_path/'runs')!r}),Path({str(tmp_path/'profiles')!r}),partial(ObservationSession,detector_factory={detector},anfis_factory=FakeANFIS))
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


def click_point(page,x,y):
    box=page.locator('#calibrationCanvas').bounding_box()
    page.mouse.click(box['x']+x/320*box['width'],box['y']+y/240*box['height'])


def calibrate(page,role):
    page.locator('#'+role+'Calibrate').click()
    expect(page.locator('#calibrationModal')).to_have_class('modal-overlay active')
    if role=='system':expect(page.locator('#toolLamp')).to_be_hidden()
    else:
        expect(page.locator('#toolLamp')).to_be_visible()
        box=page.locator('#calibrationCanvas').bounding_box()
        page.mouse.move(box['x']+10/320*box['width'],box['y']+10/240*box['height'])
        page.mouse.down();page.mouse.move(box['x']+45/320*box['width'],box['y']+45/240*box['height']);page.mouse.up()
    page.locator('#toolPolygon').click()
    for point in [(20,50),(290,50),(300,220),(10,220)]:click_point(page,*point)
    page.locator('#toolAnchors').click()
    for point in [(20,50),(290,50),(300,220),(10,220)]:click_point(page,*point)
    if role=='acquisition':
        page.locator('#toolStopline').click();click_point(page,20,190);click_point(page,290,190)
    page.locator('#saveCalibration').click()
    expect(page.locator('#calibrationModal')).not_to_have_class('modal-overlay active')
    expect(page.locator('#'+role+'Readiness')).to_have_text('Calibration saved')


def test_browser_independent_calibration_manual_prediction_and_stop(browser_server):
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1440,'height':1100})
        errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(browser_server,wait_until='domcontentloaded')
        page.wait_for_selector('#systemFeed:not([hidden])')
        expect(page.locator('#acquisitionFeed')).to_be_hidden()
        expect(page.locator('#btnStart')).to_be_disabled()
        options=page.locator('#acquisitionSource option').evaluate_all('(items)=>items.map(i=>({value:i.value,text:i.textContent}))')
        signal=next(item['value'] for item in options if 'signal.mp4' in item['text'])
        page.locator('#acquisitionSource').select_option(signal)
        page.wait_for_selector('#acquisitionFeed:not([hidden])')
        calibrate(page,'system');calibrate(page,'acquisition')
        expect(page.locator('#btnStart')).to_be_enabled()
        page.locator('.settings summary').click();page.locator('#phaseDebounce').fill('1');page.locator('#startPaused').select_option('yes')
        page.locator('#btnStart').click()
        expect(page.locator('#sessionStatus')).to_have_text('PAUSED',timeout=10000)
        expect(page.locator('#phaseBadge')).to_have_text('Phase red')
        expect(page.locator('#qualificationPolicy')).to_contain_text('motion-phase-scale-v2')
        expect(page.locator('#qualificationPolicy')).to_contain_text('scale unverified')
        # On-demand Camera 1: enabled from a fresh preview frame before any Camera 1 metrics exist.
        expect(page.locator('#queueQuality')).to_have_text('Q unavailable until a Camera 1 frame is analysed')
        expect(page.locator('#camera1AnalysisLabel')).to_contain_text('Camera 1 not analysed yet')
        expect(page.locator('#metricVw')).to_have_text('—')
        expect(page.locator('#predictNextGreen')).to_have_text('Analyze frame & predict')
        expect(page.locator('#predictNextGreen')).to_be_enabled()
        page.locator('#predictNextGreen').click()
        expect(page.locator('#recommendedGreen')).to_have_text('35.0')
        expect(page.locator('#predictNextGreen')).to_be_disabled()
        expect(page.locator('#queueQuality')).to_contain_text('single_frame_spatial_proxy')
        expect(page.locator('#camera1AnalysisLabel')).to_contain_text('Last analysed frame #')
        expect(page.locator('#camera1AnalysisLabel')).to_contain_text('not live')
        expect(page.locator('#predictionStatus')).to_contain_text('analysed (0 vehicles)')
        expect(page.locator('#recommendationCaption')).to_contain_text('Q_measurement_single_frame_spatial_proxy')
        page.reload(wait_until='domcontentloaded')
        expect(page.locator('#sessionStatus')).to_have_text('PAUSED')
        expect(page.locator('#acquisitionSource')).to_have_value(signal)
        expect(page.locator('#acquisitionFeed')).to_be_visible()
        expect(page.locator('#recommendedGreen')).to_have_text('35.0')
        page.locator('#btnStep').click()
        page.locator('#btnStop').click();expect(page.locator('#sessionStatus')).to_have_text('STOPPED')
        expect(page.locator('#exportRun')).to_be_enabled()
        assert not errors
        assert page.locator('[data-camera=system]').bounding_box()['x'] < page.locator('[data-camera=acquisition]').bounding_box()['x']
        page.screenshot(path=str(ROOT/'.venv/phase-e-resume-browser.png'),full_page=True)
        browser.close()


@pytest.mark.parametrize('browser_server',['GatedDetector'],indirect=True)
def test_browser_on_demand_analysis_busy_state_and_last_frame_label(browser_server,tmp_path):
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1440,'height':1100})
        errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(browser_server,wait_until='domcontentloaded')
        page.wait_for_selector('#systemFeed:not([hidden])')
        options=page.locator('#acquisitionSource option').evaluate_all('(items)=>items.map(i=>({value:i.value,text:i.textContent}))')
        page.locator('#acquisitionSource').select_option(next(item['value'] for item in options if 'signal.mp4' in item['text']))
        page.wait_for_selector('#acquisitionFeed:not([hidden])')
        calibrate(page,'system');calibrate(page,'acquisition')
        page.locator('.settings summary').click();page.locator('#phaseDebounce').fill('1');page.locator('#startPaused').select_option('yes')
        page.locator('#btnStart').click()
        expect(page.locator('#sessionStatus')).to_have_text('PAUSED',timeout=10000)
        expect(page.locator('#predictionReady')).to_have_text('Ready: one frame per click')
        page.locator('#predictNextGreen').click()
        expect(page.locator('#message')).to_contain_text('Analysing one still image')
        expect(page.locator('#predictionStatus')).to_contain_text('analysing frame')
        expect(page.locator('#predictNextGreen')).to_be_disabled()
        expect(page.locator('#metricVw')).to_have_text('—')
        expect(page.locator('#camera1AnalysisLabel')).to_contain_text('Camera 1 not analysed yet')
        page.locator('#btnStep').click()  # Camera 2 keeps stepping while Camera 1 is busy
        expect(page.locator('#predictionStatus')).to_contain_text('analysing frame')
        (tmp_path/'camera1.gate').write_text('go')
        expect(page.locator('#predictionStatus')).to_contain_text('analysed (1 vehicles)')
        expect(page.locator('#metricVw')).to_have_text('1.00')
        expect(page.locator('#camera1AnalysisLabel')).to_contain_text('Last analysed frame #1 ')
        expect(page.locator('#camera1AnalysisLabel')).to_contain_text('not live')
        expect(page.locator('#queueQuality')).to_contain_text('single_frame_no_motion_evidence')
        expect(page.locator('#recommendedGreen')).to_have_text('35.0')
        page.locator('#btnStop').click();expect(page.locator('#sessionStatus')).to_have_text('STOPPED')
        assert not errors
        page.screenshot(path=str(ROOT/'.venv/on-demand-camera1-browser.png'),full_page=True)
        browser.close()


@pytest.mark.parametrize('browser_server',['FailingDetector'],indirect=True)
def test_browser_reports_failed_camera1_analysis_truthfully(browser_server):
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1440,'height':1100})
        errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(browser_server,wait_until='domcontentloaded')
        page.wait_for_selector('#systemFeed:not([hidden])')
        options=page.locator('#acquisitionSource option').evaluate_all('(items)=>items.map(i=>({value:i.value,text:i.textContent}))')
        page.locator('#acquisitionSource').select_option(next(item['value'] for item in options if 'signal.mp4' in item['text']))
        page.wait_for_selector('#acquisitionFeed:not([hidden])')
        calibrate(page,'system');calibrate(page,'acquisition')
        page.locator('.settings summary').click();page.locator('#phaseDebounce').fill('1');page.locator('#startPaused').select_option('yes')
        page.locator('#btnStart').click()
        expect(page.locator('#sessionStatus')).to_have_text('PAUSED',timeout=10000)
        page.locator('#predictNextGreen').click()
        expect(page.locator('#predictionStatus')).to_contain_text('frame analysis failed: camera 1 inference failed in test')
        expect(page.locator('#predictionStatus')).to_contain_text('· failed')
        expect(page.locator('#recommendedGreen')).to_have_text('—')
        expect(page.locator('#camera1AnalysisLabel')).to_contain_text('Camera 1 not analysed yet')
        expect(page.locator('#predictNextGreen')).to_be_enabled()  # a NEW click may analyse a new frame
        page.locator('#btnStop').click();expect(page.locator('#sessionStatus')).to_have_text('STOPPED')
        assert not errors
        browser.close()


def test_mobile_layout_tabs_and_calibration_cancel(browser_server):
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':390,'height':844})
        page.goto(browser_server,wait_until='domcontentloaded');page.wait_for_selector('#systemFeed:not([hidden])')
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        a=page.locator('[data-camera=system]').bounding_box();b=page.locator('[data-camera=acquisition]').bounding_box()
        assert b['y']>=a['y']+a['height']
        page.locator('[data-camera=system] [data-tab=upload]').click()
        expect(page.locator('#systemUpload')).to_be_visible()
        expect(page.locator('[data-camera=acquisition] [data-source-tab=samples]')).to_be_visible()
        page.locator('[data-camera=system] [data-tab=samples]').click()
        page.locator('#systemCalibrate').click();expect(page.locator('#toolLamp')).to_be_hidden()
        page.locator('#toolAnchors').click();click_point(page,20,50)
        page.keyboard.press('Escape');expect(page.locator('#calibrationModal')).not_to_have_class('modal-overlay active')
        expect(page.locator('#systemReadiness')).to_have_text('Calibration required')
        page.screenshot(path=str(ROOT/'.venv/phase-e-resume-mobile.png'),full_page=True)
        browser.close()
