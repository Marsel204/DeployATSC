from playwright.sync_api import sync_playwright, expect

from test_observation_dashboard_browser import browser_server, calibrate


def test_browser_persistent_settings_and_time_decomposition(browser_server):
    with sync_playwright() as playwright:
        browser=playwright.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1440,'height':1200})
        errors=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(browser_server,wait_until='domcontentloaded')
        page.wait_for_selector('#systemFeed:not([hidden])')
        options=page.locator('#acquisitionSource option').evaluate_all('(items)=>items.map(i=>({value:i.value,text:i.textContent}))')
        signal=next(item['value'] for item in options if 'signal.mp4' in item['text'])
        page.locator('#acquisitionSource').select_option(signal)
        page.wait_for_selector('#acquisitionFeed:not([hidden])')
        calibrate(page,'system');calibrate(page,'acquisition')
        page.locator('.settings summary').click()
        page.locator('#vehicleIdentityMode').select_option('persistent_v1')
        page.locator('#startPaused').select_option('yes')
        page.locator('#phaseDebounce').fill('1')
        page.locator('#btnStart').click()
        expect(page.locator('#sessionStatus')).to_have_text('PAUSED',timeout=10000)
        data=page.request.get(browser_server+'/api/observation/telemetry').json()
        assert data['vehicle_identity']['mode']=='persistent_v1'
        row={'vehicle_id':'run:acquisition:0:1','local_id':1,'tracker_epoch':0,'identity_generation':0,
            'identity_scheme':'persistent_v1','class_name':'car','red_wait_sec':9.,
            'estimated_red_wait_sec':3.,'pending_estimated_red_wait_sec':0.,'display_red_wait_sec':12.,
            'total_stopped_sec':9.,'tracking_state':'STOPPED','cycles':[1],
            'crossed_stop_line':False,'quality_flags':[],'left_censored':False,'right_censored':True,'completed':False}
        data['vehicles']=[row];data['vehicle_total']=1
        page.route('**/api/observation/telemetry?*',lambda route:route.fulfill(json=data))
        expect(page.locator('#vehicleLedger tbody')).to_contain_text('V1')
        expect(page.locator('#vehicleLedger tbody')).to_contain_text('9.0 confirmed + 3.0 estimated = 12.0 s')
        expect(page.locator('#persistentIdentityStatus')).to_be_visible()
        assert not errors
        page.unroute('**/api/observation/telemetry?*')
        page.request.post(browser_server+'/api/observation/control',data={'run_id':data['run_id'],'action':'stop'})
        browser.close()


def test_browser_can_configure_a_signal_and_enable_automatic_ab_mode(browser_server):
    with sync_playwright() as playwright:
        browser=playwright.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1440,'height':1200})
        errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(browser_server,wait_until='domcontentloaded')
        page.wait_for_selector('#systemFeed:not([hidden])')
        options=page.locator('#acquisitionSource option').evaluate_all('(items)=>items.map(i=>({value:i.value,text:i.textContent}))')
        signal=next(item['value'] for item in options if 'signal.mp4' in item['text'])
        page.locator('#acquisitionSource').select_option(signal)
        page.wait_for_selector('#acquisitionFeed:not([hidden])')
        calibrate(page,'system');calibrate(page,'acquisition')
        page.locator('.settings summary').click()
        page.locator('#vehicleIdentityMode').select_option('persistent_v1')
        page.locator('#scheduleEnabled').check();page.locator('#unchangedClearance').check()
        page.locator('#startPaused').select_option('yes')
        page.locator('#systemCalibrate').click()
        expect(page.locator('#toolLamp')).to_be_visible()
        page.locator('#toolLamp').click()
        box=page.locator('#calibrationCanvas').bounding_box()
        page.mouse.move(box['x']+10/320*box['width'],box['y']+10/240*box['height']);page.mouse.down()
        page.mouse.move(box['x']+40/320*box['width'],box['y']+40/240*box['height']);page.mouse.up()
        page.locator('#saveCalibration').click()
        expect(page.locator('#calibrationModal')).not_to_have_class('modal-overlay active')
        page.locator('#btnStart').click()
        expect(page.locator('#sessionStatus')).to_have_text('PAUSED',timeout=10000)
        expect(page.locator('#scheduleSavingsPanel')).to_be_visible()
        expect(page.locator('#predictNextGreen')).to_have_text('Automatic A-green capture')
        expect(page.locator('#predictNextGreen')).to_be_disabled()
        data=page.request.get(browser_server+'/api/observation/telemetry').json()
        assert data['schedule_comparison']['enabled']
        assert data['a_phase']['measurement_phase']=='UNKNOWN'
        assert not errors
        page.request.post(browser_server+'/api/observation/control',data={'run_id':data['run_id'],'action':'stop'})
        browser.close()
