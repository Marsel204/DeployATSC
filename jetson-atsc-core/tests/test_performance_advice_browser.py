"""Advice uses textContent and follows additive telemetry without layout changes."""
import json
from playwright.sync_api import sync_playwright,expect
from test_observation_dashboard_browser import browser_server,calibrate


def test_browser_performance_advice_near_runtime(browser_server):
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1440,'height':1100})
        advice=['CPU clocks are not pinned <test>: run `sudo jetson_clocks`.']
        def inject(route):
            response=route.fetch();data=response.json();data['performance_advice']=advice[:]
            route.fulfill(response=response,body=json.dumps(data))
        page.route('**/api/observation/telemetry*',inject)
        page.goto(browser_server,wait_until='domcontentloaded')
        page.wait_for_selector('#systemFeed:not([hidden])')
        options=page.locator('#acquisitionSource option').evaluate_all('(items)=>items.map(i=>({value:i.value,text:i.textContent}))')
        page.locator('#acquisitionSource').select_option(next(i['value'] for i in options if 'signal.mp4' in i['text']))
        page.wait_for_selector('#acquisitionFeed:not([hidden])')
        calibrate(page,'system');calibrate(page,'acquisition')
        page.locator('.settings summary').click();page.locator('#startPaused').select_option('yes');page.locator('#btnStart').click()
        expect(page.locator('#sessionStatus')).to_have_text('PAUSED',timeout=10000)
        expect(page.locator('#performanceAdvice')).to_have_text(advice[0]);expect(page.locator('#performanceAdvice')).to_be_visible()
        assert page.locator('#performanceAdvice test').count()==0
        assert page.locator('#performanceAdvice').bounding_box()['y']>page.locator('#runtimeStatus').bounding_box()['y']
        advice.clear();expect(page.locator('#performanceAdvice')).to_be_hidden()
        page.locator('#btnStop').click();expect(page.locator('#sessionStatus')).to_have_text('STOPPED')
        browser.close()
