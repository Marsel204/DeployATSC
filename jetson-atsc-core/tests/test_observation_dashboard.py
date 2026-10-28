from pathlib import Path
from html.parser import HTMLParser

ROOT=Path(__file__).resolve().parents[1]


def test_dashboard_has_distinct_roles_and_no_duplicate_ids():
    html=(ROOT/"src/ui/static/index.html").read_text()
    class Parser(HTMLParser):
        def __init__(self):super().__init__();self.ids=[];self.roles=[]
        def handle_starttag(self,tag,attrs):
            attrs=dict(attrs)
            if "id" in attrs:self.ids.append(attrs["id"])
            if "data-camera" in attrs:self.roles.append(attrs["data-camera"])
    parser=Parser();parser.feed(html)
    assert set(parser.roles)=={"system","acquisition"}
    assert len(parser.ids)==len(set(parser.ids))
    for element in ["predictNextGreen","systemFeed","acquisitionFeed","vehicleLedger","cycleComparison","calibrationCanvas"]:
        assert element in parser.ids
    assert "Observation only" in html
    assert "Control delay" in html


def test_early_policy_quality_and_partial_labels_without_browser_server():
    """Execute the shipped renderer with a minimal DOM; no server/model/browser.

    The full browser workflows remain separate release checks.
    """
    import json,subprocess,threading
    from src.common.types import ActuationDecision
    from src.experiment.contracts import RunPolicy
    from src.experiment.engine import ObservationEngine
    from test_observation_engine import feed, synthetic_scale_evidence
    engine=ObservationEngine('labels',RunPolicy(debounce_frames=1),
        {'xmin':0,'ymin':0,'xmax':20,'ymax':20},lambda p:-1,lambda s,c:ActuationDecision(c,35),scale_evidence=synthetic_scale_evidence())
    try:
        feed(engine,0,1,'RED');engine.predict('click');engine.predictions.await_idle()
        feed(engine,.1,2,'GREEN');feed(engine,.2,3,'YELLOW')
        data=engine.snapshot()
    finally:engine.close()
    data.update(state='paused',source_mode='replay',snapshot_fresh=True,
        frame_times={'system':.2,'acquisition':.2},shared_source_test=False,
        cameras={r:{'state':'ready','sequence':3,'total_frames':10,'image_size':[20,20]} for r in ('system','acquisition')},
        runtime={r:{'mean_inference_ms':0} for r in ('system','acquisition')})
    release=threading.Event();started=threading.Event()
    def delayed_evaluate(snapshot,cycle):
        started.set()
        assert release.wait(5)
        return ActuationDecision(cycle,35)
    history_engine=ObservationEngine('labels',RunPolicy(debounce_frames=1),
        {'xmin':0,'ymin':0,'xmax':20,'ymax':20},lambda p:-1,delayed_evaluate,scale_evidence=synthetic_scale_evidence())
    try:
        feed(history_engine,0,1,'RED');history_engine.predict('late')
        assert started.wait(1)
        feed(history_engine,.1,2,'GREEN');feed(history_engine,.2,3,'YELLOW')
        release.set();history_engine.predictions.await_idle()
        feed(history_engine,.3,4,'RED');history_engine.predict('newer')
        history_engine.predictions.await_idle()
        feed(history_engine,.4,5,'GREEN');feed(history_engine,.5,6,'YELLOW')
        feed(history_engine,.6,7,'RED');feed(history_engine,.7,8,'GREEN')
        feed(history_engine,.8,9,'UNKNOWN')
        historical_data=history_engine.snapshot()
    finally:
        release.set();history_engine.predictions.await_idle();history_engine.close()
    historical_data.update({key:data[key] for key in (
        'state','source_mode','snapshot_fresh','frame_times','shared_source_test','cameras','runtime')})
    script=(ROOT/'src/ui/static/observation.js').read_text().replace(
        "document.addEventListener('DOMContentLoaded',initialize);",
        "window.renderTelemetry=renderTelemetry;boundRunId='labels';")
    harness=r'''
const vm=require('vm'),assert=require('assert');
class Element {
 constructor(){this.textContent='';this.children=[];this.value='replay';this.className='';this.style={};}
 append(...items){this.children.push(...items);}
 replaceChildren(...items){this.children=items;this.textContent='';}
 querySelector(){return this.body ||= new Element();}
}
const elements=new Map();
const get=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
const document={getElementById:get,querySelectorAll:()=>[],createElement:()=>new Element(),
 createTextNode:text=>text};
const context={document,window:{}};vm.createContext(context);
vm.runInContext(SCRIPT,context);
const text=e=>e.textContent+e.children.map(c=>typeof c==='string'?c:text(c)).join(' ');
const render=data=>context.window.renderTelemetry(data);
render(DATA);
assert.match(get('qualificationPolicy').textContent,/motion-phase-scale-v2/);
assert.match(get('qualificationPolicy').textContent,/scale verified/i);
assert.match(get('queueQuality').textContent,/measured.zero/i);
assert.match(text(get('cycleComparison').body),/Qualified.*motion-phase-scale-v2/);
assert.match(text(get('cycleComparison').body),/0 samples.*0 partial/);
assert.match(get('validCycles').textContent,/1 timing.eligible.*1 qualified/);
assert.match(get('waitCoverage').textContent,/0 samples.*0 partial.*0.*unattributed/);
const diagnostic=structuredClone(DATA);
diagnostic.comparisons[0].qualified=false;
diagnostic.comparisons[0].qualification={policy:'motion-phase-v1',state:'diagnostic',reasons:['Q_measurement_partial']};
diagnostic.metrics.quality.Q={state:'partial',complete:false,reasons:['motion_unknown']};
render(diagnostic);
assert.match(text(get('cycleComparison').body),/Timing.eligible diagnostic/);
assert.match(text(get('cycleComparison').body),/Q_measurement_partial/);
assert.match(get('queueQuality').textContent,/partial.*motion_unknown/);
const legacy=structuredClone(DATA);delete legacy.qualification_policy;
delete legacy.metrics.quality;delete legacy.summary.observed_wait_sample_count;
legacy.comparisons[0].qualified=false;delete legacy.comparisons[0].qualification;
render(legacy);
assert.match(get('qualificationPolicy').textContent,/unknown/);
assert.match(get('queueQuality').textContent,/unknown/);
assert.match(text(get('cycleComparison').body),/qualification unknown/);
assert.match(get('waitCoverage').textContent,/unknown samples/);
render(HISTORICAL_PAYLOAD);
assert.match(get('recommendationCaption').textContent,/matched/);
const rows=get('cycleComparison').body.children;
const rowFor=id=>rows.find(row=>String(row.children[0].textContent)===String(id));
assert.match(text(rowFor(1)),/late/i);
assert.match(text(rowFor(1)),/motion-phase-scale-v2/);
assert.match(text(rowFor(3)),/No recommendation/);
assert.match(text(rowFor(3)),/ambiguous.or.dark/);
assert.match(text(rowFor(3)),/qualification unknown/);
const durability=structuredClone(DATA);
durability.lifecycle={durable_revision:7,durable_generation:3,durable_finalized:false,storage_suspended:true};
durability.history={comparisons:{total:82,retained:2},predictions:{total:90,retained:2}};
durability.summary.quantiles={state:'unavailable'};
render(durability);
assert.match(get('durabilityStatus').textContent,/7.*3/);
assert.match(get('durabilityStatus').textContent,/Snapshot revision/);
assert.match(get('durabilityStatus').textContent,/suspended/i);
assert.match(get('historyStatus').textContent,/82.*90/);
assert.match(get('distributionWait').textContent,/unavailable/i);
'''.replace('SCRIPT',json.dumps(script)).replace('DATA',json.dumps(data)).replace('HISTORICAL_PAYLOAD',json.dumps(historical_data))
    result=subprocess.run(['node','-'],input=harness,text=True,capture_output=True,timeout=10)
    assert result.returncode==0,result.stdout+result.stderr
