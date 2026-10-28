"""Execute the actual calibration save handler; full browser gate stays separate."""
import json
from pathlib import Path
import subprocess


def test_calibration_handler_submits_explicit_evidence_only():
    root=Path(__file__).resolve().parents[1]/'src/ui/static'
    script=(root/'observation.js').read_text()
    handler=script[script.index('  async function saveCalibration()'):script.index('  async function loadProfile(')]
    harness=r'''
const assert=require('assert');
const elements=new Map();
const $=id=>{if(!elements.has(id))elements.set(id,{value:'',checked:false,textContent:''});return elements.get(id);};
const copy=structuredClone;
let draft,modalRole='system',sent;
const camera={system:{}};
const closeCalibration=()=>{},loadProfiles=async()=>{},updateControls=()=>{},show=()=>{};
const json=async(url,payload)=>{sent=payload;return {profile_id:'new',calibration:payload};};
HANDLER
(async()=>{
for(const method of ['', 'measured_dimensions','ruler']){
 draft={role:'system',source:'clip',anchors:[[1,1],[9,1],[9,9],[1,9]],polygon:[[1,1],[9,1],[9,9]],rulerPoints:method==='ruler'?[[1,5],[9,5]]:[]};
 $('calibrationName').value='fixture';$('groundWidth').value='8';$('groundLength').value='40';$('rulerLength').value='6';
 $('scaleMethod').value=method;$('scaleReference').value='explicit synthetic survey';$('bindingReviewed').checked=Boolean(method);
 await saveCalibration();
 if(method){assert.equal(sent.scale_evidence.method,method);assert.equal(sent.scale_evidence.reference,'explicit synthetic survey');assert.equal(sent.binding.reviewed,true);}
 else {assert.ok(!sent.scale_evidence);assert.ok(!sent.binding?.reviewed);}
 if(method==='measured_dimensions')assert.ok(!sent.ruler);
 if(method==='ruler')assert.equal(sent.ruler.length_m,6);
}
})().catch(e=>{console.error(e);process.exitCode=1;});
'''.replace('HANDLER',handler)
    result=subprocess.run(['node','-'],input=harness,text=True,capture_output=True,timeout=10)
    assert result.returncode==0,result.stderr
    html=(root/'index.html').read_text()
    for control in ('scaleMethod','scaleReference','bindingReviewed'):
        assert f'id="{control}"' in html


def test_preview_handler_registers_source_and_uses_post_body():
    root=Path(__file__).resolve().parents[1]/'src/ui/static'
    script=(root/'observation.js').read_text()
    handler=script[script.index('  async function loadPreview('):script.index('  async function upload(')]
    harness=r'''
const assert=require('assert');
const elements=new Map();
const $=id=>{if(!elements.has(id))elements.set(id,{value:'',textContent:''});return elements.get(id);};
let input='rtsp://camera:private-pass@camera.local/feed?token=private-token';
const mode=()=> 'live',source=role=>input;
const camera={system:{}};
const updateControls=()=>{},setSource=()=>{throw new Error('unexpected replay');};
const json=async(url,body)=>{assert.equal(url,'/api/observation/sources');assert.equal(body.source,input);return {source:'source_public_handle'};};
const request=async(url,options)=>{assert.equal(url,'/api/observation/snapshot');assert.equal(options.method,'POST');assert.equal(JSON.parse(options.body).source,'source_public_handle');return {blob:async()=>({})};};
const URL={createObjectURL:()=> 'blob:safe',revokeObjectURL:()=>{}};
class Image {naturalWidth=320;naturalHeight=240;set src(value){this.onload();}}
HANDLER
loadPreview('system').then(()=>assert.equal($('systemLiveSource').value,'source_public_handle')).catch(e=>{console.error(e);process.exitCode=1;});
'''.replace('HANDLER',handler)
    result=subprocess.run(['node','-'],input=harness,text=True,capture_output=True,timeout=10)
    assert result.returncode==0,result.stderr
