/* Two-camera observation workspace. Every calibration draft belongs to one role. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const roles = ['system', 'acquisition'];
  const camera = Object.fromEntries(roles.map(role => [role, {source:'', calibration:null, image:null, url:null}]));
  let profiles = [], runId = null, boundRunId = null, telemetry = null, polling = false, busy = false;
  let durableVehicles = null, exactQuantiles = null;
  let currentHistoryRequest = 0, currentQuantileRequest = 0;
  let currentQuantileStatus = 'No durable quantiles requested.';
  // Saved evidence has its own identity and request fence; never rebind live telemetry.
  let savedRuns = [], savedView = null, savedRequest = 0, quantileRequest = 0;
  let vehicleOffset = 0, captureRequestId = null, modalRole = null, draft = null, tool = 'polygon', drag = null, previousFocus = null;
  const copy = value => JSON.parse(JSON.stringify(value));
  const fmt = (value, digits=1) => typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';
  const active = () => telemetry && ['running','paused'].includes(telemetry.state);
  const mode = () => $('sourceMode').value;
  const source = role => mode() === 'live' ? $(role+'LiveSource').value.trim() : camera[role].source;
  function show(message, kind='') { $('message').textContent=message; $('message').className='notice '+kind; $('message').hidden=false; }
  function detail(value) { return Array.isArray(value) ? value.map(item => `${(item.loc||[]).join('.')}: ${item.msg}`).join('; ') : String(value); }
  async function request(url, options={}) {
    const response=await fetch(url,options);
    if (!response.ok) {
      let error;try {error=await response.json();} catch (_) {error={detail:response.statusText};}
      const failure=new Error(detail(error.detail || error));failure.status=response.status;throw failure;
    }
    return response;
  }
  async function json(url, body) {
    const options=body===undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)};
    return (await request(url,options)).json();
  }
  function switchSourceTab(tab, btn) {
    const card=btn.closest('[data-camera]');
    card.querySelectorAll('.tab-btn').forEach(b=>b.classList.toggle('active',b.dataset.tab===tab));
    card.querySelectorAll('[data-source-tab]').forEach(c=>c.hidden=c.dataset.sourceTab!==tab);
  }
  function setSource(role, value) {
    const state=camera[role];
    if (state.source !== value) { state.calibration=null;state.image=null;
      $(role+'Feed').hidden=true;$(role+'Placeholder').hidden=false;$(role+'ViewTag').textContent='Preview required'; }
    state.source=value;
    const select=$(role+'Source');
    if (value && ![...select.options].some(o=>o.value===value)) select.add(new Option(value.split('/').pop(),value));
    select.value=value;
    $(role+'SourceStatus').textContent=value ? value.split('/').pop() : 'Choose a source.';
    if (role==='system' && $('sharedTest').checked) setSource('acquisition',value);
    updateControls();
  }
  async function loadVideos() {
    const data=await json('/api/sample_videos');
    for (const role of roles) {
      const select=$(role+'Source'), selected=camera[role].source;
      select.replaceChildren(new Option('Choose video…',''));
      data.videos.forEach(v=>select.add(new Option(`${v.name} · ${v.size_mb} MB`,v.path)));
      if (selected) setSource(role,selected);
    }
    if (!camera.system.source && data.videos.length) setSource('system',data.videos[0].path);
  }
  async function loadProfiles() {
    profiles=(await json('/api/observation/profiles')).profiles;
    for (const role of roles) {
      const select=$(role+'Profile'), previous=select.value;
      select.replaceChildren(new Option('Choose profile…',''));
      profiles.filter(p=>p.calibration.role===role).forEach(p=>select.add(new Option(p.calibration.name || p.profile_id,p.profile_id)));
      select.value=previous;
    }
  }
  async function loadPreview(role) {
    if (!source(role)) throw new Error(`Choose the ${role} source first.`);
    const registered=await json('/api/observation/sources',{source:source(role)});
    if(mode()==='live')$(role+'LiveSource').value=registered.source;
    else if(registered.source!==source(role))setSource(role,registered.source);
    const blob=await (await request('/api/observation/snapshot',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({role,source:registered.source,source_mode:mode(),rectification_path:$(role+'Lens').value || null})})).blob();
    const url=URL.createObjectURL(blob), image=new Image();
    try { await new Promise((resolve,reject)=>{image.onload=resolve;image.onerror=()=>reject(new Error('Preview image could not be read'));image.src=url;}); }
    catch(error){URL.revokeObjectURL(url);throw error;}
    const state=camera[role];if(state.url)URL.revokeObjectURL(state.url);
    state.url=url;state.image=image;
    const feed=$(role+'Feed');feed.src=url;feed.hidden=false;$(role+'Placeholder').hidden=true;
    $(role+'ViewTag').textContent=`Preview · ${image.naturalWidth}×${image.naturalHeight}`;
    updateControls();return image;
  }
  async function upload(role) {
    const file=$(role+'Upload').files[0];if(!file)throw new Error('Choose a video file first.');
    show(`Uploading ${role} video…`);
    const form=new FormData();form.append('file',file);
    const data=await (await request('/api/upload_video',{method:'POST',body:form})).json();
    await loadVideos();setSource(role,data.video_path);await loadPreview(role);show('Video loaded. Calibrate this camera.','success');
  }
  async function download(role) {
    const url=$(role+'Youtube').value.trim();if(!url)throw new Error('Enter a YouTube URL.');
    show('Downloading clip…');
    const data=await json('/api/download_youtube',{url,duration_sec:Number($(role+'YoutubeDuration').value)});
    await loadVideos();setSource(role,data.video_path);await loadPreview(role);show('Clip loaded. Calibrate this camera.','success');
  }
  function calibrationValid(role) {
    const state=camera[role], value=state.calibration;
    return value && value.source===source(role) && (value.rectification_path||'')===$(role+'Lens').value;
  }
  function updateControls() {
    const running=active();
    $('btnStart').disabled=busy || running || !roles.every(calibrationValid);
    $('btnPause').disabled=busy || !running || telemetry.state==='paused';
    $('btnResume').disabled=busy || !running || telemetry.state==='running';
    $('btnStep').disabled=busy || !running || mode()!=='replay';
    $('btnStop').disabled=busy || !running;
    $('exportRun').disabled=!runId || busy;
    document.querySelectorAll('.settings input,.settings select').forEach(e=>e.disabled=running || busy);
    for (const role of roles) {
      $(role+'Sources').disabled=running || busy;
      for(const suffix of ['Lens','Profile','Preview','Calibrate']) $(role+suffix).disabled=running || busy || (['Preview','Calibrate'].includes(suffix) && !source(role));
      $(role+'Readiness').textContent=calibrationValid(role) ? 'Calibration saved' : 'Calibration required';
      $(role+'Readiness').style.color=calibrationValid(role) ? 'var(--green)' : 'var(--muted)';
    }
    $('legacyProfile').disabled=running || busy || !source('system');
    $('acquisitionSource').disabled=running || busy || $('sharedTest').checked;
    const phase=telemetry?.phase?.measurement_phase;
    const pending=(telemetry?.predictions||[]).some(p=>['predicting','awaiting_green'].includes(p.status));
    // Camera 1 runs YOLO only on this click: a fresh preview frame is enough, no prior metrics.
    const fresh=telemetry?.camera1?.frame_fresh===true;
    $('predictNextGreen').disabled=telemetry?.lifecycle?.storage_suspended || busy || !running || !fresh || pending || telemetry.prediction_busy || ['GREEN','YELLOW'].includes(phase);
  }
  function changeMode() {
    for(const role of roles){camera[role].calibration=null;camera[role].image=null;}
    for(const role of roles){$(role+'Feed').hidden=true;$(role+'Placeholder').hidden=false;$(role+'ViewTag').textContent='Preview required';}
    if(mode()==='live')$('sharedTest').checked=false;
    document.querySelectorAll('.tab-group').forEach(e=>e.hidden=mode()==='live');
    document.querySelectorAll('.live-source').forEach(e=>e.hidden=mode()!=='live');
    for(const role of roles) document.querySelectorAll(`[data-camera="${role}"] [data-source-tab]`).forEach(e=>e.hidden=mode()==='live' || e.dataset.sourceTab!=='samples');
    $('sharedLabel').hidden=mode()==='live';updateControls();
  }
  function policy() {
    const result={movement_id:$('movementId').value.trim(),offset_sec:Number($('sourceOffset').value),
      stopped_speed_mps:Number($('stoppedSpeed').value),moving_speed_mps:Number($('movingSpeed').value),
      stop_persistence_sec:Number($('stopPersistence').value),max_gap_sec:Number($('gapLimit').value),
      association_gap_sec:Number($('associationLimit').value),debounce_frames:Number($('phaseDebounce').value),phase_ratio:Number($('phaseRatio').value)};
    if(Object.values(result).some(value=>typeof value==='number' && !Number.isFinite(value)))throw new Error('Measurement settings must be finite numbers.');
    return result;
  }
  async function startRun() {
    busy=true;updateControls();show('Opening both camera sources and loading models…');
    try {
      const data=await json('/api/observation/start',{system:camera.system.calibration,acquisition:camera.acquisition.calibration,
        source_mode:mode(),step_mode:$('startPaused').value==='yes',shared_test:$('sharedTest').checked,policy:policy()});
      runId=data.run_id;vehicleOffset=0;captureRequestId=null;
      for(const role of roles){$(role+'Feed').src=`/stream/observation/${role}?run=${encodeURIComponent(runId)}`;$(role+'Feed').hidden=false;$(role+'Placeholder').hidden=true;}
      show('Observation started. Camera 2 measures continuously; Camera 1 runs YOLO only when you click Analyze frame & predict, shortly before this movement turns green.','success');
      await poll();
    } finally {busy=false;updateControls();}
  }
  async function control(action) {
    if(!runId)return;
    await json('/api/observation/control',{run_id:runId,action});await poll();
    if(action==='stop'){show('Run stopped. Final and partial observations are retained for export.','success');await loadRuns();}
  }
  function savedRunExport(){return '/api/observation/export?run_id='+encodeURIComponent($('savedRuns').value);}
  function renderRuns(runs) {
    savedRuns=runs;
    const previous=$('savedRuns').value;
    $('savedRuns').replaceChildren(...runs.map(run=>{const option=document.createElement('option');option.value=run.run_id;
      option.textContent=`${run.run_id} · ${run.status}${run.recovered_from ? ` (recovered from ${run.recovered_from})` : ''}${typeof run.durable_revision==='number' ? ` · revision ${run.durable_revision}` : ''}`;return option;}));
    $('savedRuns').value=runs.some(run=>run.run_id===previous) ? previous : (runs[0]?.run_id || '');
    $('exportSaved').disabled=!runs.length;
    $('browseSaved').disabled=!runs.length;
    const unavailable=runs.filter(run=>run.status==='unavailable').length;
    $('savedRunStatus').textContent=runs.length ? `${runs.length} saved run${runs.length===1 ? '' : 's'}${unavailable ? ` · ${unavailable} unavailable (blocks new runs until repaired)` : ''}` : 'No saved runs listed.';
  }
  async function loadRuns(){renderRuns((await json('/api/observation/runs')).runs || []);}
  function renderSaved() {
    const view=savedView, page=view?.page;
    $('viewedRunStatus').textContent=view ? `Viewing ${view.id} · ${view.status} · ${view.kind} · ${view.error || (page ? `durable generation ${page.generation ?? 'unknown (legacy)'} · revision ${page.snapshot_revision ?? 'unknown'}` : 'Loading durable evidence…')}` : 'No saved evidence selected.';
    $('savedPage').textContent=page ? `${page.total ? page.offset+1 : 0}–${Math.min(page.offset+page.rows.length,page.total)} of ${page.total} durable records` : 'No page loaded.';
    $('previousSaved').disabled=!page || page.offset===0;
    $('nextSaved').disabled=!page || page.next_offset===null;
    $('savedQuantiles').disabled=!view || !!view.error;
    const q=view?.quantiles;
    $('savedQuantileStatus').textContent=q ? `Exact durable median / P95 for ${view.id}: ${fmt(q.median_red_wait_sec)} / ${fmt(q.p95_red_wait_sec)} s · ${q.sample_count} samples · generation ${q.generation ?? 'unknown (legacy)'}${page && q.generation!==page.generation ? ' (different generation from page; reload to compare)' : ''}` : view?.quantileStatus || 'No exact quantiles requested for this saved run.';
    const columns=view?.kind==='predictions' ? ['Request ID','Cycle','ANFIS green','Status','Qualification'] : view?.kind==='comparisons' ? ['Cycle','Request ID','ANFIS green','Actual green','Difference','Qualification'] : ['Vehicle ID','Class','Red wait','Total stopped','State','Observation quality'];
    const header=document.createElement('tr');
    columns.forEach(title=>{const th=document.createElement('th');th.textContent=title;header.append(th);});
    $('savedEvidence').querySelector('thead').replaceChildren(header);
    populateTable('savedEvidence',page?.rows || [],columns.length,(row,record)=>{
      if(view.kind==='predictions'){
        cell(row,record.request_id);cell(row,record.cycle_id ?? 'unknown');cell(row,fmt(record.recommended_green_sec)+' s');cell(row,record.status);cell(row,qualificationLabel(record));
      }else if(view.kind==='comparisons'){
        cell(row,record.cycle_id);cell(row,record.prediction?.request_id || 'No recommendation');cell(row,fmt(record.recommended_green_sec)+' s');cell(row,fmt(record.actual_green_sec)+' s');cell(row,fmt(record.difference_sec)+' s');cell(row,cycleStatusLabel(record));
      }else{
        cell(row,record.vehicle_id);cell(row,record.class_name);cell(row,fmt(record.red_wait_sec)+' s');cell(row,fmt(record.total_stopped_sec)+' s');cell(row,record.tracking_state);
        const flags=[...(record.quality_flags || [])];if(record.left_censored)flags.push('partial arrival');if(record.right_censored)flags.push('partial departure');cell(row,flags.join(' · ') || 'Complete');
      }
    },view?.error ? 'Page cleared. Reload selected evidence to restart pagination.' : 'No records in this saved page.');
  }
  async function browseSaved(offset=0,continuation=false) {
    const id=$('savedRuns').value,kind=$('savedKind').value;if(!id)return;
    const previous=savedView, token=++savedRequest;
    const same=previous?.id===id;
    const query=new URLSearchParams({run_id:id,offset,limit:100});
    if(continuation && previous?.page?.generation!=null)query.set('generation',previous.page.generation);
    savedView={id,kind,status:savedRuns.find(run=>run.run_id===id)?.status || 'unknown',page:null,
      quantiles:same ? previous.quantiles : null};
    if(!same)++quantileRequest;
    renderSaved();
    try{
      const page=await json('/api/observation/history/'+kind+'?'+query);
      if(token!==savedRequest)return;
      savedView.page=page;renderSaved();
    }catch(error){
      if(token!==savedRequest)return;
      savedView.error=error.status===409 ? 'Durable generation changed; restart pagination with Reload selected evidence.' : error.message;
      savedView.quantiles=null;++quantileRequest;renderSaved();
    }
  }
  async function savedQuantiles() {
    if(!savedView)return;
    const id=savedView.id,token=++quantileRequest;
    savedView.quantileStatus='Pending · computing exact durable quantiles';renderSaved();
    try{
      const result=await json('/api/observation/quantiles?'+new URLSearchParams({run_id:id}));
      if(token!==quantileRequest || savedView?.id!==id)return;
      savedView.quantiles=result;renderSaved();
    }catch(error){
      if(token!==quantileRequest || savedView?.id!==id)return;
      savedView.quantileStatus='Exact quantiles unavailable: '+error.message;renderSaved();
    }
  }
  async function predict() {
    busy=true;updateControls();
    if(!captureRequestId)captureRequestId=globalThis.crypto?.randomUUID?.() || `${Date.now()}_${Math.random().toString(16).slice(2)}`;
    try {
      const record=await json('/api/observation/predict',{run_id:runId,request_id:captureRequestId});
      captureRequestId=null;show(record.measurement_phase_at_capture==='UNKNOWN' ? `Frame #${record.camera1_frame?.sequence ?? '?'} captured with measurement phase unknown; it cannot count as a confirmed-red comparison. Analysing one still image…` : `Frame #${record.camera1_frame?.sequence ?? '?'} captured at ${fmt(record.clicked_at,2)}s. Analysing one still image, then ANFIS. Single-frame queue is a spatial proxy (no motion evidence).`);await poll();
    } finally {busy=false;updateControls();}
  }
  function cell(row,value,title) {const element=document.createElement('td');element.textContent=value;if(title)element.title=title;row.append(element);}
  function qualificationLabel(record) {
    const q=record?.qualification;
    if(!q)return 'qualification unknown';
    const state=record.qualified ? `Qualified (${q.policy})` : record.eligible ? 'Timing-eligible diagnostic' : 'Diagnostic';
    return `${state} · ${q.policy || 'policy unknown'} · ${q.state==='unknown' ? 'qualification unknown · ' : ''}scale ${q.scale || 'unverified'}${q.reasons?.length ? ' · '+q.reasons.join(', ') : ''}`;
  }
  function cycleStatusLabel(record) {
    const status=record.prediction?.status;
    const prediction=status ? status.charAt(0).toUpperCase()+status.slice(1).replaceAll('_',' ') : 'No recommendation';
    const boundary=record.quality && record.quality!=='complete' ? ` · ${record.quality.replaceAll('_',' ')}` : '';
    return `${qualificationLabel(record)} · ${prediction}${boundary}`;
  }
  function populateTable(id,records,count,render,empty) {
    const body=$(id).querySelector('tbody');body.replaceChildren();
    if(!records.length){const row=document.createElement('tr'),td=document.createElement('td');td.colSpan=count;td.className='empty';td.textContent=empty;row.append(td);body.append(row);return;}
    records.forEach(record=>{const row=document.createElement('tr');render(row,record);body.append(row);});
  }
  function resetCurrentEvidence() {
    ++currentHistoryRequest;++currentQuantileRequest;
    durableVehicles=null;exactQuantiles=null;
    currentQuantileStatus='No durable quantiles requested.';
    $('quantileStatus').textContent=currentQuantileStatus;
  }
  function renderTelemetry(data) {
    telemetry=data;if(data.state==='idle'){if(runId || boundRunId)resetCurrentEvidence();runId=null;boundRunId=null;updateControls();return;}
    runId=data.run_id;$('runLabel').textContent=runId;
    if(boundRunId!==runId){
      resetCurrentEvidence();
      $('sourceMode').value=data.source_mode;changeMode();$('sharedTest').checked=false;
      for(const role of roles){
        const calibration=data.camera_calibrations[role];
        if(mode()==='live')$(role+'LiveSource').value=calibration.source;else setSource(role,calibration.source);
        const lens=$(role+'Lens'),path=calibration.rectification_path || '';
        if(path && ![...lens.options].some(option=>option.value===path))lens.add(new Option(path,path));
        lens.value=path;camera[role].calibration=copy(calibration);
        $(role+'Feed').src=`/stream/observation/${role}?run=${encodeURIComponent(runId)}`;
        $(role+'Feed').hidden=false;$(role+'Placeholder').hidden=true;
      }
      $('sharedTest').checked=data.shared_source_test;
      const settings={movementId:'movement_id',sourceOffset:'offset_sec',stoppedSpeed:'stopped_speed_mps',movingSpeed:'moving_speed_mps',stopPersistence:'stop_persistence_sec',gapLimit:'max_gap_sec',associationLimit:'association_gap_sec',phaseDebounce:'debounce_frames',phaseRatio:'phase_ratio'};
      for(const[id,key]of Object.entries(settings))$(id).value=data.policy[key];
      boundRunId=runId;
    }
    $('sessionStatus').textContent=data.state.toUpperCase();
    $('sessionStatus').className='badge '+(data.state==='running'?'green':data.state==='error'?'red':'');
    $('metricVw').textContent=fmt(data.metrics?.v_w_pcu,2);$('metricQ').textContent=fmt(data.metrics?.queue_meters);$('metricL').textContent=fmt(data.metrics?.occupancy_pct);
    const queue=data.metrics?.quality?.Q,analysed=data.camera1?.last_analysis;
    $('queueQuality').textContent=data.metrics ? `Q quality: ${queue?.state || 'unknown'}${queue?.reasons?.length ? ' · '+queue.reasons.join(', ') : ''} · scale ${data.scale_trust || 'unverified'}` : 'Q unavailable until a Camera 1 frame is analysed';
    $('camera1AnalysisLabel').textContent=analysed ? `Last analysed frame #${analysed.frame_sequence} (${analysed.request_id}) · ${fmt(analysed.age_sec)} s old · not live` : 'Camera 1 not analysed yet · YOLO runs only on click';
    $('qualificationPolicy').textContent=`${data.qualification_policy || 'Qualification policy unknown'} · scale ${data.scale_trust || 'unverified'}. Controlled engineering evaluation; field accuracy remains unvalidated.`;
    const lifecycle=data.lifecycle || {};
    $('durabilityStatus').textContent=`Snapshot revision ${data.snapshot_revision ?? 'unknown'} · durable revision ${lifecycle.durable_revision ?? 'unknown'} · generation ${lifecycle.durable_generation ?? 'unknown'} · ${lifecycle.durable_finalized ? 'durably finalized' : lifecycle.terminal_state || 'open'}${lifecycle.storage_suspended ? ' · measurement suspended' : ''}`;
    $('retryStorage').hidden=!data.storage_error;
    $('historyStatus').textContent=`Live windows: ${data.comparisons.length} of ${data.history?.comparisons?.total ?? data.comparisons.length} cycles · ${data.predictions.length} of ${data.history?.predictions?.total ?? data.predictions.length} predictions. Full durable evidence is available through history and export.`;
    const last=data.predictions.at(-1);
    $('recommendedGreen').textContent=fmt(last?.recommended_green_sec);
    $('recommendationCaption').textContent=last ? `${last.status.replaceAll('_',' ')} · ${qualificationLabel(last)}` : 'Awaiting manual capture';
    $('predictionStatus').replaceChildren();
    const analysis=last?.camera_analysis,analysisText=!analysis ? '' : analysis.state==='pending' ? ' · analysing frame' : analysis.state==='complete' ? ` · frame #${last.camera1_frame?.sequence} analysed (${analysis.detections} vehicles)` : ` · frame analysis ${analysis.state.replaceAll('_',' ')}${analysis.error ? ': '+analysis.error : ''}`;
    if(last){$('predictionStatus').append(document.createTextNode(`Capture at ${fmt(last.clicked_at,2)}s${analysisText} · ${last.status.replaceAll('_',' ')}${last.error && !analysis?.error ? ': '+last.error : ''}${last.completed_lead_sec!==null ? ` · total completion lead ${fmt(last.completed_lead_sec,2)}s` : ''}`));
      if(['predicting','awaiting_green'].includes(last.status)){const button=document.createElement('button');button.textContent='Cancel pending';button.onclick=()=>safe(async()=>{await json(`/api/observation/predictions/${encodeURIComponent(last.request_id)}/cancel?run_id=${encodeURIComponent(runId)}`,{});await poll();});$('predictionStatus').append(' ',button);}}
    else $('predictionStatus').textContent='Click shortly before green. Camera 1 runs YOLO once on the current frame, then ANFIS.';
    $('predictionReady').textContent=!$('predictNextGreen').disabled ? 'Ready: one frame per click' : data.camera1?.frame_problem && active() ? data.camera1.frame_problem : 'See phase and capture state below';
    const phase=data.phase?.phase || 'UNKNOWN';$('phaseBadge').textContent='Phase '+phase.toLowerCase();$('phaseBadge').className='badge '+(phase==='RED'?'red':phase==='GREEN'?'green':'signal');
    if(data.phase?.measurement_phase!==phase)$('phaseBadge').textContent=`Display ${phase.toLowerCase()} · measurement ${(data.phase?.measurement_phase || 'UNKNOWN').toLowerCase()}`;
    const lastCycle=data.comparisons.at(-1);
    $('actualGreen').textContent=data.phase.green ? `Running · ${fmt(data.time-data.phase.green.green_start)} s` : lastCycle?.actual_green_sec!==null && lastCycle?.actual_green_sec!==undefined ? `${fmt(lastCycle.actual_green_sec)} s` : '—';
    $('vehicleCount').textContent=data.vehicle_total;$('sourceTimes').textContent=`C1 ${fmt(data.frame_times?.system,2)}s / C2 ${fmt(data.frame_times?.acquisition,2)}s`;
    for(const role of roles){const health=data.cameras[role];$(role+'Health').textContent=health.state+(health.processed_frame_age_sec!==undefined ? ` · ${fmt(health.processed_frame_age_sec,2)}s processing age` : health.age_sec!==undefined ? ` · ${fmt(health.age_sec,2)}s age` : ` · ${health.sequence}/${health.total_frames} frames`);
      $(role+'ViewTag').textContent=`${data.shared_source_test?'Shared test · ':''}${data.state} · ${health.image_size.join('×')}`;}
    populateTable('cycleComparison',data.comparisons.slice(-50).reverse(),8,(row,c)=>{
      cell(row,c.cycle_id);cell(row,fmt(c.prediction?.completed_lead_sec,2)+' s');cell(row,fmt(c.recommended_green_sec)+' s');cell(row,c.running_green_sec!==undefined ? 'Running' : fmt(c.actual_green_sec)+' s');cell(row,fmt(c.difference_sec)+' s');cell(row,`${fmt(c.observed_mean_red_wait_sec)} s · ${c.observed_vehicle_count ?? 'unknown'} samples · ${c.partial_vehicle_count ?? 'unknown'} partial`);cell(row,c.n_served);cell(row,cycleStatusLabel(c));
    },'No observed green interval yet.');
    populateTable('vehicleLedger',durableVehicles?.rows || data.vehicles,8,(row,v)=>{
      cell(row,`#${v.local_id} · epoch ${v.tracker_epoch}`,v.vehicle_id);cell(row,v.class_name);cell(row,fmt(v.red_wait_sec)+' s');cell(row,fmt(v.total_stopped_sec)+' s');cell(row,v.tracking_state);cell(row,v.cycles.join(', ')+(v.history?.archived_cycles ? ' · earlier cycles in durable history' : ''));cell(row,v.crossed_stop_line ? (v.crossing_cycle==null ? 'Crossed · cycle unknown' : 'Crossed') : '—');
      const quality=[...v.quality_flags];if(v.left_censored)quality.push('partial arrival');if(v.right_censored)quality.push(v.completed?'partial departure':'in progress');cell(row,quality.length ? [...new Set(quality)].join(' · ') : 'Complete');
    },'No stable acquisition vehicle IDs yet.');
    $('vehiclePage').textContent=durableVehicles ? `${durableVehicles.total ? durableVehicles.offset+1 : 0}–${Math.min(durableVehicles.offset+100,durableVehicles.total)} of ${durableVehicles.total} durable records · generation ${durableVehicles.generation}` : `Live window: ${data.vehicles.length} of ${data.vehicle_total} records`;
    $('previousVehicles').disabled=!durableVehicles || durableVehicles.offset===0;
    $('nextVehicles').disabled=!durableVehicles || durableVehicles.next_offset===null;
    const summary=data.summary;$('meanWait').textContent=fmt(summary.observed_mean_red_wait_sec)+' s';$('distributionWait').textContent=summary.quantiles?.state==='unavailable' ? 'Unavailable · request durable quantiles' : fmt(summary.observed_median_red_wait_sec)+' / '+fmt(summary.observed_p95_red_wait_sec)+' s';$('completeVehicles').textContent=summary.complete_vehicle_count;
    $('partialVehicles').textContent=`${summary.partial_vehicle_count} partial / active records`;$('validCycles').textContent=`${summary.valid_cycle_comparisons} timing-eligible · ${summary.qualified_cycle_comparisons ?? 'unknown'} qualified`;$('losReason').textContent=data.los.reason;
    if(exactQuantiles)$('quantileStatus').textContent=`Exact durable median / P95: ${fmt(exactQuantiles.median_red_wait_sec)} / ${fmt(exactQuantiles.p95_red_wait_sec)} s · ${exactQuantiles.sample_count} samples · generation ${exactQuantiles.generation}${exactQuantiles.generation!==lifecycle.durable_generation ? ' (older generation)' : ''}`;
    else $('quantileStatus').textContent=currentQuantileStatus;
    $('waitCoverage').textContent=`${summary.observed_wait_sample_count ?? 'unknown'} samples · ${summary.partial_vehicle_count ?? 'unknown'} partial · ${summary.complete_wait_sample_count ?? 'unknown'} complete · ${summary.unattributed_crossing_count ?? 'unknown'} unattributed crossings. Unknown motion ${fmt(summary.motion_unknown_sec)} / phase ${fmt(summary.phase_unknown_sec)} vehicle-seconds (may overlap).`;
    const runtime=roles.map(role=>role==='system' ? `C1: ${data.runtime[role].frames} on-demand inference attempt(s), mean ${fmt(data.runtime[role].mean_inference_ms)}ms` : `C2: ${fmt(data.runtime[role].processed_fps)} FPS · source ${fmt(data.runtime[role].source_fps)} FPS · ${fmt(data.runtime[role].realtime_factor,2)}× playback · ${fmt(data.runtime[role].steady_inference_ms)}ms steady inference${data.detector_selection?.backend==='tensorrt' ? ' · TensorRT FP16' : ''}`).join(' · ');
    $('performanceAdvice').textContent=data.performance_advice?.[0] || '';
    $('performanceAdvice').hidden=!data.performance_advice?.length;
    $('runtimeStatus').textContent=runtime+' · timestamps follow source time; partial observations remain visible.';
    if(data.error)show(data.error,'error');else if(data.storage_error)show('Evidence storage error: '+data.storage_error,'error');
    else if(last?.error)show('Recommendation failed: '+last.error,'error');
    updateControls();
  }
  async function poll() {
    if(polling)return;polling=true;
    try{renderTelemetry(await json(`/api/observation/telemetry?limit=100`));}
    catch(error){show('Dashboard connection: '+error.message,'error');}
    finally{polling=false;}
  }
  const toolHelp={polygon:'Click at least 3 points around the road/waiting area. Clear or undo to edit.',anchors:'Click exactly 4 points: Far-Left, Far-Right, Near-Right, Near-Left.',ruler:'Click 2 endpoints on a known ground distance, then enter its real length.',lamp:'Drag a box around one signal head for the target movement. This belongs to camera 2.',stopline:'Click 2 points across the stop line. Flip the arrow to match vehicle travel.',lamp_RED:'Drag the red lamp region inside the signal-head box.',lamp_YELLOW:'Drag the yellow lamp region inside the signal-head box.',lamp_GREEN:'Drag the green lamp region inside the signal-head box.'};
  function setTool(value) {tool=value;document.querySelectorAll('[data-tool]').forEach(button=>button.classList.toggle('active',button.dataset.tool===tool));$('toolHelp').textContent=toolHelp[tool];}
  function canvasPoint(event) {const canvas=$('calibrationCanvas'),rect=canvas.getBoundingClientRect();return [Math.max(0,Math.min(canvas.width-1,(event.clientX-rect.left)*canvas.width/rect.width)),Math.max(0,Math.min(canvas.height-1,(event.clientY-rect.top)*canvas.height/rect.height))];}
  function pointKey() {return {polygon:'polygon',anchors:'anchors',ruler:'rulerPoints',stopline:'stop_line'}[tool];}
  function regionKey() {return tool==='lamp'?'signal_roi':tool.startsWith('lamp_')?tool.slice(5):null;}
  function setRegion(region) {if(tool==='lamp')draft.signal_roi=region;else draft.lamp_rois[regionKey()]=region;}
  function setupCalibrationCanvas() {
    const canvas=$('calibrationCanvas');
    if(canvas.dataset.initialized)return;canvas.dataset.initialized='yes';
    canvas.addEventListener('pointerdown',event=>{
      if(!draft)return;const point=canvasPoint(event);canvas.setPointerCapture(event.pointerId);
      if(regionKey()){drag=point;setRegion({xmin:point[0],ymin:point[1],xmax:point[0],ymax:point[1]});}
      else {const key=pointKey(),maximum=key==='anchors'?4:key==='polygon'?100:2;
        if(draft[key].length>=maximum){$('calibrationError').textContent='Clear this tool before drawing a replacement.';return;}
        draft[key].push(point);}
      draw();
    });
    canvas.addEventListener('pointermove',event=>{if(!drag)return;const point=canvasPoint(event);setRegion({xmin:Math.min(drag[0],point[0]),ymin:Math.min(drag[1],point[1]),xmax:Math.max(drag[0],point[0]),ymax:Math.max(drag[1],point[1])});draw();});
    canvas.addEventListener('pointerup',()=>{drag=null;draw();});
    canvas.addEventListener('pointercancel',()=>{drag=null;});
  }
  function draw() {
    if(!draft || !modalRole)return;
    const canvas=$('calibrationCanvas'),ctx=canvas.getContext('2d'),image=camera[modalRole].image;
    ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(image,0,0,canvas.width,canvas.height);
    const scale=Math.max(1,canvas.width/500);ctx.lineWidth=2*scale;ctx.font=`${11*scale}px sans-serif`;
    const line=(points,color,closed=false)=>{if(!points.length)return;ctx.strokeStyle=color;ctx.fillStyle=color;ctx.beginPath();points.forEach((point,index)=>index?ctx.lineTo(...point):ctx.moveTo(...point));if(closed && points.length>=3){ctx.closePath();ctx.fillStyle=color+'25';ctx.fill();}ctx.stroke();points.forEach((point,index)=>{ctx.fillStyle=color;ctx.beginPath();ctx.arc(...point,3*scale,0,Math.PI*2);ctx.fill();if(color==='#75d6d2')ctx.fillText(String(index+1),point[0]+7*scale,point[1]-5*scale);});};
    line(draft.polygon,'#75d6d2',true);line(draft.anchors,'#efc570',true);line(draft.rulerPoints,'#f49c6b');line(draft.stop_line,'#f1f4fa');
    if(draft.stop_line.length===2){const[a,b]=draft.stop_line,dx=b[0]-a[0],dy=b[1]-a[1],length=Math.hypot(dx,dy);if(length){const x=(a[0]+b[0])/2,y=(a[1]+b[1])/2,nx=-dy/length*draft.crossing_direction,ny=dx/length*draft.crossing_direction;line([[x-15*scale*nx,y-15*scale*ny],[x+15*scale*nx,y+15*scale*ny]],'#efc570');ctx.fillStyle='#efc570';ctx.fillText('COUNT →',x+18*scale*nx,y+18*scale*ny);}}
    const box=(roi,color,label)=>{if(!roi)return;ctx.strokeStyle=color;ctx.strokeRect(roi.xmin,roi.ymin,roi.xmax-roi.xmin,roi.ymax-roi.ymin);ctx.fillStyle=color;ctx.fillText(label,roi.xmin,Math.max(14*scale,roi.ymin-5*scale));};
    box(draft.signal_roi,'#efc570','SIGNAL HEAD');for(const phase of ['RED','YELLOW','GREEN'])box(draft.lamp_rois[phase],{RED:'#f29394',YELLOW:'#efc570',GREEN:'#89d8a8'}[phase],phase);
    const lines=[`${draft.anchors.length}/4 perspective anchors`,`${draft.polygon.length} road/waiting points`,`${draft.rulerPoints.length}/2 ruler points`];
    if(modalRole==='acquisition')lines.push(`${draft.signal_roi ? 'Signal head drawn' : 'Signal head required'}`,`${draft.stop_line.length}/2 stop-line points`);
    $('calibrationChecklist').replaceChildren(...lines.map(text=>{const line=document.createElement('div');line.textContent=text;return line;}));
  }
  async function openCalibration(role, initial=null) {
    previousFocus=document.activeElement;modalRole=role;await loadPreview(role);
    const image=camera[role].image,data=initial || camera[role].calibration || {};
    draft={role,source:source(role),image_size:[image.naturalWidth,image.naturalHeight],
      anchors:copy(data.anchors || []),polygon:copy(data.polygon || []),rulerPoints:data.ruler ? [copy(data.ruler.p1),copy(data.ruler.p2)] : [],
      signal_roi:copy(data.signal_roi || null),lamp_rois:copy(data.lamp_rois || {}),stop_line:copy(data.stop_line || []),crossing_direction:data.crossing_direction || 1,
      rectification_path:$(role+'Lens').value || null,lane_metrics:copy(data.lane_metrics || {})};
    $('calibrationName').value=data.name || (role==='system'?'jamika_system':'jamika_acquisition');$('groundWidth').value=data.road_width_m || 8;$('groundLength').value=data.road_length_m || 40;$('rulerLength').value=data.ruler?.length_m || 2;
    $('scaleMethod').value=data.scale_evidence?.method || '';$('scaleReference').value=data.scale_evidence?.reference || '';$('bindingReviewed').checked=false;
    draft.parent_revision=data.binding?.geometry_digest ? camera[role].revision : undefined;
    $('calibrationTitle').textContent=role==='system'?'Camera 1 · System calibration':'Camera 2 · Acquisition calibration';
    $('calibrationSubtitle').textContent=role==='system'?'Road perspective and metric scale for ANFIS inputs.':'Waiting area, metric scale, signal head and vehicle crossing direction.';
    for(const id of ['toolLamp','toolStopline','flipStopline','toolRed','toolYellow','toolGreen'])$(id).hidden=role!=='acquisition';
    $('calibrationCanvas').width=image.naturalWidth;$('calibrationCanvas').height=image.naturalHeight;
    $('calibrationError').textContent='';$('calibrationModal').classList.add('active');setTool(role==='system'?'polygon':'lamp');draw();$('closeCalibration').focus();
  }
  function closeCalibration(){ $('calibrationModal').classList.remove('active');draft=null;modalRole=null;drag=null;previousFocus?.focus(); }
  async function saveCalibration() {
    $('calibrationError').textContent='';
    if(draft.anchors.length!==4)throw new Error('Draw exactly four perspective anchors for this camera.');
    if(draft.polygon.length<3)throw new Error('Draw at least three road/waiting-area points.');
    if(modalRole==='acquisition' && (!draft.signal_roi || draft.stop_line.length!==2))throw new Error('Draw the acquisition signal head and two stop-line points.');
    const payload={...copy(draft),name:$('calibrationName').value.trim(),road_width_m:Number($('groundWidth').value),road_length_m:Number($('groundLength').value)};
    delete payload.rulerPoints;
    if(modalRole==='system') {delete payload.stop_line;delete payload.signal_roi;delete payload.lamp_rois;}
    else {
      const pixels=roi=>Object.fromEntries(Object.entries(roi).map(([key,value])=>[key,Math.round(value)]));
      payload.signal_roi=pixels(payload.signal_roi);
      payload.lamp_rois=Object.fromEntries(Object.entries(payload.lamp_rois).map(([phase,roi])=>[phase,pixels(roi)]));
    }
    const method=$('scaleMethod').value;
    if(method==='measured_dimensions')delete payload.ruler;
    else {
      if(draft.rulerPoints.length===1)throw new Error('Finish both ruler endpoints or clear the ruler.');
      if(draft.rulerPoints.length===2)payload.ruler={p1:draft.rulerPoints[0],p2:draft.rulerPoints[1],length_m:Number($('rulerLength').value)};
    }
    if(method){
      const reference=$('scaleReference').value.trim();
      if(!reference || !$('bindingReviewed').checked)throw new Error('Provide the measurement reference and confirm the source, lens and geometry review.');
      if(method==='ruler' && !payload.ruler)throw new Error('Draw two ruler endpoints and enter the measured distance.');
      payload.scale_evidence={method,reference};payload.binding={reviewed:true};
    }
    const result=await json('/api/observation/calibration',payload),role=modalRole;
    camera[role].calibration=result.calibration;camera[role].revision=result.revision;closeCalibration();await loadProfiles();$(role+'Profile').value=result.profile_id;updateControls();show(`${role==='system'?'System':'Acquisition'} calibration saved. Playback has not started.`,'success');
  }
  async function loadProfile(role) {
    const profile=profiles.find(p=>p.profile_id===$(role+'Profile').value);if(!profile)return;
    const data=copy(profile.calibration);camera[role].revision=profile.revision;
    if(mode()==='live')$(role+'LiveSource').value=data.source;else setSource(role,data.source);
    if(data.rectification_path && ![...$(role+'Lens').options].some(option=>option.value===data.rectification_path))$(role+'Lens').add(new Option(data.rectification_path,data.rectification_path));
    $(role+'Lens').value=data.rectification_path || '';camera[role].calibration=data;await loadPreview(role);updateControls();
    show('Saved calibration loaded. Review it if the camera or lens position has changed.');
  }
  async function loadLegacy() {
    if(!$('legacyProfile').value)return;
    const image=await loadPreview('system');
    const result=await json('/api/observation/legacy_profile',{config_path:$('legacyProfile').value,source:source('system'),image_size:[image.naturalWidth,image.naturalHeight]});
    const data=result.calibration;$('systemLens').value=data.rectification_path || '';await openCalibration('system',data);
    show('Legacy road geometry loaded for review. Saving creates a separate observation profile.');
  }
  async function safe(task, modal=false) {try{await task();}catch(error){if(modal)$('calibrationError').textContent=error.message;else show(error.message,'error');updateControls();}}
  function initialize() {
    setupCalibrationCanvas();
    for(const role of roles){$(role+'Source').addEventListener('change',()=>safe(async()=>{setSource(role,$(role+'Source').value);if(source(role))await loadPreview(role);}));
      $(role+'LiveSource').addEventListener('change',()=>{camera[role].calibration=null;camera[role].image=null;updateControls();});
      $(role+'Lens').addEventListener('change',()=>safe(async()=>{camera[role].calibration=null;await loadPreview(role);}));
      $(role+'Preview').onclick=()=>safe(()=>loadPreview(role));$(role+'Calibrate').onclick=()=>safe(()=>openCalibration(role));$(role+'Profile').onchange=()=>safe(()=>loadProfile(role));}
    $('sharedTest').onchange=()=>safe(async()=>{if($('sharedTest').checked){setSource('acquisition',source('system'));if(source('acquisition'))await loadPreview('acquisition');}else{setSource('acquisition','');$('acquisitionFeed').hidden=true;$('acquisitionPlaceholder').hidden=false;}updateControls();});
    $('sourceMode').onchange=changeMode;$('legacyProfile').onchange=()=>safe(loadLegacy);
    document.querySelectorAll('[data-upload]').forEach(button=>button.onclick=()=>safe(()=>upload(button.dataset.upload)));
    document.querySelectorAll('[data-youtube]').forEach(button=>button.onclick=()=>safe(()=>download(button.dataset.youtube)));
    $('btnStart').onclick=()=>safe(startRun);$('predictNextGreen').onclick=()=>safe(predict);
    for(const[action,id]of[['pause','btnPause'],['play','btnResume'],['step','btnStep'],['stop','btnStop']])$(id).onclick=()=>safe(()=>control(action));
    $('exportRun').onclick=()=>{window.location.href='/api/observation/export';};
    $('refreshRuns').onclick=()=>safe(loadRuns);
    $('exportSaved').onclick=()=>{if($('savedRuns').value)window.location.href=savedRunExport();};
    $('browseSaved').onclick=()=>safe(()=>browseSaved());
    $('savedRuns').onchange=$('savedKind').onchange=()=>safe(()=>browseSaved());
    $('previousSaved').onclick=()=>safe(()=>browseSaved(Math.max(0,savedView.page.offset-100),true));
    $('nextSaved').onclick=()=>safe(()=>browseSaved(savedView.page.next_offset,true));
    $('savedQuantiles').onclick=()=>safe(savedQuantiles);
    async function historyPage(offset=0) {
      const id=runId,token=++currentHistoryRequest;if(!id)return;
      const current=()=>id===runId && token===currentHistoryRequest;
      const query=new URLSearchParams({run_id:id,offset,limit:100});
      if(durableVehicles?.generation!=null)query.set('generation',durableVehicles.generation);
      try{
        const page=await json('/api/observation/history/vehicles?'+query);
        if(!current())return;
        durableVehicles=page;
      }catch(error){
        if(!current())return;
        if(error.status===409){resetCurrentEvidence();if(telemetry?.run_id===id)renderTelemetry(telemetry);}
        throw error;
      }
      await poll();
    }
    $('historyVehicles').onclick=()=>safe(()=>historyPage());
    $('liveVehicles').onclick=()=>{resetCurrentEvidence();poll();};
    $('previousVehicles').onclick=()=>safe(()=>historyPage(Math.max(0,durableVehicles.offset-100)));
    $('nextVehicles').onclick=()=>safe(()=>historyPage(durableVehicles.next_offset));
    $('requestQuantiles').onclick=()=>safe(async()=>{
      const id=runId,token=++currentQuantileRequest;if(!id)return;
      const current=()=>id===runId && token===currentQuantileRequest;
      exactQuantiles=null;currentQuantileStatus='Pending · computing exact durable quantiles';
      $('quantileStatus').textContent=currentQuantileStatus;
      try{
        const result=await json('/api/observation/quantiles?'+new URLSearchParams({run_id:id}));
        if(!current())return;
        exactQuantiles=result;
      }catch(error){
        if(!current())return;
        currentQuantileStatus='Exact quantiles unavailable: '+error.message;
        $('quantileStatus').textContent=currentQuantileStatus;throw error;
      }
      await poll();
    });
    $('retryStorage').onclick=()=>safe(async()=>{await json('/api/observation/recover?'+new URLSearchParams({run_id:runId}),{});await poll();});
    document.querySelectorAll('[data-tool]').forEach(button=>button.onclick=()=>setTool(button.dataset.tool));
    $('clearTool').onclick=()=>{if(regionKey()){if(tool==='lamp')draft.signal_roi=null;else delete draft.lamp_rois[regionKey()];}else draft[pointKey()]=[];draw();};
    $('undoTool').onclick=()=>{const key=pointKey();if(key)draft[key].pop();draw();};$('flipStopline').onclick=()=>{draft.crossing_direction*=-1;draw();};
    $('closeCalibration').onclick=$('cancelCalibration').onclick=closeCalibration;$('saveCalibration').onclick=()=>safe(saveCalibration,true);
    document.addEventListener('keydown',event=>{if(!$('calibrationModal').classList.contains('active'))return;if(event.key==='Escape'){event.preventDefault();closeCalibration();}if(event.key==='Tab'){const focus=[...$('calibrationModal').querySelectorAll('button,input,select')].filter(e=>!e.hidden && !e.disabled && e.offsetParent!==null);const first=focus[0],last=focus.at(-1);if(! $('calibrationModal').contains(document.activeElement)){event.preventDefault();first.focus();}else if(event.shiftKey && document.activeElement===first){event.preventDefault();last.focus();}else if(!event.shiftKey && document.activeElement===last){event.preventDefault();first.focus();}}});
    safe(async()=>{
      await loadVideos();await loadProfiles();
      const configs=await json('/api/configs');configs.configs.forEach(c=>$('legacyProfile').add(new Option(c.name,c.path)));
      const bootstrap=await json('/api/observation/bootstrap');
      if(bootstrap.source_mode){$('sourceMode').value=bootstrap.source_mode;changeMode();}
      for(const[role,key]of[['system','source_sys'],['acquisition','source_acq']]){
        if(!bootstrap[key])continue;
        if(mode()==='live')$(role+'LiveSource').value=bootstrap[key];else setSource(role,bootstrap[key]);
      }
      if(bootstrap.config_path)$('legacyProfile').value=bootstrap.config_path;
      await poll();await loadRuns();
      if(!runId && mode()==='replay')for(const role of roles)if(source(role))await loadPreview(role);
    }).finally(()=>setInterval(poll,500));
    updateControls();
  }
  window.atsc={switchSourceTab,setupCalibrationCanvas};
  document.addEventListener('DOMContentLoaded',initialize);
})();
