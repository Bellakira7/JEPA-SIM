(() => {
  'use strict';
  const $=s=>document.querySelector(s), cases=window.JEPA_DATA.companyDemo.cases, records=window.JEPA_REPLAY;
  const videos={gs:$('#drive-gs'),jepa:$('#drive-jepa'),gsBev:$('#bev-gs'),jepaBev:$('#bev-jepa')};
  const streamSuffix={gs:'gs',jepa:'layout',gsBev:'gs-bev',jepaBev:'jepa-bev'};
  const slider=$('#rollout-time'), play=$('#play-rollout');
  const mediaCache=new Map();
  function mediaUrl(src){if(!mediaCache.has(src))mediaCache.set(src,fetch(src).then(r=>{if(!r.ok)throw new Error('Video unavailable');return r.blob();}).then(blob=>URL.createObjectURL(blob)).catch(e=>{mediaCache.delete(src);throw e;}));return mediaCache.get(src);}
  const motion=matchMedia('(prefers-reduced-motion: reduce)');
  let current=cases[0],time=0,running=false,ready=false,repeat=true,generation=0,last=0,inView=false,autoStart=!motion.matches;
  function sample(arm,t){const rows=records[current.id+'-'+arm].frames;return rows[Math.min(30,Math.floor((t+1e-6)*5))];}
  function syncControls(){play.setAttribute('aria-label',(running?'Pause':'Play')+' recorded rollout');$('#play-label').textContent=running?'Pause':'Play';}
  function pause(){running=false;Object.values(videos).forEach(v=>v.pause());syncControls();}
  async function start(){if(!ready)return;if(time>=6)seek(0);running=true;last=0;syncControls();const results=await Promise.allSettled(Object.values(videos).map(v=>v.play()));if(results.some(r=>r.status==='rejected')){pause();$('#demo-status').textContent='Press Play to start the paired replay.';}}
  function seek(t){time=Math.max(0,Math.min(6,t));Object.values(videos).forEach(v=>{if(v.readyState>=1)v.currentTime=time;});render();}
  cases.forEach((c,i)=>{const b=document.createElement('button');b.className='case-button';b.textContent=c.title;b.dataset.case=i;b.setAttribute('aria-pressed','false');b.addEventListener('click',()=>select(i));$('.case-switch').append(b);});
  function loadVideo(video,src){return new Promise((resolve,reject)=>{const clear=()=>{video.removeEventListener('loadeddata',ok);video.removeEventListener('error',bad);};const ok=()=>{clear();resolve();},bad=()=>{clear();reject();};video.addEventListener('loadeddata',ok);video.addEventListener('error',bad);video.src=src;video.load();});}
  async function select(i){
    pause();ready=false;play.disabled=slider.disabled=true;const g=++generation;current=cases[i];time=0;
    document.querySelectorAll('.case-button').forEach((b,k)=>{b.classList.toggle('active',i===k);b.setAttribute('aria-pressed',String(i===k));});
    $('#drive-scene').textContent=current.title;$('#demo-status').textContent='';$('#play-label').textContent='Loading';
    for(const arm of ['gs','jepa']){$('#error-'+arm).textContent=current.finalError[arm==='gs'?'3DGS':'JEPA-SIM'].toFixed(2)+' m';}
    for(const [key,video] of Object.entries(videos))video.poster='assets/replay/'+current.id+'-'+streamSuffix[key]+'.jpg';
    render();
    try{const urls=await Promise.all(Object.keys(videos).map(key=>mediaUrl('assets/replay/'+current.id+'-'+streamSuffix[key]+'.mp4')));if(g!==generation)return;await Promise.all(Object.keys(videos).map((key,k)=>loadVideo(videos[key],urls[k])));if(g!==generation)return;ready=true;play.disabled=slider.disabled=false;syncControls();if(autoStart&&inView)start();}
    catch{if(g!==generation)return;$('#demo-status').textContent='Video could not load. Select the scene to retry.';$('#play-label').textContent='Unavailable';}
  }
  play.addEventListener('click',()=>{autoStart=false;if(running)pause();else start();});
  slider.addEventListener('input',()=>{autoStart=false;pause();seek(Number(slider.value));});
  $('#drive-repeat').addEventListener('click',()=>{repeat=!repeat;$('#drive-repeat').setAttribute('aria-pressed',String(repeat));});
  motion.addEventListener('change',()=>{if(motion.matches){autoStart=false;pause();}});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
  new IntersectionObserver(entries=>{inView=entries[0].isIntersecting;if(!inView)pause();else if(autoStart&&ready)start();},{threshold:.08}).observe($('.drive-theater'));
  function render(){slider.value=time;slider.style.setProperty('--progress',time/6*100+'%');slider.setAttribute('aria-valuetext',time.toFixed(1)+' seconds');$('#time-output').innerHTML=time.toFixed(1)+' <span>/ 6.0 s</span>';$('#drive-frame').textContent=String(Math.min(31,Math.floor(time*5)+1)).padStart(2,'0')+' / 31';for(const arm of ['gs','jepa']){$('#velocity-'+arm).textContent=(sample(arm,time).speed*3.6).toFixed(0);}}
  function tick(now){
    if(running){
      // A shared media clock stops the world view while a replay stream is buffering.
      if(Object.values(videos).every(v=>v.readyState>=3)){
        time=Math.min(6,videos.gs.currentTime);
        for(const [key,video] of Object.entries(videos)){if(key!=='gs'&&Math.abs(video.currentTime-time)>.07)video.currentTime=time;}
        if(now-last>30){render();last=now;}
        if(time>=6){if(repeat){seek(0);Object.values(videos).forEach(v=>v.play().catch(pause));}else{pause();render();}}
      }
    }
    requestAnimationFrame(tick);
  }
  select(0);requestAnimationFrame(tick);
})();
