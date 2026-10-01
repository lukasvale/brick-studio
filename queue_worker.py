"""Durable local folder queue. Atomic state, bounded retries and resumable shoots."""
import argparse,fcntl,html,json,os,signal,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CHILD=None
STOP=False
RESERVE=2*1024**3

def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def read(path,default=None):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default

def interrupted(*_):
    global STOP
    STOP=True
    if CHILD and CHILD.poll() is None:CHILD.terminate()

def existing_parent(path):
    while not path.exists():path=path.parent
    return path

def check_space(job,recipe):
    remaining=max(1,len(job['sources'])-job.get('completed',0))
    variants=2 if recipe.get('framing')=='both' else 1
    need=RESERVE+remaining*int(recipe.get('size',2400))**2*5*variants
    if shutil.disk_usage(existing_parent(Path(job['output']))).free<need:
        raise OSError(f'Free at least {need/1024**3:.1f} GB on the export drive before resuming.')
    if shutil.disk_usage(ROOT/'work').free<RESERVE+remaining*8*1024**2:
        raise OSError('Free more space on the app drive before resuming (at least 2 GB plus mask storage).')

def clean_temporary(job):
    # Only incomplete writes owned by this job. Masks and finished images stay.
    for name in ('metadata','output'):
        folder=Path(job[name])
        if folder.exists():
            for file in folder.rglob('*.tmp'):file.unlink(missing_ok=True)

def summary(state,path):
    rows=[];attention=[]
    for job in state['jobs']:
        meta=Path(job.get('metadata') or ROOT/'work'/'none')
        m=read(meta/'manifest.json',{})
        for r in m.get('records',[]):
            if r.get('review'):attention.append(dict(batch=job['name'],photo=r['name'],source=r['source'],reason='; '.join(r['review'])))
        for error in m.get('errors',[]):
            attention.append(dict(batch=job['name'],photo=Path(error['source']).name,source=error['source'],reason=error['error']))
        if job.get('message') and job.get('status') in ('error','paused'):
            attention.append(dict(batch=job['name'],photo='Batch',source=job['folder'],reason=job['message']))
        sheet=meta/'contact-sheet.jpg'
        img=f'<a href="{sheet.as_uri()}"><img src="{sheet.as_uri()}" alt="Contact sheet for {html.escape(job["name"],quote=True)}"></a>' if sheet.exists() else ''
        output=html.escape(job.get('output') or '')
        rows.append(f'<section><h2>{html.escape(job["name"])}</h2><p>{html.escape(job["status"])} · {job.get("completed",0)} / {len(job["sources"])} photos</p><p>{output}</p>{img}</section>')
    table=''.join('<tr>'+''.join(f'<td>{html.escape(str(r[k]))}</td>' for k in ('batch','photo','reason'))+'</tr>' for r in attention)
    document='''<!doctype html><meta charset="utf-8"><title>Brick Studio — batch summary</title>
<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#20232b;background:#fff}img{max-width:100%;height:auto}td,th{text-align:left;vertical-align:top;padding:10px;border-bottom:1px solid #ddd;overflow-wrap:anywhere}table{width:100%;border-collapse:collapse}section{margin:40px 0}p{overflow-wrap:anywhere}</style>'''
    document+=f'<h1>Batch processing summary</h1><p>{sum(j.get("completed",0) for j in state["jobs"])} photos exported · {len(attention)} items to review</p>'
    document+='<h2>Needs attention</h2>'+(('<table><tr><th>Batch</th><th>Photo</th><th>Reason</th></tr>'+table+'</table>') if attention else '<p>No processing errors or automatic review flags. Check the contact sheets before delivery.</p>')
    document+=''.join(rows)
    target=path.parent/'summary.html';tmp=target.with_suffix('.tmp');tmp.write_text(document);tmp.replace(target)
    atomic(path.parent/'summary.json',dict(attention=attention,batches=state['jobs']))
    state['summary']=str(target)

def run(path):
    global CHILD,STOP
    path=path.resolve();path.parent.mkdir(parents=True,exist_ok=True)
    lock=(path.parent/'worker.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('This queue is already running')
    state=read(path)
    if not state:raise SystemExit('No saved queue')
    pause=path.parent/'pause.request'
    # The native app clears an old pause before launch. Preserve a new pause
    # clicked while this worker was starting (including immediately quitting).
    if state.get('status')!='starting':pause.unlink(missing_ok=True)
    # Stop ends the run and discards the unfinished export (the app moves its
    # folder to the Trash); unlike pause it is never resumed.
    stop=path.parent/'stop.request'
    if state.get('status')!='starting':stop.unlink(missing_ok=True)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    started=time.monotonic();prior_elapsed=state.get('elapsed_seconds',0) or 0
    state.update(status='running',worker_pid=os.getpid(),phase='Preparing queue')
    atomic(path,state)
    awake=None
    if sys.platform=='darwin':awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
    try:
        for index,job in enumerate(state['jobs']):
            if STOP or pause.exists() or stop.exists():break
            if job.get('archived') is True or job.get('include') is False:continue  # Not selected for this export.
            if job.get('status')=='complete':continue
            if job.get('status')=='error':continue
            recipe=job.get('started_recipe') or job.get('override') or state['shared_recipe']
            # Snapshot settings on first start; later edits cannot alter a
            # half-finished product and quietly mix its export recipes.
            job['started_recipe']=recipe.copy()
            job.setdefault('output',str(Path(state['output_root'])/(job['name']+'-'+job['id'][:8])))
            job.setdefault('metadata',str(ROOT/'work'/'shoots'/('queue-'+job['id'])))
            # Swift optional paths can be null in an older saved queue.
            if not job['output']:job['output']=str(Path(state['output_root'])/(job['name']+'-'+job['id'][:8]))
            if not job['metadata']:job['metadata']=str(ROOT/'work'/'shoots'/('queue-'+job['id']))
            meta=Path(job['metadata']);meta.mkdir(parents=True,exist_ok=True)
            listing=meta/'inputs.json';atomic(listing,job['sources'])
            while job.get('attempts',0)<3:
                try:check_space(job,recipe)
                except OSError as exc:
                    job.update(status='paused',message=str(exc));state.update(status='paused',phase=str(exc));STOP=True;break
                if STOP or pause.exists() or stop.exists():break
                clean_temporary(job)
                path.with_name('progress.json').unlink(missing_ok=True)
                job.update(status='running',message=None);state.update(active_name=job['name'],active_job_id=job['id'],phase='Preparing '+job['name'],photo_number=0,photo_total=len(job['sources']),photo_stage='preparation')
                atomic(path,state)
                command=[sys.executable,str(ROOT/'studio.py'),'--input-list',str(listing),'--output',job['output'],'--metadata',str(meta),'--owner-pid',str(os.getpid())]
                command.append('--replace' if job.get('replace') else '--resume')
                allowed=('size','aspect','exposure','warmth','fill','shadow','contrast','sharpness','denoise','mask_mode','mask_backend','framing','shadow_method','shadow_detection','centered_scale','whites')
                for key in allowed:
                    if recipe.get(key) is not None:command+=['--'+key.replace('_','-'),str(recipe[key])]
                if job.get('adjustment_recipes'):
                    adjustment_settings=meta/'adjustment-recipes.json';atomic(adjustment_settings,job['adjustment_recipes'])
                    command+=['--adjustment-recipes',str(adjustment_settings)]
                if job.get('photo_recipes') is not None:
                    photo_settings=meta/'photo-recipes.json';atomic(photo_settings,job['photo_recipes'])
                    command+=['--photo-recipes',str(photo_settings)]
                elif recipe.get('package_cleanup'):command.append('--clean-package-code')
                if recipe.get('small_parts'):command.append('--small-parts')
                if recipe.get('recover_parts'):command.append('--recover-parts')
                if recipe.get('recover_white'):command.append('--recover-white')
                if recipe.get('recover_dark'):command.append('--recover-dark')
                job['replace']=False
                CHILD=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                state['child_pid']=CHILD.pid;atomic(path,state);last_disk=0
                with (meta/'processing.log').open('a') as log:
                    for line in CHILD.stdout:
                        log.write(line);log.flush()
                        if pause.exists() or stop.exists():STOP=True
                        if time.monotonic()-last_disk>3:
                            last_disk=time.monotonic()
                            try:
                                if min(shutil.disk_usage(ROOT/'work').free,shutil.disk_usage(existing_parent(Path(job['output']))).free)<RESERVE:
                                    STOP=True;job['message']='Low disk space. Free space and resume.'
                            except OSError as exc:STOP=True;job['message']=str(exc)
                        if STOP and CHILD.poll() is None:CHILD.terminate()
                        try:event=json.loads(line)
                        except ValueError:continue
                        if event.get('stage')=='progress':
                            job['completed']=event.get('completed',0)
                            state.update(phase=event.get('phase','Processing'),photo_number=event.get('photo_number',0),
                                         photo_total=event.get('total',len(job['sources'])),photo_stage=event.get('photo_stage','rendering'),
                                         photo_elapsed=event.get('photo_elapsed',0),photo_progress=event.get('photo_progress',0),
                                         elapsed_seconds=prior_elapsed+time.monotonic()-started)
                            chosen=[j for j in state['jobs'] if j.get('include') is not False and j.get('archived') is not True]
                            remaining=sum(len(j['sources'])-j.get('completed',0) for j in state['jobs'][index+1:] if j.get('status') not in ('complete','error') and j.get('include') is not False and j.get('archived') is not True)
                            state['eta_seconds']=event.get('eta_seconds',0)+remaining*state.get('seconds_per_photo',28)
                            done=sum(j.get('completed',0) for j in chosen)
                            total=sum(len(j['sources']) for j in chosen)
                            state.update(completed_photos=done,total_photos=total,
                                         completed_batches=sum(j.get('status')=='complete' for j in state['jobs']),
                                         error_count=sum(j.get('errors',0) for j in state['jobs']),
                                         progress=min(.99,done/max(1,total)))
                            # Frequent timer updates stay tiny even with 3,600
                            # source paths. Durable queue checkpoints are saved
                            # after photos and batch transitions, not every tick.
                            keys=('worker_pid','active_job_id','active_name','phase','photo_number','photo_total','photo_stage','photo_elapsed','photo_progress',
                                  'elapsed_seconds','eta_seconds','completed_photos','total_photos','completed_batches','error_count','progress')
                            snapshot={k:state[k] for k in keys if k in state}
                            snapshot['active_completed']=job['completed']
                            atomic(path.with_name('progress.json'),snapshot)
                        if event.get('stage')=='photo_done':
                            seconds=event['record'].get('seconds',5)
                            # Rendering alone excludes preparation; cached vs
                            # fresh timings are learned from completed batches.
                            state['last_render_seconds']=seconds
                            job['completed']=len(read(meta/'manifest.json',{}).get('records',[]))
                            atomic(path,state)
                CHILD.stdout.close();code=CHILD.wait();CHILD=None
                state.pop('child_pid',None)
                m=read(meta/'manifest.json',{})
                job['completed']=len(m.get('records',[]));job['errors']=len(m.get('errors',[]))
                if stop.exists():job.update(status='stopped',message='Stopped · export discarded');break
                if STOP or pause.exists():job['status']='paused';break
                if code==0 and job['completed']==len(job['sources']):
                    job.update(status='complete',message=None)
                    elapsed=m.get('elapsed_seconds',0)
                    if elapsed and not job.get('attempts'):state['seconds_per_photo']=max(1,elapsed/max(1,len(job['sources'])))
                    break
                job['attempts']=job.get('attempts',0)+1
                job['message']=f'Attempt {job["attempts"]} failed; completed images are saved.'
                if code==2:job['attempts']=3;job['message']='The source files or recipe changed, or the batch configuration is invalid. Review the processing log.'
                job['status']='error' if job['attempts']>=3 else 'retrying'
                atomic(path,state)
            if job.get('status')=='error' and not job.get('errors'):job['errors']=max(1,len(job['sources'])-job.get('completed',0))
            clean_temporary(job);summary(state,path);atomic(path,state)
        if stop.exists():
            state.update(status='stopped',phase='Stopped · unfinished export discarded')
            for job in state['jobs']:
                if job.get('status') in ('running','retrying'):job.update(status='stopped',message='Stopped · export discarded')
            stop.unlink(missing_ok=True)
        else:
            state['status']='paused' if STOP or pause.exists() else 'complete'
            state['phase']='Paused · completed photos are saved' if state['status']=='paused' else 'Finished · review the summary'
    except Exception as exc:
        state.update(status='paused',phase=str(exc))
        for job in state['jobs']:
            if job.get('status')=='running':job.update(status='paused',message=str(exc))
    finally:
        if CHILD and CHILD.poll() is None:CHILD.terminate();CHILD.wait(timeout=15)
        if awake:awake.terminate();awake.wait(timeout=5)
        state.update(worker_pid=None,child_pid=None,elapsed_seconds=prior_elapsed+time.monotonic()-started,
                     completed_photos=sum(j.get('completed',0) for j in state['jobs']),
                     completed_batches=sum(j.get('status')=='complete' for j in state['jobs']),
                     error_count=sum(j.get('errors',0) for j in state['jobs']))
        if state['status']=='complete':state['eta_seconds']=0;state['progress']=1
        summary(state,path);atomic(path,state)
        lock.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('state',type=Path)
    run(parser.parse_args().state)
