"""Persistent local preview renderer. Latest request wins; no AI on slider edits."""
import json
import os
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import studio
from masking import source_key,cached_mask,get_mask,mask_diagnostics,edge_ready

ROOT=Path(__file__).resolve().parent
LIVE=None
condition=threading.Condition()
pending=None
closing=False
cache=OrderedDict()

def send(**event):
    studio.emit(**event)

def read_requests():
    global pending,closing
    for line in sys.stdin:
        try:
            request=json.loads(line)
            if request.get('command')=='quit':break
            with condition:
                pending=request
                condition.notify()
        except (ValueError,TypeError):
            send(event='preview_error',id=None,message='Invalid preview request')
    with condition:
        closing=True
        condition.notify()

def prepared_folder(path,mode,small_parts,backend,recover_parts=False,recover_white=False,recover_dark=False):
    from preview_cache import key
    from masking import PARTS_VERSION
    from white_recovery import WHITE_VERSION
    parts=[f'parts-v{PARTS_VERSION}'] if recover_parts else []
    if recover_white:parts.append(f'white-v{WHITE_VERSION}')
    if recover_dark:
        from dark_recovery import DARK_VERSION
        parts.append(f'dark-v{DARK_VERSION}')
    return ROOT/'work/preview-cache/prepared'/key([source_key(path),mode,small_parts,backend,'crop-v3']+parts)


def prepare(path, mode, request_id, small_parts=False, backend='mps',recover_parts=False,recover_white=False,recover_dark=False):
    from preview_cache import locked,trim,touch
    folder=prepared_folder(path,mode,small_parts,backend,recover_parts,recover_white,recover_dark)
    with locked(folder):
        data=_prepare(path,mode,request_id,small_parts,backend,folder,recover_parts,recover_white,recover_dark)
        touch(folder)
    trim(ROOT/'work/preview-cache',folder)
    return data


def _prepare(path, mode, request_id, small_parts, backend, folder,recover_parts=False,recover_white=False,recover_dark=False):
    from masking import needs_recovery
    studio.configure_backend(backend)
    key=source_key(path)
    mask_file=cached_mask(path,mode,small_parts,recover_parts,recover_white,recover_dark)
    version=mask_file.stat().st_mtime_ns if mask_file.exists() else 0
    memory_key=(key,mode,small_parts,backend,recover_parts,recover_white,recover_dark,version)
    upgrade=needs_recovery(path,mode,small_parts,recover_parts,recover_white,recover_dark)
    if memory_key in cache and not upgrade:
        cache.move_to_end(memory_key)
        return cache[memory_key]
    rgb_file=folder/'subject.npy';info_file=folder/'subject-info.json'
    original_file=folder/'original.jpg'
    started=time.monotonic()
    send(event='preview_progress',id=request_id,source=str(path),phase='Preparing full-detail preview',fraction=.05)
    data=None
    if edge_ready(mask_file) and rgb_file.exists() and info_file.exists() and original_file.exists() and not upgrade:
        try:
            info=json.loads(info_file.read_text())
            if info.get('crop_version')!=3:raise ValueError('Old preview cache')
            if info['mask_version']!=version:raise ValueError('Changed mask')
            if info['mask_path']!=str(mask_file):raise ValueError('Changed mask engine')
            rgb=np.load(rgb_file,allow_pickle=False).astype(np.float32)/65535
            with Image.open(mask_file) as image:mask_full=np.asarray(image).astype(np.float32)/65535
            l,t,r,b=info['crop']
            mask=mask_full[t:b,l:r]
            if rgb.shape[:2]!=mask.shape or rgb.ndim!=3 or not np.isfinite(rgb).all():raise ValueError('Invalid source cache')
            data=dict(rgb=rgb,mask=mask,bbox=info['bbox'],neutral=info['neutral'],origin=info['crop'][:2],source_size=info['source_size'])
        except (OSError,ValueError,KeyError,EOFError):pass
    if data is None:
        # Develop once at full resolution, then keep only the subject region.
        # Downsampling the entire booth was throwing away most product detail.
        rgb_full=studio.decode(path)
        original=studio.pil(rgb_full);original.thumbnail((1600,1600))
        original.save(original_file,quality=94,icc_profile=studio.ICC)
        send(event='preview_source',id=request_id,source=str(path),path=str(original_file))
        neutral=studio.neutral_gains(rgb_full).tolist()
        mask_full,_,_=get_mask(rgb_full,path,mode,
            lambda phase,fraction:send(event='preview_progress',id=request_id,source=str(path),phase=phase,fraction=fraction),
            small_parts=small_parts,recover_parts=recover_parts,recover_white=recover_white,recover_dark=recover_dark)
        ys,xs=np.where(mask_full>.5)
        bounds=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
        x,y,right,bottom=bounds
        pad=int(max(right-x,bottom-y)*.18)+4
        l=max(0,x-pad);t=max(0,y-pad);r=min(rgb_full.shape[1],right+pad);b=min(rgb_full.shape[0],bottom+pad)
        rgb=rgb_full[t:b,l:r].copy();mask=mask_full[t:b,l:r].copy()
        temporary=rgb_file.with_suffix('.tmp')
        with temporary.open('wb') as file:np.save(file,np.round(rgb*65535).astype(np.uint16),allow_pickle=False)
        temporary.replace(rgb_file)
        mask_file=cached_mask(path,mode,small_parts,recover_parts,recover_white,recover_dark)
        info=dict(crop=[l,t,r,b],crop_version=3,bbox=bounds,neutral=neutral,mask_version=mask_file.stat().st_mtime_ns,mask_path=str(mask_file),source_size=[rgb_full.shape[1],rgb_full.shape[0]])
        temporary=info_file.with_suffix('.tmp');temporary.write_text(json.dumps(info));temporary.replace(info_file)
        data=dict(rgb=rgb,mask=mask,bbox=bounds,neutral=neutral,origin=[l,t],source_size=info['source_size'])
    from contact_shadow import shadow_layer
    data['shadow']=None
    if recover_white or recover_dark:
        from masking import addition_protection
        full=addition_protection(path,mode,small_parts,recover_parts,recover_white,recover_dark)
        x,y=data['origin'];h,w=data['mask'].shape
        data['addition_protection']=full[y:y+h,x:x+w].copy()
    data.update(original=str(original_file),prepared_seconds=round(time.monotonic()-started,3),
                review=mask_diagnostics(path,mode,small_parts,recover_parts,recover_white,recover_dark).get('review',[]))
    mask_file=cached_mask(path,mode,small_parts,recover_parts,recover_white,recover_dark)
    cache[(key,mode,small_parts,backend,recover_parts,recover_white,recover_dark,mask_file.stat().st_mtime_ns)]=data
    while len(cache)>4:
        cache.popitem(last=False)
    return data


def rendered_folder(request,path,mode,backend):
    from preview_cache import key
    studio.configure_backend(backend)
    settings=request.get('settings',{})
    mask=cached_mask(path,mode,bool(settings.get('small_parts')),bool(settings.get('recover_parts')),bool(settings.get('recover_white')),bool(settings.get('recover_dark')))
    stamp=mask.stat().st_mtime_ns if mask.exists() else 0
    content={k:v for k,v in request.items() if k not in ('id','source')}
    content['batch_geometry']=sorted([r for r in request.get('batch_geometry',[]) if r.get('source')!=str(path)],key=lambda r:json.dumps(r,sort_keys=True))
    version='floor-cleanup-v1' if request.get('settings',{}).get('shadow_detection')=='soft' else 'render-v1'
    return ROOT/'work/preview-cache/rendered'/key([version,source_key(path),str(mask),stamp,content])

def render_request(request):
    started=time.monotonic()
    request_id=request['id']
    path=Path(request['source']).resolve()
    mode=request.get('mask_mode','efficient')
    settings=request.get('settings',{})
    backend=request.get('mask_backend') or settings.get('mask_backend','mps')
    if backend not in ('onnx','mps'):raise ValueError('Invalid mask backend')
    from preview_cache import locked,atomic_json,touch,trim
    folder=rendered_folder(request,path,mode,backend)
    from masking import needs_recovery
    recover_parts=bool(settings.get('recover_parts'))
    recover_white=bool(settings.get('recover_white'))
    recover_dark=bool(settings.get('recover_dark'))
    if request.get('geometry_only') and not needs_recovery(path,mode,bool(settings.get('small_parts')),recover_parts,recover_white,recover_dark):
        prepared=prepared_folder(path,mode,bool(settings.get('small_parts')),backend,recover_parts,recover_white,recover_dark)
        try:
            with locked(prepared):
                info=json.loads((prepared/'subject-info.json').read_text())
                mask=cached_mask(path,mode,bool(settings.get('small_parts')),recover_parts,recover_white,recover_dark)
                if info['crop_version']!=3 or info['mask_path']!=str(mask) or info['mask_version']!=mask.stat().st_mtime_ns:raise ValueError('Changed mask')
                touch(prepared)
            send(event='geometry_result',id=request_id,source=str(path),bbox=info['bbox'],source_size=info['source_size'],cache_hit=True)
            return
        except (OSError,ValueError,KeyError):pass
    if not request.get('geometry_only') and not needs_recovery(path,mode,bool(settings.get('small_parts')),recover_parts,recover_white,recover_dark):
        try:
            with locked(folder):
                saved=json.loads((folder/'result.json').read_text())
                if not Path(saved['path']).is_file() or not Path(saved['original']).is_file():raise ValueError('Evicted preview')
                with Image.open(saved['path']) as im:im.verify()
                touch(folder)
            saved.update(id=request_id,source=str(path),seconds=round(time.monotonic()-started,3),cache_hit=True)
            send(**saved);return
        except (OSError,ValueError,KeyError):pass
    data=prepare(path,mode,request_id,bool(settings.get('small_parts')),backend,recover_parts,recover_white,recover_dark)
    if request.get('geometry_only'):
        # A batch's shared scale needs another photo's silhouette, typically
        # much wider or narrower than the one on screen, so the caller can
        # learn its bbox without paying for a render nobody will look at.
        send(event='geometry_result',id=request_id,source=str(path),
             bbox=data['bbox'],source_size=data['source_size'])
        return
    with condition:
        if pending is not None:return  # Render the newest recipe, not queued old slider positions.
    def setting(name,default,low,high):
        value=float(settings.get(name,default))
        if not np.isfinite(value) or not low<=value<=high:raise ValueError(f'Invalid {name}')
        return value
    exposure=setting('exposure',.65,-2,2)
    warmth=setting('warmth',.25,-1,1)
    contrast=setting('contrast',0,-1,1)
    whites=setting('whites',0,-1,1)
    shadow=setting('shadow',2,0,2)
    fill=setting('fill',.84,.5,.94)
    sharpness=setting('sharpness',.75,0,2)
    denoise=setting('denoise',0,0,2)
    export_size=int(setting('size',2400,512,6000))
    width=height=export_size
    aspect=settings.get('aspect','square')
    if aspect=='landscape':height=round(width*2/3)
    elif aspect=='portrait':width=round(height*4/5)
    elif aspect=='wide':height=round(width*9/16)
    elif aspect!='square':raise ValueError('Invalid canvas shape')
    shadow_method=settings.get('shadow_method','classic')
    detection=settings.get('shadow_detection') or 'dark'
    if detection not in ('dark','soft'):raise ValueError('Invalid shadow detection')
    mask=data['mask']
    if detection=='soft':
        from contact_shadow import soften_shadow_mask
        if 'floor_cleanup_mask' not in data:
            data['floor_cleanup_mask']=soften_shadow_mask(data['rgb'],mask)
            if recover_white or recover_dark:
                data['floor_cleanup_mask']=np.maximum(data['floor_cleanup_mask'],data['addition_protection'])
        mask=data['floor_cleanup_mask']
    shadow_key='shadow_'+detection
    if shadow and shadow_method=='local' and data.get(shadow_key) is None:
        from contact_shadow import shadow_layer
        data[shadow_key]=shadow_layer(data['rgb'],mask,detection)
    source_rgb=data['rgb'];reviews=list(data['review'])
    if settings.get('package_cleanup'):
        if 'package_rgb' not in data:
            from packaging_cleanup import clean_package_code
            data['package_rgb'],data['package_info']=clean_package_code(source_rgb)
        source_rgb=data['package_rgb']
        if data['package_info']['status']=='review':reviews.append(data['package_info']['message'])
    corrected,_=studio.correct(source_rgb,exposure,warmth,contrast,neutral=data['neutral'],whites=whites)
    from framing import batch_frame
    geometry=request.get('batch_geometry',[])
    geometry=[r for r in geometry if r.get('source')!=str(path)]
    geometry.append(dict(bbox=data['bbox'],source_size=data['source_size']))
    frame=batch_frame(geometry,width,height,fill,settings.get('framing','centered'),settings.get('centered_scale')=='independent')
    image,_,_=studio.render(corrected,mask,width,height,fill,shadow,sharpness,denoise,
                            include_transparent=False,frame=frame,source_width=data['source_size'][0],origin=data['origin'],shadow_layer=data.get(shadow_key),shadow_method=shadow_method)
    if request.get('thumbnail'):image.thumbnail((640,640),Image.Resampling.LANCZOS)
    folder=rendered_folder(request,path,mode,backend)
    target=folder/'preview.jpg'
    result=dict(event='preview_result',id=request_id,source=str(path),path=str(target),original=data['original'],
                bbox=data['bbox'],source_size=data['source_size'],width=width,height=height,
                seconds=round(time.monotonic()-started,3),settings=settings,review=reviews,cache_hit=False)
    with locked(folder):
        temporary=target.with_suffix('.tmp')
        image.save(temporary,format='JPEG',quality=96,subsampling=0,icc_profile=studio.ICC)
        temporary.replace(target)
        atomic_json(folder/'result.json',result);touch(folder)
    send(**result)
    trim(ROOT/'work/preview-cache',folder)

def main():
    global pending
    threading.Thread(target=read_requests,daemon=True).start()
    send(event='preview_ready')
    while True:
        with condition:
            condition.wait_for(lambda:pending is not None or closing)
            if closing:return
            request=pending;pending=None
        try:render_request(request)
        except Exception as exc:send(event='preview_error',id=request.get('id'),message=str(exc))

if __name__=='__main__':
    import argparse
    from preview_storage import PreviewStorage
    parser=argparse.ArgumentParser()
    parser.add_argument('--session-dir',type=Path)
    parser.add_argument('--owner-pid',type=int)
    args=parser.parse_args()
    storage=PreviewStorage(args.session_dir,args.owner_pid)
    LIVE=storage.path
    try:main()
    finally:
        storage.close()
        studio.SESSION=None
        studio.SESSION_BACKEND=None
        import gc
        gc.collect()
