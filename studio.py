"""Local RAW-to-catalog pipeline. Source photographs are never overwritten."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time, zipfile, threading
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '4')
import cv2
import numpy as np
import rawpy
from PIL import Image, ImageCms, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent
EXTENSIONS = {'.dng', '.cr2', '.cr3', '.nef', '.arw', '.raf', '.rw2', '.orf', '.pef', '.jpg', '.jpeg', '.png', '.tif', '.tiff'}
ICC = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
SESSION = None
SESSION_BACKEND = None
MASK_BACKEND = 'mps'  # 'mps' (PyTorch Metal, default) or 'onnx' (rembg CPU)
EMIT_LOCK = threading.Lock()

def configure_backend(backend):
    """Select the subject-separation engine; clears a loaded session on change."""
    global MASK_BACKEND, SESSION, SESSION_BACKEND
    if backend not in ('onnx', 'mps'):
        raise ValueError(f'Unknown mask backend: {backend}')
    if SESSION is not None and SESSION_BACKEND != backend:
        SESSION = None
        SESSION_BACKEND = None
        import gc
        gc.collect()
    MASK_BACKEND = backend

def session():
    global SESSION, SESSION_BACKEND
    if SESSION is None or SESSION_BACKEND != MASK_BACKEND:
        if MASK_BACKEND == 'mps':
            emit(stage='model', message='Loading Metal subject separation model…')
            from birefnet_mps import new_session
            SESSION = new_session()
        else:
            emit(stage='model', message='Loading subject separation model…')
            from rembg import new_session
            SESSION = new_session('birefnet-general', providers=['CPUExecutionProvider'])
        SESSION_BACKEND = MASK_BACKEND
    return SESSION

def emit(**event):
    with EMIT_LOCK:
        print(json.dumps(event), flush=True)

def write_manifest(out, manifest):
    temporary = out / 'manifest.json.tmp'
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(out / 'manifest.json')

def decode(path, half_size=False):
    if path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}:
        from PIL import ImageOps
        im = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
        return np.asarray(im).astype(np.float32) / 255
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(use_camera_wb=True, output_color=rawpy.ColorSpace.sRGB, half_size=half_size,
                              output_bps=16, gamma=(2.4, 12.92),
                              highlight_mode=rawpy.HighlightMode.Blend)
    return rgb.astype(np.float32) / 65535

def pil(rgb):
    return Image.fromarray(np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8))

def neutral_gains(rgb):
    h, w = rgb.shape[:2]
    # Sample only the lit, neutral side walls, excluding clipped pixels.
    samples = np.concatenate([rgb[int(.20*h):int(.65*h):6, int(.03*w):int(.13*w):6].reshape(-1,3),
                              rgb[int(.20*h):int(.65*h):6, int(.87*w):int(.97*w):6].reshape(-1,3)])
    valid = samples[(samples.min(1) > .60) & (samples.max(1) < .97) & (np.ptp(samples,axis=1) < .12)]
    gains = np.ones(3, dtype=np.float32)
    if len(valid) > 50:
        neutral = np.median(valid, axis=0)
        gains = np.clip(neutral.mean() / neutral, .94, 1.06)
    return gains

def correct(rgb, exposure, warmth, contrast=0, neutral=None, whites=0):
    gains=neutral_gains(rgb) if neutral is None else np.array(neutral,dtype=np.float32).copy()
    gains *= np.array([1+warmth*.035, 1, 1-warmth*.035])
    balanced = np.clip(rgb * gains, 0, 1)
    linear = np.where(balanced <= .04045, balanced / 12.92, ((balanced+.055)/1.055)**2.4)
    # Exposure with a shoulder protecting white bricks and specular highlights.
    gain = 2 ** exposure
    linear = linear * gain / (1 + (gain - 1) * linear)
    result = np.where(linear <= .0031308, linear*12.92, 1.055*linear**(1/2.4)-.055)
    # Gentle monotonic S-curve: richer blacks without clipping shadow detail.
    result += .045 * np.sin(2 * np.pi * (result - .5))
    if contrast:
        # Symmetric contrast around mid-gray; preserve black/white endpoints.
        value=np.clip(result,0,1)
        value=np.where(value<1e-7,0,np.where(value>1-1e-7,1,value))
        power=2**contrast
        low=value**power;high=(1-value)**power
        result=low/(low+high)
    if whites:
        # Whites: change the slope of the tone curve above a high threshold, so blown white
        # bricks come down while mid-tones, shadows and colour are left alone. A slope keeps
        # the order of tones, so a bright brick keeps its gradation instead of going flat.
        knee = .7
        top = result @ np.array([.2126, .7152, .0722], np.float32)
        slope = 1 + whites * .8
        target = np.where(top > knee, knee + (top - knee) * slope, top)
        # Scale all three channels alike: the tone moves, the colour does not.
        result = np.clip(result * (target / np.maximum(top, 1e-6))[:, :, None], 0, 1)
    luminance = result @ np.array([.2126, .7152, .0722], np.float32)
    result = np.clip(luminance[:, :, None] + (result - luminance[:, :, None]) * 1.05, 0, 1)
    return result.astype(np.float32), gains.tolist()

def components(mask,preserve_frame=False):
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask>.45).astype('uint8'))
    if n <= 1:
        raise ValueError('No product detected. Try a clearer photo with space around the set.')
    h,w = mask.shape
    # Prefer the central product over the opening of the photo box at the top edge.
    scores=[]
    for i in range(1,n):
        x,y,bw,bh,area = stats[i]
        center = 1 - min(1, abs(x+bw/2-w/2)/w + abs(y+bh/2-h*.58)/h)
        scores.append(area*center*(.03 if y == 0 else 1))
    main = int(np.argmax(scores))+1
    x,y,bw,bh,area = stats[main]
    # A set can include separate figures, vehicles and accessories. Keep
    # every interior foreground component, not only those touching the main
    # model. Exclude frame-connected booth/equipment regions unless they are
    # the principal subject. The crop must enclose the full retained union.
    keep=np.zeros(mask.shape,np.uint8)
    min_area=max(3,round(h*w*.000003))
    for i in range(1,n):
        xx,yy,ww,hh,aa=stats[i]
        touches_frame=xx==0 or yy==0 or xx+ww>=w or yy+hh>=h
        if i==main or (aa>min_area and (preserve_frame or not touches_frame)):
            keep[labels==i]=1
    ys,xs=np.where(keep)
    box=(int(xs.min()),int(ys.min()),int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1))
    support=cv2.dilate(keep,np.ones((7,7),np.uint8))
    return mask*support,box

def subject_mask(rgb, cache):
    h,w=rgb.shape[:2]
    if cache.exists():
        try:
            with Image.open(cache) as image:
                mask=np.asarray(image).astype(np.float32)/65535
            if mask.shape==(h,w) and np.isfinite(mask).all() and np.any(mask>.5): return mask
        except (OSError, ValueError):
            pass  # An interrupted cache write must not make the source unusable.
    from masking import build_mask
    mask,_=build_mask(rgb,mode='detailed')
    cache.parent.mkdir(parents=True,exist_ok=True)
    temporary=cache.with_suffix('.tmp')
    Image.fromarray((mask*65535).astype(np.uint16)).save(temporary,format='PNG')
    temporary.replace(cache)
    return mask

def contact_foreground(rgb, alpha, density):
    """Remove the photographed floor mix, not an assumed pure-white mix.

    White subtraction double-darkened antialiasing over real dark contacts.
    Nearby uncovered source pixels provide the floor colour; low confidence
    leaves the source untouched. Opaque pixels and transparent exports stay exact.
    """
    edge_weight=density if np.ndim(density)==2 else density.max(axis=2)
    edge=(alpha>.04)&(alpha<.98)&(edge_weight>.012)
    if not np.any(edge):return rgb
    visible=(alpha<.02).astype(np.float32)
    weight=cv2.GaussianBlur(visible,(0,0),3)
    floor=cv2.GaussianBlur(rgb*visible[:,:,None],(0,0),3)/np.maximum(weight[:,:,None],1e-6)
    edge &= weight>.05
    result=rgb.copy()
    a=alpha[edge,None]
    result[edge]=np.clip((rgb[edge]-(1-a)*floor[edge])/np.maximum(a,.15),0,1)
    return result


def _denoise(image, amount, detail_scale=1):
    """Edge-preserving smooth; amount 0–2. Above 1 stacks a second pass."""
    pixels=np.asarray(image).astype(np.float32)/255
    strength=float(np.clip(amount,0,2))
    diameter=3+2*int(round(2+2*strength))  # odd: 7…15
    if detail_scale<.75:diameter=max(5,diameter-2)
    space=max(.7,2.2*detail_scale)*(1+.45*strength)
    color=.03+.14*strength
    smooth=cv2.bilateralFilter(pixels,diameter,color,space)
    if strength>1:
        # Second pass cleans residual grain without needing a heavier blend.
        extra=cv2.bilateralFilter(smooth,diameter,.03+.1*(strength-1),space)
        smooth=smooth*(2-strength)+extra*(strength-1)
    blend=min(.96,.55*strength)
    return pil(pixels*(1-blend)+smooth*blend)


def render(rgb, mask, width, height, fill, shadow, sharpness=.75, denoise=0, max_aspect=None, detail_scale=1, include_transparent=True, frame=None, source_width=None, origin=(0,0), shadow_layer=None, shadow_method="classic"):
    ys,xs=np.where(mask>.5)
    if not len(xs): raise ValueError('Empty subject mask')
    x0,x1=int(xs.min()),int(xs.max()+1);y0,y1=int(ys.min()),int(ys.max()+1)
    bw,bh=x1-x0,y1-y0
    pad=int(max(bw,bh)*.06)+4
    l=max(0,x0-pad);t=max(0,y0-pad);r=min(rgb.shape[1],x1+pad);b=min(rgb.shape[0],y1+pad)
    fg=rgb[t:b,l:r];alpha=mask[t:b,l:r]
    background=np.ones_like(fg)
    density=None
    wide_density=None
    if shadow:
        if shadow_method=='classic':
            dist=cv2.distanceTransform((alpha<.5).astype(np.uint8),cv2.DIST_L2,5)
            yy=np.arange(t,b)[:,None]
            support=np.exp(-(dist/(bw*.033+1))**2)
            support*=np.clip((yy-(y1-bh*.12))/(bh*.09),0,1)
            lum=fg@np.array([.2126,.7152,.0722],np.float32)
            density=np.clip(1-lum,0,.24)*support*.65*float(shadow)
        elif shadow_method=='local':
            from contact_shadow import recover_shadow,shadow_bounds,cap
            sl,st,sr,sb=shadow_bounds(mask)
            wide_density=(recover_shadow(rgb[st:sb,sl:sr],mask[st:sb,sl:sr],bw,shadow) if shadow_layer is None
                          else cap(shadow_layer*float(shadow)))
            density=wide_density[t-st:b-st,l-sl:r-sl]
        else:raise ValueError('Unknown shadow method')
        if density.ndim==2:background-=density[:,:,None]
        else:background-=density
    composite_fg=fg
    if shadow and shadow_method=='local':
        composite_fg=contact_foreground(fg,alpha,density)
    composite=composite_fg*alpha[:,:,None]+background*(1-alpha[:,:,None])
    scale=min(width*fill/bw,height*fill/bh)
    if max_aspect:
        scale=min(scale,width*fill/max_aspect/bh)
    center=[(x0+x1)/2,(y0+y1)/2]
    if frame is not None:
        from framing import placement
        full_bbox=[x0+origin[0],y0+origin[1],x1+origin[0],y1+origin[1]]
        shared,center=placement(frame,full_bbox,source_width or rgb.shape[1],origin)
        if shared is not None:scale=shared
    rw,rh=max(1,round((r-l)*scale)),max(1,round((b-t)*scale))
    ox=round(width/2-(center[0]-l)*scale)
    oy=round(height/2-(center[1]-t)*scale)
    canvas=Image.new('RGB',(width,height),'white')
    if wide_density is not None:
        # Wider floor only: keep the original foreground/PNG sampling grid.
        floor=np.clip(1-wide_density,0,1)
        if floor.ndim==2:floor=np.repeat(floor[:,:,None],3,axis=2)
        floor_image=pil(floor)
        floor_image=floor_image.resize((max(1,round((sr-sl)*scale)),max(1,round((sb-st)*scale))),Image.Resampling.LANCZOS)
        canvas.paste(floor_image,(round(width/2-(center[0]-sl)*scale),round(height/2-(center[1]-st)*scale)))
    crop=pil(composite).resize((rw,rh),Image.Resampling.LANCZOS)
    if denoise>0:crop=_denoise(crop,denoise,detail_scale)
    if sharpness>0:
        crop=crop.filter(ImageFilter.UnsharpMask(radius=max(.3,.85*detail_scale),percent=round(sharpness*100),threshold=3))
    canvas.paste(crop,(ox,oy))
    if not include_transparent:return canvas,None,[x0,y0,x1,y1]
    # Apply the same detail treatment to the transparent export's RGB pixels.
    foreground=pil(fg).resize((rw,rh),Image.Resampling.LANCZOS)
    if denoise>0:foreground=_denoise(foreground,denoise,detail_scale)
    if sharpness>0:
        foreground=foreground.filter(ImageFilter.UnsharpMask(radius=max(.3,.85*detail_scale),percent=round(sharpness*100),threshold=3))
    rgba=foreground.convert('RGBA')
    rgba.putalpha(Image.fromarray((alpha*255).astype('uint8')).resize((rw,rh),Image.Resampling.LANCZOS))
    transparent=Image.new('RGBA',(width,height));transparent.paste(rgba,(ox,oy))
    return canvas,transparent,[x0,y0,x1,y1]

def contact_sheet(records,out):
    if not records:return
    columns=6;tw=300;th=330
    sheet=Image.new('RGB',(columns*tw,((len(records)+columns-1)//columns)*th),'#eef0f3')
    draw=ImageDraw.Draw(sheet)
    for i,record in enumerate(records):
        im=Image.open(record['jpeg']);im.thumbnail((tw-12,th-34))
        x=i%columns*tw;y=i//columns*th
        sheet.paste(im,(x+(tw-im.width)//2,y+(th-34-im.height)//2))
        draw.text((x+10,y+th-25),f'{i+1:02d}  {record["name"][:8]}',fill='#343942')
    sheet.save(out/'contact-sheet.jpg',quality=92,icc_profile=ICC)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs',nargs='*',type=Path)
    p.add_argument('--input-list',type=Path)
    p.add_argument('--adjustment-recipes',type=Path,help='Per-photo learned tone/detail settings and manual provenance')
    p.add_argument('--photo-recipes',type=Path,help='Explicit packaging-photo recipes')
    p.add_argument('--output',type=Path,default=ROOT/'output'/'catalog')
    p.add_argument('--metadata',type=Path,help='Private processing files, outside image deliverables')
    p.add_argument('--framing',choices=['centered','fixed','both'],default='centered')
    p.add_argument('--frame-plan',type=Path,help='Reuse a saved batch framing plan so a re-rendered photo aligns with the shoot it belongs to')
    p.add_argument('--centered-scale',choices=['shared','independent'],default='shared',help='Centred crops: one scale for the shoot, or each photo filled on its own')
    p.add_argument('--size',type=int,default=2400)
    p.add_argument('--aspect',choices=['square','landscape','portrait','wide'],default='square')
    p.add_argument('--exposure',type=float,default=.65)
    p.add_argument('--warmth',type=float,default=.25)
    p.add_argument('--fill',type=float,default=.84)
    p.add_argument('--clean-package-code',action='store_true',help='Remove the vertical code beside a packaging QR')
    p.add_argument('--no-shadow',action='store_true')
    p.add_argument('--shadow',type=float,default=2,help='Contact shadow strength: 0 to 2')
    p.add_argument('--shadow-method',choices=['classic','local'],default='classic')
    p.add_argument('--shadow-detection',choices=['dark','soft'],default='dark')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--replace',action='store_true',
                   help='Overwrite selected photos in an existing export; keep other photos and the saved framing plan')
    p.add_argument('--owner-pid',type=int)
    p.add_argument('--contrast',type=float,default=0,help='Contrast adjustment: -1 to 1')
    p.add_argument('--whites',type=float,default=0,help='Bring the brightest tones down or up: -1 to 1')
    p.add_argument('--sharpness',type=float,default=.75)
    p.add_argument('--denoise',type=float,default=0)
    p.add_argument('--mask-mode',choices=['efficient','detailed'],default='efficient')
    p.add_argument('--mask-backend',choices=['onnx','mps'],default='mps',
                   help='mps: BiRefNet on Apple Silicon GPU (default). onnx: rembg BiRefNet on CPU.')
    p.add_argument('--small-parts',action='store_true',help='Look harder for small loose pieces such as studs, bones and tiles')
    p.add_argument('--recover-white',action='store_true',help='Recover faint attached white parts on difficult photos')
    p.add_argument('--recover-dark',action='store_true',help='Recover dark parts near the floor that were taken for shadow')
    p.add_argument('--recover-parts',action='store_true',help='Recover flat neutral parts and thin dark tools with extra local detection')
    args=p.parse_args()
    configure_backend(args.mask_backend)
    if not 512<=args.size<=6000 or not .5<=args.fill<=.94 or not -2<=args.exposure<=2 or not -1<=args.warmth<=1:
        p.error('Settings out of range')
    if not 0<=args.sharpness<=2 or not 0<=args.denoise<=2:p.error('Detail settings out of range')
    if not -1<=args.contrast<=1 or not 0<=args.shadow<=2:p.error('Contrast or shadow out of range')
    if not -1<=args.whites<=1:p.error('Whites out of range')
    shadow_strength=0 if args.no_shadow else args.shadow
    inputs=args.inputs
    if args.input_list:inputs += [Path(x) for x in json.loads(args.input_list.read_text())]
    paths=[]
    for path in inputs:
        paths.extend(sorted(x for x in path.iterdir() if x.suffix.lower() in EXTENSIONS) if path.is_dir() else [path])
    paths=list(dict.fromkeys(x.resolve() for x in paths))
    if not paths:p.error('Choose at least one RAW image or folder')
    # Deliverables keep exactly the source file's name. Two different sources
    # sharing a name would silently overwrite each other's export, so such
    # photos are excluded up front rather than picking a name for them.
    by_stem={}
    for path in paths:by_stem.setdefault(path.stem,[]).append(path)
    duplicate_names={stem:group for stem,group in by_stem.items() if len(group)>1}
    if duplicate_names:
        paths=[path for path in paths if len(by_stem[path.stem])==1]
        if not paths:p.error('Every selected photo shares its name with another selected photo. Rename the files so each is unique, then export again.')
    from photo_recipes import normalize,options,canvas as recipe_canvas,modes as recipe_modes
    try:photo_recipes=normalize(json.loads(args.photo_recipes.read_text()),paths) if args.photo_recipes else {}
    except (OSError,ValueError) as exc:p.error(str(exc))
    configs={str(path):options(args,path,photo_recipes) for path in paths}
    from recipe_learning import normalize as normalize_adjustments
    try:adjustments=normalize_adjustments(json.loads(args.adjustment_recipes.read_text()),paths) if args.adjustment_recipes else {}
    except (OSError,ValueError,TypeError) as exc:p.error(str(exc))
    for source,adjustment in adjustments.items():
        for key,value in adjustment['values'].items():setattr(configs[source],key,value)
        for key,value in adjustment.get('options',{}).items():setattr(configs[source],key,value=='true' if key in ('small_parts','recover_parts','recover_white','recover_dark') else value)
    out=args.output.resolve()
    metadata=(args.metadata or ROOT/'work'/'shoots'/(out.name+'-'+hashlib.sha256(str(out).encode()).hexdigest()[:8])).resolve()
    if metadata==out or out in metadata.parents:p.error('Processing folder must be outside the image output folder')
    # Reusing a populated destination can mix old angles or recipes into delivery,
    # unless --resume continues an interrupted shoot or --replace updates selected photos.
    resume_state=None
    replace_state=None
    locked_frames=None
    if args.resume and args.replace:p.error('Use either --resume or --replace, not both')
    if args.replace:
        if not (metadata/'manifest.json').exists():
            p.error('Replace needs an existing shoot. Open that export’s processing folder, or export into a new folder.')
        try:replace_state=json.loads((metadata/'manifest.json').read_text())
        except (OSError,ValueError):p.error('Cannot read the saved shoot; choose a new export folder')
        if replace_state.get('output')!=str(out):
            p.error('This processing folder belongs to a different export path')
    elif args.resume and (metadata/'manifest.json').exists():
        try:resume_state=json.loads((metadata/'manifest.json').read_text())
        except (OSError,ValueError):p.error('Cannot read the saved batch; choose a new export folder')
    if out.exists() and any(out.iterdir()) and resume_state is None and replace_state is None:
        p.error('Choose an empty output folder for this shoot')
    out.mkdir(parents=True,exist_ok=True);metadata.mkdir(parents=True,exist_ok=True)
    import fcntl
    batch_lock=(metadata/'batch.lock').open('w')
    fcntl.flock(batch_lock,fcntl.LOCK_EX)
    (metadata/'before').mkdir(exist_ok=True)
    modes=['centered','fixed'] if args.framing=='both' else [args.framing]
    folders={'centered':'centered','fixed':'fixed-frame'}
    for mode in set(m for cfg in configs.values() for m in recipe_modes(cfg)):
        for kind in ['jpeg','transparent']:(out/folders[mode]/kind).mkdir(parents=True,exist_ok=True)
    width=height=args.size
    if args.aspect=='landscape':height=round(width*2/3)
    if args.aspect=='portrait':width=round(height*4/5)
    if args.aspect=='wide':height=round(width*9/16)
    settings={'size':args.size,'aspect':args.aspect,'exposure':args.exposure,'warmth':args.warmth,'fill':args.fill,'shadow':shadow_strength,'contrast':args.contrast,'sharpness':args.sharpness,'denoise':args.denoise,'mask_mode':args.mask_mode,'tone':'catalog-v2','edge_version':10,'framing':args.framing,'shadow_method':args.shadow_method,'shadow_version':10 if args.shadow_method=='local' else 2}
    # Only recorded when on, so batches saved before this option still resume.
    if args.small_parts:settings['small_parts']=True
    if args.recover_parts:settings['recover_parts']=True
    if args.recover_white:settings['recover_white']=True
    if any(cfg.recover_white for cfg in configs.values()):
        from white_recovery import WHITE_VERSION
        settings['white_version']=WHITE_VERSION
    if args.recover_dark:settings['recover_dark']=True
    if any(cfg.recover_dark for cfg in configs.values()):
        from dark_recovery import DARK_VERSION
        settings['dark_version']=DARK_VERSION
    settings['mask_backend']=args.mask_backend
    manifest={'version':2,'settings':settings,'total':len(paths),'records':[],'errors':[],
              'output':str(out),'metadata':str(metadata)}
    for stem,group in duplicate_names.items():
        for path in group:
            message=f'Skipped: another selected photo is also named "{stem}". Rename the files so each is unique, then export again.'
            manifest['errors'].append({'source':str(path),'error':message})
            emit(stage='error',name=path.name,message=message)
    # Only recorded when on, so batches saved before this option still resume.
    if args.centered_scale=='independent':settings['centered_scale']='independent'
    if args.whites:settings['whites']=args.whites
    if args.shadow_detection!='dark':settings['shadow_detection']=args.shadow_detection
    if adjustments:settings['adjustment_recipes']=adjustments
    if args.clean_package_code:settings.update(package_cleanup=True,package_cleanup_version=2)
    if args.photo_recipes:settings.update(photo_recipes=photo_recipes,package_cleanup_version=2)
    from masking import source_key,cached_mask,get_mask,mask_diagnostics
    def fingerprint(path):
        try:return source_key(path)
        except OSError:return 'unavailable'
    source_keys={str(path):fingerprint(path) for path in paths}
    selected={str(path) for path in paths}
    if replace_state is not None:
        prior_sources=replace_state.get('sources') or []
        if not prior_sources:p.error('The saved shoot has no source list to update')
        unknown=sorted(selected-set(prior_sources))
        if unknown:p.error('Selected photos are not part of this export: '+', '.join(Path(u).name for u in unknown[:5]))
        prior_settings=replace_state.get('settings') or {}
        framing_keys=('size','aspect','fill','framing','centered_scale')
        framing_changed=any(prior_settings.get(k)!=settings.get(k) for k in framing_keys)
        if framing_changed and selected!=set(prior_sources):
            p.error('Framing settings changed. Process the whole shoot so every angle matches.')
        def record_ok(record):
            try:
                for pair in record['variants'].values():
                    for file in pair.values():
                        with Image.open(file) as im:
                            im.load()
                            if im.size!=(record['width'],record['height']):raise ValueError('Changed export size')
                return True
            except (OSError,ValueError,KeyError):return False
        kept=[r for r in replace_state.get('records',[]) if r.get('source') not in selected and record_ok(r)]
        kept_errors=[e for e in replace_state.get('errors',[]) if e.get('source') not in selected]
        prior_keys=dict(replace_state.get('source_keys') or {})
        prior_keys.update(source_keys)
        prior_prepared=dict(replace_state.get('prepared') or {})
        for source in selected:prior_prepared.pop(source,None)
        if not framing_changed and replace_state.get('frames'):
            locked_frames=replace_state['frames']
        manifest.update(sources=prior_sources,source_keys=prior_keys,records=kept,errors=kept_errors,
                        prepared=prior_prepared,frames=replace_state.get('frames') or {},
                        total=len(prior_sources),settings=settings,output=str(out),metadata=str(metadata))
    elif resume_state is not None:
        if (resume_state.get('settings')!=settings or resume_state.get('output')!=str(out)
            or resume_state.get('source_keys')!=source_keys or resume_state.get('sources')!=[str(p) for p in paths]):
            p.error('The recipe or source files changed. Start a new batch to keep framing consistent.')
        valid=[]
        for record in resume_state.get('records',[]):
            try:
                for pair in record['variants'].values():
                    for file in pair.values():
                        with Image.open(file) as im:
                            im.load()
                            if im.size!=(record['width'],record['height']):raise ValueError('Changed export size')
                valid.append(record)
            except (OSError,ValueError,KeyError):pass
        manifest['records']=valid
        # Keep the last committed framing plan through an interrupted mask
        # preflight; otherwise the next resume would re-render valid photos.
        manifest['frames']=resume_state.get('frames',{})
        manifest.update(sources=[str(p) for p in paths],source_keys=source_keys)
    else:
        manifest.update(sources=[str(p) for p in paths],source_keys=source_keys)
    start=time.time()

    from framing import mask_bounds,batch_frame
    from progress import ProgressReporter
    from masking import needs_recovery,edge_ready
    estimates=[]
    for path in paths:
        cfg=configs[str(path)];mask_mode=cfg.mask_mode
        configure_backend(cfg.mask_backend)
        try:cached=('-edges-v' in cached_mask(path,mask_mode,cfg.small_parts,cfg.recover_parts,cfg.recover_white,cfg.recover_dark).stem) and not needs_recovery(path,mask_mode,cfg.small_parts,cfg.recover_parts,cfg.recover_white,cfg.recover_dark)
        except OSError:cached=False
        base=3 if args.mask_backend=='mps' else (24 if args.mask_mode=='efficient' else 48)
        estimates.append(.6 if cached else base*(1.5 if cfg.small_parts else 1))
    # Two bounded-memory stages: find every mask first, then render straight
    # from RAW at the final scale. No intermediate full-size RGB copies on disk
    # and no second resizing of exports. Cached masks need no RAW preflight.
    n=len(paths)
    def progress_event(**event):
        if args.owner_pid:
            try:os.kill(args.owner_pid,0)
            except ProcessLookupError:os._exit(75)
        if event.get('stage')=='progress':
            event['photo_stage']='preparation' if event['index']<n else 'rendering'
            event['index']%=n;event['photo_number']=event['index']+1
            event['completed']=len(manifest['records']);event['total']=n
        emit(**event)
    reporter=ProgressReporter([p.name for p in paths]*2,estimates+[5*len(modes)+args.denoise*2 for _ in paths],progress_event).start()
    prepared={}
    for i,path in enumerate(paths):
        cfg=configs[str(path)]
        configure_backend(cfg.mask_backend)
        reporter.begin_photo(i)
        reporter.phase('Checking masks for shared framing',.05)
        try:
            key=source_key(path);cache=cached_mask(path,cfg.mask_mode,cfg.small_parts,cfg.recover_parts,cfg.recover_white,cfg.recover_dark)
            mask=None;info={};was_cached=False;method='cached'
            if edge_ready(cache) and cache.exists() and not needs_recovery(path,cfg.mask_mode,cfg.small_parts,cfg.recover_parts,cfg.recover_white,cfg.recover_dark):
                try:
                    with Image.open(cache) as im:mask=np.asarray(im).astype(np.float32)/65535
                    if mask.ndim!=2 or not np.isfinite(mask).all() or not np.any(mask>.5):mask=None
                except (OSError,ValueError):mask=None
            if mask is None:
                rgb=decode(path)
                mask,was_cached,method=get_mask(rgb,path,cfg.mask_mode,reporter.phase,info,cfg.small_parts,cfg.recover_parts,cfg.recover_white,cfg.recover_dark)
                del rgb
            else:
                was_cached=True;info=mask_diagnostics(path,cfg.mask_mode,cfg.small_parts,cfg.recover_parts,cfg.recover_white,cfg.recover_dark)
            h,w=mask.shape;bbox=mask_bounds(mask)
            prepared[str(path)]=dict(id=key,bbox=bbox,source_size=[w,h],mask_cached=was_cached,
                                     mask_method=method,mask_diagnostics=info)
            del mask
            if replace_state is not None:
                prepared_all=dict(manifest.get('prepared') or {});prepared_all.update(prepared)
                manifest['prepared']=prepared_all
            else:
                manifest['prepared']=prepared
            reporter.complete_photo()
        except Exception as exc:
            reporter.complete_photo(adapt=False)
            manifest['errors'].append({'source':str(path),'error':str(exc)})
            emit(stage='error',name=path.name,message=str(exc))
        write_manifest(metadata,manifest)
    regular=[v for path,v in prepared.items() if path not in photo_recipes]
    plans={mode:batch_frame(regular,width,height,args.fill,mode,args.centered_scale=='independent') for mode in modes} if regular else {}
    if locked_frames is not None:
        missing=[mode for mode in modes if mode not in locked_frames]
        if missing:p.error('The saved framing plan has no '+', '.join(missing)+' entry')
        plans={mode:locked_frames[mode] for mode in modes}
        if 'packaging' in locked_frames:plans['packaging']=locked_frames['packaging']
    elif args.frame_plan:
        saved=json.loads(args.frame_plan.read_text())
        missing=[mode for mode in modes if mode not in saved]
        if missing:p.error('The saved framing plan has no '+', '.join(missing)+' entry')
        plans={mode:saved[mode] for mode in modes}
    package_plans={}
    custom_layout={path for path,cfg in configs.items() if any(getattr(cfg,k)!=getattr(args,k) for k in ('size','aspect','fill','framing','centered_scale'))}
    for path in set(photo_recipes)|custom_layout:
        if path not in prepared:continue
        cfg=configs[path];pw,ph=recipe_canvas(cfg)
        refs=[prepared[path]] if path in photo_recipes else regular
        package_plans[path]={mode:batch_frame(refs,pw,ph,cfg.fill,mode,cfg.centered_scale=='independent') for mode in recipe_modes(cfg)}
    if package_plans:
        prior_pack=locked_frames.get('packaging') if isinstance((locked_frames or {}).get('packaging'),dict) else {}
        merged=dict(prior_pack);merged.update(package_plans);plans['packaging']=merged
    if resume_state and resume_state.get('frames')!=plans:
        # A recovered missing view can enlarge the union: rerender all views
        # together rather than silently mixing two framing plans.
        manifest['records']=[]
    manifest['frames']=plans
    completed_sources={r['source'] for r in manifest['records']}
    write_manifest(metadata,manifest)
    for i,path in enumerate(paths):
        cfg=configs[str(path)];width,height=recipe_canvas(cfg);modes=recipe_modes(cfg)
        configure_backend(cfg.mask_backend)
        shadow_strength=0 if cfg.no_shadow else cfg.shadow
        frame_plans=package_plans.get(str(path),plans)
        reporter.begin_photo(n+i)
        if str(path) in completed_sources:
            reporter.complete_photo(adapt=False);continue
        item=prepared.get(str(path))
        if item is None:
            reporter.complete_photo(adapt=False);continue
        emit(stage='processing',index=i,total=n,name=path.name,message=f'Rendering photo {i+1} of {n}')
        try:
            rgb=decode(path)
            if [rgb.shape[1],rgb.shape[0]]!=item['source_size']:raise ValueError('Mask dimensions changed; regenerate the mask before exporting')
            name=path.stem
            before=metadata/'before'/f'{name}.jpg'
            prev=pil(rgb);prev.thumbnail((1600,1600));prev.save(before,quality=94,icc_profile=ICC)
            reporter.phase('Correcting color',.15)
            with Image.open(cached_mask(path,cfg.mask_mode,cfg.small_parts,cfg.recover_parts,cfg.recover_white,cfg.recover_dark)) as im:mask=np.asarray(im).astype(np.float32)/65535
            if cfg.shadow_detection=='soft':
                from contact_shadow import soften_shadow_mask
                mask=soften_shadow_mask(rgb,mask)
                if cfg.recover_white or cfg.recover_dark:
                    from masking import addition_protection
                    mask=np.maximum(mask,addition_protection(path,cfg.mask_mode,cfg.small_parts,cfg.recover_parts,cfg.recover_white,cfg.recover_dark))
            from contact_shadow import shadow_layer
            package_info=None
            if cfg.clean_package_code:
                from packaging_cleanup import clean_package_code
                reporter.phase('Cleaning packaging code',.18)
                rgb,package_info=clean_package_code(rgb)
            soft_floor=shadow_layer(rgb,mask,'soft') if shadow_strength and cfg.shadow_method=='local' and cfg.shadow_detection=='soft' else None
            corrected,gains=correct(rgb,cfg.exposure,cfg.warmth,cfg.contrast,whites=cfg.whites)
            del rgb
            # Strut repair already ran once, in the cached mask, on the fixed mask
            # view. Repeating it here on the recipe-darkened image drew extra
            # 1 px bars that the preview never showed.
            floor_shadow=soft_floor if cfg.shadow_detection=='soft' else (shadow_layer(corrected,mask) if shadow_strength and cfg.shadow_method=="local" else None)
            variants={}
            for j,mode in enumerate(modes):
                reporter.phase('Rendering '+('centred crop' if mode=='centered' else 'fixed frame'),.25+.35*j)
                canvas,transparent,bbox=render(corrected,mask,width,height,cfg.fill,shadow_strength,cfg.sharpness,cfg.denoise,frame=frame_plans[mode],shadow_layer=floor_shadow,shadow_method=cfg.shadow_method)
                jpg=out/folders[mode]/'jpeg'/f'{name}.jpg';png=out/folders[mode]/'transparent'/f'{name}.png'
                jpg_tmp=jpg.with_suffix('.jpg.tmp');png_tmp=png.with_suffix('.png.tmp')
                canvas.save(jpg_tmp,format='JPEG',quality=96,subsampling=0,icc_profile=ICC)
                transparent.save(png_tmp,format='PNG',icc_profile=ICC)
                jpg_tmp.replace(jpg);png_tmp.replace(png)
                variants[mode]={'jpeg':str(jpg),'png':str(png)}
                del canvas,transparent
            del corrected,mask,floor_shadow
            flags=list(item['mask_diagnostics'].get('review',[]))
            if package_info and package_info['status']=='review':flags.append(package_info['message'])
            x,y,r,b=item['bbox'];w,h=item['source_size']
            if min(x,y,w-r,h-b)<4:flags.append('Subject touches source edge')
            duration=reporter.complete_photo()
            record=dict(item,name=path.name,source=str(path),before=str(before),width=width,height=height,
                        wb_gains=gains,review=flags,seconds=round(duration,3),variants=variants,
                        framing=frame_plans[modes[0]],**variants[modes[0]])
            if package_info is not None:record.update(package_cleanup=package_info,photo_type='packaging')
            # Every successful photo carries its exact recipe, even when a partial
            # re-export later changes the manifest's batch-level settings.
            from recipe_learning import FIELDS, record_export
            import uuid
            effective={k:getattr(cfg,k) for k in FIELDS}
            effective.update({k:getattr(cfg,k) for k in ('size','aspect','fill','framing','centered_scale','shadow_method','mask_mode','mask_backend','small_parts','recover_parts','recover_white','recover_dark')})
            effective['shadow']=shadow_strength
            effective['shadow_detection']=cfg.shadow_detection
            record.update(effective_recipe=effective,processed_at=time.time(),processing_event=str(uuid.uuid4()))
            if str(path) in adjustments:record['adjustment']=adjustments[str(path)]
            if str(path) in photo_recipes:record.update(photo_type='packaging',recipe=photo_recipes[str(path)])
            manifest['records'].append(record)
            write_manifest(metadata,manifest)
            try:record_export(record,effective,adjustments.get(str(path)),ROOT)
            except Exception as exc:
                # Exports stay valid if history storage fails. The next import
                # recovers this event from the committed manifest.
                emit(stage='learning_warning',message='Recipe history will retry: '+str(exc))
            emit(stage='photo_done',index=i+1,total=n,record=record,message=f'Exported {i+1} of {n}')
        except Exception as exc:
            reporter.complete_photo(adapt=False)
            # Remove this photo's partial deliverables if either variant failed.
            for mode in modes:
                for kind,ext in [('jpeg','jpg'),('transparent','png')]:
                    (out/folders[mode]/kind/f'{path.stem}.{ext}').unlink(missing_ok=True)
            manifest['errors'].append({'source':str(path),'error':str(exc)})
            emit(stage='error',name=path.name,message=str(exc))
        write_manifest(metadata,manifest)
    manifest['records'].sort(key=lambda r:manifest['sources'].index(r['source']))
    reporter.begin_packaging()
    contact_sheet(manifest['records'],metadata)
    manifest['elapsed_seconds']=round(time.time()-start,1)
    write_manifest(metadata,manifest)
    reporter.finish()
    emit(stage='done',output=str(out),metadata=str(metadata),count=len(manifest['records']),errors=len(manifest['errors']),seconds=manifest['elapsed_seconds'])
    return 1 if manifest['errors'] else 0

if __name__=='__main__':
    # Helpers import studio; share this module instead of creating a second
    # copy with a separately owned native inference session.
    sys.modules['studio']=sys.modules[__name__]
    try:
        exit_code=main()
    finally:
        # Release inference sessions before Python tears down native modules.
        SESSION=None
        SESSION_BACKEND=None
        import gc
        gc.collect()
    sys.exit(exit_code)
