"""Explicit packaging assignments and their independent recipes."""
from argparse import Namespace
from pathlib import Path
import math

NUMBERS={'size':(512,6000),'exposure':(-2,2),'warmth':(-1,1),'contrast':(-1,1),
         'fill':(.5,.94),'sharpness':(0,2),'denoise':(0,2),'shadow':(0,2),'whites':(-1,1)}
CHOICES={'aspect':('square','landscape','portrait','wide'),'framing':('centered','fixed','both'),
         'mask_mode':('efficient','detailed'),'shadow_method':('classic','local'),
         'mask_backend':('onnx','mps')}
CHOICES['shadow_detection']=('dark','soft')
CHOICES['centered_scale']=('shared','independent')

def normalize(mapping, sources):
    if not isinstance(mapping,dict):raise ValueError('Photo recipes must be an object')
    allowed={str(Path(p).resolve()) for p in sources};result={}
    for path,recipe in mapping.items():
        path=str(Path(path).resolve())
        if path not in allowed:continue
        if not isinstance(recipe,dict):raise ValueError('Invalid packaging recipe')
        clean={}
        for key,(low,high) in NUMBERS.items():
            if key in recipe:
                v=recipe[key]
                if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not low<=v<=high:raise ValueError('Invalid packaging '+key)
                if key=='size' and int(v)!=v:raise ValueError('Invalid packaging size')
                clean[key]=int(v) if key=='size' else v
        for key,choices in CHOICES.items():
            if key in recipe:
                if recipe[key] not in choices:raise ValueError('Invalid packaging '+key)
                clean[key]=recipe[key]
        clean['package_cleanup']=True
        for option in ('small_parts','recover_parts','recover_white','recover_dark'):
            if option in recipe:
                if not isinstance(recipe[option],bool):raise ValueError('Invalid packaging '+option)
                clean[option]=recipe[option]
        result[path]=clean
    return result

def options(base, path, mapping):
    # Packaging defaults stand alone: no missing field inherits a LEGO recipe.
    values=dict(vars(base))
    if str(path) in mapping:
        values.update(size=2400,aspect='square',framing='centered',mask_mode='efficient',mask_backend='mps',
                      exposure=.65,warmth=.25,contrast=0,fill=.84,sharpness=.75,denoise=0,
                      shadow=2,shadow_method='classic',shadow_detection='dark',no_shadow=False,whites=0,small_parts=False,recover_parts=False,recover_white=False,recover_dark=False)
        values.update({k:v for k,v in mapping[str(path)].items() if k!='package_cleanup'})
        values['clean_package_code']=True
    elif mapping or getattr(base,'photo_recipes',None):
        values['clean_package_code']=False
    return Namespace(**values)

def canvas(cfg):
    w=h=cfg.size
    if cfg.aspect=='landscape':h=round(w*2/3)
    if cfg.aspect=='portrait':w=round(h*4/5)
    if cfg.aspect=='wide':h=round(w*9/16)
    return w,h

def modes(cfg):return ['centered','fixed'] if cfg.framing=='both' else [cfg.framing]
