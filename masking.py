"""Reusable subject masks and a fast guide for precise one-pass segmentation."""
import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scene_detection import (scene_evidence,evidence_boxes,union_boxes,padded_box,
                             clipped_sides,missing_regions,partial_dark_parts)

ROOT=Path(__file__).resolve().parent
PARTS_VERSION=2
# The subject model, edge cleanup and piece recovery always look at one fixed
# rendering of the photo, never at the user's recipe. Learned recipes drifted to
# exposure 0.14-0.30 (default 0.65); on that darker view the model and the
# brightness-based cleanup took floor shadow for dark plastic, and because masks
# are cached per photo, whichever recipe ran first fixed the cutout for good.
MASK_VIEW_VERSION=1

def mask_view(rgb):
    from studio import correct
    return correct(rgb,.65,.25)[0]

def _backend_tag():
    # onnx keeps the historical filenames so existing caches stay valid.
    from studio import MASK_BACKEND
    return '' if MASK_BACKEND=='onnx' else f'-{MASK_BACKEND}'

def source_key(path):
    from source_identity import source_key as content_key
    return content_key(path,ROOT)

def base_mask(path, mode='efficient', small_parts=False, recover_parts=False, recover_white=False, recover_dark=False):
    key=source_key(path)
    # Small-part masks are cached apart: the option changes what counts as a
    # piece, so a mask found without it must never be reused once it is on.
    # Engine tags keep Metal and ONNX masks from colliding.
    fine='-fine' if small_parts else ''
    engine=_backend_tag()+f'-view{MASK_VIEW_VERSION}'+(f'-parts-v{PARTS_VERSION}' if recover_parts else '')
    if recover_white:
        from white_recovery import WHITE_VERSION
        engine+=f'-white-v{WHITE_VERSION}-e{EDGE_VERSION}'
    # Appended after the white tag, so white-only caches keep their filenames.
    if recover_dark:
        from dark_recovery import DARK_VERSION
        engine+=f'-dark-v{DARK_VERSION}'+('' if recover_white else f'-e{EDGE_VERSION}')
    detailed=ROOT/'work'/'masks'/f'{key}{fine}{engine}-coverage-v3.png'
    guided=ROOT/'work'/'masks'/f'{key}{fine}{engine}-guided-coverage-v3.png'
    if detailed.exists():return detailed
    return guided if mode=='efficient' else detailed

EDGE_VERSION=10

def _edge_stem(name):
    stem=name
    for version in range(EDGE_VERSION,0,-1):
        tag=f'-edges-v{version}'
        if stem.endswith(tag):return stem[:-len(tag)]
    return stem

def edge_ready(path):
    return path.stem.endswith(f'-edges-v{EDGE_VERSION}')

def cached_mask(path,mode='efficient',small_parts=False,recover_parts=False,recover_white=False,recover_dark=False):
    base=base_mask(path,mode,small_parts,recover_parts,recover_white,recover_dark)
    # v5 bridges a strut only on photographed evidence; v4 kept dark contact
    # antialiasing; v3 repaired white struts; v2 cleaned floor.
    for version in range(EDGE_VERSION,1,-1):
        refined=base.with_name(base.stem+f'-edges-v{version}.png')
        if refined.exists():return refined
    return base

def mask_diagnostics(path,mode='efficient',small_parts=False,recover_parts=False,recover_white=False,recover_dark=False):
    try:return json.loads(base_mask(path,mode,small_parts,recover_parts,recover_white,recover_dark).with_suffix('.json').read_text())
    except (OSError,ValueError):return {}

def finish_edges(rgb,mask,cache):
    from edge_refinement import refine_colored_edges,repair_white_bridges,smooth_contact_edges,drop_floor_islands
    if edge_ready(cache):return mask
    source=mask
    # Derived caches may already have hardened dark edges; rebuild from the AI mask.
    if '-edges-v' in cache.stem:
        base_file=cache.with_name(_edge_stem(cache.stem)+cache.suffix)
        if base_file.exists():
            try:
                with Image.open(base_file) as image:
                    loaded=np.asarray(image).astype(np.float32)/65535
                if loaded.shape==mask.shape and np.isfinite(loaded).all():source=loaded
            except (OSError,ValueError):pass
    refined=drop_floor_islands(rgb,smooth_contact_edges(rgb,repair_white_bridges(rgb,refine_colored_edges(rgb,source))))
    target=cache.with_name(_edge_stem(cache.stem)+f'-edges-v{EDGE_VERSION}.png')
    temporary=target.with_suffix('.tmp')
    Image.fromarray((refined*65535).astype(np.uint16)).save(temporary,format='PNG')
    temporary.replace(target)
    return refined


def _vision_guide(small):
    from studio import pil,components
    helper=ROOT/'native'/'foreground-guide'
    if helper.exists():
        try:
            with tempfile.TemporaryDirectory(dir=ROOT/'work') as temp:
                source=Path(temp)/'guide.jpg';target=Path(temp)/'mask.png'
                small.save(source,quality=95)
                subprocess.run([str(helper),str(source),str(target)],check=True,capture_output=True,timeout=45)
                mask=np.asarray(Image.open(target)).astype(np.float32)/255
                if mask.ndim==3:mask=mask[:,:,0]
            _,box=components(mask)
            x,y,w,h=box
            return (x,y,x+w,y+h)
        except (OSError,ValueError,subprocess.SubprocessError):
            pass
    return None


def _guide(rgb):
    """Compatibility helper; both AI and full-scene evidence determine bounds."""
    from studio import pil
    small=pil(rgb);small.thumbnail((1600,1600))
    boxes=evidence_boxes(scene_evidence(small))
    vision=_vision_guide(small)
    if vision:boxes.append(vision)
    l,t,r,b=union_boxes(boxes)
    return (l,t,r-l,b-t),small.size,.12


def _predict_region(rgb,box):
    from studio import pil,session
    l,t,r,b=box
    close=pil(rgb[t:b,l:r]);close.thumbnail((1600,1600))
    return np.asarray(session().predict(close)[0]).astype(np.float32)/255


def _place(mask,fine,box):
    l,t,r,b=box
    alpha=cv2.resize(fine,(r-l,b-t),interpolation=cv2.INTER_LINEAR)
    mask[t:b,l:r]=np.maximum(mask[t:b,l:r],np.clip((alpha-.035)/.93,0,1))


def build_mask(rgb,mode='efficient',on_phase=None,small_parts=False):
    """One full-quality model pass normally; bounded recovery only if needed."""
    from studio import pil,components,session
    started=time.monotonic()
    h,w=rgb.shape[:2]
    small=pil(rgb);small.thumbnail((1600,1600))
    gw,gh=small.size
    if on_phase:on_phase('Checking the whole photo for pieces',.18)
    seeds=scene_evidence(small,small_parts)
    boxes=evidence_boxes(seeds)
    passes=0
    if mode=='detailed':
        coarse=np.asarray(session().predict(small)[0]).astype(np.float32)/255
        passes+=1
        try:
            _,(x,y,bw,bh)=components(coarse)
            boxes.append((x,y,x+bw,y+bh))
        except ValueError:pass
    else:
        vision=_vision_guide(small)
        if vision:boxes.append(vision)
    guide=union_boxes(boxes) if boxes else (0,0,gw,gh)
    box=padded_box((guide[0]*w/gw,guide[1]*h/gh,guide[2]*w/gw,guide[3]*h/gh),rgb.shape)
    guide_seconds=time.monotonic()-started
    if on_phase:on_phase('Refining product edges',.25)
    fine=_predict_region(rgb,box);passes+=1
    clipped=clipped_sides(fine,box,rgb.shape)
    if any(clipped):
        # Expand the actual clipped ROI, not another guide which can repeat
        # the same mistake. Inspect unfiltered predictions on both attempts.
        if on_phase:on_phase('Checking pieces at the crop boundary',.55)
        box=padded_box(box,rgb.shape,.35)
        fine=_predict_region(rgb,box);passes+=1
        clipped=clipped_sides(fine,box,rgb.shape)
    # Internal crop boundaries aren't the camera frame. Do not discard an
    # accessory there even if the model still needs a review after the retry.
    try:fine,_=components(fine,preserve_frame=any(clipped))
    except ValueError:fine=np.zeros_like(fine)
    mask=np.zeros((h,w),np.float32);_place(mask,fine,box)
    missing=missing_regions(seeds,mask,small_parts)
    recovery=[]
    if missing:
        # A local second look gives omitted accessories MORE model pixels,
        # and never replaces the already good main-product mask.
        if on_phase:on_phase('Checking separate accessories',.62)
        # Loose small pieces come many at a time, and one sweep leaves the rest
        # behind, so the budget follows the option rather than the usual pair.
        sweeps=3 if small_parts else 1
        if small_parts:
            # Sharing one crop between several nearby studs shrinks each one in
            # the model's input, and only the best-placed survive. Give every
            # piece its own tight crop instead, capped for worst-case runtime.
            # Smallest islands first so white/colored studs are not starved by
            # larger unresolved floor skirts still on the dark layer.
            jobs=[(r,[r]) for r in _prefer_compact(missing,10)]
        else:
            jobs=[(region,[region]) for region in missing[:2]]
        recovery=_recover(rgb,mask,seeds,jobs)
        for _ in range(sweeps):
            extra=extend_recovery(rgb,mask,seeds,recovery,on_phase,small_parts=small_parts)
            if not extra:break
            recovery+=extra
        passes+=len(recovery)
        missing=missing_regions(seeds,mask,small_parts)
    review=[]
    if any(clipped):review.append('Check product at processing boundary')
    if missing:review.append('Check separate pieces against the original')
    if not np.any(mask>.5):raise ValueError('No product detected. Check the source photo.')
    info=dict(version='coverage-v3',small_parts=bool(small_parts),model_passes=passes,processing_box=list(box),
              recovery_boxes=recovery,unresolved_regions=[list(map(int,b)) for b in missing],
              guide_seconds=round(guide_seconds,3),seconds=round(time.monotonic()-started,3),review=review,
              recovery_round=RECOVERY_ROUND)
    return mask,info


RECOVERY_ROUND=6


def _prefer_compact(regions,limit):
    """Recover tiny loose pieces before large unresolved skirts."""
    return sorted(regions,key=lambda r:(r[2]-r[0])*(r[3]-r[1]))[:limit]


def _interior_touch(accepted,patch,shape):
    """Sides where a kept piece runs off the crop without reaching the frame."""
    h,w=shape[:2];pl,pt,pr,pb=patch
    ys,xs=np.where(accepted>0)
    if not len(xs):return (False,)*4
    return (xs.min()==0 and pl>0,ys.min()==0 and pt>0,
            xs.max()==accepted.shape[1]-1 and pr<w,ys.max()==accepted.shape[0]-1 and pb<h)


def _grow(patch,sides,shape):
    h,w=shape[:2];pl,pt,pr,pb=patch
    step=max(pr-pl,pb-pt)*.6
    return (max(0,int(pl-step)) if sides[0] else pl,max(0,int(pt-step)) if sides[1] else pt,
            min(w,int(pr+step)) if sides[2] else pr,min(h,int(pb+step)) if sides[3] else pb)


def _accept(rgb,mask,seeds,members,candidate,patch,recover_parts=False):
    """Components of one close look that pass every acceptance rule."""
    h,w=rgb.shape[:2];gh,gw=seeds.shape
    pl,pt,pr,pb=patch
    target_seeds=np.zeros_like(seeds)
    for ml,mt,mr,mb in members:target_seeds[mt:mb,ml:mr]=seeds[mt:mb,ml:mr]
    # Only the suspected missing islands anchor a recovery, not a
    # neighboring portion of the already accepted main product.
    gl=max(0,int(pl*gw/w));gt=max(0,int(pt*gh/h))
    gr=min(gw,int(np.ceil(pr*gw/w)));gb=min(gh,int(np.ceil(pb*gh/h)))
    anchors=target_seeds[gt:gb,gl:gr]
    anchors=cv2.resize(anchors,candidate.shape[::-1],interpolation=cv2.INTER_NEAREST)>0
    existing=cv2.resize(mask[pt:pb,pl:pr],candidate.shape[::-1],interpolation=cv2.INTER_AREA)>.45
    found=candidate>.45
    # A piece lying against the product can come back joined to it, and the
    # joined blob fails the overlap and border rules. Judge the part outside
    # the existing mask on its own as well. That part must also be clearly
    # coloured: a dark shadow or tinted reflection under the base comes back
    # joined to the product in exactly the same way.
    apart=found&~cv2.dilate(existing.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
    colour=cv2.resize(rgb[pt:pb,pl:pr],candidate.shape[::-1],interpolation=cv2.INTER_AREA)
    high=colour.max(axis=2);saturation=(high-colour.min(axis=2))/np.maximum(high,1e-6)
    accepted=np.zeros(candidate.shape,np.uint8)
    # Only a patch side lying on the camera frame can hide a partial
    # neighbour. An interior side is a crop the pipeline chose, and a piece
    # sized like its own island necessarily reaches it.
    frame_side=(pl<=0,pt<=0,pr>=w,pb>=h)
    for joined,binary in ((False,found),(True,apart)):
        count,labels,stats,_=cv2.connectedComponentsWithStats(binary.astype(np.uint8))
        keep=np.zeros(count,np.uint8)
        for i in range(1,count):
            x,y,cw,ch,area=stats[i]
            # Partial neighboring objects and booth at the camera frame
            # must not contaminate a valid main mask.
            if ((x==0 and frame_side[0]) or (y==0 and frame_side[1])
                or (x+cw>=candidate.shape[1] and frame_side[2])
                or (y+ch>=candidate.shape[0] and frame_side[3])):continue
            if np.count_nonzero(existing&(labels==i))>area*.20:continue
            if np.count_nonzero(anchors&(labels==i))<3:continue
            part=labels==i
            if joined:
                if recover_parts:
                    # Opt-in flat parts may be grey or black and touch the main
                    # model. Require substantial photographed evidence instead
                    # of the usual saturated-colour rule. Alpha still comes
                    # entirely from the local model prediction, never the seeds.
                    if np.count_nonzero(anchors&part)<area*.10:continue
                elif saturation[part].mean()<.45 or high[part].mean()<.15:continue
            keep[i]=1
        accepted|=keep[labels]
    return accepted


def _recover(rgb,mask,seeds,jobs,recover_parts=False,stats=None):
    """One close model look per job; keep only isolated pieces its own islands anchor.

    A piece as large as the island that flagged it fills its first crop and runs
    off the edge, so the crop grows toward that side until the piece fits and is
    placed whole instead of sliced along the crop boundary.
    """
    h,w=rgb.shape[:2];gh,gw=seeds.shape
    patches=[]
    for (l,t,r,b),members in jobs:
        patch=padded_box((l*w/gw,t*h/gh,r*w/gw,b*h/gh),rgb.shape,.40)
        best=None
        for _ in range(3):
            candidate=_predict_region(rgb,patch)
            if stats is not None:stats['model_passes']=stats.get('model_passes',0)+1
            accepted=_accept(rgb,mask,seeds,members,candidate,patch,recover_parts)
            if not np.any(accepted):break
            best=(candidate,accepted,patch)
            sides=_interior_touch(accepted,patch,rgb.shape)
            if not any(sides):break
            grown=_grow(patch,sides,rgb.shape)
            if grown==patch:break
            patch=grown
        if best is not None:
            candidate,accepted,patch=best
            accepted=_drop_floods(accepted,patch,rgb.shape)
            if np.any(accepted):
                support=cv2.dilate(accepted,np.ones((7,7),np.uint8))
                _place(mask,candidate*support,patch)
        patches.append(list(patch))
    return patches


def _drop_floods(accepted,patch,shape):
    """Remove regions that still span the crop after it has finished growing.

    A real piece that runs off the crop makes the crop grow 60% toward that side,
    up to three times, until the piece fits. On an empty crop the model instead
    paints the booth, and that region keeps filling whatever crop it is given.
    Seen on 30578-1 at 140 degrees: the booth horizon flagged a 'missing piece',
    and a 588x232 px rectangle of plain backdrop touching the crop's left, right
    and bottom edges was accepted, which multiplied the photo's width by 5 and
    shrank the shared scale for the whole batch. Only interior sides count; a
    side on the camera frame is already handled by the frame rule.
    """
    h,w=shape[:2];pl,pt,pr,pb=patch
    count,labels,stats,_=cv2.connectedComponentsWithStats(accepted.astype(np.uint8))
    keep=np.ones(count,np.uint8);keep[0]=0
    ch,cw=accepted.shape
    for i in range(1,count):
        x,y,bw,bh,_=stats[i]
        left=x==0 and pl>0;right=x+bw>=cw and pr<w
        top=y==0 and pt>0;bottom=y+bh>=ch and pb<h
        if (left and right) or (top and bottom):keep[i]=0
    return keep[labels].astype(accepted.dtype)


def _tried(region,boxes,gw,gh,w,h):
    """Was this island the target of an earlier look, not merely inside its padding?

    Each look crops padded_box(target,.40), which adds 2/9 of the crop's longer
    side on every side. A look anchors only its own target, so an island lying
    in that padding was never examined even though the crop contained it.
    """
    l0,t0,r0,b0=region[0]*w/gw,region[1]*h/gh,region[2]*w/gw,region[3]*h/gh
    area=max(1e-6,(r0-l0)*(b0-t0))
    for l,t,r,b in boxes:
        inset=max(r-l,b-t)*2/9
        overlap=max(0,min(r0,r-inset)-max(l0,l+inset))*max(0,min(b0,b-inset)-max(t0,t+inset))
        if overlap>=area*.5:return True
    return False


def _inside(region,group):
    return region[0]>=group[0] and region[1]>=group[1] and region[2]<=group[2] and region[3]<=group[3]


def _clusters(regions,gap=40):
    groups=[list(r) for r in regions]
    merged=True
    while merged:
        merged=False
        for i in range(len(groups)):
            for j in range(i+1,len(groups)):
                a,b=groups[i],groups[j]
                if a[0]-gap<=b[2] and b[0]-gap<=a[2] and a[1]-gap<=b[3] and b[1]-gap<=a[3]:
                    groups[i]=[min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3])]
                    del groups[j];merged=True;break
            if merged:break
    return [tuple(g) for g in groups]


def extend_recovery(rgb,mask,seeds,tried,on_phase=None,retry=False,small_parts=False):
    """Close looks for omitted islands that no earlier look examined.

    The first recovery stops after two islands, so a set with several loose
    accessories, such as swords scattered on the floor, kept the rest out.
    Islands lying near each other share one look. A region an earlier look
    already examined and rejected, such as a coloured floor reflection, is not
    retried, so a photo whose omissions were all examined gains no time.
    """
    h,w=rgb.shape[:2];gh,gw=seeds.shape
    missing=missing_regions(seeds,mask,small_parts)
    fresh=[r for r in missing if not _tried(r,tried,gw,gh,w,h)]
    if small_parts:
        # One piece, one crop: a shared crop shrinks each piece in the model's
        # input, and that is exactly what left some behind the first time.
        # Compact first so pale neutral studs keep pace with colored ones.
        jobs=[(r,[r]) for r in _prefer_compact(fresh,10)]
    else:
        groups=_clusters(fresh,40)
        jobs=[(g,[r for r in fresh if _inside(r,g)]) for g in groups]
    if retry:
        # Earlier rounds could reject a piece joined to the product, or share
        # one crop between several studs, or merge a stud with cast-shadow
        # fringe. Give those islands another dedicated look.
        jobs+=[(r,[r]) for r in missing if _tried(r,tried,gw,gh,w,h)]
    if not jobs:return []
    if on_phase:on_phase('Checking more separate pieces',.66)
    if small_parts and len(jobs)>10:
        # Keep the compact preference when retry inflates the list.
        order={r:i for i,r in enumerate(_prefer_compact([j[0] for j in jobs],len(jobs)))}
        jobs=sorted(jobs,key=lambda j:order.get(j[0],10**9))[:10]
    return _recover(rgb,mask,seeds,jobs)


def needs_recovery(path,mode='efficient',small_parts=False,recover_parts=False,recover_white=False,recover_dark=False):
    """A cached mask still flags omitted islands that no look has examined."""
    # Opt-in masks have their own versioned, completed recovery pass. Standard
    # recovery upgrades must never switch them back to the normal cache.
    if recover_white or recover_dark:
        base=base_mask(path,mode,small_parts,recover_parts,recover_white,recover_dark)
        return not all(p.is_file() for p in (
            base.with_name(base.stem+f'-edges-v{EDGE_VERSION}.png'),
            base.with_name(base.stem+'-protection.png'),base.with_suffix('.json')))
    if recover_parts:return False
    base=base_mask(path,mode,small_parts,recover_parts,recover_white)
    try:
        info=json.loads(base.with_suffix('.json').read_text())
        with Image.open(base) as image:w,h=image.size
    except (OSError,ValueError):return False
    if info.get('recovery_round',1)>=RECOVERY_ROUND:return False
    # Before round 5 a cast-shadow fringe could merge with a nearby stud into
    # one recovery crop, so any sign of loose pieces warrants one re-check.
    # Current masks re-check unexamined islands.
    if info.get('recovery_round',1)<RECOVERY_ROUND:return bool(info.get('unresolved_regions') or info.get('recovery_boxes'))
    if not info.get('unresolved_regions'):return False
    scale=min(1,1600/max(w,h));gw,gh=max(1,round(w*scale)),max(1,round(h*scale))
    return any(not _tried(r,info.get('recovery_boxes',[]),gw,gh,w,h) for r in info['unresolved_regions'])


def _upgrade_recovery(rgb,path,mode,on_phase=None,small_parts=False):
    """Extend a cached mask's recovery in place; the main mask is never rebuilt."""
    from studio import pil
    base=base_mask(path,mode,small_parts);info_path=base.with_suffix('.json')
    info=json.loads(info_path.read_text())
    with Image.open(base) as image:mask=np.asarray(image).astype(np.float32)/65535
    small=pil(rgb);small.thumbnail((1600,1600))
    seeds=scene_evidence(small,small_parts)
    extra=extend_recovery(rgb,mask,seeds,info.get('recovery_boxes',[]),on_phase,retry=info.get('recovery_round',1)<RECOVERY_ROUND,small_parts=small_parts)
    info['recovery_round']=RECOVERY_ROUND
    if extra:
        missing=missing_regions(seeds,mask,small_parts)
        info['recovery_boxes']=info.get('recovery_boxes',[])+extra
        info['model_passes']=info.get('model_passes',0)+len(extra)
        info['unresolved_regions']=[list(map(int,b)) for b in missing]
        review=[m for m in info.get('review',[]) if m!='Check separate pieces against the original']
        if missing:review.append('Check separate pieces against the original')
        info['review']=review
        temporary=base.with_suffix('.tmp')
        Image.fromarray((mask*65535).astype(np.uint16)).save(temporary,format='PNG')
        temporary.replace(base)
        for version in range(2,EDGE_VERSION+1):
            base.with_name(base.stem+f'-edges-v{version}.png').unlink(missing_ok=True)
    temporary=info_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(info,indent=2));temporary.replace(info_path)


def recover_flat_parts(rgb,mask,on_phase=None,partial_only=False):
    """Bounded close looks for neutral ramps and thin tools, on explicit request."""
    from studio import pil
    started=time.monotonic();stats={}
    small=pil(rgb);small.thumbnail((1600,1600))
    seeds=scene_evidence(small,True,True)
    missing=missing_regions(seeds,mask,True,True)
    # Large flat pieces first, then individual tools. No repeated whole-photo
    # inference and no unlimited retries on shadows the model cannot resolve.
    jobs=[] if partial_only else [(r,[r]) for r in sorted(missing,key=lambda r:-(r[2]-r[0])*(r[3]-r[1]))[:16]]
    patches=[]
    for i,job in enumerate(jobs):
        if on_phase:on_phase(f'Recovering flat parts and tools {i+1} of {len(jobs)}',.68+.20*i/max(1,len(jobs)))
        patches+=_recover(rgb,mask,seeds,[job],recover_parts=True,stats=stats)
    detail_seeds,partial=partial_dark_parts(rgb,mask)
    detail_patches=[]
    for i,region in enumerate(partial):
        if on_phase:on_phase(f'Recovering incomplete tools {i+1} of {len(partial)}',.90+.06*i/max(1,len(partial)))
        detail_patches+=_recover(rgb,mask,detail_seeds,[(region,[region])],recover_parts=True,stats=stats)
    return mask,dict(parts_recovery_boxes=patches,parts_unresolved_regions=missing_regions(seeds,mask,True,True),
                     partial_tool_boxes=detail_patches,parts_version=PARTS_VERSION,
                     parts_model_passes=stats.get('model_passes',0),parts_seconds=round(time.monotonic()-started,3))


def addition_protection(path,mode='efficient',small_parts=False,recover_parts=False,recover_white=True,recover_dark=False):
    """Only the opt-in recovered alpha, so floor cleanup cannot erase the new part."""
    base=base_mask(path,mode,small_parts,recover_parts,recover_white,recover_dark)
    with Image.open(base.with_name(base.stem+'-protection.png')) as image:
        return np.asarray(image).astype(np.float32)/65535


def white_protection(path,mode='efficient',small_parts=False,recover_parts=False):
    return addition_protection(path,mode,small_parts,recover_parts,True,False)


def _get_addition_mask(rgb,path,mode,on_phase,diagnostics,small_parts,recover_parts,recover_white,recover_dark):
    import fcntl
    base=base_mask(path,mode,small_parts,recover_parts,recover_white,recover_dark)
    base.parent.mkdir(parents=True,exist_ok=True)
    with base.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _load_or_build_addition_mask(rgb,path,mode,on_phase,diagnostics,small_parts,recover_parts,recover_white,recover_dark)


def _load_or_build_addition_mask(rgb,path,mode,on_phase,diagnostics,small_parts,recover_parts,recover_white,recover_dark):
    base=base_mask(path,mode,small_parts,recover_parts,recover_white,recover_dark)
    final=base.with_name(base.stem+f'-edges-v{EDGE_VERSION}.png')
    protection_file=base.with_name(base.stem+'-protection.png')
    try:
        with Image.open(final) as image:mask=np.asarray(image).astype(np.float32)/65535
        protection=addition_protection(path,mode,small_parts,recover_parts,recover_white,recover_dark)
        info=json.loads(base.with_suffix('.json').read_text())
        if (mask.shape!=rgb.shape[:2] or protection.shape!=mask.shape
                or not np.isfinite(mask).all() or not np.any(mask>.5)):
            raise ValueError('Invalid recovery cache')
        if diagnostics is not None:diagnostics.update(info)
        return mask,True,'cached'
    except (OSError,ValueError):pass
    info={}
    original,_,_=get_mask(rgb,path,mode,on_phase,info,small_parts,recover_parts)
    mask=original;review=[];methods=[]
    if recover_white:
        from white_recovery import recover_white_parts
        mask,extra=recover_white_parts(rgb,mask,on_phase)
        info.update(extra,recover_white=True)
        info['seconds']=round(info.get('seconds',0)+extra['white_seconds'],3)
        review.append('Check recovered white parts against the original; faint outlines may remain incomplete')
        methods.append('white-parts-recovery')
    if recover_dark:
        from dark_recovery import recover_dark_parts
        mask,extra=recover_dark_parts(rgb,mask,on_phase)
        info.update(extra,recover_dark=True)
        info['seconds']=round(info.get('seconds',0)+extra['dark_seconds'],3)
        if extra['dark_recovery_boxes']:
            review.append('Check recovered dark parts near the floor against the original')
        methods.append('dark-parts-recovery')
    protection=np.where(mask>original+1/65535,mask,0)
    info['review']=list(dict.fromkeys(info.get('review',[])+review))
    base.parent.mkdir(parents=True,exist_ok=True)
    # The source mask already passed normal edge cleanup. Save this result as
    # finished: repeating that cleanup would classify the additions as floor.
    for target,alpha in ((base,mask),(protection_file,protection),(final,mask)):
        temporary=target.with_suffix('.tmp')
        Image.fromarray(np.round(alpha*65535).astype(np.uint16)).save(temporary,format='PNG')
        temporary.replace(target)
    temporary=base.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(info,indent=2));temporary.replace(base.with_suffix('.json'))
    if diagnostics is not None:diagnostics.update(info)
    return mask,False,'+'.join(methods)


def get_mask(rgb, path, mode='efficient', on_phase=None, diagnostics=None, small_parts=False, recover_parts=False, recover_white=False, recover_dark=False):
    """`rgb` is the unadjusted decode. White/dark recovery read it directly;
    everything else sees mask_view(rgb), independent of the export recipe."""
    if recover_white or recover_dark:
        return _get_addition_mask(rgb,path,mode,on_phase,diagnostics,small_parts,recover_parts,recover_white,recover_dark)
    rgb=mask_view(rgb)
    h,w=rgb.shape[:2]
    for candidate in dict.fromkeys([cached_mask(path,mode,small_parts,recover_parts),base_mask(path,mode,small_parts,recover_parts,recover_white)]):
        if candidate.exists():
            try:
                with Image.open(candidate) as image:mask=np.asarray(image).astype(np.float32)/65535
                if mask.shape==(h,w) and np.isfinite(mask).all() and np.any(mask>.5):
                    if needs_recovery(path,mode,small_parts,recover_parts):
                        _upgrade_recovery(rgb,path,mode,on_phase,small_parts)
                        candidate=cached_mask(path,mode,small_parts)
                        with Image.open(candidate) as image:mask=np.asarray(image).astype(np.float32)/65535
                    if diagnostics is not None:
                        diagnostics.update(mask_diagnostics(path,mode,small_parts,recover_parts))
                    return finish_edges(rgb,mask,candidate),True,'cached'
            except (OSError,ValueError):pass
            if '-edges-v' in candidate.stem:candidate.unlink(missing_ok=True)
    cache=base_mask(path,mode,small_parts,recover_parts,recover_white)
    if recover_parts:
        info={}
        # Extend the user's previous recovery result. There is no need to
        # repeat all its model passes merely to repair incomplete handles.
        prior=cache.with_name(cache.name.replace(f'-parts-v{PARTS_VERSION}','-parts-v1'))
        detailed_prior=prior.with_name(prior.name.replace('-guided-coverage-v3','-coverage-v3'))
        if detailed_prior.exists():prior=detailed_prior
        prior_edge=prior.with_name(prior.stem+f'-edges-v{EDGE_VERSION}.png')
        upgraded=False
        try:
            with Image.open(prior_edge if prior_edge.exists() else prior) as image:
                mask=np.asarray(image).astype(np.float32)/65535
            info=json.loads(prior.with_suffix('.json').read_text())
            upgraded=mask.shape==(h,w) and np.isfinite(mask).all() and np.any(mask>.5)
        except (OSError,ValueError):pass
        if not upgraded:
            info={}
            mask,_,_=get_mask(rgb,path,mode,on_phase,info,small_parts)
        previous_boxes=info.get('parts_recovery_boxes',[]) if upgraded else []
        mask,extra=recover_flat_parts(rgb,mask.copy(),on_phase,partial_only=upgraded)
        extra['parts_recovery_boxes']=previous_boxes+extra['parts_recovery_boxes']
        info.update(extra,recover_parts=True)
        info['model_passes']=info.get('model_passes',0)+extra['parts_model_passes']
        info['seconds']=round(info.get('seconds',0)+extra['parts_seconds'],3)
    else:
        mask,info=build_mask(rgb,mode,on_phase,small_parts)
    if diagnostics is not None:diagnostics.update(info)
    cache.parent.mkdir(parents=True,exist_ok=True)
    temporary=cache.with_suffix('.tmp')
    Image.fromarray((mask*65535).astype(np.uint16)).save(temporary,format='PNG')
    temporary.replace(cache)
    info_path=cache.with_suffix('.json');temporary=info_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(info,indent=2));temporary.replace(info_path)
    method='coverage-two-pass' if mode=='detailed' else 'coverage-one-pass'
    if info['model_passes']>(2 if mode=='detailed' else 1):method+='-recovery'
    return finish_edges(rgb,mask,cache),False,method
