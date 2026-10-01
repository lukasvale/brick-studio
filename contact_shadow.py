"""Recover photographed floor shadows near local lower subject edges.

Uses a small analysis image, existing alpha and unadjusted source colour.
No inference, invented drop shadow, or persistent image cache is involved.
Photographed bounce keeps its colour; only saturated floor stains are dropped.
"""
import cv2
import numpy as np


CEILING=1.3


def shade(density):
    """Luminance of a 2D or RGB shadow density."""
    if density.ndim==2:return density
    return density@np.array([.2126,.7152,.0722],np.float32)


def cap(density):
    """Approach the density ceiling smoothly.

    A hard clip turns a deep contact, such as a tyre pressing into the floor,
    into one saturated patch with no tonal variation inside it.
    Hue is kept: channels scale together from the strongest one.
    """
    d=np.maximum(density,0)
    mag=d if d.ndim==2 else d.max(axis=2,keepdims=True)
    scaled=mag/np.cbrt(1+(mag/CEILING)**3)
    if d.ndim==2:return scaled
    return d*(scaled/np.maximum(mag,1e-6))


def soften_bays(density, mask, subject_width):
    """Dissolve hard floor debris recovered as a sharp shadow island.

    A loose tile next to the cutout is photographed floor darkening, so it
    survives as a hard scrap while the rest of the contact is a soft wash.
    A small median on uncovered floor follows that wash; product pixels and
    the steep outer contact stay as photographed.
    """
    mag=shade(density)
    peak=float(mag.max())
    if peak<1e-5:return density
    size=int(np.clip(int(round(subject_width*.037))|1,5,13))
    median=cv2.medianBlur(np.clip(np.round(mag/peak*255),0,255).astype(np.uint8),size).astype(np.float32)/255*peak
    scale=np.where(mask<.5,median/np.maximum(mag,1e-6),1).astype(np.float32)
    if density.ndim==2:return density*scale
    return density*scale[:,:,None]


def booth_horizon(luminance, mask):
    """Weight that is zero along and above the photo-box horizon, or None.

    The horizon is measured only in the booth margins either side of the
    subject, where the wall meets the brighter floor, and fitted as a gentle
    curve across. Both margins must agree and the fit must be tight; otherwise
    nothing is suppressed.
    """
    gh,gw=luminance.shape
    solid=mask>.5
    if not solid.any():return None
    lum=cv2.GaussianBlur(luminance,(0,0),1.5)
    columns=np.flatnonzero(solid.any(axis=0));lowest=int(np.where(solid)[0].max())
    points=[]
    for side in (np.arange(0,columns.min()-2,2),np.arange(columns.max()+3,gw,2)):
        found=[]
        for c in side:
            col=lum[:lowest,c]
            if len(col)<12:continue
            step=col[6:]-col[:-6];y=int(np.argmax(step))
            if step[y]<.035:continue
            below=col[y+3:];lit=np.flatnonzero(below>=np.percentile(below,90)-.012)
            found.append((c,y+3,int(lit[0]) if len(lit) else 0))
        if len(found)<4:return None
        points+=found
    points=np.array(points,float)
    coef=np.polyfit(points[:,0],points[:,1],2)
    residual=np.abs(np.polyval(coef,points[:,0])-points[:,1])
    keep=residual<=np.median(residual)*3+2
    coef=np.polyfit(points[keep,0],points[keep,1],2)
    if np.median(np.abs(np.polyval(coef,points[keep,0])-points[keep,1]))>2:return None
    band=float(np.clip(np.median(points[keep,2]),3,gh*.08))
    horizon=np.polyval(coef,np.arange(gw))
    return np.clip((np.arange(gh)[:,None]-(horizon[None,:]+band))/max(3,band),0,1).astype(np.float32)


def soften_shadow_mask(rgb, mask):
    """Compatibility entry point for saved shadow_detection=soft recipes."""
    from floor_cleanup import remove_floor_patches
    return remove_floor_patches(rgb,mask)


def recover_shadow(rgb, alpha, subject_width, strength, detection='dark'):
    if detection not in ('dark','soft'):raise ValueError('Invalid shadow detection')
    h,w=alpha.shape
    factor=min(1,900/max(h,w))
    size=(max(1,round(w*factor)),max(1,round(h*factor)))
    pixels=cv2.resize(rgb,size,interpolation=cv2.INTER_AREA)
    mask=cv2.resize(alpha,size,interpolation=cv2.INTER_AREA)
    solid=mask>.8
    yy,xx=np.where(solid)
    if not len(xx):return np.zeros((h,w,3),np.float32)
    gh,gw=solid.shape
    # The lower outline follows feet, wheels and separated bases at different
    # depths. Restrict to the lower scene to avoid roofs and the booth horizon.
    bottom=gh-1-np.argmax(solid[::-1],axis=0)
    valid=solid.any(axis=0)&(bottom>=yy.max()-(yy.max()-yy.min())*.45)
    seeds=np.zeros((gh,gw),np.uint8)
    columns=np.flatnonzero(valid)
    seeds[bottom[columns],columns]=1
    if not len(columns):return np.zeros((h,w,3),np.float32)
    # Join nearby supports at the same floor depth so an arch's photographed
    # shadow can continue through the opening. White floor still contributes
    # no shadow: these seeds only define where to inspect source luminance.
    for left,right in zip(columns[:-1],columns[1:]):
        if 1<right-left<=subject_width*factor*.45 and abs(int(bottom[right])-int(bottom[left]))<=max(3,(yy.max()-yy.min())*.12):
            xs=np.arange(left+1,right)
            ys=np.rint(np.interp(xs,[left,right],[bottom[left],bottom[right]])).astype(int)
            seeds[ys,xs]=1
    distance=cv2.distanceTransform(1-seeds,cv2.DIST_L2,5)
    radius=max(3,subject_width*factor*.028)
    # Keep photographed shadows across gaps between feet/base sections. A
    # plateau avoids immediately fading every shadow into isolated dark dots.
    reach_floor=max(6,subject_width*factor*.065)
    support=np.exp(-(np.maximum(0,distance-reach_floor)/(radius*1.7))**2)
    # Never retain the upper scene, even within the fade of a lower edge.
    floor_start=yy.max()-(yy.max()-yy.min())*.48
    support*=np.clip((np.arange(gh)[:,None]-floor_start)/max(3,radius),0,1)
    luminance=pixels@np.array([.2126,.7152,.0722],np.float32)
    # The shadow is darkest in the band the antialiased outline partly covers.
    # Dropping every covered pixel left the blur below to fill that band from
    # lighter distant floor, erasing the shadow where the set meets the floor.
    # Unmix it: observed = covered*product + (1-covered)*floor.
    solid_weight=(mask>.8).astype(np.float32)
    near=cv2.GaussianBlur(solid_weight,(0,0),2.5)
    product=cv2.GaussianBlur(pixels*solid_weight[:,:,None],(0,0),2.5)/np.maximum(near[:,:,None],1e-6)
    covered=np.clip(mask,0,1)
    unmixed=np.clip((pixels-covered[:,:,None]*product)/np.maximum(1-covered,.3)[:,:,None],0,1)
    floor_pix=np.where(((covered>=.02)&(near>.02))[:,:,None],unmixed,pixels)
    # Mostly-covered pixels stay out: their unmixed floor is too sensitive to
    # the product's own colour to be a trustworthy measurement.
    visible=np.clip((.72-covered)/.30,0,1)
    weights=cv2.GaussianBlur(visible,(0,0),.85)
    # Mask-normalized smoothing cannot bleed dark plastic into a blank floor.
    smooth=cv2.GaussianBlur(floor_pix*visible[:,:,None],(0,0),.85)/np.maximum(weights[:,:,None],1e-6)
    smooth_luma=shade(smooth).astype(np.float32)
    # Median only the brightness so a warm bounce is not mixed toward grey.
    filtered=cv2.medianBlur(smooth_luma,3)
    smooth=smooth*(filtered/np.maximum(smooth_luma,1e-6))[:,:,None]
    smooth_luma=filtered
    # Estimate the lit floor locally, excluding product pixels. This removes
    # broad floor grey/lighting gradients instead of mistaking them for shadow.
    background=np.where((mask<.02)[:,:,None],smooth,0)
    reach=max(5,round(subject_width*factor*.14))
    floor=cv2.dilate(background,np.ones((reach*2+1,reach*2+1),np.uint8))
    floor=cv2.GaussianBlur(floor,(0,0),max(1,radius*.3))
    deficit=np.maximum(0,(floor-smooth)/np.maximum(floor,.1)-.008)
    # A painted stain is a bright saturated blob (high chroma). Dark bounce
    # from orange plastic has low chroma; judging it by saturation instead
    # punched a white hole in the darkest part of the shadow, because sat
    # rises as the pixel gets darker.
    chroma=smooth.max(2)-smooth.min(2)
    allow=np.clip((.40-chroma)/.12,0,1)
    # Reproduce the measured per-channel deficit. Grey contacts stay grey;
    # warm floor bounce stays warm. At strength 2 this is the camera's darkening.
    density=deficit*(support*allow*float(strength)*.5)[:,:,None]
    # The booth's horizon is a dark band crossing behind the model. On a tall
    # model it lies below the upper-scene cutoff, and support from nearby
    # raised parts drew it out sideways as a streak. Floor shadow can only lie
    # below the horizon, so density fades out along and above it.
    horizon=booth_horizon(luminance,mask)
    if horizon is not None:density*=horizon[:,:,None]
    # Extend the photographed shadow beneath the antialiased object edge.
    # The foreground covers this overlap; it closes the gap caused by color
    # rejection/uncertain alpha without shrinking the product mask.
    overlap=max(1,round(subject_width*factor*.003))
    # Fully covered product pixels contain no measured floor, so their
    # extrapolated dark values are excluded: they seeded a black rim outside
    # the cutout. Spreading only measured floor closes the thin rim on both
    # sides of the edge without that risk.
    keep=((covered<.72)&(weights>.05))[:,:,None]
    measured=np.where(keep,density,0)
    expanded=cv2.dilate(measured,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(overlap*2+1,overlap*2+1)))
    density=np.maximum(measured,expanded)
    density=cap(density)
    density=soften_bays(density,mask,subject_width*factor)
    # Cleanup changes the subject mask upstream. Real contact shadows keep
    # their normal density rather than being globally weakened by this toggle.
    # Feather at the retained source boundary, never a rectangular cutoff.
    border=np.minimum(np.minimum(np.arange(gw),np.arange(gw)[::-1])[None,:],
                      np.minimum(np.arange(gh),np.arange(gh)[::-1])[:,None])
    density*=np.clip(border/max(3,radius),0,1)[:,:,None]
    return cv2.resize(density,(w,h),interpolation=cv2.INTER_LINEAR)


def shadow_bounds(mask):
    from framing import mask_bounds
    x,y,right,bottom=mask_bounds(mask)
    pad=int(max(right-x,bottom-y)*.18)+4
    l=max(0,x-pad);t=max(0,y-pad)
    r=min(mask.shape[1],right+pad);b=min(mask.shape[0],bottom+pad)
    return l,t,r,b


def shadow_layer(rgb, mask, detection='dark'):
    from framing import mask_bounds
    x,y,right,bottom=mask_bounds(mask)
    l,t,r,b=shadow_bounds(mask)
    return recover_shadow(rgb[t:b,l:r],mask[t:b,l:r],right-x,1,detection)
