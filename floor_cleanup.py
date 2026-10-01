"""Optional reclassification of floor accidentally included in a subject mask."""
import cv2
import numpy as np


def remove_floor_patches(rgb, alpha):
    solid=alpha>.5
    ys,xs=np.where(solid)
    if not len(xs):return alpha.copy()
    span=max(int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1))
    pad=max(12,round(span*.08))
    l=max(0,int(xs.min())-pad);r=min(alpha.shape[1],int(xs.max())+pad+1)
    t=max(0,int(ys.min())-pad);b=min(alpha.shape[0],int(ys.max())+pad+1)
    pixels=rgb[t:b,l:r];mask=alpha[t:b,l:r]
    lum=pixels.mean(2);chroma=np.ptp(pixels,axis=2)
    distance=cv2.distanceTransform((mask>.5).astype('uint8'),cv2.DIST_L2,5)
    lower=np.arange(t,b)[:,None]>ys.min()+(ys.max()-ys.min())*.53
    # Reconsider even opaque pixels, but only neutral, shallow lower contours.
    candidate=lower&(lum>.27)&(chroma<.13)&(distance<span*.055)&(mask>.01)
    if not candidate.any():return alpha.copy()
    labels=np.where(mask>.5,cv2.GC_FGD,cv2.GC_BGD).astype('uint8')
    labels[candidate]=cv2.GC_PR_FGD
    labels[(mask<=.5)&(mask>.01)&~candidate]=cv2.GC_PR_BGD
    if np.count_nonzero(labels==cv2.GC_FGD)<5 or np.count_nonzero(labels==cv2.GC_BGD)<5:return alpha.copy()
    # The colour model sees the local photographed floor on both sides of
    # an incorrect alpha contour, while tyres and coloured bricks anchor it.
    bg=np.zeros((1,65),np.float64);fg=bg.copy()
    cv2.setRNGSeed(0)
    cv2.grabCut(np.clip(pixels*255,0,255).astype('uint8'),labels,None,bg,fg,4,cv2.GC_INIT_WITH_MASK)
    removed=candidate&((labels==cv2.GC_BGD)|(labels==cv2.GC_PR_BGD))
    if not removed.any():return alpha.copy()
    # Drop antialias remnants of a small isolated floor island only after
    # its own photographed colours have overwhelmingly voted background.
    count,components,stats,_=cv2.connectedComponentsWithStats((mask>.1).astype('uint8'))
    for i in range(1,count):
        region=components==i
        if stats[i,cv2.CC_STAT_AREA]<span*span*.012 and removed[region].mean()>.8:
            removed|=region
    retained=(mask>.5)&~removed
    # Keep an antialiased boundary at the new cut; never grow the old mask.
    feather=cv2.GaussianBlur(retained.astype(np.float32),(0,0),.55)
    affected=cv2.dilate(removed.astype('uint8'),np.ones((3,3),np.uint8))>0
    out=alpha.copy();out[t:b,l:r]=np.where(affected,np.minimum(mask,feather),mask)
    return out
