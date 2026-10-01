"""Conservative removal of the vertical stock code beside a packaging QR.

Local, non-generative repair. A yellow label, QR geometry and a narrow column
of dark ink must agree. Never changes the QR or pixels outside the repair mask.
"""
import time
import cv2
import numpy as np


def clean_package_code(rgb):
    started=time.perf_counter();h,w=rgb.shape[:2]
    factor=min(1,1600/max(h,w))
    small=cv2.resize(np.clip(rgb*255,0,255).astype(np.uint8),None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA)
    hsv=cv2.cvtColor(small,cv2.COLOR_RGB2HSV)
    yellow=((hsv[:,:,0]>12)&(hsv[:,:,0]<40)&(hsv[:,:,1]>100)&(hsv[:,:,2]>65)).astype(np.uint8)
    count,_,stats,_=cv2.connectedComponentsWithStats(yellow)
    candidates=sorted(stats[1:],key=lambda s:s[4],reverse=True)[:4]
    detector=cv2.QRCodeDetector()
    for x,y,bw,bh,area in candidates:
        if area<120 or bw<20 or bh<20:continue
        l=max(0,int(x/factor)-12);t=max(0,int(y/factor)-12)
        r=min(w,int((x+bw)/factor)+12);b=min(h,int((y+bh)/factor)+12)
        pixels=np.clip(rgb[t:b,l:r]*255,0,255).astype(np.uint8)
        gray=cv2.cvtColor(pixels,cv2.COLOR_RGB2GRAY)
        for scale in [1,2,3]:
            test=cv2.resize(gray,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
            found,points=detector.detect(test)
            if found:break
        if not found:continue
        points=points[0]/scale
        side=float(np.mean(np.linalg.norm(points-np.roll(points,1,axis=0),axis=1)))
        if side<35:continue
        # OpenCV's finder orientation gives QR top-left/top-right/bottom-right/
        # bottom-left. Project a narrow strip beside its right edge.
        transform=cv2.getPerspectiveTransform(np.array([[0,0],[1,0],[1,1],[0,1]],np.float32),points.astype(np.float32))
        polygon=cv2.perspectiveTransform(np.array([[[1.04,.04],[1.34,.04],[1.34,1.02],[1.04,1.02]]],np.float32),transform)[0]
        roi=np.zeros(gray.shape,np.uint8);cv2.fillConvexPoly(roi,np.rint(polygon).astype(np.int32),1)
        protected=np.zeros_like(roi);cv2.fillConvexPoly(protected,np.rint(points).astype(np.int32),1)
        margin=max(2,round(side*.035));protected=cv2.dilate(protected,np.ones((margin*2+1,margin*2+1),np.uint8))
        roi[protected>0]=0
        lab=cv2.cvtColor(pixels,cv2.COLOR_RGB2HSV)
        paper=(lab[:,:,0]>12)&(lab[:,:,0]<40)&(lab[:,:,1]>80)&(lab[:,:,2]>90)
        sample=pixels[(roi>0)&paper]
        if len(sample)<max(30,np.count_nonzero(roi)*.65):continue
        reference=np.median(sample.astype(np.float32),axis=0)
        # Ink is darker than the local label in all channels. This also picks
        # up soft letter edges while excluding ordinary label lighting grain.
        ink=(gray < float(reference@np.array([.299,.587,.114]))*.80)&(roi>0)
        n,labels,parts,_=cv2.connectedComponentsWithStats(ink.astype(np.uint8))
        keep=np.zeros(n,bool)
        for i,(px,py,pw,ph,pa) in enumerate(parts[1:],1):
            if pa>=2 and pw<side*.30 and ph<side*.30:keep[i]=True
        ink=keep[labels]
        ys,xs=np.where(ink)
        if len(xs)<12 or np.ptp(ys)<side*.45 or np.ptp(xs)>side*.35:continue
        # The whole code column must be surrounded by label paper, not overlap
        # a bag edge or unrelated artwork. Inpaint only ink plus a tiny fringe.
        radius=max(1,round(side*.012))
        repair=cv2.dilate(ink.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1)))
        repair[roi==0]=0;repair[protected>0]=0
        # Replace the complete ink column using real neighbouring paper grain.
        # Fit only the broad lighting gradient; retain donor texture around it.
        pad=radius+2
        x0=max(0,int(xs.min())-pad);x1=min(gray.shape[1],int(xs.max())+pad+1)
        y0=max(0,int(ys.min())-pad);y1=min(gray.shape[0],int(ys.max())+pad+1)
        ph,pw=y1-y0,x1-x0
        donor=None
        offsets=[(sign*d,0) for d in range(pw+2,max(pw+3,round(side*.65)),2) for sign in (1,-1)]
        offsets += [(dx,-dy) for dy in range(ph+2,round(side*2),3) for dx in (0,-pw,pw)]
        for dx,dy in offsets:
            a=x0+dx;z=a+pw;c=y0+dy;d=c+ph
            if a<0 or z>gray.shape[1] or c<0 or d>gray.shape[0]:continue
            region=(slice(c,d),slice(a,z))
            if np.any(protected[region]) or np.mean(paper[region])<.99:continue
            if np.min(gray[region])<float(reference@np.array([.299,.587,.114]))*.85:continue
            donor=(a,z,c,d);break
        if donor is None:continue
        # Estimate label lighting from clean paper around the repair, excluding QR.
        yy,xx=np.mgrid[:gray.shape[0],:gray.shape[1]]
        vicinity=(abs(xx-(x0+x1)/2)<side*.65)&(yy>=y0)&(yy<y1)
        fit=vicinity&paper&(protected==0)&(cv2.dilate(ink.astype(np.uint8),np.ones((5,5),np.uint8))==0)
        fit &= gray>float(reference@np.array([.299,.587,.114]))*.85
        if np.count_nonzero(fit)<50:continue
        original=rgb[t:b,l:r]
        design=np.stack([np.ones_like(xx),xx/side,yy/side],axis=-1)
        coefficients=np.linalg.lstsq(design[fit],original[fit],rcond=None)[0]
        lighting=design@coefficients
        a,z,c,d=donor
        texture=original[c:d,a:z]-lighting[c:d,a:z]
        texture-=np.mean(texture,axis=(0,1),keepdims=True)
        replacement=np.clip(lighting[y0:y1,x0:x1]+texture,0,1)
        py,px=np.mgrid[:ph,:pw]
        feather=np.clip(np.minimum.reduce([px,py,pw-1-px,ph-1-py])/2,0,1).astype(np.float32)
        feather[protected[y0:y1,x0:x1]>0]=0
        feather[~paper[y0:y1,x0:x1] & ~cv2.dilate(ink.astype(np.uint8),np.ones((5,5),np.uint8))[y0:y1,x0:x1].astype(bool)]=0
        result=rgb.copy();patch=result[t+y0:t+y1,l+x0:l+x1]
        patch[:]=patch*(1-feather[:,:,None])+replacement*feather[:,:,None]
        return result,dict(status='removed',box=[l+x0,t+y0,l+x1,t+y1],qr=(points+[l,t]).tolist(),
                           changed_pixels=int(np.count_nonzero(np.any(result!=rgb,axis=2))),method='paper-texture-v2',
                           seconds=round(time.perf_counter()-started,3))
    return rgb,dict(status='review',message='Packaging code was not confidently located; photo left unchanged.',seconds=round(time.perf_counter()-started,3))
