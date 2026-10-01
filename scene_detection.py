"""Cheap full-scene coverage checks for a neutral product-photo booth.

These are crop/QA hints, never foreground alpha. BiRefNet still decides the
edges, holes and transparency using the photographed pixels.
"""
import cv2
import numpy as np


def union_boxes(boxes):
    boxes=list(boxes)
    if not boxes:raise ValueError('No product region found')
    return (min(b[0] for b in boxes),min(b[1] for b in boxes),
            max(b[2] for b in boxes),max(b[3] for b in boxes))


def scene_evidence(image,small_parts=False,recover_parts=False):
    """Return strong color and compact neutral-object hints at guide resolution.

    Keep the two tests separate: otherwise a booth horizon can connect to and
    swallow a colored product component. Broad neutral bands aren't objects.

    Values: 1 = neutral/dark hint, 2 = color hint. Missing-piece recovery reads
    the layers apart so a cast-shadow skirt cannot glue itself to a red stud and
    drag both into one huge close-look crop.

    small_parts makes the neutral test sensitive to faint pieces. A white or
    translucent piece on the white floor only darkens it slightly, so at the
    normal threshold just a thin crescent of its shading registers and the
    shape tests below discard it as a seam. Such a piece is then never reported
    missing, no close look is taken, and it is lost silently: the render keeps
    its contact shadow, so the floor shows a shadow with nothing casting it.
    The size floors move only a little; the darkness threshold does the work.
    """
    rgb=np.asarray(image).astype(np.float32)/255
    h,w=rgb.shape[:2]
    high=rgb.max(2);chroma=high-rgb.min(2)
    color=(chroma>.085)&(chroma/(high+.001)>.20)&(high>.12)
    lum=rgb@np.array([.2126,.7152,.0722],np.float32)
    edge=max(1,round(w*.14))
    background=np.quantile(np.concatenate((lum[:,:edge],lum[:,-edge:]),axis=1),.65,axis=1)
    dark=background[:,None]-lum>(.19 if recover_parts else .05 if small_parts else .19)
    seeds=np.zeros((h,w),np.uint8)
    min_area=max(6,round(h*w*.000006)) if small_parts else max(8,round(h*w*.00001))
    thinnest=2 if recover_parts else 5 if small_parts else 7
    longest=6
    # Neutral first, color second so a saturated stud keeps layer 2 even where
    # its own shading also trips the dark test.
    for value,neutral,raw in ((1,True,dark),(2,False,color)):
        closed=cv2.morphologyEx(raw.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        count,labels,stats,_=cv2.connectedComponentsWithStats(closed)
        accepted=np.zeros(count,np.uint8)
        for i in range(1,count):
            x,y,bw,bh,area=stats[i]
            if area<min_area or x==0 or y==0 or x+bw>=w or y+bh>=h:continue
            if neutral and (min(bw,bh)<thinnest or bw>bh*(24 if recover_parts else longest) or bw>w*(.60 if recover_parts else .20) or bh>h*.35):continue
            if neutral:
                # A diagonal seam has a deceptively tall bounding rectangle.
                # Measure its actual thickness rather than that rectangle.
                part=(labels[y:y+bh,x:x+bw]==i).astype(np.uint8)
                m=cv2.moments(part)
                covariance=np.array([[m['mu20'],m['mu11']],[m['mu11'],m['mu02']]])/m['m00']
                thin,along=np.linalg.eigvalsh(covariance)
                if thin<(0.5 if recover_parts else 2.25) and along>max(thin,0.01)*64:continue
            accepted[i]=1
        seeds[accepted[labels]>0]=value
    return seeds


def evidence_boxes(seeds):
    count,_,stats,_=cv2.connectedComponentsWithStats((seeds>0).astype(np.uint8))
    return [(int(x),int(y),int(x+bw),int(y+bh))
            for x,y,bw,bh,area in stats[1:] if area>=3]


def partial_dark_parts(rgb,mask):
    """Find small photographed tools with missing sections in an existing mask.

    The usual 1600px overview and four-pixel exclusion band can hide a thin
    handle between two retained ends. Inspect dark pieces at twice that detail,
    and use each whole photographed component as the next model crop.
    These pixels are only anchors; the model must still predict the alpha.
    """
    h,w=rgb.shape[:2];scale=min(1,3200/max(h,w))
    size=(max(1,round(w*scale)),max(1,round(h*scale)))
    small=cv2.resize(rgb,size,interpolation=cv2.INTER_AREA)
    alpha=cv2.resize(mask,size,interpolation=cv2.INTER_AREA)
    dark=(small.mean(2)<.42)&(np.ptp(small,axis=2)<.18)
    dark=cv2.morphologyEx(dark.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    count,labels,stats,_=cv2.connectedComponentsWithStats(dark)
    distance=cv2.distanceTransform((alpha<.5).astype(np.uint8),cv2.DIST_L2,5)
    seeds=np.zeros(alpha.shape,np.uint8);regions=[]
    longest=max(40,max(size)*.05);largest=max(100,alpha.size*.00045)
    for i in range(1,count):
        x,y,bw,bh,area=(int(v) for v in stats[i])
        if not 12<=area<=largest or min(bw,bh)<2 or max(bw,bh)>longest:continue
        if x==0 or y==0 or x+bw>=size[0] or y+bh>=size[1]:continue
        part=labels[y:y+bh,x:x+bw]==i
        covered=np.mean(alpha[y:y+bh,x:x+bw][part]>.5)
        missing=np.count_nonzero(part&(distance[y:y+bh,x:x+bw]>1.5))
        if not .10<covered<.95 or missing<max(8,area*.05):continue
        seeds[y:y+bh,x:x+bw][part]=1
        regions.append((missing,(x,y,x+bw,y+bh)))
    return seeds,[r for _,r in sorted(regions,reverse=True)[:8]]


def missing_regions(seeds,mask,small_parts=False,recover_parts=False):
    """Find substantial omitted islands, not small antialiased edge differences.

    Color hints (2) and neutral hints (1) are judged as separate layers so a
    cast-shadow skirt under the model cannot glue itself to a nearby stud.
    Dilating all missed pixels used to merge them into one recovery crop that
    swallowed the piece and the close look failed.

    A seed mostly already in the mask may still hold a real accessory that
    touches it; peel compact uncovered blobs out of that fringe, and skip only
    the thin cast-shadow skirt.
    """
    if mask.shape!=seeds.shape:
        mask=cv2.resize(mask,(seeds.shape[1],seeds.shape[0]),interpolation=cv2.INTER_AREA)
    foreground=(mask>.45).astype(np.uint8)
    distance=cv2.distanceTransform(1-foreground,cv2.DIST_L2,5)
    outside=distance>4
    minimum=max(12,round(seeds.size*.000008)) if small_parts else max(30,round(seeds.size*.000025))
    smallest=2 if recover_parts else 6 if small_parts else 16
    filled=2 if recover_parts else 4 if small_parts else 6
    # Prefer color layer first: those are the studs that must not share a crop
    # with the soft floor contact of the main product.
    layers=(2,1) if np.any(seeds>1) else (None,)
    regions=[]
    seen=set()
    for layer in layers:
        binary=(seeds==layer) if layer is not None else (seeds>0)
        count,labels,stats,_=cv2.connectedComponentsWithStats(binary.astype(np.uint8))
        for i in range(1,count):
            x,y,w,h,_=(int(v) for v in stats[i])
            part=labels[y:y+h,x:x+w]==i
            band=outside[y:y+h,x:x+w]
            uncovered=part&band
            pixels=int(uncovered.sum())
            if pixels<minimum:continue
            covered=int((part&~band).sum())
            from_fringe=covered>pixels*2
            # Color studs are never "fringe" of the main product in the dark
            # layer sense; only neutral skirts use the elongated-fringe reject.
            fringe=from_fringe and layer!=2 and not recover_parts
            blobs=[uncovered] if not from_fringe else _uncovered_blobs(uncovered)
            for blob in blobs:
                if int(blob.sum())<minimum:continue
                region=_missing_box(blob,x,y,minimum,smallest,filled,fringe)
                if not region or region in seen:continue
                seen.add(region);regions.append(region)
    return regions


def _uncovered_blobs(uncovered):
    count,labels,stats,_=cv2.connectedComponentsWithStats(uncovered.astype(np.uint8))
    return [labels==j for j in range(1,count)]


def _missing_box(pixels,ox,oy,minimum,smallest,filled,from_fringe=False):
    total=int(pixels.sum())
    if total<minimum:return None
    ys,xs=np.where(pixels)
    l=int(xs.min())+int(ox);t=int(ys.min())+int(oy)
    r=int(xs.max())+int(ox)+1;b=int(ys.max())+int(oy)+1
    bw,bh=r-l,b-t
    # Soft contact-shadow skirts peeled off an already-found object are long and
    # thin; a real accessory beside the product is compact.
    if from_fringe and max(bw,bh)>min(bw,bh)*4:return None
    # Require a genuinely two-dimensional island; a thin floor reflection under
    # wheels is not a missing piece. A piece seen edge-on, such as a sword
    # lying flat, is thin but filled.
    if min(bw,bh)<smallest and not (min(bw,bh)>=filled and total>=max(bw,bh)*4):return None
    if total<max(bw,bh)*1.5:return None
    return (l,t,r,b)


def padded_box(box,shape,margin=.12):
    h,w=shape[:2];l,t,r,b=box
    pad=max(r-l,b-t)*margin
    return (max(0,int(np.floor(l-pad))),max(0,int(np.floor(t-pad))),
            min(w,int(np.ceil(r+pad))),min(h,int(np.ceil(b+pad))))


def clipped_sides(mask,box,shape):
    """Must run on raw prediction BEFORE frame-connected cleanup."""
    l,t,r,b=box;h,w=shape[:2]
    band=max(2,round(min(mask.shape)*.004))
    return (l>0 and bool(np.any(mask[:,:band]>.45)),
            t>0 and bool(np.any(mask[:band]>.45)),
            r<w and bool(np.any(mask[:,-band:]>.45)),
            b<h and bool(np.any(mask[-band:]>.45)))
