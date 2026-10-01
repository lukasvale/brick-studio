"""Remove weak neutral mask debris and recover enclosed colored omissions.

Designed for the neutral lightbox workflow. Leaves neutral backdrop and white
mesh openings unchanged; uses photographed pixels, not another AI inference.
"""
import cv2
import numpy as np


MARGIN = .008        # a real post is at least this much darker than clear floor
MIN_EVIDENCE = .75   # and is darker along nearly the whole gap, not in patches
STUB_LIMIT = 120     # a gap flanked by anything wider than this is open floor


def _stub_width(row, x, limit):
    """Width of the solid run holding x, stopping once it exceeds limit."""
    if not row[x]:
        return 0
    n = row.shape[0]
    left = right = x
    while left > 0 and row[left - 1] and x - left <= limit:
        left -= 1
    while right < n - 1 and row[right + 1] and right - x <= limit:
        right += 1
    return right - left + 1


def repair_white_bridges(rgb, mask, max_gap=320, stub_limit=STUB_LIMIT):
    """Reconnect thin struts the model left with holes.

    Under a lightbox a thin white LEGO post is only slightly darker than the
    open backdrop, so the model keeps the bar and the body but drops the
    mid-strut. A gap is bridged only where the photograph itself shows a
    continuous darker line between two thin stubs of that same post. A gap
    between two wide parts, such as an engine above a baseplate, is open
    floor: bridging those drew vertical bars across the backdrop.
    """
    solid = mask >= .85
    ys, xs = np.where(solid)
    if not len(xs):
        return mask
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    pad = 16
    t, b = max(0, y0 - pad), min(mask.shape[0], y1 + pad + 1)
    l, r = max(0, x0 - pad), min(mask.shape[1], x1 + pad + 1)
    m = mask[t:b, l:r].copy()
    pix = rgb[t:b, l:r]
    h, w = m.shape
    solid_c = m >= .85
    luma = pix.mean(2).astype(np.float32)
    chroma = (pix.max(2) - pix.min(2)).astype(np.float32)
    uncovered = m < .15
    open_luma = float(np.percentile(luma[uncovered], 85)) if uncovered.any() else 1.

    fill = np.zeros((h, w), bool)
    for x in range(w):
        on = np.flatnonzero(solid_c[:, x])
        if len(on) < 2:
            continue
        breaks = np.where(np.diff(on) > 1)[0]
        if len(breaks) == 0:
            continue
        starts = np.concatenate(([on[0]], on[breaks + 1]))
        ends = np.concatenate((on[breaks], [on[-1]]))
        for i in range(len(starts) - 1):
            lo, hi = int(ends[i] + 1), int(starts[i + 1] - 1)
            if not 10 <= hi - lo + 1 <= max_gap:
                continue
            stub = min(_stub_width(solid_c[lo - 1], x, stub_limit) if lo > 0 else 0,
                       _stub_width(solid_c[hi + 1], x, stub_limit) if hi + 1 < h else 0)
            if not 0 < stub <= stub_limit:
                continue
            # Sample clear backdrop beyond the post's own width on both sides;
            # a reference taken inside the post would show no contrast at all.
            offset = stub // 2 + 14
            left, right = x - offset, x + offset
            if left < 0 or right >= w:
                continue
            gap = slice(lo, hi + 1)
            centre = luma[gap, x]
            side = np.minimum(luma[gap, left], luma[gap, right])
            darker = (centre < side - MARGIN) & (chroma[gap, x] < .18)
            if darker.mean() >= MIN_EVIDENCE:
                fill[gap, x] = darker

    if fill.any():
        # A real post spans several columns. Isolated 1-2 px columns are the
        # antialiased edge of a slanted rod or seam, and filling them drew thin
        # grey bars down the backdrop (71784-1, 30611-1).
        n, lab, st, _ = cv2.connectedComponentsWithStats(fill.astype(np.uint8))
        narrow = np.zeros(n, bool)
        narrow[1:] = st[1:, cv2.CC_STAT_WIDTH] < 4
        # A gap at ground level between two parts is floor darkened by their
        # contact shadow, not a broken post (30611-1 at 240 and 250 degrees).
        # The struts this repair exists for are raised off the floor.
        ground = h - 1 - (y1 - y0) * .2 - pad
        narrow[1:] |= st[1:, cv2.CC_STAT_TOP] + st[1:, cv2.CC_STAT_HEIGHT] > ground
        fill &= ~narrow[lab]
        m = np.where(fill, np.maximum(m, .98), m)

    # White plastic with soft alpha vanishes on a white export; harden it.
    # Dark plastic and contact-shadow edges are also low-chroma, so a luma
    # gate is required: snapping those to opaque leaves a jagged cut.
    s = m >= .85
    near = cv2.dilate(s.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    white_plastic = (chroma < .14) & (luma > open_luma - .08) & (luma < open_luma - .003)
    soft = near & white_plastic & (m > .12) & (m < .9)
    m = np.where(soft, np.maximum(m, .95), m)

    out = mask.copy()
    out[t:b, l:r] = m
    return out


def refine_colored_edges(rgb,mask):
    ys,xs=np.where(mask>.85)
    if not len(xs):return mask
    span=max(int(xs.max()-xs.min()),int(ys.max()-ys.min()))
    radius=max(3,min(24,round(span*.012)))
    x,y,w,h=cv2.boundingRect((mask>.02).astype(np.uint8))
    l=max(0,x-radius);r=min(mask.shape[1],x+w+radius)
    t=max(0,y-radius);b=min(mask.shape[0],y+h+radius)
    original=mask[t:b,l:r];pixels=rgb[t:b,l:r]
    # Patchy, low-confidence floor/shadow fragments are not opaque plastic.
    # Keep opaque pixels and their antialiasing band, plus colored thin parts.
    # This does not erode every contour or discard detached solid components.
    luma=pixels.mean(2)
    chroma=np.ptp(pixels,axis=2)
    # Black/brown parts near a contact are low-chroma like the floor, and the
    # model often scores them only as medium. Treat that compact dark core as
    # product so the silhouette is not eaten back to a jagged opaque remnant.
    opaque=original>=.9
    dark_plastic=(original>=.45)&(luma<.38)&(chroma<.16)
    solid=opaque|dark_plastic
    edge_distance=cv2.distanceTransform((~solid).astype(np.uint8),cv2.DIST_L2,5)
    edge_width=max(3,min(6,span*.002))
    transition=np.clip((edge_width-edge_distance)/(edge_width*.5),0,1)
    bright_floor=(chroma<.10)&(luma>.55)
    dark_wisp=(chroma<.10)&(luma<=.55)&~dark_plastic
    # Preserve soft transparency enclosed by an opaque frame (windscreens,
    # windows). Only exposed floor fragments are eligible for removal.
    outside=np.pad((~solid).astype(np.uint8),1,constant_values=1)
    cv2.floodFill(outside,None,(0,0),2)
    enclosed_soft=outside[1:-1,1:-1]==1
    debris=(original<.9)&(bright_floor|dark_wisp)&~enclosed_soft
    alpha=np.where(debris,original*transition,original)
    anchor=alpha>.85
    distance=cv2.distanceTransform((~anchor).astype(np.uint8),cv2.DIST_L2,5)
    high=pixels.max(2);chroma=high-pixels.min(2);sat=chroma/(high+.001)
    confidence=np.minimum(np.clip((chroma-.08)/.12,0,1),np.clip((sat-.22)/.20,0,1))
    confidence*=np.clip((high-.07)/.08,0,1)
    # Repair small holes/concavities, not outward reflections below a base.
    enclosed=cv2.morphologyEx(anchor.astype(np.uint8),cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1)),
        borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    candidate=(confidence>.1)&(distance<=radius)&enclosed
    # A true omission has plastic on both sides of it. Coloured glow on the floor
    # hangs under a part with nothing below it; closing only called it enclosed
    # because a separate piece stood beside it (60072-1 at 10 degrees: the
    # bucket's yellow bounce between its teeth and the cones became opaque, and
    # the shadow then started at that hard, lumpy line).
    a8=anchor.astype(np.uint8)
    above=cv2.dilate(a8,np.ones((radius+1,1),np.uint8),anchor=(0,radius))>0
    # Plastic below only counts if no open, lit floor separates it: a cone base
    # standing a few pixels under the glow does not enclose it.
    clear=(original<.15)&(chroma<.10)&(luma>.55)
    reach_below=np.full(anchor.shape[1],np.inf,np.float32)
    below=np.zeros(anchor.shape,bool)
    for yy in range(anchor.shape[0]-1,-1,-1):
        below[yy]=reach_below<=radius
        reach_below=np.where(anchor[yy],0,np.where(clear[yy],np.inf,reach_below+1))
    # Only at floor level: a minifigure's hand curl also hangs over nothing, but
    # well above the ground. Glow lies within a few percent of the model's size
    # of where nearby parts meet the floor (about 30 px there, the hand 160 px).
    hh,ww=anchor.shape
    rows=np.arange(hh)[:,None]
    lowest=np.where(anchor.any(0),hh-1-np.argmax(anchor[::-1],axis=0),-1).astype(np.float32)
    reach=max(3,round(span*.08))
    ground=cv2.dilate(lowest[None,:],np.ones((1,reach*2+1),np.uint8))[0]
    near_floor=rows>=ground[None,:]-span*.06
    candidate&=~(above&~below&near_floor)
    count,labels=cv2.connectedComponents((candidate|anchor).astype(np.uint8))
    anchored=np.zeros(count,bool);anchored[np.unique(labels[anchor])]=True;anchored[0]=False
    repair=np.where(candidate&anchored[labels],confidence,0)
    result=mask.copy();result[t:b,l:r]=np.where(opaque,original,np.maximum(alpha,repair))
    return result


def smooth_contact_edges(rgb, mask, sigma=.75):
    """Restore a 1px antialiased cut on dark silhouettes.

    Floor cleanup and a hard mask both leave stair-steps on black/brown
    parts, which show as a jagged cut against the contact shadow. Bright
    floor is left empty, so the shadow compositor still owns that region.
    """
    luma=rgb.mean(2).astype(np.float32)
    chroma=np.ptp(rgb,axis=2).astype(np.float32)
    dark=(chroma<.18)&(luma<.42)
    solid=mask>=.85
    if not np.any(solid):return mask
    dist_out=cv2.distanceTransform((~solid).astype(np.uint8),cv2.DIST_L2,5)
    dist_in=cv2.distanceTransform(solid.astype(np.uint8),cv2.DIST_L2,5)
    band=dark&((dist_in<=1.5)|(dist_out<=1.2))
    # Only grow onto pixels as dark as plastic, not shaded floor.
    band&=(dist_in>0)|(luma<.28)
    if not np.any(band):return mask
    return np.where(band,cv2.GaussianBlur(mask,(0,0),sigma),mask)


def drop_floor_islands(rgb, mask):
    """Remove detached, soft-edged, colourless islands lying at floor level.

    The subject model sometimes cuts out a floor mark or a shadow scrap as a
    separate 'piece'. On the current rig a mark at the turntable's centre shows
    through whenever the model does not cover it, at the same pixels in every
    set, rotating with each angle. Measured on 30274, 41360, 60067 and 60411:
    those islands have a blurred outline (edge sharpness 0.13-0.19) and are
    about as bright as the floor around them. Real loose parts (handcuffs,
    wheels, small pieces) have crisp outlines (0.41-0.56), so a soft outline is
    the deciding test, backed by low colour and floor-level position.
    """
    b = (mask > .5).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b)
    if n < 3:
        return mask
    main = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    ys = np.where(lab == main)[0]
    top, bot = int(ys.min()), int(ys.max())
    lum = cv2.GaussianBlur(rgb.mean(2).astype(np.float32), (0, 0), 1)
    chroma = np.ptp(rgb, axis=2)
    grad = np.hypot(cv2.Sobel(lum, cv2.CV_32F, 1, 0), cv2.Sobel(lum, cv2.CV_32F, 0, 1))
    out = mask.copy()
    k3 = np.ones((3, 3), np.uint8)
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if i == main or area > st[main, cv2.CC_STAT_AREA] * .03:
            continue
        if y + h < top + (bot - top) * .6:          # floor level only
            continue
        pad = 20
        l, t = max(0, x - pad), max(0, y - pad)
        r, bb = min(b.shape[1], x + w + pad), min(b.shape[0], y + h + pad)
        comp = (lab[t:bb, l:r] == i).astype(np.uint8)
        edge = (cv2.dilate(comp, k3) > 0) & ~(cv2.erode(comp, k3) > 0)
        if np.median(grad[t:bb, l:r][edge]) >= .28:
            continue                                # crisp outline: a real part
        if np.median(chroma[t:bb, l:r][comp > 0]) >= .12:
            continue                                # coloured: a real part
        # Its soft antialiased rim goes too, or a faint dotted outline of the
        # removed blob stays on the white background.
        rim = (cv2.dilate(comp, np.ones((7, 7), np.uint8)) > 0) & ~(b[t:bb, l:r] > 0)
        out[t:bb, l:r] = np.where((comp > 0) | rim, 0, out[t:bb, l:r])
    return out
