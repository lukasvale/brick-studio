"""Opt-in recovery of dark parts the subject model handed to the floor shadow.

A black plate or bumper resting just above the booth floor sits directly on top
of its own contact shadow, and the model sometimes splits the two at the wrong
line. Measured on 31056-1 at 20 degrees, a column through the gap under the
front bumper reads: kept grey plate 0.25 -> lost black plastic 0.12 (flat) ->
sharp step -> floor shadow 0.20 (flat) -> lit floor 0.89.

The part is darker than the shadow below it, and a sharp step separates two flat
levels. A shadow alone does not produce that: it is one dark level that fades
out. So a pixel band is only returned to the subject when, in its own column,
it lies directly under the existing cutout, is genuinely dark, is noticeably
darker than the still-shadowed floor below it, and ends in a sharp step rather
than a ramp. Neighbouring columns must agree, because LEGO edges are straight.

Original RGB and already accepted alpha are never replaced; this only adds.
"""
import time

import cv2
import numpy as np

DARK_VERSION = 4

# Luminance weights for linear sRGB.
_LUMA = np.array([.2126, .7152, .0722], np.float32)


def _column_depth(profile, floor, step_window, max_depth):
    """Rows of dark part below the cutout in one column, or 0.

    `profile` starts at the first row under the existing mask edge.
    """
    n = len(profile)
    limit = min(max_depth, n - step_window - 4)
    for k in range(2, limit):
        low, high = profile[k], profile[k + step_window]
        # A sharp rise: at least 25% brighter within a few rows.
        if high - low < max(.03, low * .25):
            continue
        before = float(np.median(profile[:k + 1]))
        tail = profile[k + step_window:k + step_window + 12]
        if len(tail) < 4:
            return 0
        after = float(np.median(tail))
        # What lies below the step must still be shadow, not lit floor:
        # a dark band that ends on bright floor is exactly as likely to be
        # the shadow itself, so it is left for manual review.
        if after >= floor * .55:
            return 0
        # The part must be genuinely dark and darker than that shadow.
        if before >= floor * .25 or before >= after * .78:
            return 0
        return k + step_window // 2
    return 0


def find_dark_parts(rgb, mask):
    """Return (added alpha, boxes) for dark bands under the lower outline."""
    solid = mask > .5
    ys, xs = np.where(solid)
    if not len(xs):
        return np.zeros(mask.shape, np.float32), []
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    span = max(x1 - x0 + 1, y1 - y0 + 1)
    lum = cv2.GaussianBlur(np.ascontiguousarray(rgb @ _LUMA), (0, 0), 1.0)
    h, w = mask.shape

    # Lit floor level, measured beside and below the subject.
    t = max(0, y0 + (y1 - y0) // 2)
    b = min(h, y1 + max(8, round(span * .12)))
    l, r = max(0, x0 - round(span * .1)), min(w, x1 + round(span * .1) + 1)
    outside = lum[t:b, l:r][mask[t:b, l:r] < .02]
    if outside.size < 100:
        return np.zeros(mask.shape, np.float32), []
    floor = float(np.percentile(outside, 95))

    # Edge sharpness comes from the lens, not from how big the model is, and the
    # rig shoots every set from the same distance. A span-scaled window grew to
    # 14 px on a 2,343 px model (71776-1) and read a gentle 45 px shadow ramp
    # under the chassis as a sharp edge; the real bumper edge on 31056-1 rises
    # within ~5 px. Likewise a plate's thickness in pixels is fixed by the rig
    # (about 25-35 px there), so the search depth is capped, not scaled freely.
    step_window = 4
    max_depth = int(np.clip(round(span * .07), 12, 64))
    # Only the lower part of the outline can sit on its own floor shadow.
    lower_limit = y0 + (y1 - y0) * .5

    depth = np.zeros(w, np.int32)
    edge = np.full(w, -1, np.int32)
    for x in range(x0, x1 + 1):
        col = np.flatnonzero(solid[:, x])
        if not len(col):
            continue
        bottom = int(col.max())
        if bottom < lower_limit:
            continue
        start = bottom + 1
        stop = min(h, start + max_depth + step_window + 16)
        if stop - start < step_window + 8:
            continue
        d = _column_depth(lum[start:stop, x], floor, step_window, max_depth)
        if d:
            depth[x] = d
            edge[x] = bottom

    # Neighbouring columns must agree: a real part is a run of similar depths.
    # And it must be long. Measured on 31056-1, every true bumper/chassis band
    # was 90-177 px wide while every false one (floor seen under the car beside
    # a tyre, which also shows a hard umbra plus a lighter outer shadow) was
    # 26-58 px. Short dark pockets are left for manual review.
    min_run = max(12, round(span * .12))
    added = np.zeros(mask.shape, np.float32)
    boxes = []
    x = x0
    while x <= x1:
        if not depth[x]:
            x += 1
            continue
        run_start = x
        while x <= x1 and depth[x]:
            x += 1
        cols = np.arange(run_start, x)
        if len(cols) < min_run:
            continue
        smooth = cv2.medianBlur(depth[cols].astype(np.uint8).reshape(1, -1),
                                max(3, (min(len(cols), round(span * .02)) // 2) * 2 + 1)).ravel()
        # Reject runs whose thickness jumps around: shadow penumbra does, plastic does not.
        if np.std(smooth) > max(3, np.median(smooth) * .45):
            continue
        # A plate or bumper edge is a strip, clearly wider than it is deep.
        if len(cols) < np.median(smooth) * 2.2:
            continue
        # LEGO edges are straight: fit the bottom as a line so it does not
        # stair-step with the per-column estimate. Keep the measured shape only
        # if a line clearly does not describe it.
        tops = edge[cols] + 1
        measured = (edge[cols] + smooth).astype(np.float64)
        slope, offset = np.polyfit(cols, measured, 1)
        line = slope * cols + offset
        bottoms = line if np.max(np.abs(line - measured)) <= max(2, np.median(smooth) * .3) else measured
        # Close hairline gaps to neighbouring cutout at each end, so a strip
        # meets the fender or wheel beside it instead of leaving a white sliver.
        # Reach a couple of columns INTO that neighbour too: its own outline is
        # antialiased, so stopping exactly at it left a half-transparent column.
        overlap = 2   # also how far the strip reaches up into the existing cutout
        gap = max(3, round(span * .01))
        ext_cols, ext_bottom, ext_top = list(cols), list(bottoms), list(tops)
        for end, step in ((0, -1), (-1, 1)):
            c, bot, top = int(cols[end]), float(bottoms[end]), int(tops[end])
            for k in range(1, gap + 1):
                nc = c + step * k
                if nc < 0 or nc >= w:
                    break
                if np.any(solid[top:int(bot) + 1, nc]):
                    for j in range(1, min(k + overlap, gap + overlap) + 1):
                        if 0 <= c + step * j < w:
                            ext_cols.append(c + step * j); ext_bottom.append(bot); ext_top.append(top)
                    break
        for c, top, bot in zip(ext_cols, ext_top, ext_bottom):
            whole = int(np.floor(bot))
            added[max(0, top - overlap):whole + 1, c] = 1
            if whole + 1 < h:
                # Sub-pixel coverage gives one antialiased row along the straight edge.
                added[whole + 1, c] = max(added[whole + 1, c], bot - whole)
        boxes.append([int(min(ext_cols)), int(min(ext_top)), int(max(ext_cols) + 1), int(np.ceil(max(ext_bottom)) + 1)])
    return np.clip(added, 0, 1), boxes


def recover_dark_parts(rgb, mask, on_phase=None):
    started = time.monotonic()
    if not np.any(mask > .5):
        raise ValueError('No existing subject mask for dark-part recovery')
    if on_phase:
        on_phase('Recovering dark parts near the floor', .86)
    added, boxes = find_dark_parts(rgb, mask)
    result = np.maximum(mask, added)
    return result, dict(
        dark_version=DARK_VERSION, dark_recovery_boxes=boxes,
        dark_added_pixels=int(np.count_nonzero((result > .5) & (mask <= .5))),
        dark_seconds=round(time.monotonic() - started, 3))
