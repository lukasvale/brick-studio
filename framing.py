"""Shared source-coordinate framing; never equalize silhouette heights."""
import numpy as np


def mask_bounds(mask):
    ys, xs = np.where(mask > .5)
    if not len(xs):
        raise ValueError('Empty subject mask')
    return [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)]


def batch_frame(records, width, height, fill, mode='centered', independent=False):
    """Use source-width units so equivalent full/half-resolution inputs align.

    independent centred framing gives each photo its own scale, so every view
    reaches the product fill instead of the widest view setting one scale.
    """
    if mode not in ('centered', 'fixed'):
        raise ValueError('Unknown framing mode')
    boxes = np.array([np.array(r['bbox'], dtype=float)/r['source_size'][0] for r in records])
    if not len(boxes):
        raise ValueError('No photos available for framing')
    if independent and mode == 'centered':
        return dict(mode=mode, scale=None, center=None, independent=True)
    low = boxes[:, :2].min(axis=0)
    high = boxes[:, 2:].max(axis=0)
    extent = high-low if mode == 'fixed' else (boxes[:, 2:]-boxes[:, :2]).max(axis=0)
    return dict(mode=mode, scale=float(min(width*fill/extent[0], height*fill/extent[1])),
                center=((low+high)/2).tolist() if mode == 'fixed' else None)


def placement(frame, bbox, source_width, origin=(0, 0)):
    """Return pixel scale and center in the supplied (possibly cropped) RGB.

    A None scale means the photo is scaled to the product fill on its own.
    """
    scale = None if frame['scale'] is None else frame['scale']/source_width
    center = frame['center']
    if center is None:
        center = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
    else:
        center = [v*source_width for v in center]
    return scale, [center[0]-origin[0], center[1]-origin[1]]
