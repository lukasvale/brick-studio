"""Local, versioned recipe memory. Only human choices supply training labels."""
from pathlib import Path
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import re
import os
import sqlite3
import time
import uuid

import cv2
import numpy as np
from PIL import Image, ImageOps

VERSION = 1
FIELDS = {'exposure': (-2, 2), 'contrast': (-1, 1), 'whites': (-1, 1),
          'warmth': (-1, 1), 'sharpness': (0, 2), 'denoise': (0, 2), 'shadow': (0, 2)}
DEFAULTS = dict(exposure=.65, contrast=0., whites=0., warmth=.25, sharpness=.75, denoise=0., shadow=2.)
PHOTO_OPTIONS = {'shadow_method': ('classic', 'local'), 'shadow_detection': ('dark', 'soft')}
PHOTO_OPTIONS.update(aspect=('square','landscape','portrait','wide'),framing=('centered','fixed','both'),
                     centered_scale=('shared','independent'),mask_mode=('efficient','detailed'),
                     mask_backend=('onnx','mps'),small_parts=('true','false'),recover_parts=('true','false'),recover_white=('true','false'),recover_dark=('true','false'))
ROOT = Path(__file__).resolve().parent


@contextmanager
def connect(root=ROOT):
    folder = Path(os.environ.get('BRICK_STUDIO_LEARNING_ROOT', root)) / 'work/learning'
    folder.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(folder / 'history.sqlite3', timeout=30)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS events (event TEXT PRIMARY KEY, identity TEXT, source TEXT, '
               'family TEXT, kind TEXT, stamp REAL, features TEXT, recipe TEXT, labels TEXT, provenance TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS models (signature TEXT PRIMARY KEY, report TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS imports (path TEXT PRIMARY KEY, signature TEXT)')
    try:
        with db: yield db
    finally: db.close()


def features(image):
    """Identical inexpensive descriptor for saved originals and new RAW decodes.

    Subject hints exclude booth walls; their convex hull retains neutral/white
    parts between colored pieces. This is a matching descriptor, never a mask.
    """
    im = ImageOps.exif_transpose(image).convert('RGB')
    im.thumbnail((384, 384), Image.Resampling.LANCZOS)
    rgb = np.asarray(im).astype(np.float32) / 255
    from scene_detection import scene_evidence
    seeds = scene_evidence(im)
    h, w = seeds.shape
    seeds[:int(h*.12)] = 0
    seeds[:, :int(w*.10)] = 0
    seeds[:, int(w*.90):] = 0
    points = cv2.findNonZero((seeds > 0).astype('uint8'))
    region = np.zeros((h, w), np.uint8)
    if points is not None and len(points) >= 20:
        cv2.fillConvexPoly(region, cv2.convexHull(points), 1)
    else:
        region[int(h*.30):int(h*.85), int(w*.25):int(w*.75)] = 1
    pixels = rgb[region > 0]
    lum = pixels @ np.array([.2126, .7152, .0722])
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    hist = np.histogram(lum, bins=8, range=(0, 1))[0] / len(lum)
    hue = np.histogram(hsv[:, 0], bins=8, range=(0, 360), weights=hsv[:, 1])[0] / len(lum)
    edges = np.concatenate((rgb[:, :max(1, w//10)].reshape(-1, 3), rgb[:, -max(1, w//10):].reshape(-1, 3)))
    values = np.r_[np.quantile(lum, [.1, .25, .5, .75, .9]), hist, hue,
                   pixels.mean(0), hsv[:, 1].mean(), np.mean(lum > .92), np.median(edges, axis=0)]
    return np.round(values, 6).tolist()


def clean_recipe(recipe):
    result = {}
    for key, (lo, hi) in FIELDS.items():
        v = recipe.get(key, DEFAULTS[key])
        if v is None: v = DEFAULTS[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
            raise ValueError('Invalid learned adjustment: ' + key)
        result[key] = float(v)
    return result


def normalize(mapping, sources):
    if not isinstance(mapping, dict): raise ValueError('Adjustments must be an object')
    allowed = {str(Path(p).resolve()) for p in sources}
    result = {}
    for path, item in mapping.items():
        path = str(Path(path).resolve())
        if path not in allowed: continue
        if not isinstance(item, dict) or not isinstance(item.get('values'), dict): raise ValueError('Invalid photo adjustment')
        values = clean_recipe(item['values'])
        # Partial suggestions inherit unspecified fields from the user's recipe.
        values = {k: v for k, v in values.items() if k in item['values']}
        for k,(lo,hi) in {'fill':(.5,.94),'size':(512,6000)}.items():
            if k not in item['values']:continue
            v=item['values'][k]
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not lo<=v<=hi or (k=='size' and int(v)!=v):
                raise ValueError('Invalid photo adjustment: '+k)
            values[k]=int(v) if k=='size' else v
        manual = item.get('manual_fields', [])
        if not isinstance(manual, list) or any(k not in FIELDS for k in manual): raise ValueError('Invalid manual fields')
        options = item.get('options') or {}
        if not isinstance(options, dict) or any(k not in PHOTO_OPTIONS or v not in PHOTO_OPTIONS[k] for k,v in options.items()):
            raise ValueError('Invalid photo options')
        result[path] = dict(item, values=values, manual_fields=manual)
        if options: result[path]['options'] = options
    return result


def add_event(db, record, recipe, stamp, event, provenance='historical', adjustment=None):
    recipe = dict(recipe, **clean_recipe(recipe))
    labels = list(FIELDS) if adjustment is None else adjustment.get('manual_fields', [])
    descriptor = record.get('input_features')
    if descriptor is None:
        with Image.open(record['before']) as im: descriptor = features(im)
    if len(descriptor) != 29 or not np.isfinite(descriptor).all(): raise ValueError('Invalid image features')
    identity = record.get('id') or record['source']
    kind = record.get('photo_type', 'product')
    db.execute('INSERT OR IGNORE INTO events VALUES (?,?,?,?,?,?,?,?,?,?)',
               (event, identity, record['source'], str(Path(record['source']).parent), kind, stamp,
                json.dumps(descriptor), json.dumps(recipe), json.dumps(labels), provenance))


def record_export(record, recipe, adjustment=None, root=ROOT):
    """Commit only after all output variants and the manifest were saved."""
    with connect(root) as db:
        add_event(db, record, recipe, record['processed_at'], record['processing_event'],
                  'manual' if adjustment is None else 'assisted', adjustment)


def backfill(root=ROOT):
    counts = dict(manifests=0, imported=0, skipped=0, errors=[])
    files = sorted(set((Path(root)/'work/shoots').glob('*/manifest.json')) |
                   set((Path(root)/'output').glob('*/manifest.json')))
    with connect(root) as db:
        for file in files:
            counts['manifests'] += 1
            signature = 'legacy-v4:' + str(file.stat().st_mtime_ns) + ':' + str(file.stat().st_size)
            if db.execute('SELECT signature FROM imports WHERE path=?', (str(file),)).fetchone() == (signature,): continue
            try:
                m = json.loads(file.read_text())
                skipped_before = counts['skipped']
                for r in m.get('records', []):
                    try:
                        if r.get('package_cleanup') or m.get('settings', {}).get('package_cleanup'):
                            r = dict(r, photo_type='packaging')
                            db.execute("UPDATE events SET kind='packaging' WHERE identity=? AND source=? AND provenance='historical'",
                                       (r.get('id') or r['source'],r['source']))
                        recipe = r.get('effective_recipe') or dict(m.get('settings', {}), **r.get('recipe', {}))
                        if r.get('photo_type') == 'packaging' and not r.get('effective_recipe'): recipe = dict(DEFAULTS, **r.get('recipe', recipe))
                        if isinstance(recipe.get('shadow'), bool): recipe = dict(recipe, shadow=2. if recipe['shadow'] else 0.)
                        event = r.get('processing_event') or hashlib.sha256((str(file)+r['source']+json.dumps(recipe, sort_keys=True)).encode()).hexdigest()
                        if db.execute('SELECT 1 FROM events WHERE event=?', (event,)).fetchone():
                            db.execute('UPDATE events SET kind=?,recipe=? WHERE event=?',
                                       (r.get('photo_type','product'),json.dumps(dict(recipe,**clean_recipe(recipe))),event))
                            continue
                        add_event(db, r, recipe, r.get('processed_at', file.stat().st_mtime), event,
                                  'historical', r.get('adjustment'))
                        counts['imported'] += 1
                    except (OSError, ValueError, KeyError, TypeError) as exc:
                        counts['skipped'] += 1
                        counts['errors'].append(str(file)+': '+str(exc))
                if counts['skipped']==skipped_before:
                    db.execute('INSERT OR REPLACE INTO imports VALUES (?,?)', (str(file), signature))
                else: db.execute('DELETE FROM imports WHERE path=?',(str(file),))
            except (OSError, ValueError, TypeError) as exc: counts['errors'].append(str(file)+': '+str(exc))
    return counts


def fit_color_model(rows):
    """Regularized color/brightness regression, balanced by shoot, not angles."""
    families = {}
    for r in rows: families.setdefault(r['family'], []).append(r)
    x = np.array([np.median([r['features'] for r in rs],axis=0) for rs in families.values()])
    y = np.array([np.median([r['value'] for r in rs]) for rs in families.values()])
    center=x.mean(0);scale=np.maximum(x.std(0),.025);z=(x-center)/scale
    coefficients=np.linalg.solve(z.T@z+10*np.eye(z.shape[1]),z.T@(y-y.mean()))
    return dict(center=center.tolist(),scale=scale.tolist(),coefficients=coefficients.tolist(),
                intercept=float(y.mean()),minimum=float(y.min()),maximum=float(y.max()))


def color_prediction(model, descriptor):
    prediction=model['intercept']+((np.asarray(descriptor)-model['center'])/model['scale'])@np.asarray(model['coefficients'])
    return float(np.clip(prediction,model['minimum'],model['maximum']))


class Memory:
    def __init__(self, root=ROOT):
        # Latest human label per content + field; re-exports and 36 angles must
        # not outvote a correction or another product family.
        with connect(root) as db:
            rows = db.execute('SELECT identity,source,family,kind,stamp,features,recipe,labels FROM events ORDER BY stamp').fetchall()
        latest = {}
        self.excluded_fixtures = 0
        for identity, source, family, kind, stamp, descriptor, recipe, labels in rows:
            try:
                first = Path(source).relative_to(Path(root)/'work').parts[0].lower()
                if first.startswith('tmp') or 'test' in first or 'qa' in first:
                    self.excluded_fixtures += 1; continue
            except (ValueError,IndexError): pass
            for field in json.loads(labels):
                latest[(identity, kind, field)] = dict(identity=identity, source=source, family=family, kind=kind,
                    stamp=stamp, features=json.loads(descriptor), field=field, value=json.loads(recipe)[field])
        self.rows = list(latest.values())
        self.root = root
        self.calibration = None
        self.index = {}
        for kind in ('product', 'packaging'):
            for field in FIELDS:
                selected = [r for r in self.rows if r['kind']==kind and r['field']==field]
                self.index[(kind,field)] = (selected, np.array([r['features'] for r in selected]))

    def suggest(self, descriptor, kind='product', identity=None, exclude_family=None):
        values, evidence = {}, []
        for field in FIELDS:
            # Shadow strength depends on the selected shadow algorithm, not just color.
            if field == 'shadow': continue
            all_rows, matrix = self.index[(kind,field)]
            keep = [i for i,r in enumerate(all_rows) if r['family'] != exclude_family]
            rows = [all_rows[i] for i in keep]
            if not rows: continue
            matrix = matrix[keep]
            exact = [r for r in rows if identity and r['identity'] == identity]
            if exact:
                best = max(exact, key=lambda r:r['stamp'])
                values[field] = best['value']; evidence.append(dict(source=best['source'], distance=0., field=field))
                continue
            distances = np.sqrt(np.mean((matrix - descriptor)**2, axis=1))
            # One angle from each shoot, then up to five similar shoots.
            chosen, families = [], set()
            for i in np.argsort(distances):
                r = rows[i]
                if distances[i] > .14: break
                if r['family'] in families: continue
                families.add(r['family']); chosen.append((r, float(distances[i])))
                if len(chosen) == 5: break
            # Unfamiliar photos retain the established recipe.
            if len(chosen) < 3 or chosen[0][1] > .09: continue
            weights = np.array([1/(.025+d)**2 for _, d in chosen])
            targets = np.array([r['value'] for r, _ in chosen])
            estimate = float(np.average(targets, weights=weights))
            # Disagreeing neighbors cannot justify an automatic adjustment.
            if self.calibration is None and np.sqrt(np.average((targets-estimate)**2, weights=weights)) > .32: continue
            if self.calibration is not None:
                strategy = self.calibration.get(kind, {}).get(field, {}).get('strategy', 'default')
                if strategy == 'default': continue
                if strategy == 'similar' and np.sqrt(np.average((targets-estimate)**2,weights=weights)) > .32: continue
                if strategy == 'color': estimate = color_prediction(self.calibration[kind][field]['color_model'],descriptor)
                if strategy == 'personal':
                    family_values = {}
                    for r in rows: family_values.setdefault(r['family'], []).append(r['value'])
                    estimate = float(np.median([np.median(v) for v in family_values.values()]))
            values[field] = round(estimate, 4)
            evidence.append(dict(source=chosen[0][0]['source'], distance=round(chosen[0][1], 4), field=field))
        return dict(values=values, manual_fields=[], model_version=VERSION, model_revision=getattr(self,'revision',None),
                    confidence='matched' if identity and any(e['distance']==0 for e in evidence) else 'similar' if values else 'fallback',
                    examples=evidence)

    def calibrate(self):
        """Choose per-field strategies using held-out whole product folders.

        This measures agreement with recorded choices, not visual quality.
        Sparse fields stay manual; exact previous-photo corrections still win.
        """
        signature = hashlib.sha256(('strategy-v4:' + json.dumps([(r['identity'],r['family'],r['kind'],r['field'],r['stamp'],r['value']) for r in self.rows],sort_keys=True)).encode()).hexdigest()
        self.revision=signature
        with connect(self.root) as db:
            cached = db.execute('SELECT report FROM models WHERE signature=?', (signature,)).fetchone()
        if cached:
            self.calibration = json.loads(cached[0]); return self.calibration
        result = {}
        for kind in ('product','packaging'):
            families = {}
            for r in self.rows:
                if r['kind']==kind: families.setdefault(r['family'], {}).setdefault(r['identity'], {})[r['field']]=r
            errors = {k:[] for k in FIELDS}
            for family, photos in families.items():
                rows = photos[sorted(photos)[len(photos)//2]]
                descriptor = next(iter(rows.values()))['features']
                prediction = self.suggest(descriptor,kind,exclude_family=family)['values']
                for field, row in rows.items():
                    others = {}
                    for r in self.index[(kind,field)][0]:
                        if r['family']!=family: others.setdefault(r['family'], []).append(r['value'])
                    if not others: continue
                    prior = float(np.median([np.median(v) for v in others.values()]))
                    target = row['value']
                    model=fit_color_model([r for r in self.index[(kind,field)][0] if r['family']!=family])
                    errors[field].append([abs(prediction.get(field,DEFAULTS[field])-target),abs(prior-target),
                                          abs(DEFAULTS[field]-target),abs(color_prediction(model,descriptor)-target)])
            result[kind] = {}
            for field, samples in errors.items():
                if not samples: continue
                nearest, personal, baseline, color = np.mean(samples,axis=0)
                strategy = 'similar' if nearest < personal else 'personal'
                if color < min(nearest,personal)*.95: strategy='color'
                selected_error={'similar':nearest,'personal':personal,'color':color}[strategy]
                if len(samples)<8 or selected_error>=baseline*.98 or field=='shadow': strategy='default'
                result[kind][field] = dict(strategy=strategy,held_out_folders=len(samples),
                    similar_mae=float(nearest),personal_mae=float(personal),default_mae=float(baseline),color_mae=float(color))
                if strategy=='color': result[kind][field]['color_model']=fit_color_model(self.index[(kind,field)][0])
        self.calibration = result
        with connect(self.root) as db:
            db.execute('INSERT OR REPLACE INTO models VALUES (?,?)',(signature,json.dumps(result)))
        return result

    def report(self):
        report = dict(excluded_fixture_events=self.excluded_fixtures, training_photos=len({(r['identity'], r['kind']) for r in self.rows}),
                      product_families=len({r['family'] for r in self.rows}), fields={})
        for field in FIELDS:
            values = [r['value'] for r in self.rows if r['field'] == field]
            if values: report['fields'][field] = dict(samples=len(values), minimum=min(values), median=float(np.median(values)), maximum=max(values))
        return report


class InputAnalysis:
    """Reuse descriptors from exported originals or a previous matching run."""
    def __init__(self,root):
        self.root=root
        with connect(root) as db:
            db.execute('CREATE TABLE IF NOT EXISTS input_analysis (path TEXT PRIMARY KEY, signature TEXT, identity TEXT, descriptor TEXT)')
            self.known={identity:json.loads(value) for identity,value in db.execute('SELECT identity,features FROM events ORDER BY stamp')}

    def get(self,path):
        from source_identity import source_key
        from studio import decode,pil
        path=Path(path).resolve();stat=path.stat()
        signature=json.dumps([1,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns,stat.st_dev,stat.st_ino])
        with connect(self.root) as db:
            row=db.execute('SELECT identity,descriptor FROM input_analysis WHERE path=? AND signature=?',(str(path),signature)).fetchone()
        if row:return row[0],json.loads(row[1]),True
        identity=source_key(path,self.root)
        descriptor=self.known.get(identity)
        reused=descriptor is not None
        if descriptor is None:descriptor=features(pil(decode(path,half_size=True)))
        with connect(self.root) as db:
            db.execute('INSERT OR REPLACE INTO input_analysis VALUES (?,?,?,?)',(str(path),signature,identity,json.dumps(descriptor)))
        self.known[identity]=descriptor
        return identity,descriptor,reused


def folder_samples(items):
    """At most four turntable views, with a deterministic spread otherwise."""
    ordered=sorted(items,key=lambda item:item['source'])
    angles={}
    for item in ordered:
        match=re.search(r'(\d{1,3})deg',Path(item['source']).name,re.I)
        if match:angles[item['source']]=int(match[1])%360
    if len(angles)==len(ordered):
        remaining=list(ordered);chosen=[]
        for target in (0,90,180,270):
            if not remaining:break
            best=min(remaining,key=lambda item:min((angles[item['source']]-target)%360,(target-angles[item['source']])%360))
            chosen.append(best);remaining.remove(best)
        return chosen
    return [ordered[i] for i in sorted(set(np.linspace(0,len(ordered)-1,min(4,len(ordered)),dtype=int)))] if ordered else []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--progress',type=Path)
    args = parser.parse_args()
    request=json.loads(args.request.read_text()) if args.request else {'photos':[]}
    report=dict(adjustments={},errors=[])
    last_progress=0
    def progress(message,completed=0,force=False):
        nonlocal last_progress
        if not args.progress:return
        now=time.monotonic()
        if not force and now-last_progress<.5:return
        last_progress=now
        args.progress.parent.mkdir(parents=True,exist_ok=True)
        tmp=args.progress.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(report,message=message,completed=completed,total=len(request['photos']))))
        tmp.replace(args.progress)
    progress('Checking saved processing history…',force=True)
    imported = backfill(args.root)
    progress('Updating the suggestion model…',force=True)
    memory = Memory(args.root)
    validation = memory.calibrate()
    report.update(history=imported, analysis=memory.report(), validation=validation)
    if args.request:
        analysis=InputAnalysis(args.root)
        reused=0
        groups={}
        for item in request['photos']:
            groups.setdefault((str(Path(item['source']).resolve().parent),item.get('kind','product')),[]).append(item)
        completed=0;sampled=0
        for folder_number,((folder,kind),items) in enumerate(groups.items(),1):
            samples=folder_samples(items);predictions=[]
            for number,item in enumerate(samples,1):
                progress(f"Folder {folder_number}/{len(groups)} · sample {number}/{len(samples)} · {Path(folder).name}",completed,force=True)
                try:
                    identity,descriptor,hit=analysis.get(item['source']);reused+=int(hit);sampled+=1
                    predictions.append(memory.suggest(descriptor,kind,identity))
                except Exception as exc:report['errors'].append(dict(source=item['source'],error=str(exc)))
            # A common recipe for the rotation, robust to one unusually lit angle.
            values={field:round(float(np.median([p['values'][field] for p in predictions if field in p['values']])),4)
                    for field in FIELDS if field!='shadow' and any(field in p['values'] for p in predictions)}
            for item in items:
                try:
                    path=Path(item['source']).resolve();stat=path.stat()
                    report['adjustments'][str(path)]=dict(values=values,manual_fields=[],model_version=VERSION,
                        model_revision=memory.revision,confidence='folder' if values else 'fallback',
                        source_size=stat.st_size,source_mtime=stat.st_mtime,examples=[],sample_sources=[p['source'] for p in samples])
                except Exception as exc:
                    if not any(e['source']==item['source'] for e in report['errors']):report['errors'].append(dict(source=item['source'],error=str(exc)))
            completed+=len(items)
            progress(f"Applied {folder_number}/{len(groups)} folders · analysed {sampled} sample photos",completed,force=True)
        report['reused_analysis']=reused
        report['sampled_photos']=sampled
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix('.tmp')
    temp.write_text(json.dumps(report, indent=2)); temp.replace(args.output)
    print(json.dumps(dict(analysis=report['analysis'], imported=imported['imported'], errors=len(report['errors']))))


if __name__ == '__main__': main()
