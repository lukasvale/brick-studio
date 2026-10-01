"""Bounded, rebuildable preview cache shared across app sessions."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid


def key(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


@contextmanager
def locked(folder):
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'cache.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        yield


def atomic_json(path,value):
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps(value));tmp.replace(path)


def trim(base,keep,budget=2*1024**3):
    """Evict only our generated entries, skipping any in active use."""
    folders=[p for kind in ('prepared','rendered') for p in (base/kind).glob('*') if p.is_dir() and not p.is_symlink()]
    sizes={};stamps={}
    for p in folders:
        try:
            sizes[p]=sum(f.stat().st_size for f in p.iterdir() if f.is_file())
            stamps[p]=p.stat().st_mtime
        except OSError:sizes.pop(p,None)
    total=sum(sizes.values())
    for folder in sorted(sizes,key=lambda p:stamps[p]):
        if total<=budget:break
        if folder==keep:continue
        try:
            with (folder/'cache.lock').open('a') as handle:
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                shutil.rmtree(folder)
                total-=sizes[folder]
        except (OSError,FileNotFoundError):continue


def touch(folder):
    os.utime(folder,None)
