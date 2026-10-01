"""Content identities with one-time recovery of legacy path-keyed masks."""
import hashlib
import json
import sqlite3
from functools import lru_cache
from pathlib import Path


def legacy_key(path,size,mtime):
    return hashlib.sha256(f'{path}:{size}:{mtime}:mask-v1'.encode()).hexdigest()[:20]


@lru_cache(maxsize=8)
def history(root):
    root=Path(root);result={}
    manifests=list((root/'work/shoots').glob('*/manifest.json'))+list((root/'output').glob('*/manifest.json'))
    for file in manifests:
        try:
            data=json.loads(file.read_text())
            pairs=list(data.get('source_keys',{}).items())
            pairs += [(r['source'],r['id']) for r in data.get('records',[]) if 'source' in r and 'id' in r]
            for path,key in pairs:result.setdefault(Path(path).name,set()).add((path,key))
        except (OSError,ValueError,TypeError):continue
    return result


def has_mask(folder,key):
    return any(folder.glob(f'{key}*coverage-v3*.png'))


@lru_cache(maxsize=2048)
def identify(path,root,size,mtime,ctime,device,inode):
    with open(path,'rb') as file:digest=hashlib.file_digest(file,'sha256').hexdigest()
    folder=Path(root)/'work/masks';folder.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(folder/'source-identities.sqlite3',timeout=15) as db:
        db.execute('CREATE TABLE IF NOT EXISTS identities (digest TEXT PRIMARY KEY, cache_key TEXT NOT NULL)')
        row=db.execute('SELECT cache_key FROM identities WHERE digest=?',(digest,)).fetchone()
        if row:return row[0]
        key=legacy_key(path,size,mtime)
        if not has_mask(folder,key):
            key=digest[:20]
            # Old caches have no content digest. Require their recorded old
            # path + current size/timestamp to reproduce the exact old key.
            for old,old_key in sorted(history(root).get(Path(path).name,())):
                if legacy_key(old,size,mtime)==old_key and has_mask(folder,old_key):
                    key=old_key;break
        db.execute('INSERT OR IGNORE INTO identities VALUES (?,?)',(digest,key))
        return db.execute('SELECT cache_key FROM identities WHERE digest=?',(digest,)).fetchone()[0]


def source_key(path,root):
    path=Path(path).resolve();info=path.stat()
    # ctime/inode also invalidate the in-memory digest after same-size edits
    # or replacement, even if an editor restores the modification timestamp.
    return identify(str(path),str(root),info.st_size,info.st_mtime_ns,info.st_ctime_ns,info.st_dev,info.st_ino)
