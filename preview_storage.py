"""Session-only preview storage. Persistent subject masks live elsewhere."""
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'work/live-preview'

def alive(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False
    except PermissionError:return True

def remove_abandoned(base=BASE):
    base.mkdir(parents=True,exist_ok=True)
    for folder in base.glob('session-*'):
        if folder.is_symlink() or not folder.is_dir():continue
        try:pid=int((folder/'owner.pid').read_text())
        except (OSError,ValueError):continue
        if pid>0 and not alive(pid):shutil.rmtree(folder,ignore_errors=True)

def remove_legacy():
    # An older open app may still display these files. Wait until only the
    # current app/worker is using this project before removing the old cache.
    try:
        rows=subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines()
        for row in rows:
            parts=row.strip().split(None,1)
            if len(parts)<2:continue
            pid=int(parts[0]);command=parts[1]
            if pid in {os.getpid(),os.getppid()}:continue
            if command.startswith(str(ROOT/'Brick Studio.app/Contents/MacOS/BrickStudio')) or (str(ROOT/'preview_worker.py') in command and 'Python' in command):return
    except (OSError,ValueError,subprocess.SubprocessError):return
    for path in BASE.iterdir():
        if path.is_symlink():continue
        if path.is_dir() and re.fullmatch(r'[0-9a-f]{20}',path.name):shutil.rmtree(path)
        elif path.is_file() and re.fullmatch(r'preview-\d+-\d+\.(jpg|tmp)',path.name):path.unlink()

    thumbnails=ROOT/'work/previews'
    if thumbnails.is_dir() and not thumbnails.is_symlink():shutil.rmtree(thumbnails)

class PreviewStorage:
    def __init__(self, directory=None, owner_pid=None):
        remove_abandoned()
        remove_legacy()
        self.owned=directory is None
        self.path=Path(directory).resolve() if directory else BASE/f'session-{uuid.uuid4()}'
        if self.path.parent!=BASE.resolve() or not self.path.name.startswith('session-'):
            raise ValueError('Preview session must be inside work/live-preview')
        self.path.mkdir(parents=True,exist_ok=True)
        (self.path/'owner.pid').write_text(str(owner_pid or os.getpid()))
    def close(self):
        if self.owned:shutil.rmtree(self.path,ignore_errors=True)
