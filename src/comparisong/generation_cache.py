"""Small persistent fingerprint cache; immutable source files during a run."""
from pathlib import Path
import json
import os
from threading import RLock


class FingerprintCache:
    def __init__(self,path):
        self.path=Path(path);self.changed=0;self.lock=RLock()
        try:self.entries=json.loads(self.path.read_text())
        except (OSError,ValueError):self.entries={}

    def peek(self,path):
        """Fast resume assumes frozen inputs; do not stat every source again."""
        with self.lock:return self.entries.get(os.path.abspath(path),{}).get('digest')

    def get(self,path,compute):
        # realpath/resolve performs repeated parent lstat calls on Drive FUSE.
        # Absolute spelling is enough: the content digest still keys results.
        path=Path(os.path.abspath(path));stat=path.stat()
        key=str(path);stamp=[stat.st_size,stat.st_mtime_ns]
        with self.lock:cached=self.entries.get(key,{})
        if cached.get('stamp')==stamp:return cached['digest']
        digest=compute(str(path))
        with self.lock:
            self.entries[key]=dict(stamp=stamp,digest=digest);self.changed+=1
            if self.changed>=50:self.save()
        return digest

    def save(self):
        with self.lock:
            if not self.changed:return
            self.path.parent.mkdir(parents=True,exist_ok=True)
            temp=self.path.with_suffix('.tmp.json')
            temp.write_text(json.dumps(self.entries));temp.replace(self.path);self.changed=0
