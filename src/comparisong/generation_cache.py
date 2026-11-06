"""Small persistent fingerprint cache; immutable source files during a run."""
from pathlib import Path
import json


class FingerprintCache:
    def __init__(self,path):
        self.path=Path(path);self.changed=0
        try:self.entries=json.loads(self.path.read_text())
        except (OSError,ValueError):self.entries={}

    def get(self,path,compute):
        path=Path(path).resolve();stat=path.stat()
        key=str(path);stamp=[stat.st_size,stat.st_mtime_ns]
        cached=self.entries.get(key,{})
        if cached.get('stamp')==stamp:return cached['digest']
        digest=compute(str(path))
        self.entries[key]=dict(stamp=stamp,digest=digest);self.changed+=1
        if self.changed>=50:self.save()
        return digest

    def save(self):
        if not self.changed:return
        self.path.parent.mkdir(parents=True,exist_ok=True)
        temp=self.path.with_suffix('.tmp.json')
        temp.write_text(json.dumps(self.entries));temp.replace(self.path);self.changed=0
