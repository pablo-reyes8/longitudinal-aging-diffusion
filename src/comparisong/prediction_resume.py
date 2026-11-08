"""Bulk metadata inventory and persistent validation for Drive-backed outputs."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from threading import RLock
from PIL import Image


class ResumeIndex:
    def __init__(self,path,root,identities,*,workers=16):
        self.path=Path(path);self.root=Path(root);self.lock=RLock();self.dirty=False
        try:self.records=json.loads(self.path.read_text()).get('records',{})
        except (ValueError,OSError):self.records={}
        def listing(identity):
            try:
                with os.scandir(self.root/str(identity)) as entries:return str(identity),{entry.name:entry for entry in entries}
            except (FileNotFoundError,NotADirectoryError):return str(identity),{}
        try:
            with os.scandir(self.root) as entries:available={entry.name for entry in entries}
        except FileNotFoundError:available=set()
        wanted=[identity for identity in dict.fromkeys(identities) if str(identity) in available]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            self.inventory=dict(pool.map(listing,wanted))

    @staticmethod
    def stamp(entry):
        stat=entry.stat();return [stat.st_size,stat.st_mtime_ns]

    def inspect(self,path,signature,*,verify=False):
        path=Path(path);key=path.relative_to(self.root).as_posix()
        folder=self.inventory.get(path.parent.name,{})
        marker=folder.get(path.with_suffix('.json').name)
        if marker is None:return None,False,'missing_marker'
        try:marker_stamp=self.stamp(marker)
        except OSError:return None,False,'missing_marker'
        with self.lock:previous=self.records.get(key,{})
        cached=previous.get('marker') if previous.get('marker_stamp')==marker_stamp else None
        if cached is None:
            try:cached=path.with_suffix('.json').read_text();cached=json.loads(cached)
            except (OSError,ValueError):return None,False,'invalid_marker'
        if cached.get('signature')!=signature:return cached,False,'signature_mismatch'
        if cached.get('status')=='failed':
            self._store(key,cached,marker_stamp,None)
            return cached,True,'cached_failure'
        if cached.get('status','ok')!='ok':return cached,False,'invalid_marker'
        image=folder.get(path.name)
        if image is None:return cached,False,'missing_png'
        try:png_stamp=self.stamp(image)
        except OSError:return cached,False,'missing_png'
        if not png_stamp[0]:return cached,False,'corrupt_png'
        if verify or previous.get('png_stamp')!=png_stamp or previous.get('marker_stamp')!=marker_stamp:
            try:
                with Image.open(path) as opened:opened.verify()
            except (OSError,ValueError,SyntaxError):return cached,False,'corrupt_png'
        self._store(key,cached,marker_stamp,png_stamp)
        return cached,True,'ok'

    def fast_inspect(self,path):
        """Legacy import for an unchanged run: filenames only, no PNG/JSON opens.

        Failed markers without PNGs are read to preserve retry_failed semantics.
        Once imported, ProgressResume skips even this directory inventory.
        """
        path=Path(path);folder=self.inventory.get(path.parent.name,{})
        marker=path.with_suffix('.json').name
        if path.name in folder and marker in folder:return dict(status='ok'),True,'legacy_pair'
        if marker in folder:
            try:
                data=json.loads(path.with_suffix('.json').read_text())
                if data.get('status')=='failed':return data,True,'cached_failure'
            except (OSError,ValueError):pass
        return None,False,'missing_pair'

    def _store(self,key,marker,marker_stamp,png_stamp):
        record=dict(marker=marker,marker_stamp=marker_stamp,png_stamp=png_stamp)
        with self.lock:
            if self.records.get(key)!=record:self.records[key]=record;self.dirty=True

    def record(self,path,marker):
        """Record an atomic PNG+marker write; no PNG decode is needed again."""
        path=Path(path)
        try:
            stamp=self.stamp(path.with_suffix('.json'))
            png_stamp=self.stamp(path) if marker.get('status')=='ok' else None
            self._store(path.relative_to(self.root).as_posix(),marker,stamp,png_stamp)
        except OSError:pass  # Missing metadata falls back to import next time.

    def save(self):
        with self.lock:
            if not self.dirty:return
            self.path.parent.mkdir(parents=True,exist_ok=True)
            temp=self.path.with_suffix('.tmp.json')
            temp.write_text(json.dumps(dict(version=1,root=str(self.root),records=self.records)))
            temp.replace(self.path);self.dirty=False


class ProgressResume:
    """One progress file for all model phases; intentionally no artifact scan."""
    def __init__(self,path):
        self.path=Path(path)
        try:self.models=json.loads(self.path.read_text()).get('models',{})
        except (ValueError,OSError):self.models={}

    def load(self,name,context):
        record=self.models.get(name)
        if record and record.get('context')!=context:
            raise ValueError(f'{name}: parameters or paths differ from the saved progress. '
                'Use resume_validation="strict" to validate individual signatures, or restore the previous configuration.')
        return record

    def save(self,rows,contexts,expected,blocked=None):
        blocked=blocked or {}
        for name,context in contexts.items():
            model_rows=[row for row in rows if row['model_name']==name]
            # Preserve phases not yet reached. A new partial phase must never
            # delete a previously recorded later model's progress.
            pending=blocked.get(name,set())-{row['sample_id'] for row in model_rows}
            if model_rows or pending:
                self.models[name]=dict(context=context,complete={row['sample_id'] for row in model_rows}==expected,
                    rows=model_rows,pending=sorted(pending))
        self.path.parent.mkdir(parents=True,exist_ok=True)
        temp=self.path.with_suffix('.tmp.json')
        temp.write_text(json.dumps(dict(version=1,models=self.models),default=str));temp.replace(self.path)
