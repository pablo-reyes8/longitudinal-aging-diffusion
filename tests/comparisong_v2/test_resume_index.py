import json
from pathlib import Path
from PIL import Image
from src.comparisong.prediction_resume import ResumeIndex


def prediction(tmp_path):
    root=tmp_path/'generated'/'SAM';path=root/'id_1'/'sample.png'
    path.parent.mkdir(parents=True);Image.new('RGB',(8,8),'gray').save(path)
    path.with_suffix('.json').write_text(json.dumps(dict(signature='same',status='ok')))
    return root,path


def test_legacy_markers_import_once_then_unchanged_pngs_are_not_opened(tmp_path,monkeypatch):
    root,path=prediction(tmp_path);index_path=tmp_path/'resume'/'SAM.json'
    cache=ResumeIndex(index_path,root,['id_1'],workers=2)
    cached,valid,reason=cache.inspect(path,'same')
    assert valid and reason=='ok';cache.save()
    cache=ResumeIndex(index_path,root,['id_1'],workers=2)
    def forbidden(*args,**kwargs):raise AssertionError('unchanged PNG must not be reopened')
    monkeypatch.setattr(Image,'open',forbidden)
    read=Path.read_text
    def text(self,*args,**kwargs):
        assert self!=path.with_suffix('.json'),'unchanged marker must not be reopened'
        return read(self,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',text)
    assert cache.inspect(path,'same')[1]


def test_changed_or_removed_png_and_changed_signature_are_not_reused(tmp_path):
    root,path=prediction(tmp_path);index_path=tmp_path/'resume.json'
    cache=ResumeIndex(index_path,root,['id_1']);assert cache.inspect(path,'same')[1];cache.save()
    cache=ResumeIndex(index_path,root,['id_1'])
    assert cache.inspect(path,'different')[2]=='signature_mismatch'
    path.write_bytes(b'broken image')
    cache=ResumeIndex(index_path,root,['id_1'])
    assert cache.inspect(path,'same')[2]=='corrupt_png'
    path.unlink();cache=ResumeIndex(index_path,root,['id_1'])
    assert cache.inspect(path,'same')[2]=='missing_png'


def test_failed_markers_need_no_png(tmp_path):
    root,path=prediction(tmp_path);path.unlink()
    path.with_suffix('.json').write_text(json.dumps(dict(signature='same',status='failed',error='no face')))
    cache=ResumeIndex(tmp_path/'resume.json',root,['id_1'])
    marker,valid,reason=cache.inspect(path,'same')
    assert valid and marker['error']=='no face' and reason=='cached_failure'
