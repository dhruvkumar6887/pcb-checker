import numpy as np
from backend.ml.yolo import detect, fuse
from backend.ml.generate_dataset import generate
from backend.vision.synthetic import fixture,PARTS
from backend.knowledge.enrichment import lookup,enrich,DatasheetFacts
from backend.app import store

def test_optional_off():
    assert detect(np.zeros((20,20,3),np.uint8))==([],None)
    reading = {'source':'rules'}
    assert enrich({'mpn':'NE555'},reading)==reading

def test_fusion_requires_overlap():
    score = dict(presence_score=.72,status='uncertain',confidence=.5,signals={})
    assert fuse(score,[20,20,30,30],[([100,100,130,130],.99)])==score
    assert fuse(score,[20,20,30,30],[([20,20,50,50],.99)])['status']=='present'

def test_dataset(tmp_path):
    generate(fixture(),PARTS,tmp_path,10)
    assert len(list((tmp_path/'images/train').glob('*.jpg')))==8
    assert len(list((tmp_path/'images/val').glob('*.jpg')))==2

def test_lookup_cache_schema_and_fallback(tmp_path,monkeypatch):
    monkeypatch.setattr(store,'DATA',tmp_path)
    class Search:
        def search(self,mpn): return ['https://example.com/datasheet.pdf']
    class Extract:
        def extract(self,mpn,text,schema):
            return dict(mpn=mpn,function='Timer',recommended_voltage_min=4.5,recommended_voltage_max=16,
                        operating_temperature_min=None,operating_temperature_max=None,confidence=.8)
    facts,url = lookup('NE555',Search(),Extract(),fetch=lambda url:'document')
    assert facts.function=='Timer'
    facts,url = lookup('NE555',Search(),Extract(),fetch=lambda url:1/0)
    assert facts.recommended_voltage_min==4.5
    class Bad:
        def extract(self,*args): return {'function':'unvalidated'}
    assert lookup('UNKNOWN',Search(),Bad(),fetch=lambda url:'document') is None
