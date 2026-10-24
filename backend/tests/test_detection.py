import cv2
import numpy as np
import pytest
from backend.vision.synthetic import fixture, augment, PARTS
from backend.vision.alignment import rectify
from backend.vision.detection import prepare_pair, score_component, classify

def test_boundaries():
    assert classify(.99) == 'present'
    assert classify(.05) == 'missing'
    assert classify(.65) == 'uncertain'

@pytest.mark.parametrize('seed', range(50))
def test_missing_without_false_positives(seed):
    ideal = fixture()
    missing = {PARTS[seed % len(PARTS)]['designator'], PARTS[(seed+2) % len(PARTS)]['designator']}
    photo,_ = augment(ideal, PARTS, missing, seed)
    reference,_,_ = rectify(ideal)
    test,_,_ = rectify(photo)
    reference,test = prepare_pair(reference,test)
    scores = {p['designator']: score_component(reference,test,p['roi']) for p in PARTS}
    actual = {name for name, result in scores.items() if result['status']=='missing'}
    assert actual == missing, {name:(v['status'],v['presence_score']) for name,v in scores.items()}
    assert all(scores[name]['status']=='present' for name in scores if name not in missing)

def test_displaced_component():
    image = fixture()
    part = PARTS[0]
    x,y,w,h = part['roi']
    moved = image.copy()
    body = image[y:y+h,x:x+w].copy()
    moved[y:y+h,x:x+w] = (53,94,46)
    moved[y+20:y+h+20,x+20:x+w+20] = body
    reference,test = prepare_pair(image,moved)
    assert score_component(reference,test,part['roi'])['status']=='displaced'
