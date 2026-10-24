from pathlib import Path
import cv2
from backend.tests.test_api import client,wait_result
from backend.vision.synthetic import fixture,PARTS

def test_three_uploads_find_changes_without_inventing_designators(client):
    root = Path(__file__).resolve().parents[2]/'fixtures/ne555'
    response = client.post('/api/compare',files={
        'ideal':('ideal.png',(root/'ideal.png').read_bytes()),
        'test':('phone.png',(root/'test_missing.png').read_bytes()),
        'schematic':('reference.net',(root/'schematic.net').read_bytes())})
    assert response.status_code==202,response.text
    result = wait_result(client,response.json()['analysis_id'])
    assert result['status']=='completed',result
    assert result['mode']=='visual_compare'
    assert result['components']
    for ref in ['R2','C1']:
        expected = next(p['roi'] for p in PARTS if p['designator']==ref)
        cx,cy = expected[0]+expected[2]/2,expected[1]+expected[3]/2
        assert any(x<=cx<=x+w and y<=cy<=y+h for x,y,w,h in [p['roi_canonical'] for p in result['components']])
    assert all(part['designator'].startswith('V') and part['status']=='uncertain' for part in result['components'])
    assert len(result['schematic_components'])==6
    assert not result['missing']
    assert client.get(f"/api/analyses/{result['analysis_id']}/report?format=pdf").status_code==200

def test_identical_photos_have_no_changes(client):
    encoded = cv2.imencode('.png',fixture())[1].tobytes()
    response = client.post('/api/compare',files={
        'ideal':('ideal.png',encoded),'test':('same.png',encoded),
        'schematic':('ref.kicad_sch',b'(kicad_sch (symbol (property "Reference" "R1") (property "Value" "1k")))')})
    result = wait_result(client,response.json()['analysis_id'])
    assert result['status']=='completed',result
    assert result['components']==[]
    assert result['summary']=='No significant visual differences detected'
