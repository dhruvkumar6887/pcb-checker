import time
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app import store

@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setattr(store,'DATA',tmp_path)
    with TestClient(app) as session: yield session

def wait_result(client,analysis_id):
    for _ in range(100):
        response = client.get('/api/analyses/'+analysis_id).json()
        if response['status'] not in ('queued','running'): return response
        time.sleep(.05)
    pytest.fail('analysis timeout')

def test_full_pipeline_cache_and_downloads(client):
    profile = client.post('/api/demo').json()
    image = client.get('/api/demo/test').content
    response = client.post(f"/api/boards/{profile['id']}/analyze?debug=1",files={'test':('test.png',image,'image/png')})
    assert response.status_code==202,response.text
    analysis_id = response.json()['analysis_id']
    result = wait_result(client,analysis_id)
    assert result['status']=='completed',result
    assert set(result['missing'])=={'R2','C1'}
    assert result['elapsed_seconds']<5
    for part in result['components']:
        assert len(part['polygon_px_in_original_test_image'])==4
    assert client.get(f'/api/analyses/{analysis_id}/image').content.startswith(b'\x89PNG')
    assert client.get(f'/api/analyses/{analysis_id}/image?rectified=true&show_present=true').status_code==200
    assert client.get(f'/api/analyses/{analysis_id}/report?format=pdf').content.startswith(b'%PDF')
    assert client.get(f'/api/analyses/{analysis_id}/report').json()['missing']==result['missing']
    assert len(list((store.DATA/'analyses'/analysis_id).glob('diff_*.png')))==6
    again = client.post(f"/api/boards/{profile['id']}/analyze?debug=1",files={'test':('test.png',image)})
    assert again.json()=={'analysis_id':analysis_id,'cached':True}

def test_create_confirm_and_friendly_errors(client):
    root = Path(__file__).resolve().parents[2]/'fixtures/ne555'
    response = client.post('/api/boards',files={'ideal':('ideal.png',(root/'ideal.png').read_bytes()),'bom':('bom.csv',(root/'bom.csv').read_bytes())})
    assert response.status_code==201,response.text
    board = response.json()
    assert client.post(f"/api/boards/{board['id']}/analyze",files={'test':('photo.png',b'bad')}).json()['error']['code']=='profile_incomplete'
    from backend.vision.synthetic import PARTS
    confirmed = client.put(f"/api/boards/{board['id']}/components",json={'components':PARTS})
    assert confirmed.status_code==200,confirmed.text
    bad = client.post(f"/api/boards/{board['id']}/analyze",files={'test':('photo.png',b'bad')})
    assert wait_result(client,bad.json()['analysis_id'])['error']['code']=='invalid_image'
    assert client.get('/api/boards/nonexistent').status_code==404
    invalid = client.post('/api/boards',files={'ideal':('bad.exe',b'bad'),'bom':('bom.csv',b'bad')})
    assert invalid.json()['error']['code']=='unsupported_file'

def test_reference_schematic_needs_bom(client):
    root = Path(__file__).resolve().parents[2]/'fixtures/ne555'
    response = client.post('/api/boards',files={'ideal':('ideal.png',(root/'ideal.png').read_bytes()),'schematic':('schematic.pdf',b'%PDF')})
    assert response.json()['error']['code']=='bom_required'
