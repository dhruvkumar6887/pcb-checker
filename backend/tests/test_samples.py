import io
import json
import zipfile
from pathlib import Path
import numpy as np
from backend.tests.test_api import client,wait_result
from backend.schematic.parser import parse_parts

def test_sample_pack_download_and_upload(client):
    response = client.get('/api/samples/sample_uploads.zip')
    assert response.status_code==200
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    names = ['01_correct_pcb.png','02_phone_missing_parts.png','03_reference.kicad_sch']
    files = {}
    for field,name in zip(['ideal','test','schematic'],names):
        download = client.get('/api/samples/'+name)
        assert download.status_code==200
        assert download.content==archive.read('sample_uploads/'+name)
        files[field] = (name,download.content)
    response = client.post('/api/compare',files=files)
    assert response.status_code==202,response.text
    result = wait_result(client,response.json()['analysis_id'])
    assert result['status']=='completed',result
    assert result['components']
    assert {part['designator'] for part in result['schematic_components']}=={'U1','R1','R2','C1','Q1','D1'}
    assert client.get('/api/samples/not-allowed.txt').status_code==404

def test_marker_distances_and_reference_match():
    folder = Path(__file__).resolve().parents[2]/'sample_uploads'
    data = json.loads((folder/'marker_distances.json').read_text())
    parts = parse_parts('reference.kicad_sch',(folder/'03_reference.kicad_sch').read_bytes())
    assert len(parts)==len(data['components'])==6
    for part in data['components']:
        distances = np.array(part['distance_to_marker_mm'])
        assert np.allclose(distances/distances.sum(),part['normalized_ratios'])
