from pathlib import Path
import yaml

def test_compose_and_static_assets():
    root = Path(__file__).resolve().parents[2]
    compose = yaml.safe_load((root/'docker-compose.yml').read_text())
    service = compose['services']['pcb-checker']
    assert service['ports']==['127.0.0.1:${PCB_PORT:-8000}:8000']
    assert service['environment']['PCB_DATA_DIR']=='/app/data'
    html = (root/'frontend/index.html').read_text(encoding='utf-8')
    assert 'id="annotationCanvas"' in html and 'id="resultCanvas"' in html
    for asset in ['app.js','styles.css']:
        assert (root/'frontend'/asset).is_file()
