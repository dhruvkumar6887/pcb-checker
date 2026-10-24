from pathlib import Path
import pytest
from backend.schematic.parser import parse_parts, sexpr, placement_proposals
from backend.app.profiles import confirm_components
from backend.vision.synthetic import PARTS
from backend.app.errors import AppError

def test_csv_and_net():
    root = Path(__file__).resolve().parents[2]/'fixtures/ne555'
    csv = parse_parts('bom.csv', (root/'bom.csv').read_bytes())
    net = parse_parts('schematic.net', (root/'schematic.net').read_bytes())
    assert {p['designator'] for p in csv} == {p['designator'] for p in net}
    assert 'TIMING' in next(p for p in net if p['designator']=='C1')['nets']

def test_kicad_and_sexpr():
    data = b'(kicad_sch (symbol (property "Reference" "R1") (property "Value" "10k")))'
    assert parse_parts('test.kicad_sch', data)[0]['value'] == '10k'
    with pytest.raises(ValueError):
        sexpr('(bad')
    with pytest.raises(AppError):
        parse_parts('bad.net', b'<!DOCTYPE bad><export/>')

def test_confirm_profile_and_placements():
    board = dict(width_mm=86.25, height_mm=61.25, components=PARTS)
    result = confirm_components(board, PARTS)
    assert sum(result[0]['ratios']) == pytest.approx(1)
    with pytest.raises(AppError):
        confirm_components(board, PARTS[:-1])
    result = placement_proposals('test.csv', b'Ref,PosX,PosY\nU1,10,20', [dict(PARTS[0])], (800,600))
    assert result[0]['roi'] == [376.,284.,48.,32.]
