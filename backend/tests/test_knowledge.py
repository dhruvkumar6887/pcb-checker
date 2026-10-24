import pytest
from backend.knowledge.engine import describe, nominal_value, category

@pytest.mark.parametrize('text,expected',[('1k',1000),('4k7',4700),('10uF',1e-5),('100nF',1e-7),('2.2M',2.2e6)])
def test_values(text,expected):
    assert nominal_value(text)==pytest.approx(expected)

def test_reading_and_sources():
    answer = describe(dict(designator='R2',value='10k 5%',nets=['TIMING']), 'missing')
    assert answer['ideal_reading']['source']=='schematic-derived'
    assert answer['ideal_reading']['tolerance']=='5%'
    assert all('source' in cause and 'confidence' in cause for cause in answer['probable_defects'])
    assert 'TIMING' in answer['downstream_effect']
    assert not describe(dict(designator='U1',value='NE555'),'present')['probable_defects']
    assert category(dict(designator='C1',footprint='Radial'))=='electrolytic_cap'
