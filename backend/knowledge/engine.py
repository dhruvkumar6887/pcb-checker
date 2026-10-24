import re
import copy
from pathlib import Path
import yaml

RULES = yaml.safe_load(Path(__file__).with_name('rules.yaml').read_text())

def category(part):
    ref = part['designator'].upper()
    text = f"{part.get('value','')} {part.get('footprint','')}".lower()
    if ref.startswith('TP'): return 'test_point'
    if ref.startswith('R'): return 'resistor'
    if ref.startswith('C'): return 'electrolytic_cap' if any(x in text for x in ['electroly','radial','polar']) else 'capacitor'
    if ref.startswith(('LED', 'D')): return 'led' if 'led' in text or ref.startswith('LED') else 'diode'
    if ref.startswith('Q'): return 'transistor'
    if ref.startswith('U'): return 'ic'
    if ref.startswith(('J','P','CN')): return 'connector'
    return 'unknown'

def nominal_value(value):
    text = value.strip().replace('Ω','').replace('µ','u').replace('μ','u')
    match = re.match(r'^(\d+)([RrKkMm])(\d+)(?:\s|$)', text)
    if match:
        number = float(match[1]+'.'+match[3])
        return number*{'R':1,'r':1,'K':1e3,'k':1e3,'M':1e6,'m':1e-3}[match[2]]
    match = re.match(r'^(\d+(?:\.\d+)?)\s*([pnuUmMkKMG]?)(?:[FfHhVv]|[Oo]hm)?(?:\s|$)',text)
    if not match: return None
    return float(match[1])*{'':1,'p':1e-12,'n':1e-9,'u':1e-6,'U':1e-6,'m':1e-3,'k':1e3,'K':1e3,'M':1e6,'G':1e9}[match[2]]

def describe(part, status):
    kind = category(part)
    rule = {**RULES['defaults'], **RULES['categories'].get(kind,{})}
    value = part.get('value','unknown')
    nominal = nominal_value(value) if kind in ('resistor','capacitor','electrolytic_cap') else None
    tolerance = re.search(r'(\d+(?:\.\d+)?)\s*%',value)
    reading = dict(measurement=rule['measurement'], expected=rule['expected'].format(value=value),
                   tolerance=tolerance.group(0) if tolerance else None, unit=rule.get('unit'),
                   source='schematic-derived' if nominal is not None else 'rules',
                   confidence=.85 if nominal is not None else .55)
    if nominal is not None:
        reading['expected'] = f"{nominal:g} {rule['unit']} nominal. " + reading['expected']
    # Do not silently substitute a generic LED's forward voltage for an unknown color.
    if kind=='led' and 'red' not in value.lower():
        reading['expected'] = 'Forward voltage depends on LED color/MPN and test current; check datasheet. Reverse typically OL.'
    nets = part.get('nets',[])
    effect = rule['effect'] + (f" Connected nets: {', '.join(nets)}. Circuit-specific behavior needs review." if nets else '')
    return dict(part_type=kind, probable_defects=copy.deepcopy(rule['probable_defects']) if status in ('missing','displaced') else [],
                downstream_effect=effect if status in ('missing','displaced') else 'No missing-part effect established.',
                downstream_effect_source='rules', ideal_reading=reading)
