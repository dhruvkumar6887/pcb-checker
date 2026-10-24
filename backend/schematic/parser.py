import csv
import io
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from backend.app.errors import AppError
from backend.app.models import Component

def sexpr(text):
    """Small non-evaluating S-expression reader; quoted strings retain escapes."""
    tokens = re.findall(r'"(?:\\.|[^"\\])*"|[()]|[^\s()]+', text)
    root, stack = [], []
    current = root
    for token in tokens:
        if token == '(':
            child = []
            current.append(child)
            stack.append(current)
            current = child
            if len(stack) > 100:
                raise ValueError('S-expression nesting is too deep')
        elif token == ')':
            if not stack:
                raise ValueError('Unbalanced parentheses')
            current = stack.pop()
        else:
            current.append(token[1:-1].replace('\\"', '"').replace('\\\\', '\\') if token.startswith('"') else token)
    if stack or len(root) != 1:
        raise ValueError('Unbalanced or multiple root expressions')
    return root[0]

def children(node, key):
    return [value for value in node if isinstance(value, list) and value and value[0] == key]

def child(node, key, default=None):
    return next(iter(children(node, key)), default or [])

def properties(node):
    return {item[1].lower(): item[2] for item in children(node, 'property') if len(item) > 2}

def parse_parts(filename: str, raw: bytes) -> list[dict]:
    extension = Path(filename).suffix.lower()
    try:
        text = raw.decode('utf-8-sig')
        parts = []
        if extension == '.csv':
            for row in csv.DictReader(io.StringIO(text)):
                row = {str(k).strip().lower(): (v or '').strip() for k,v in row.items() if k}
                reference = row.get('designator', row.get('reference', row.get('ref', '')))
                for ref in re.split(r'[,;\s]+', reference):
                    if ref:
                        parts.append(Component(designator=ref, value=row.get('value',''),
                                               footprint=row.get('footprint',''), mpn=row.get('mpn','')).model_dump())
        elif extension == '.net':
            if '<!DOCTYPE' in text.upper() or '<!ENTITY' in text.upper():
                raise ValueError('XML entities and doctypes are not supported')
            root = ET.fromstring(text)
            for comp in root.findall('./components/comp'):
                fields = {f.get('name','').lower(): f.text for f in comp.findall('./fields/field')}
                ref = comp.get('ref','')
                nets = sorted({net.get('name','') for net in root.findall('./nets/net')
                               if any(n.get('ref') == ref for n in net.findall('node'))})
                parts.append(Component(designator=ref, value=comp.findtext('value',''),
                                       footprint=comp.findtext('footprint',''), mpn=fields.get('mpn') or '', nets=nets).model_dump())
        elif extension == '.kicad_sch':
            root = sexpr(text)
            if root[0] != 'kicad_sch':
                raise ValueError('Expected kicad_sch root')
            for symbol in children(root, 'symbol'):
                props = properties(symbol)
                ref = props.get('reference', '')
                if ref and not ref.startswith('#'):
                    parts.append(Component(designator=ref, value=props.get('value',''),
                                           footprint=props.get('footprint',''), mpn=props.get('mpn','')).model_dump())
        else:
            raise AppError('unsupported_schematic', 'Use .kicad_sch, KiCad XML .net, or BOM .csv. For PDF/image schematics attach a BOM CSV too.')
        if not parts:
            raise ValueError('No components found (CSV requires a Designator column)')
        if len({part['designator'] for part in parts}) != len(parts):
            raise ValueError('Duplicate designators; export a flattened netlist for hierarchical or multi-unit symbols')
        return parts
    except AppError:
        raise
    except (ValueError, KeyError, IndexError, ET.ParseError, UnicodeError, TypeError) as exc:
        raise AppError('invalid_schematic', f'Cannot parse schematic/BOM: {exc}') from exc

def placement_proposals(filename, raw, components, frame_size):
    """Center placement coordinates in holder. All suggested boxes need confirmation."""
    try:
        text = raw.decode('utf-8-sig')
        positions = {}
        if filename.lower().endswith('.kicad_pcb'):
            for fp in children(sexpr(text), 'footprint'):
                ref = properties(fp).get('reference')
                if not ref:
                    ref = next((item[2] for item in children(fp,'fp_text') if item[1]=='reference'), None)
                at = child(fp, 'at')
                if ref and len(at) >= 3:
                    positions[ref] = (float(at[1]), -float(at[2]))
        else:
            for row in csv.DictReader(io.StringIO(text)):
                row = {k.strip().lower(): v for k,v in row.items() if k}
                ref = row.get('ref', row.get('designator', ''))
                positions[ref] = (float(row.get('posx', row.get('x', ''))), float(row.get('posy', row.get('y',''))))
        if not positions:
            raise ValueError('No placements found')
        from backend.app.config import CONFIG
        import numpy as np
        center = np.array(list(positions.values())).mean(axis=0)
        scale = CONFIG['geometry']['pixels_per_mm']
        for part in components:
            if part['designator'] in positions:
                x, y = (np.array(positions[part['designator']])-center)*[scale,-scale]+np.array(frame_size)/2
                part['roi'] = [max(0,float(x-24)), max(0,float(y-16)), 48., 32.]
        return components
    except (ValueError, KeyError, IndexError, UnicodeError, TypeError) as exc:
        raise AppError('invalid_placement', f'Cannot read placement file: {exc}. Use KiCad PCB or CSV Ref,PosX,PosY in mm.') from exc
