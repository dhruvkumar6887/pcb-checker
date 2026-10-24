"""Rebuild the sample upload pack with deterministic, self-contained KiCad symbols."""
import csv
import json
import shutil
import uuid
import zipfile
from pathlib import Path
import cv2
import numpy as np
from backend.vision.synthetic import fixture,augment,PARTS
from backend.vision.alignment import geometry,distance_ratios

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'sample_uploads'

def uid(name): return str(uuid.uuid5(uuid.NAMESPACE_URL,'pcb-checker/sample/'+name))
def quote(text): return json.dumps(str(text))

def schematic():
    root_id = uid('root')
    font = '(effects (font (size 1.27 1.27)))'
    small = '(effects (font (size 1 1)))'
    libraries,instances,labels = [],[],[]
    connections = {
        'U1':['GND','TIMING','OUT','VCC',None,'TIMING','DISCHARGE','VCC'],
        'R1':['VCC','DISCHARGE'],'R2':['DISCHARGE','TIMING'],
        'C1':['TIMING','GND'],'Q1':['LOAD','OUT','GND'],'D1':['LOAD','VCC']}
    for index,part in enumerate(PARTS):
        ref,value = part['designator'],part['value']
        name = 'Part_'+ref
        lib_id = 'Sample:'+name
        count = len(connections[ref])
        # Simple original reference blocks, not copied manufacturer symbol artwork.
        pin_names = ['GND','TRIG','OUT','RESET','CTRL','THRESH','DISCH','VCC'] if ref=='U1' else [str(i) for i in range(1,count+1)]
        pins,coords = [],[]
        for i in range(count):
            right = i>=((count+1)//2)
            x = 11.43 if right else -11.43
            row = i-((count+1)//2) if right else i
            y = 7.62-row*5.08
            coords.append((x,y))
            pins.append(f'(pin passive line (at {x} {y} {180 if right else 0}) (length 5.08) (name {quote(pin_names[i])} {small}) (number "{i+1}" {small}))')
        libraries.append(f'''(symbol {quote(lib_id)} (pin_names (offset 0.5)) (in_bom yes) (on_board yes)
            (property "Reference" "{ref[0]}" (at 0 13 0) {font})
            (property "Value" {quote(value)} (at 0 10.5 0) {font})
            (symbol "{name}_0_1" (rectangle (start -6.35 10.16) (end 6.35 -10.16) (stroke (width 0.254) (type default)) (fill (type background))))
            (symbol "{name}_1_1" {''.join(pins)}))''')
        sx,sy = [65,145,225][index%3],[75,140][index//3]
        instance_pins = ''.join(f'(pin "{i+1}" (uuid "{uid(ref+str(i))}"))' for i in range(count))
        instances.append(f'''(symbol (lib_id {quote(lib_id)}) (at {sx} {sy} 0) (unit 1) (in_bom yes) (on_board yes) (dnp no) (uuid "{uid(ref)}")
            (property "Reference" {quote(ref)} (at {sx} {sy-14} 0) {font})
            (property "Value" {quote(value)} (at {sx} {sy-11.5} 0) {font})
            (property "Footprint" {quote(part['footprint'])} (at {sx} {sy} 0) (effects (font (size 1.27 1.27)) hide))
            {instance_pins}
            (instances (project "03_reference" (path "/{root_id}" (reference {quote(ref)}) (unit 1)))))''')
        for i,((x,y),net) in enumerate(zip(coords,connections[ref])):
            px,py = sx+x,sy-y
            if net:
                labels.append(f'(label {quote(net)} (at {px} {py} 0) (effects (font (size 1 1)) (justify left bottom)) (uuid "{uid(ref+"label"+str(i))}"))')
            else:
                labels.append(f'(no_connect (at {px} {py}) (uuid "{uid(ref+"NC")}"))')
    return f'''(kicad_sch (version 20250114) (generator "eeschema")
      (uuid "{root_id}") (paper "A4")
      (title_block (title "PCB detector - synthetic component reference") (comment 1 "Reference only: not a build-ready electrical design or physical PCB layout."))
      (lib_symbols {''.join(libraries)})
      {''.join(labels)} {''.join(instances)}
      (sheet_instances (path "/" (page "1")))
    )'''

def generate():
    OUTPUT.mkdir(exist_ok=True)
    ideal = fixture()
    missing,_ = augment(ideal,PARTS,['R2','C1'],7)
    healthy,_ = augment(ideal,PARTS,[],7)
    for filename,image in [('01_correct_pcb.png',ideal),('02_phone_missing_parts.png',missing),('04_phone_correct_board.png',healthy)]:
        cv2.imwrite(str(OUTPUT/filename),image)
    (OUTPUT/'03_reference.kicad_sch').write_text(schematic(),encoding='utf-8')
    for name in ['bom.csv','schematic.net','components.json']:
        shutil.copyfile(ROOT/'fixtures/ne555'/name,OUTPUT/name)
    centers,size = geometry()
    distances = []
    with (OUTPUT/'placement.csv').open('w',newline='',encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Ref','PosX','PosY'])
        for part in PARTS:
            x,y,w,h = part['roi']
            point = np.array([x+w/2,y+h/2])
            ratios,baseline = distance_ratios(point,centers)
            writer.writerow([part['designator'],(point[0]-size[0]/2)/8,(size[1]/2-point[1])/8])
            distances.append(dict(designator=part['designator'],center_px=point.tolist(),
                distance_to_marker_mm=(np.linalg.norm(centers-point,axis=1)/8).tolist(),
                normalized_ratios=ratios,baseline_ratios=baseline))
    (OUTPUT/'marker_distances.json').write_text(json.dumps(dict(marker_ids=[0,1,2,3],
        marker_size_mm=10,baseline_width_mm=86.25,baseline_height_mm=61.25,pixels_per_mm=8,components=distances),indent=2))
    (OUTPUT/'READ_ME.txt').write_text('''PCB CHECKER SAMPLE UPLOADS

These are synthetic demonstration images, not actual phone photographs.

Open http://127.0.0.1:8000/ and upload:
  Correct PCB:       01_correct_pcb.png
  Phone PCB:         02_phone_missing_parts.png
  KiCad schematic:   03_reference.kicad_sch
Then click Compare images.

Ground truth: R2 (10k resistor) and C1 (10uF capacitor) were erased.
Simple comparison highlights changes with V labels; it does not infer part names.
Use 04_phone_correct_board.png instead to try a board with no parts removed.

All four ArUco tags are already included. Keep the default holder dimensions:
86.25 mm horizontal / 61.25 mm vertical between marker centers, 10 mm tags.

Optional advanced files:
  bom.csv                 Part identities and values
  schematic.net           XML netlist reference
  placement.csv           Approximate placement proposals, not final body boxes
  components.json         Exact ground-truth component boxes for this fixture
  marker_distances.json   Ground-truth center distances to marker IDs 0,1,2,3

The KiCad schematic is a self-contained component reference sheet with named
nets. It is not a build-ready electrical circuit or PCB manufacturing layout.
Schematic symbol positions do not describe physical PCB or tag locations.
''',encoding='utf-8')
    with zipfile.ZipFile(ROOT/'sample_uploads.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUTPUT.iterdir()):
            entry = zipfile.ZipInfo('sample_uploads/'+path.name,date_time=(2026,1,1,0,0,0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry,path.read_bytes())

if __name__=='__main__': generate()
