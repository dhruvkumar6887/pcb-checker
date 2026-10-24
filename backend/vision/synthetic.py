"""Deterministic six-part fixture and phone-photo augmentation (no external assets)."""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np

PARTS = [
    dict(designator='U1', value='NE555', footprint='DIP-8', roi=[325, 215, 110, 100]),
    dict(designator='R1', value='1k', footprint='Axial', roi=[185, 180, 72, 30]),
    dict(designator='R2', value='10k', footprint='Axial', roi=[510, 180, 72, 30]),
    dict(designator='C1', value='10uF', footprint='Radial', roi=[200, 360, 55, 55]),
    dict(designator='Q1', value='2N2222', footprint='TO-92', roi=[510, 355, 65, 55]),
    dict(designator='D1', value='Red LED', footprint='LED-5mm', roi=[350, 395, 45, 45]),
]

def fixture():
    image = np.full((600, 800, 3), 245, np.uint8)
    cv2.rectangle(image, (120, 120), (680, 480), (53, 94, 46), -1)
    # Copper tracks are deliberately outside component bodies.
    for y in [155, 335, 455]:
        cv2.line(image, (145, y), (650, y), (82, 140, 92), 3)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for marker_id, (x, y) in enumerate([(15, 15), (705, 15), (705, 505), (15, 505)]):
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 80)
        image[y:y+80, x:x+80] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    for index, part in enumerate(PARTS):
        x, y, w, h = part['roi']
        cv2.rectangle(image, (x-7, y+8), (x+w+7, y+h-8), (185, 190, 185), -1)
        color = [(38, 38, 40), (130, 175, 205), (120, 165, 200),
                 (170, 100, 30), (35, 35, 35), (45, 45, 225)][index]
        cv2.rectangle(image, (x, y), (x+w-1, y+h-1), color, -1)
        if part['designator'].startswith('R'):
            for offset, stripe in [(15, (20, 35, 80)), (30, (0, 0, 0)), (45, (20, 25, 160))]:
                cv2.line(image, (x+offset, y), (x+offset, y+h-1), stripe, 5)
        else:
            cv2.line(image, (x+8, y+8), (x+w-9, y+8), (205, 205, 205), 2)
        cv2.putText(image, part['designator'], (x, y-12), cv2.FONT_HERSHEY_SIMPLEX, .45, (225, 235, 220), 1)
    return image

def augment(image, components, missing=(), seed=0):
    rng = np.random.default_rng(seed)
    edited = image.copy()
    mask = np.zeros(image.shape[:2], np.uint8)
    for part in components:
        if part['designator'] in missing:
            x, y, w, h = map(int, part['roi'])
            mask[y-2:y+h+2, x-2:x+w+2] = 255
    if mask.any():
        edited = cv2.inpaint(edited, mask, 7, cv2.INPAINT_TELEA)
    h, w = image.shape[:2]
    source = np.float32([[0,0], [w-1,0], [w-1,h-1], [0,h-1]])
    # Keep the full holder inside the image while varying perspective/rotation/scale.
    target = source * rng.uniform(.78, .92) + [w*.07, h*.07]
    target += rng.uniform(-16, 16, (4,2))
    homography = cv2.getPerspectiveTransform(source, target.astype(np.float32))
    edited = cv2.warpPerspective(edited, homography, (w,h), borderValue=(245,245,245))
    edited = np.clip(edited.astype(float)*rng.uniform(.85, 1.12)+rng.uniform(-8,8), 0,255).astype(np.uint8)
    return edited, homography

def generate(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    ideal = fixture()
    cv2.imwrite(str(directory/'ideal.png'), ideal)
    test, _ = augment(ideal, PARTS, ['R2', 'C1'], 7)
    cv2.imwrite(str(directory/'test_missing.png'), test)
    (directory/'components.json').write_text(json.dumps(PARTS, indent=2))
    with (directory/'bom.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['Designator','Value','Footprint','MPN'])
        writer.writeheader()
        for part in PARTS:
            writer.writerow({key: part.get(key.lower(), '') for key in writer.fieldnames})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('fixtures/ne555'))
    parser.add_argument('--seed', type=int)
    parser.add_argument('--missing', nargs='*', default=['R2', 'C1'])
    args = parser.parse_args()
    generate(args.output)
    if args.seed is not None:
        photo, _ = augment(fixture(), PARTS, args.missing, args.seed)
        cv2.imwrite(str(args.output/'custom_test.png'), photo)
