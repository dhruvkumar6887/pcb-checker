"""Generate YOLO positives and erased-part negatives from a confirmed board profile."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import yaml
from backend.vision.synthetic import augment

def generate(ideal,components,output,count=200):
    output = Path(output)
    for seed in range(count):
        split = 'val' if seed%5==0 else 'train'
        for directory in ['images','labels']: (output/directory/split).mkdir(parents=True,exist_ok=True)
        rng = np.random.default_rng(seed)
        missing = [part['designator'] for part in components if rng.random()<.35]
        image,homography = augment(ideal,components,missing,seed)
        if seed%3==0: image = cv2.GaussianBlur(image,(3,3),.6)
        cv2.imwrite(str(output/'images'/split/f'{seed:05d}.jpg'),image)
        h,w = image.shape[:2]
        lines = []
        for part in components:
            if part['designator'] in missing: continue
            x,y,rw,rh = part['roi']
            points = np.float32([[[x,y],[x+rw,y],[x+rw,y+rh],[x,y+rh]]])
            polygon = cv2.perspectiveTransform(points,homography)[0]
            low = np.maximum(polygon.min(axis=0),[0,0])
            high = np.minimum(polygon.max(axis=0),[w,h])
            center,size = (low+high)/2,(high-low)
            lines.append(f'0 {center[0]/w:.7f} {center[1]/h:.7f} {size[0]/w:.7f} {size[1]/h:.7f}')
        (output/'labels'/split/f'{seed:05d}.txt').write_text('\n'.join(lines))
    (output/'dataset.yaml').write_text(yaml.safe_dump(dict(path=str(output.resolve()),train='images/train',val='images/val',names={0:'component'})))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ideal',type=Path,required=True)
    parser.add_argument('--profile',type=Path,required=True,help='Saved profile.json, or fixture components.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--count',type=int,default=200)
    args = parser.parse_args()
    components = json.loads(args.profile.read_text())
    if isinstance(components,dict): components = components['components']
    ideal = cv2.imread(str(args.ideal))
    if ideal is None or not components or any(not part.get('roi') for part in components):
        parser.error('Supply a readable rectified image and confirmed component ROIs.')
    if args.count<10: parser.error('Use at least 10 images for train/validation splits.')
    generate(ideal,components,args.output,args.count)
