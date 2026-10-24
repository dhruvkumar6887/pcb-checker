"""Lazy, optional single-class detector. Positive evidence only; absence is not proof."""
from functools import lru_cache
from pathlib import Path
import logging
from backend.app.config import CONFIG
from backend.vision.detection import classify

@lru_cache(maxsize=1)
def load_model(weights):
    if not weights or not Path(weights).is_file():
        raise ValueError('Configure features.yolo_weights with a trained local weights file.')
    from ultralytics import YOLO
    return YOLO(weights)

def detect(image):
    cfg = CONFIG['features']
    if not cfg['yolo']: return [],None
    try:
        model = load_model(cfg['yolo_weights'])
        result = model.predict(image,device='cpu',conf=cfg['yolo_min_confidence'],verbose=False)[0]
        return [(box.xyxy[0].cpu().tolist(),float(box.conf[0])) for box in result.boxes],None
    except Exception as exc:
        logging.getLogger('pcb.ml').warning('YOLO unavailable; using classical scores: %s',type(exc).__name__)
        return [],'Optional YOLO detector unavailable; used classical detection.'

def fuse(score,roi,detections):
    cfg = CONFIG['features']
    x,y,w,h = roi
    confidence = None
    for (left,top,right,bottom),detected_confidence in detections:
        intersection = max(0,min(x+w,right)-max(x,left))*max(0,min(y+h,bottom)-max(y,top))
        union = w*h+max(0,right-left)*max(0,bottom-top)-intersection
        if union>0 and intersection/union>=cfg['yolo_min_iou']:
            confidence = max(confidence or 0,detected_confidence)
    if confidence is not None:
        weight = cfg['yolo_fusion_weight']
        score = dict(score)
        presence = max(score['presence_score'],score['presence_score']*(1-weight)+confidence*weight)
        score['presence_score'] = presence
        if score['status']!='displaced': score['status'] = classify(presence)
        score['confidence'] = presence if score['status']=='present' else 1-presence if score['status']=='missing' else .5
        score['signals'] = {**score['signals'],'yolo':confidence}
    return score
