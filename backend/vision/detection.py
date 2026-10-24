import cv2
import numpy as np
from skimage.exposure import match_histograms
from skimage.metrics import structural_similarity
from backend.app.config import CONFIG
from backend.vision.alignment import normalize_lighting

def prepare_pair(ideal, test):
    ideal = normalize_lighting(ideal)
    test = normalize_lighting(test)
    return ideal, np.clip(match_histograms(test, ideal, channel_axis=2), 0,255).astype(np.uint8)

def classify(score):
    cfg = CONFIG['detection']
    if score >= cfg['present_threshold']:
        return 'present'
    if score <= cfg['missing_threshold']:
        return 'missing'
    return 'uncertain'

def background_similarity(ideal,test,components):
    """Reject grossly different layouts; this is not a board-identity guarantee."""
    cfg = CONFIG['geometry']
    margin = int((cfg['margin_mm']+cfg['marker_mm']/2)*cfg['pixels_per_mm'])
    mask = np.zeros(ideal.shape[:2],np.uint8)
    mask[margin:-margin,margin:-margin] = 1
    padding = CONFIG['detection']['padding_px']
    for part in components:
        x,y,w,h = map(int,part['roi'])
        mask[max(0,y-padding):y+h+padding,max(0,x-padding):x+w+padding] = 0
    a,b = cv2.cvtColor(ideal,cv2.COLOR_BGR2GRAY),cv2.cvtColor(test,cv2.COLOR_BGR2GRAY)
    _,similarities = structural_similarity(a,b,data_range=255,full=True)
    return float(np.mean(similarities[mask>0])) if mask.any() else 1.

def crop(image, roi, padding=0):
    x,y,w,h = map(int, roi)
    return image[max(0,y-padding):min(image.shape[0],y+h+padding),
                 max(0,x-padding):min(image.shape[1],x+w+padding)]

def template_score(ideal, test, roi, search):
    template = crop(ideal, roi)
    region = crop(test, roi, search)
    # Normalized squared difference remains meaningful for uniform component bodies.
    scores = cv2.matchTemplate(region, template, cv2.TM_SQDIFF_NORMED)
    minimum, _, location, _ = cv2.minMaxLoc(scores)
    x, y = map(int, roi[:2])
    offset = np.array(location)+[max(0,x-search)-x, max(0,y-search)-y]
    return float(np.clip(1-minimum, 0,1)), offset

def score_component(ideal, test, roi):
    cfg = CONFIG['detection']
    reference, actual = crop(ideal,roi,cfg['padding_px']), crop(test,roi,cfg['padding_px'])
    gray_a = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(actual, cv2.COLOR_BGR2GRAY)
    similarity = max(0., float(structural_similarity(gray_a, gray_b, data_range=255)))
    lab_a = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(float)
    lab_b = cv2.cvtColor(actual, cv2.COLOR_BGR2LAB).astype(float)
    color = float(np.exp(-np.mean(np.abs(lab_a-lab_b))/cfg['color_scale']))
    edges_a = cv2.Canny(gray_a,cfg['canny_low'],cfg['canny_high']) > 0
    edges_b = cv2.Canny(gray_b,cfg['canny_low'],cfg['canny_high']) > 0
    edges = float(max(0,1-abs(edges_a.mean()-edges_b.mean())/cfg['edge_scale']))
    template, _ = template_score(ideal, test, roi, cfg['search_px'])
    signals = dict(ssim=similarity, color=color, edges=edges, template=template)
    presence = sum(signals[name]*weight for name,weight in cfg['weights'].items())
    status = classify(presence)
    displaced_score, offset = template_score(ideal,test,roi,cfg['displaced_search_px'])
    if (status != 'present' and displaced_score >= cfg['displaced_template_min']
            and np.linalg.norm(offset) >= cfg['displaced_offset_px']):
        status = 'displaced'
    # Distance from the decision boundary is a heuristic confidence, not probability.
    confidence = (presence if status=='present' else 1-presence if status=='missing'
                  else displaced_score if status=='displaced' else .5)
    return dict(status=status, presence_score=round(float(presence),6),
                confidence=round(float(confidence),6), signals=signals,
                offset_px=offset.tolist())
