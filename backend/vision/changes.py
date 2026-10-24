"""Locate visual differences without guessing schematic-to-photo correspondence."""
import cv2
import numpy as np
from skimage.metrics import structural_similarity
from backend.app.config import CONFIG
from backend.vision.alignment import geometry, distance_ratios

def changed_regions(ideal, test, width_mm, height_mm):
    cfg = CONFIG['quick_compare']
    kernel = cfg['blur_kernel']
    a = cv2.GaussianBlur(ideal,(kernel,kernel),0)
    b = cv2.GaussianBlur(test,(kernel,kernel),0)
    lab_a = cv2.cvtColor(a,cv2.COLOR_BGR2LAB).astype(float)
    lab_b = cv2.cvtColor(b,cv2.COLOR_BGR2LAB).astype(float)
    difference = np.max(np.abs(lab_a-lab_b),axis=2)
    _,similarity = structural_similarity(cv2.cvtColor(a,cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(b,cv2.COLOR_BGR2GRAY),data_range=255,full=True)
    mask = ((difference>cfg['lab_difference_threshold']) &
            (similarity<1-cfg['ssim_difference_threshold'])).astype(np.uint8)*255
    centers,size = geometry(width_mm,height_mm)
    # Exclude the holder's fiducials and outer frame, not the board's components.
    margin = int((CONFIG['geometry']['margin_mm']+CONFIG['geometry']['marker_mm']/2)*CONFIG['geometry']['pixels_per_mm'])
    mask[:margin] = mask[-margin:] = 0
    mask[:,:margin] = mask[:,-margin:] = 0
    shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(cfg['morphology_kernel'],)*2)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,shape)
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(contour) for contour in contours if cv2.contourArea(contour)>=cfg['min_changed_area']]
    boxes = sorted(boxes,key=lambda box:(box[1],box[0]))[:cfg['max_regions']]
    parts = []
    for index,(x,y,w,h) in enumerate(boxes,1):
        pad = cfg['padding_px']
        x0,y0 = max(0,x-pad),max(0,y-pad)
        x1,y1 = min(size[0],x+w+pad),min(size[1],y+h+pad)
        ratios,baseline = distance_ratios(np.array([(x0+x1)/2,(y0+y1)/2]),centers)
        parts.append(dict(designator=f'V{index}',value='Unassigned visual change',roi=[x0,y0,x1-x0,y1-y0],
                          ratios=ratios,baseline_ratios=baseline,nets=[]))
    return parts
