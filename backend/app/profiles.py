import copy
import cv2
import numpy as np
from backend.app.config import CONFIG
from backend.app.errors import AppError
from backend.vision.alignment import geometry, distance_ratios

def confirm_components(board, components):
    limit = CONFIG['detection']['max_components']
    if len(components) > limit or not components:
        raise AppError('invalid_components', f'Select between 1 and {limit} components.')
    names = [item['designator'] for item in components]
    if len(set(names)) != len(names):
        raise AppError('duplicate_designators', 'Each designator may have only one box.')
    known = {part['designator']: part for part in board['components']}
    if set(names) != set(known):
        raise AppError('component_mismatch', 'Confirm one box for every schematic/BOM designator. Do not omit parts from the profile.')
    centers, (width, height) = geometry(board['width_mm'], board['height_mm'])
    confirmed = []
    for item in components:
        if not item.get('roi'):
            raise AppError('roi_required', f"Draw a box for {item['designator']} before saving.")
        x,y,w,h = item['roi']
        if x+w > width or y+h > height:
            raise AppError('roi_out_of_bounds', f"Box for {item['designator']} extends beyond the rectified image.")
        part = copy.deepcopy(known[item['designator']])
        part['roi'] = item['roi']
        part['ratios'], part['baseline_ratios'] = distance_ratios(np.array([x+w/2,y+h/2]), centers)
        confirmed.append(part)
    return confirmed

def contour_proposals(image):
    cfg = CONFIG['detection']
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0,0,0]), np.array([179,255,cfg['proposal_value_min']]))
    # Restrict suggestions to the board interior, avoiding holder markers.
    margin = int(CONFIG['geometry']['marker_mm']*CONFIG['geometry']['pixels_per_mm']*1.5)
    mask[:margin] = 0
    mask[-margin:] = 0
    mask[:,:margin] = 0
    mask[:,-margin:] = 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [list(cv2.boundingRect(contour)) for contour in contours
             if cfg['proposal_min_area'] <= cv2.contourArea(contour) <= cfg['proposal_max_area']]
    return sorted(boxes, key=lambda box:(box[1],box[0]))[:cfg['proposal_limit']]
