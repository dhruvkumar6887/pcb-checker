import json
import logging
import time
import cv2
import numpy as np
from backend.app.config import CONFIG
from backend.app.errors import AppError
from backend.app.models import ComponentResult
from backend.app import store
from backend.vision.alignment import decode_image, rectify, polygon_for_roi, ratio_check
from backend.vision.detection import prepare_pair, score_component, crop
from backend.knowledge.engine import describe
from backend.knowledge.enrichment import enrich
from backend.ml.yolo import detect as yolo_detect, fuse

logger = logging.getLogger('pcb.analysis')
COLORS = {'missing':(65,65,235),'present':(105,185,28),'uncertain':(30,180,240),'displaced':(220,100,180)}

def annotate(image, results, rectified=False, show_present=False):
    image = image.copy()
    for part in results:
        if part['status']=='present' and not show_present: continue
        if rectified:
            x,y,w,h = part['roi_canonical']
            polygon = np.int32([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])
        else:
            polygon = np.int32(part['polygon_px_in_original_test_image'])
        color = COLORS[part['status']]
        cv2.polylines(image,[polygon],True,color,2,cv2.LINE_AA)
        x,y = polygon[0]
        cv2.putText(image,f"{part['designator']} {part['value']}",(max(0,x),max(18,y-7)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2,cv2.LINE_AA)
    return image

def analyze(analysis_id, board, raw, debug=False):
    started = time.perf_counter()
    directory = store.DATA/'analyses'/analysis_id
    directory.mkdir(parents=True,exist_ok=True)
    store.update(analysis_id,'running')
    try:
        image = decode_image(raw)
        test, homography, centers = rectify(image,board['width_mm'],board['height_mm'])
        ideal = cv2.imread(str(store.DATA/'boards'/board['id']/'ideal.png'))
        if ideal is None: raise AppError('profile_image_missing','Ideal photo is unavailable. Create the board profile again.')
        reference, normalized = prepare_pair(ideal,test)
        inverse = np.linalg.inv(homography)
        results = []
        warnings = []
        detections,ml_warning = yolo_detect(normalized)
        if ml_warning: warnings.append(ml_warning)
        for component in board['components']:
            roi = component['roi']
            score = fuse(score_component(reference,normalized,roi),roi,detections)
            x,y,w,h = roi
            predicted = cv2.perspectiveTransform(np.float32([[[x+w/2,y+h/2]]]),inverse)[0,0]
            deviation, _ = ratio_check(component['ratios'],centers,predicted)
            part_warnings = []
            if deviation > CONFIG['geometry']['ratio_warning_px']:
                score['confidence'] *= CONFIG['geometry']['ratio_confidence_factor']
                part_warnings.append(f'Distance-ratio check differs by {deviation:.1f} px; camera tilt or marker localization may affect confidence.')
            polygon = polygon_for_roi(roi,inverse)
            low, high = polygon.min(axis=0), polygon.max(axis=0)
            detail = describe(component,score['status'])
            detail['ideal_reading'] = enrich(component,detail['ideal_reading'])
            result = ComponentResult(designator=component['designator'],value=component['value'],
                status=score['status'],confidence=score['confidence'],presence_score=score['presence_score'],
                signals={**score['signals'],'ratio_deviation_px':deviation},roi_canonical=roi,
                roi_px_in_original_test_image=[*low.tolist(),*(high-low).tolist()],
                polygon_px_in_original_test_image=polygon.tolist(),warnings=part_warnings,**detail).model_dump()
            results.append(result)
            if debug:
                diff = cv2.absdiff(crop(reference,roi),crop(normalized,roi))
                heat = cv2.applyColorMap(cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY),cv2.COLORMAP_JET)
                cv2.imwrite(str(directory/f"diff_{component['designator']}.png"),heat)
        missing = [part['designator'] for part in results if part['status']=='missing']
        uncertain = [part['designator'] for part in results if part['status']=='uncertain']
        displaced = [part['designator'] for part in results if part['status']=='displaced']
        if uncertain: warnings.append('Review uncertain parts before making repair decisions: '+', '.join(uncertain))
        if any(part['warnings'] for part in results): warnings.append('Distance-ratio verification raised alignment warnings. Consider a more overhead photo.')
        elapsed = time.perf_counter()-started
        report = dict(analysis_id=analysis_id,board_id=board['id'],board_name=board['name'],components=results,
            summary=f"{len(missing)} of {len(results)} components missing"+(': '+', '.join(missing) if missing else ''),
            missing=missing,uncertain=uncertain,displaced=displaced,warnings=warnings,
            original_size=[image.shape[1],image.shape[0]],canonical_size=[test.shape[1],test.shape[0]],
            inverse_homography=inverse.tolist(),elapsed_seconds=round(elapsed,3),debug=debug,
            confidence_note='Heuristic scores, not calibrated probabilities. Visual comparison does not establish electrical health.')
        cv2.imwrite(str(directory/'original.png'),image)
        cv2.imwrite(str(directory/'rectified.png'),test)
        cv2.imwrite(str(directory/'annotated.png'),annotate(image,results))
        if debug:
            cv2.imwrite(str(directory/'ideal_rectified.png'),ideal)
            cv2.imwrite(str(directory/'normalized_test.png'),normalized)
        (directory/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        store.update(analysis_id,'completed',report)
        logger.info(json.dumps(dict(event='analysis_completed',analysis_id=analysis_id,elapsed_seconds=elapsed,missing=missing)))
    except Exception as exc:
        error = dict(code=exc.code,message=exc.message) if isinstance(exc,AppError) else dict(code='analysis_failed',message='Analysis could not complete. Check the image and profile, then retry.')
        store.update(analysis_id,'failed',error=error)
        logger.exception(json.dumps(dict(event='analysis_failed',analysis_id=analysis_id,error=error)))
