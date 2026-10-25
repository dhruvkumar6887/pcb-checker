import cv2
import numpy as np
import pytest
from backend.app.errors import AppError
from backend.tests.test_api import client, wait_result
from backend.vision.alignment import prepare_reference, align_features, polygon_for_roi
from backend.vision.synthetic import fixture


def board_photo():
    # Crop off the entire holder, so no marker can aid the registration.
    return fixture()[110:490, 110:690].copy()


def transformed(image):
    h, w = image.shape[:2]
    source = np.float32([[0,0],[w,0],[w,h],[0,h]])
    destination = np.float32([[38,30],[w+30,15],[w+48,h+40],[20,h+26]])
    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(image, matrix, (w+80,h+70), borderValue=(245,245,245)), matrix


def encode(image):
    return cv2.imencode('.png', image)[1].tobytes()


def test_markerless_transform_and_inverse():
    image = board_photo()
    reference, mode = prepare_reference(image)
    assert mode == 'features'
    test, known = transformed(image)
    aligned, actual = align_features(test, reference)
    # Recover the independently known scale/padding of the reference photo.
    from backend.app.config import CONFIG
    cfg = CONFIG['geometry']
    margin = int((cfg['margin_mm']+cfg['marker_mm']/2)*cfg['pixels_per_mm'])+8
    h,w = image.shape[:2]
    rh,rw = reference.shape[:2]
    scale = min((rw-2*margin)/w,(rh-2*margin)/h)
    offset = np.array([(rw-round(w*scale))//2,(rh-round(h*scale))//2])
    points = np.float32([[[100,100],[400,100],[400,280],[100,280]]])
    phone = cv2.perspectiveTransform(points, known)
    recovered = cv2.perspectiveTransform(phone, actual)
    assert np.max(np.linalg.norm(recovered-(points*scale+offset),axis=2)) < 2
    polygon = polygon_for_roi([200,200,30,40], np.linalg.inv(actual))
    assert np.isfinite(polygon).all()
    assert aligned.shape == reference.shape


def test_markerless_unrelated_and_blank_photos_rejected():
    reference, _ = prepare_reference(board_photo())
    for image in [np.full((400,600,3),128,np.uint8),
                  np.random.default_rng(10).integers(0,256,(400,600,3),dtype=np.uint8)]:
        with pytest.raises(AppError) as error:
            align_features(image, reference)
        assert error.value.code in ('blurry_image', 'alignment_failed')


def test_tagged_profile_accepts_photo_without_tags():
    from backend.vision.alignment import align_to_reference
    ideal = fixture()
    reference, mode = prepare_reference(ideal)
    assert mode == 'aruco'
    # Remove every tag without changing the PCB.
    photo = ideal.copy()
    for x,y in [(0,0),(690,0),(690,490),(0,490)]:
        photo[y:y+110,x:x+110] = 245
    aligned, homography, centers, mode = align_to_reference(
        photo, reference, {'width_mm':86.25,'height_mm':61.25})
    assert mode == 'features'
    assert centers is None
    points = np.float32([[[150,150],[650,150],[650,450],[150,450]]])
    recovered = cv2.perspectiveTransform(points, homography)
    assert np.max(np.linalg.norm(recovered-points,axis=2)) < 2


def test_markerless_saved_profile_and_component_overlay(client):
    ideal = board_photo()
    response = client.post('/api/boards',files={
        'ideal':('ideal.png',encode(ideal)),
        'bom':('bom.csv',b'Designator,Value\nR1,1k\n')})
    assert response.status_code == 201, response.text
    board = response.json()
    assert board['alignment_mode'] == 'features'
    roi = [200,200,30,40]
    response = client.put(f"/api/boards/{board['id']}/components",
                          json={'components':[{'designator':'R1','roi':roi}]})
    assert response.status_code == 200, response.text
    photo, _ = transformed(ideal)
    response = client.post(f"/api/boards/{board['id']}/analyze",files={'test':('test.png',encode(photo))})
    result = wait_result(client,response.json()['analysis_id'])
    assert result['status'] == 'completed', result
    part = result['components'][0]
    assert part['status'] == 'present'
    assert part['marker_baseline_ratios'] is None
    assert len(part['polygon_px_in_original_test_image']) == 4


@pytest.mark.parametrize('missing', [False, True])
def test_markerless_comparison_and_exports(client, missing):
    ideal = board_photo()
    test = ideal.copy()
    if missing:
        # Deliberately erase a component while preserving the rest of the board.
        test[88:125, 165:230] = (70,110,50)
    test, _ = transformed(test)
    response = client.post('/api/compare',files={
        'ideal':('ideal.png',encode(ideal)), 'test':('phone.png',encode(test)),
        'schematic':('bom.csv',b'Designator,Value\nR1,1k\n')})
    assert response.status_code == 202, response.text
    result = wait_result(client,response.json()['analysis_id'])
    assert result['status'] == 'completed', result
    assert result['alignment_mode'] == 'features'
    assert bool(result['components']) == missing
    for part in result['components']:
        assert part['marker_distance_ratios'] is None
        assert 'ratio_deviation_px' not in part['signals']
    for endpoint in ('image','report?format=pdf'):
        assert client.get(f"/api/analyses/{result['analysis_id']}/{endpoint}").status_code == 200
