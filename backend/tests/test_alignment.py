import cv2
import numpy as np
import pytest
from backend.vision.alignment import rectify, geometry, distance_ratios, polygon_for_roi, ratio_check
from backend.vision.synthetic import fixture, augment, PARTS
from backend.app.errors import AppError

def test_homography_and_inverse():
    image = fixture()
    test, known = augment(image, PARTS, seed=3)
    rectified, homography, _ = rectify(test)
    centers, _ = geometry()
    for p in centers:
        expected = cv2.perspectiveTransform(p.reshape(1,1,2), known)[0,0]
        recovered = cv2.perspectiveTransform(expected.reshape(1,1,2), homography)[0,0]
        assert np.linalg.norm(recovered-p) < 2
    assert rectified.shape == image.shape
    polygon = polygon_for_roi([200,200,30,40], np.linalg.inv(homography))
    assert polygon.shape == (4,2)

def test_ratios_scaling_and_solution():
    centers, _ = geometry()
    point = np.array([300., 250.])
    ratios, baseline = distance_ratios(point, centers)
    shifted = centers*2.4+23
    actual, actual_baseline = distance_ratios(point*2.4+23, shifted)
    assert np.allclose(actual, ratios)
    assert np.allclose(actual_baseline, baseline)
    error, _ = ratio_check(ratios, shifted, point*2.4+23)
    assert error < .001

def test_missing_marker_is_friendly():
    image = fixture()
    image[:110,:110] = 255
    with pytest.raises(AppError, match='') as error:
        rectify(image)
    assert error.value.code == 'markers_missing'

def test_exif_orientation_and_heic():
    import io
    from PIL import Image
    from backend.vision.alignment import decode_image
    image = Image.new('RGB',(60,30),(200,60,40))
    exif = Image.Exif()
    exif[274] = 6
    output = io.BytesIO()
    image.save(output,format='JPEG',exif=exif)
    assert decode_image(output.getvalue()).shape[:2]==(60,30)
    output = io.BytesIO()
    image.save(output,format='HEIF')
    assert decode_image(output.getvalue()).shape[:2]==(30,60)
