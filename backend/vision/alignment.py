import io
import warnings
import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener
from scipy.optimize import least_squares
from backend.app.config import CONFIG
from backend.app.errors import AppError

register_heif_opener()
cv2.setNumThreads(1)

def decode_image(raw: bytes) -> np.ndarray:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                if image.width * image.height > CONFIG['images']['max_pixels']:
                    raise AppError('image_too_large', 'Image has too many pixels. Resize it before uploading.', 413)
                image = ImageOps.exif_transpose(image).convert('RGB')
                return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise AppError('invalid_image', 'Cannot decode this image. Use a valid JPEG, PNG or HEIC photo.') from exc

def geometry(width_mm=None, height_mm=None):
    cfg = CONFIG['geometry']
    scale, margin = cfg['pixels_per_mm'], cfg['margin_mm'] * cfg['pixels_per_mm']
    width = float(width_mm or cfg['baseline_width_mm']) * scale
    height = float(height_mm or cfg['baseline_height_mm']) * scale
    size = (int(round(width + 2*margin)), int(round(height + 2*margin)))
    if min(width, height) <= 0 or max(size) > cfg['max_side_px']:
        raise AppError('invalid_holder', 'Holder dimensions must be positive and fit the configured maximum frame size.')
    centers = np.float32([[margin, margin], [margin+width, margin],
                          [margin+width, margin+height], [margin, margin+height]])
    return centers, size

def rectify(image, width_mm=None, height_mm=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < CONFIG['images']['min_blur_variance']:
        raise AppError('blurry_image', 'Photo is too blurry. Stabilize the phone and focus on the entire holder.')
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    corners, ids, _ = detector.detectMarkers(gray)
    found = {}
    if ids is not None:
        for corner, marker_id in zip(corners, ids.flatten()):
            if int(marker_id) in found:
                raise AppError('duplicate_markers', 'Duplicate marker IDs found. Keep only this holder in frame.')
            found[int(marker_id)] = corner.reshape(4,2).mean(axis=0)
    missing = sorted(set(range(4))-set(found))
    if missing:
        raise AppError('markers_missing', f'Markers {missing} were not detected. Include the whole holder, improve lighting, and retake a sharp photo.')
    raw_centers = np.float32([found[i] for i in range(4)])
    if not cv2.isContourConvex(raw_centers.astype(np.int32)):
        raise AppError('marker_layout', 'Marker IDs must follow the holder corners: 0 TL, 1 TR, 2 BR, 3 BL.')
    canonical, size = geometry(width_mm, height_mm)
    homography = cv2.getPerspectiveTransform(raw_centers, canonical)
    if not np.isfinite(homography).all() or abs(np.linalg.det(homography)) < 1e-12:
        raise AppError('bad_alignment', 'Holder geometry is degenerate. Retake the photo from above.')
    return cv2.warpPerspective(image, homography, size), homography, raw_centers

def distance_ratios(point, centers):
    distances = np.linalg.norm(np.asarray(centers)-point, axis=1)
    baseline = np.linalg.norm(centers[1]-centers[0])
    return (distances/distances.sum()).tolist(), (distances/baseline).tolist()

def ratio_check(ratios, raw_centers, predicted):
    # Similarity invariance only; projective tilt produces expected residuals.
    target = np.asarray(ratios)
    solved = least_squares(lambda p: np.asarray(distance_ratios(p, raw_centers)[0])-target,
                           np.asarray(predicted, dtype=float))
    return float(np.linalg.norm(solved.x-predicted)), solved.x.tolist()

def polygon_for_roi(roi, inverse):
    x, y, w, h = roi
    points = np.float32([[x,y], [x+w,y], [x+w,y+h], [x,y+h]])
    return cv2.perspectiveTransform(points[None], inverse)[0]

def normalize_lighting(image):
    cfg = CONFIG['images']
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = cv2.createCLAHE(cfg['clahe_clip'], (cfg['clahe_tiles'],)*2).apply(lab[:,:,0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
