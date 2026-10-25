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


def prepare_reference(image, width_mm=None, height_mm=None):
    """Use the holder when available, otherwise preserve the photo's aspect ratio."""
    try:
        rectified, _, _ = rectify(image, width_mm, height_mm)
        return rectified, 'aruco'
    except AppError as exc:
        if exc.code != 'markers_missing':
            raise
    _, size = geometry(width_mm, height_mm)
    cfg = CONFIG['geometry']
    margin = int((cfg['margin_mm'] + cfg['marker_mm']/2)*cfg['pixels_per_mm']) + 8
    height, width = image.shape[:2]
    scale = min((size[0]-2*margin)/width, (size[1]-2*margin)/height)
    new_width, new_height = max(1, round(width*scale)), max(1, round(height*scale))
    x, y = (size[0]-new_width)//2, (size[1]-new_height)//2
    transform = np.array([[scale, 0, x], [0, scale, y], [0, 0, 1]], dtype=float)
    reference = cv2.warpPerspective(image, transform, size, borderValue=(245,245,245))
    return reference, 'features'


def align_features(image, reference):
    """Match PCB texture with SIFT and reject weak or implausible registrations."""
    message = ('Could not reliably align these photos without tags. Use sharp, close-up photos '
               'of the same board side, with the whole PCB visible and a similar viewing angle. '
               'Avoid glare and a busy background, or use ArUco tags.')

    def fail():
        raise AppError('alignment_failed', message)

    def features(photo):
        scale = min(1., 1600/max(photo.shape[:2]))
        gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=scale, fy=scale)
        points, descriptors = cv2.SIFT_create(nfeatures=5000).detectAndCompute(gray, None)
        return points, descriptors, scale

    if cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < CONFIG['images']['min_blur_variance']:
        raise AppError('blurry_image', 'Photo is too blurry. Focus on the entire PCB and retake it.')
    source, source_desc, source_scale = features(image)
    target, target_desc, target_scale = features(reference)
    if source_desc is None or target_desc is None or min(len(source), len(target)) < 12:
        fail()
    matches = cv2.BFMatcher().knnMatch(source_desc, target_desc, k=2)
    good = [pair[0] for pair in matches if len(pair)==2 and pair[0].distance < .72*pair[1].distance]
    # Repeated silkscreen/features must not count as many independent observations.
    good = sorted(good, key=lambda m:m.distance)
    unique, used = [], set()
    for match in good:
        if match.trainIdx not in used:
            unique.append(match)
            used.add(match.trainIdx)
    if len(unique) < 12:
        fail()
    a = np.float32([source[m.queryIdx].pt for m in unique])/source_scale
    b = np.float32([target[m.trainIdx].pt for m in unique])/target_scale
    homography, mask = cv2.findHomography(a, b, cv2.RANSAC, 3., maxIters=4000, confidence=.995)
    if homography is None or mask is None or not np.isfinite(homography).all():
        fail()
    inliers = mask.ravel().astype(bool)
    if inliers.sum() < 12 or inliers.mean() < .45:
        fail()
    height, width = reference.shape[:2]
    hull = cv2.convexHull(b[inliers])
    if cv2.contourArea(hull) < .025*width*height:
        fail()
    h, w = image.shape[:2]
    corners = np.float32([[0,0],[w,0],[w,h],[0,h]])
    denominators = np.c_[corners, np.ones(4)] @ homography[2]
    if not (np.all(denominators > 1e-8) or np.all(denominators < -1e-8)):
        fail()
    projected = cv2.perspectiveTransform(corners[None], homography)[0]
    area = cv2.contourArea(projected, oriented=True)
    if not np.isfinite(projected).all() or not cv2.isContourConvex(projected) or not .15*width*height < area < 20*width*height:
        fail()
    # Require the central inspection area to be visible; cropped photos must not
    # turn uncovered pixels into apparent missing components.
    cfg = CONFIG['geometry']
    margin = int((cfg['margin_mm']+cfg['marker_mm']/2)*cfg['pixels_per_mm'])
    valid = cv2.warpPerspective(np.ones(image.shape[:2], np.uint8), homography, (width,height))
    if valid[margin:-margin,margin:-margin].mean() < .90:
        fail()
    return cv2.warpPerspective(image, homography, (width,height), borderValue=(245,245,245)), homography


def align_to_reference(image, reference, board):
    if board.get('alignment_mode', 'aruco') == 'aruco':
        try:
            test, homography, centers = rectify(image, board['width_mm'], board['height_mm'])
            return test, homography, centers, 'aruco'
        except AppError as exc:
            if exc.code != 'markers_missing':
                raise
    test, homography = align_features(image, reference)
    return test, homography, None, 'features'

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
