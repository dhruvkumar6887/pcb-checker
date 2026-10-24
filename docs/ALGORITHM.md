# Alignment and missing-part scoring

## Simple comparison without a location map

The default three-upload mode rectifies and normalizes both images, then combines a blurred LAB difference mask with local SSIM disagreement. Morphological closing groups nearby changed pixels; configured minimum area and holder margins suppress small residuals and marker edges. Padded region boxes are mapped back to the phone photo with the same inverse homography. Each region receives a V label, marker-distance ratios and uncertain status. The schematic catalog remains separate because no physical correspondence has been established. Identical normalized inputs produce no regions. Local shadows or glare can still create regions, and some real defects may be visually indistinguishable. This mode highlights changes rather than proving component absence or electrical faults.

## ArUco homography

Detect all four DICT_4X4_50 IDs, average each marker's four corners, and order the centers by ID 0,1,2,3. Four non-collinear point correspondences define a planar homography H from the camera image to the canonical rectangle. Canonical marker spacing is the measured center baseline in mm times configured pixels/mm; margins keep the markers inside the output. Each image gets its own homography. Individual printed marker rotation does not affect the center method.

OpenCV `warpPerspective` resamples both images into the same frame. No raw pixel positions are compared across independently sized phone photos. Components must remain rigidly fixed relative to the holder. A planar transform does not correct lens distortion, parallax from tall components, motion blur, or nonplanar boards. Four centers give an exact solution, without RANSAC redundancy; retaking a sharp overhead photo is preferable to trusting poorly located centers.

For each canonical component rectangle, transform all four corners by H inverse into the original EXIF-oriented test image. Return this quadrilateral and its axis-aligned bounding rectangle. Draw the quadrilateral in the original view; a rectangular overlay would over-cover tilted parts.

## Distance ratios

At canonical component center p and marker centers m_i, compute d_i = norm(p - m_i), r_i = d_i / sum(d_j), and b_i = d_i / norm(m_1 - m_0). Save both arrays in the profile.

Translation, rotation, and uniform scaling preserve these ratios. Perspective does not. In a test photo, initialize a least-squares solver at H inverse(p) and minimize the difference between raw-image ratios and stored canonical r_i. Compare its recovered position with H inverse(p). A configured disagreement threshold reduces confidence and adds a warning, while the homography remains responsible for locating the component. Do not force the ratio solution to match under tilt or treat disagreement as absence.

## Photometric normalization

Convert each canonical image to LAB, apply CLAHE to L, then histogram-match the test's color channels to the normalized ideal. This reduces broad illumination changes but cannot eliminate glare, local shadows or reflected text. A background SSIM check excludes annotated bodies and holder markers to reject gross mismatches, rather than calling every part missing on a completely different layout.

## Presence fusion

For each confirmed ROI, use padded canonical patches to compute:

- Grayscale SSIM, clipped at zero.
- Mean absolute LAB difference, mapped to exp(-difference / color_scale).
- Absolute Canny edge-density difference, mapped to max(0, 1 - difference / edge_scale).
- Color template match in a small neighborhood, mapped to 1 minus normalized squared difference. Color is retained so differently colored bodies and exposed substrate with similar luminance remain distinguishable.

The weighted sum is a score in [0,1]. Defaults are 0.30 SSIM, 0.20 color, 0.15 edges and 0.35 template. Scores >= 0.76 are present, <= 0.55 missing, and the gap uncertain. A larger template search can classify a nearby strongly matching body as displaced when its offset exceeds the configured threshold. It does not prove rotation or distinguish identical nearby parts; rotation-specific classification is not implemented.

Reported confidence is presence score for present, one minus presence score for missing, 0.5 for uncertain, or template score for displaced, reduced by alignment warnings. These are heuristic confidence values, not calibrated probabilities. All image processing uses a single OpenCV thread and fixed operations; random synthetic generation uses a seed. Hardware/library versions can still cause floating-point differences. Optional YOLO positive overlap evidence raises the score when explicitly enabled; absent model detections do not prove missing parts.

## Knowledge and reports

Designator/value/footprint determine a deterministic YAML category. Nominal R/C values are parsed into base units. Rules describe safe measurement conditions, generic possible causes and effects. Connectivity from XML netlists adds net names; no unsupported functional topology is invented. Optional strict-schema datasheet enrichment can replace a reading with sourced operating ranges. Each finding retains its source and confidence. The original photo, annotated output, JSON and downloadable PDF are stored locally with a persisted analysis record.
