# Validation

Validated in the supplied Windows workspace with Python 3.11.9 and the versions in `requirements.txt`.

- Backend: 76 passing tests cover the synthetic fixture, marker homography and inverse mapping, distance-ratio scaling and solving, HEIC/EXIF decoding, CSV/XML/KiCad property parsing, profile validation, 50 seeded two-part erasures, displacement, offline rules, async API status, cache reuse, debug files, portrait-safe PNG/JSON/PDF exports, gross profile mismatch rejection, packaging configuration, and optional interface/fallback behavior.
- Frontend: Node syntax check plus installed Chrome using Playwright. Exercised sample analysis, both view toggles, expandable row details, PDF download, component box editing and saving, reference/BOM upload, and a 390 px mobile viewport without document overflow.
- The six-component sample reported `R2` and `C1` missing. A browser-run analysis reported approximately 0.2 seconds before final file output; this is a measurement on this fixture, not a broad performance guarantee.
- Browser screenshots and an exported PDF are saved in ignored `artifacts/` when `scripts/browser-check.cjs` is run.
- Docker was not installed, so the Docker image was not built or executed. Compose configuration is parsed in a unit test; Docker health checks run after deployment.
- No trained YOLO weights or live datasheet provider/API keys were supplied. Their contracts and fallbacks were tested with deterministic substitutes; no model training or live extraction accuracy claim is made.
- No real phone-photo dataset or physical PCB was supplied. Synthetic assertions do not establish real-world precision/recall. Calibrate against representative real boards before using results to guide rework.
- A non-failing upstream Starlette/AnyIO TestClient deprecation warning may appear during tests.

The default three-upload workflow adds two API tests (78 total): seeded missing-part regions overlap the actual erased locations without fabricated schematic identities, and identical photos yield no changes. The browser check also exercises the simple comparison and mobile layout before the advanced workflow.

Run `python -m pytest backend/tests -q` for the authoritative current count. Check `git log --oneline` for the staged implementation commits.
