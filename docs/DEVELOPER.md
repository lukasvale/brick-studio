# Developer guide

Brick Studio has two halves: a native SwiftUI app for the interface, and a Python engine that does all image processing. The app starts the Python scripts as separate processes from `.venv` in the project folder. That is why the app must stay inside the project folder: it finds the scripts relative to its own location.

```text
Brick Studio.app  (native/BrickStudio.swift)
 ├─ preview_worker.py   long-running; live previews over JSON lines on stdin/stdout
 ├─ queue_worker.py     runs the batch queue; starts studio.py once per folder
 │    └─ studio.py      exports one folder (also usable on its own from the command line)
 └─ recipe_learning.py  suggests settings from the learning history

native/foreground-guide  (native/ForegroundGuide.swift) Apple Vision helper used by the engine
```

## Files

| File | Role |
| --- | --- |
| `native/BrickStudio.swift` | The whole app: state (`Studio` class), queue handling, and every view |
| `native/ForegroundGuide.swift` | Command-line helper returning Apple Vision subject bounds; built to `native/foreground-guide` |
| `studio.py` | Exporter CLI and the core RAW → image pipeline |
| `preview_worker.py` | Persistent preview renderer; keeps prepared subject crops in memory so slider edits need no AI pass |
| `queue_worker.py` | Processes `work/queue/state.json` folder by folder, with checkpoints, pause, stop and retries |
| `recipe_learning.py` | Learning history (SQLite) and setting suggestions |
| `masking.py` | Subject masks: guide, model passes, coverage checks for loose pieces, mask cache |
| `birefnet_mps.py` | BiRefNet on the Apple GPU (Metal) with the same interface as rembg's CPU session |
| `scene_detection.py` | Whole-scene checks for pieces the first guide missed |
| `edge_refinement.py` | Mask edge cleanup: floor debris, enclosed colour gaps, strut repair |
| `floor_cleanup.py` | Optional "Remove floor patches" |
| `white_recovery.py`, `dark_recovery.py` | Optional recovery of white parts and dark parts near the floor |
| `contact_shadow.py` | Recovers the photographed floor shadow under the product |
| `framing.py` | Shared framing plan: one scale per rotation, centred or fixed frame |
| `packaging_cleanup.py` | Optional removal of the printed code beside a packaging QR |
| `photo_recipes.py` | Per-photo packaging recipes |
| `source_identity.py` | Content-based photo identities, so moved or renamed files keep their cached masks |
| `preview_cache.py`, `preview_storage.py` | Persistent preview cache (2 GB cap) and per-session preview files |
| `progress.py` | Measured elapsed time and estimated per-photo progress |
| `seed/learning-history.sqlite3` | Starting learning history, installed by `setup.sh` |
| `setup.sh`, `build_app.sh` | One-time install (Python environment, packages, seed history, app build) and app rebuild |
| `Set up Brick Studio.command` | Double-click wrapper for `setup.sh`: checks for Python 3.13 and the developer tools, runs setup, opens the app |

## Processing pipeline

`studio.py` processes a folder in two stages, so memory stays bounded however many photos there are:

1. **Masks for every photo.**
   - Decode the RAW with rawpy/LibRaw: camera white balance, 16-bit, sRGB.
   - Find the subject. A fast guide (Apple Vision plus a scan for coloured parts) picks a padded region, and BiRefNet makes the mask at 1024 × 1024. The model always sees a fixed rendering of the photo (`mask_view`), so the cutout does not depend on the recipe sliders.
   - Check coverage: clipped crops get one wider retry, and loose pieces outside the main mask get their own close look.
   - Run edge refinement and any optional recovery steps, then cache the mask in `work/masks/`.
2. **Framing plan.** Build one plan from all masks: a shared scale for the rotation, centred or fixed frame.
3. **Render each photo from the RAW at final size.**
   - Neutral correction from the booth walls.
   - Exposure with highlight compression, contrast, whites, warmth.
   - Contact shadow recovered from the photographed floor, composited on white.
   - Sharpening and noise reduction.
   - Write a JPEG (quality 96, sRGB) and an RGBA PNG.

Each export writes `manifest.json` and `processing.log` into its `--metadata` folder. The manifest holds the recipe, bounding boxes, timings and review flags per photo.

Example of a direct export without the app:

```bash
.venv/bin/python studio.py /path/to/product-folder --output output/test \
  --framing both --size 2400 --exposure 0.65 --warmth 0.25 --shadow 1.15 --shadow-method local
.venv/bin/python studio.py --help   # every option
```

## Caches and version numbers

Masks are keyed by photo content (not path), engine and cutout options. Version constants decide when cached results are reused:

- `EDGE_VERSION`, `PARTS_VERSION` and `MASK_VIEW_VERSION` in `masking.py`
- `WHITE_VERSION` in `white_recovery.py`
- `DARK_VERSION` in `dark_recovery.py`
- `shadow_version` (and `edge_version`) in the export settings in `studio.py`, which decide whether a part-finished batch can resume

When you change one of those algorithms, bump its version so old cached masks or partly finished batches are not reused with new logic. The app's preview cache (`work/preview-cache/`) rebuilds itself when versions or recipes change.

## Queue and app state

- `work/queue/state.json`: the queue and each folder's recipe. The worker writes checkpoints here after every photo. `progress.json` beside it carries the live per-photo state.
- `pause.request` and `stop.request` in the same folder are how the app asks the worker to pause or stop.
- A folder's recipe is frozen when its export starts (`started_recipe`), so a half-finished folder never mixes settings.
- Remembered folder recipes and packaging settings live in `~/Library/Application Support/Brick Studio/`.

## Learning

`recipe_learning.py` stores processing events in `work/learning/history.sqlite3`. Each event holds an image descriptor (brightness, highlight, colour and saturation measurements of the unedited photo), the recipe used, and which fields a person set by hand. Only fields a person set by hand become training labels. Suggestions are validated by holding out whole product folders.

The shipped seed holds only folder and file names; it has no machine paths. The model cache is empty and rebuilds on first use, which takes a little longer once.

## Building

```bash
./build_app.sh   # compiles native/ForegroundGuide.swift and native/BrickStudio.swift, assembles and ad-hoc signs the app
```

The app is ad-hoc signed for local use. To distribute a prebuilt app to other Macs without Gatekeeper warnings, sign it with a Developer ID certificate and notarize it. The app would also still need the Python environment beside it, so building with `setup.sh` on each Mac is the simpler path.

## Notes for production use

- `birefnet_mps.py` loads `ZhengPeng7/BiRefNet` from Hugging Face with `trust_remote_code=True`, which runs model code downloaded from that repository. Pin a specific revision (`revision=` in `from_pretrained`) before relying on it in production.
- The Python packages in `requirements*.txt` are pinned to the versions this was built and used with.
- The regression test suite used during development is not part of this handoff.

## Third-party components

Licences as published by each project at the time of writing. Verify them before any commercial distribution.

| Component | Use | Licence |
| --- | --- | --- |
| rawpy / LibRaw | RAW decoding | MIT / LGPL-2.1 or CDDL-1.0 |
| OpenCV (opencv-python-headless) | Image processing | Apache-2.0 |
| rembg | CPU segmentation session | MIT |
| BiRefNet (ZhengPeng7) | Segmentation model and weights | MIT |
| ONNX Runtime | CPU inference | MIT |
| PyTorch, torchvision | Metal inference | BSD-3-Clause |
| Hugging Face transformers, huggingface_hub | Model loading | Apache-2.0 |
| NumPy, SciPy, scikit-image | Numerical work | BSD-3-Clause |
