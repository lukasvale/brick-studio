# Brick Studio

Brick Studio is a macOS app that turns RAW photographs of LEGO sets, shot on a turntable in a white photo booth, into catalogue-ready images. For each photo it removes the background, keeps the real contact shadow, corrects tone and colour, and frames every angle of a 360° rotation consistently. It writes a white-background JPEG and a transparent PNG per photo.

It handles many products at once: each product is a folder of photos, folders queue up, and the queue can run overnight with pause and resume.

Everything runs on the Mac. Photos are never uploaded; the only network use is the one-time download of the segmentation model.

## Requirements

- A Mac with Apple Silicon (M1 or later) and macOS 14 or later
- Python 3.13 (from [python.org](https://www.python.org/downloads/) or `brew install python@3.13`)
- Xcode Command Line Tools: `xcode-select --install`
- About 3 GB of disk for the Python environment and model, plus space for exports and caches
- Internet during setup and for the first cutout; offline afterwards

## Setup

```bash
git clone <repository-url> brick-studio
cd brick-studio
./setup.sh
```

`setup.sh` creates a Python environment in `.venv`, installs the pinned packages, installs the starting learning history and builds `Brick Studio.app` in the same folder. It takes a few minutes. For a smaller install without the GPU engine, run `./setup.sh --cpu-only` (cutouts are then roughly ten times slower).

Open **Brick Studio.app**. Keep it inside this folder: the app runs the Python engine that sits next to it. The first cutout downloads the segmentation model (about 430 MB) and is slow; later ones are quick.

After changing any Swift code, run `./build_app.sh` and reopen the app. Python changes take effect when the app is reopened.

## Using the app

1. **Add folders.** Drag product folders onto the window, or click **Add folders** (⌘O). One folder per product, holding the photos of one rotation. A parent folder with product subfolders also works. RAW (DNG, CR2/CR3, NEF, ARW, RAF, RW2, ORF, PEF), JPEG, PNG and TIFF are accepted.
2. **Review a folder.** Click it to preview. Step through the angles with ‹ › or the ← → keys, or jump to 0°, 90°, 180° and 270°. Hover over the photo for a magnifier. Choose its zoom from the zoom button, or scroll over the photo.
3. **Adjust.** The panel on the right holds Tone, Detail, Shadow, Output and Cutout. Sliders apply when released. Double-click a slider's name to reset it. Edits apply to every selected folder. Click selects one folder, Shift-click a range, and ⌘-click adds or removes one. Hover over any control for an explanation.
4. **Fine-tune single photos** (optional). **All photos** opens the folder's photos. Edits apply to the checked photos. Shift-click checks a range, and ⌘-click checks or unchecks one.
5. **Export.** Choose the destination on the card above the Process button, then click **Process**. The footer shows the folder being processed, a cell per photo and progress for the whole run. **Pause** keeps finished photos, and **Resume** continues later. **Stop** discards the unfinished folder's export, which goes to the Trash.

The folder list can sit on the left or along the bottom, or be hidden with ⌃⌘S (useful on a laptop screen). Use the sidebar button in the toolbar to switch.

### Output

Each product exports to `<destination>/<folder name>-<id>/`:

```text
centered/jpeg/         white background, product centred at one scale for the whole rotation
centered/transparent/  matching PNGs with transparency
fixed-frame/jpeg/      one crop for the whole rotation, original positions kept
fixed-frame/transparent/
```

Only the chosen framing is exported (Output → Centred, Fixed or Both). File names match the source photos.

### Suggested settings

The app suggests tone and detail settings for new folders by comparing the photos with earlier work. It ships with a starting history of about 12,600 settings from roughly 300 earlier products. From then on it learns only from your own manual corrections. Use the wand button to re-suggest settings for the selection. Turn off automatic suggestions in the ⋯ menu ("Learn settings for new photos"). To start with no history, delete `work/learning/history.sqlite3` while the app is closed.

## Where things are stored

| Location | Contents |
| --- | --- |
| `work/masks/` | Reusable cutouts, so re-exports skip the slow detection step |
| `work/preview-cache/` | Rendered previews, capped at 2 GB |
| `work/queue/` | Queue state and the worker log |
| `work/shoots/` | Per-export processing records (manifest, log, contact sheet) |
| `work/learning/` | Learning history and current suggestions |
| `work/hf-cache/` | Metal engine model weights (about 430 MB) |
| `~/.rembg/models/` | CPU engine model weights (about 930 MB, downloaded only if the CPU engine is used) |
| `~/Library/Application Support/Brick Studio/` | Remembered folder settings and packaging settings |

Source photos are never modified. Deleting `work/` resets the queue, caches and learning history, but leaves exports untouched.

## Troubleshooting

- **Preview stays on "Preparing"**: see `work/preview-worker.log`. A new folder's first preview has to find the subject once, which takes longer.
- **A folder fails during export**: see `work/queue/worker.log` and the folder's `processing.log` under `work/shoots/`. Failed folders are retried up to three times; **Retry failed** in the toolbar tries again.
- **The export card is yellow**: the destination drive is not connected.
- **Long runs**: keep the Mac plugged in with the lid open. The queue prevents idle sleep, but closing the lid can still suspend it.

## Known limitations

- Designed for a neutral white photo booth. Other backdrops need a different colour calibration.
- Review fine railings, transparent bricks and pale pieces before publishing. Automatic review flags (yellow triangle) do not catch every defect.
- Colours are corrected, not calibrated: there is no grey card or colour chart in the pipeline.
- Sharpening cannot restore detail missing from a soft source photo.

For how the code is organised, see [docs/DEVELOPER.md](docs/DEVELOPER.md).

## Licence

You may use and modify Brick Studio, including for your own business, but not sell, rent, sublicense or redistribute it. See [LICENSE](LICENSE). The libraries and models it installs keep their own licences.
