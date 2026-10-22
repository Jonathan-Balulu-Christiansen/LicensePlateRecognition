# Visus — Real-Time Israeli License Plate Recognition

<img src="assets/branding/visus_icon.png" width="72" align="right" alt="Visus app icon">

Visus detects, tracks and reads Israeli license plates in real time, from a camera or a video
file, and shows the result in a desktop application. *Visus* is Latin for eyesight.

A fine-tuned YOLO26 detector finds the plates and ByteTrack follows each one across frames.
A CNN+CTC network then reads the whole number in a single pass, without segmenting it into
digits, and the number is confirmed by voting over several frames.
Israeli civilian plates carry digits only (7 or 8), so the reader's decoding is constrained to
exactly those two formats.

## How it works

```text
camera / video
  -> YOLO26m detector        find plate boxes (NMS-free head), 640 px input
  -> ByteTrack               a stable id per plate across frames
  -> crop                    every plate at least 90 px wide, every 4th frame until locked
  -> CNN+CTC reader          whole crop -> 7- or 8-digit number + per-digit confidence;
                             all plates of a frame in one batch
  -> temporal voting         lock the number once enough confident reads agree
  -> plate origin            country from the plate colour (yellow -> Israel)
```

**The reader.** The reader converts the crop to grey and resizes it to 48×192. A 2D CNN collapses the
height while keeping 48 columns along the plate. A residual stack of dilated 1D convolutions then
gives every column a view of the whole plate. The network has 1.85 M parameters. It is trained with
CTC loss, which needs only the number string as a label: the network learns by itself where each
digit is. The network is pretrained on synthetic plates and fine-tuned on real crops. Keras 3 runs
on the PyTorch backend, so TensorFlow is not needed.

## Results

| Stage | Metric | Value |
|---|---|---|
| Detector (YOLO26m, 640) | mAP50 on the held-out test split | **0.978** |
| | mAP50-95 | 0.750 |
| | precision / recall | 0.952 / 0.937 |
| Plate reader (CNN+CTC) | whole plates read exactly (5-fold cross-validation by plate) | **98.1%** |
| | speed | ~1.1 ms per plate (batch of 8, RTX 4070 Ti) |
| End to end | demo clips that lock the correct number | 7 / 8 |

How these numbers were measured:
- **Detector.** It is scored once, at its own input size, on a test split that is never used for
  early stopping or checkpoint selection.
- **Reader.** It is scored on whole plates: a plate counts only if every digit is right. Every plate
  is tested by a model that never saw it.
- **The reader's design.** Two results drove it:
  - The classic alternative (segment the digits, then classify each one) reached 64.5–67.3% under
    the same protocol.
  - The same network trained on real plates alone reached about 55%. Synthetic pretraining is what
    makes it work.

## Repository layout

```text
src/                     the system
  app.py                 desktop application (PySide6): source / detector / reader pickers,
                         live video with overlays, per-stage timings, plate panel, passages log
  pipeline.py            real-time engine (worker thread): capture, detect + track, batched
                         reading, voting, timings, gate log
  plate_reader.py        CNN+CTC reader: preprocessing, network, constrained decoding, voting
  plate_origin.py        plate country (from colour) and number formatting
  theme.py               colours, fonts, stylesheet, app icon
  common.py              shared helpers: paths, dataset YAML, CUDA, run folders
  predict.py             detector only, on images / video / camera -> annotated output + CSV
  train_detector.py      fine-tune YOLO26m, then score the held-out test split
  train_reader.py        pretrain on synthetic plates, fine-tune on real crops; --cv, --promote
tools/                   data preparation (not part of the live system)
  dataset_check.py       validate YOLO labels and check splits for leakage
  split_dataset.py       deterministic train/val/test split from an immutable pool
  auto_label.py          detector-assisted pre-labels for human review
  extract_frames_from_video.py, downsample_fps.py   video helpers
  synth_plates.py        synthetic Israeli plate crops with exact labels
  gen_scenes.py          plan balanced batches of synthetic vehicle scenes
  number_scenes.py       print realistic numbers onto the blank plates of those scenes
  visualize_yolo_augmentations.py, build_branding.py
tests/                   pytest suite
configs/dataset.yaml     dataset paths and class name for Ultralytics
docs/DATASETS.md         building the datasets from scratch
docs/RUNBOOK.md          every command, from install to training to evaluation
```

Data and trained models are not part of this repository. The code expects them in `data/` and
`weights/` next to `src/`, and `.gitignore` keeps those folders out of git.

## Getting started

Requirements: Python 3.12. A CUDA GPU is recommended for training and for real-time use.
Everything also runs on the CPU, more slowly.

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux / macOS: source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the direct dependencies, with the CUDA 12.1 PyTorch build. On a machine
without a GPU, remove the `+cu121` suffixes and the extra index line.
`requirements-exact-versions.txt` is the full `pip freeze` of the verified environment.

**Models.** Trained weights are not published yet; they will be added once trained on a larger
dataset. Until then, the models can be trained from the steps below:
1. Build the datasets ([docs/DATASETS.md](docs/DATASETS.md)).
2. Train both models ([docs/RUNBOOK.md](docs/RUNBOOK.md)).

The app loads `weights/detector/best.pt` and `weights/reader/plate_reader.keras`.

**Run the app:**

```bash
python src/app.py
```

Pick a **Source** (a camera, or *Video file…*), a **Detector** and a **Reader** (the promoted
models, every training run, or any file), and the number of agreeing **Votes** needed to lock a
plate, then press **Start**. Locked plates appear in the side panel and in *Recent passages*.
Each one is also appended to `reports/gate_log.jsonl`:

```json
{"date":"21/10/2026","time":"17:24:43","plate":"12345601","origin":"IL","detector_confidence":0.78,"ocr_digit_mean":0.9996,"ocr_plate_confidence":0.9972}
```

`ocr_plate_confidence` is the product of the per-digit confidences, so one weak digit lowers the
score of the whole number.

## Tests and code quality

```bash
pip install -r requirements-dev.txt
pytest                 # fast suite
pytest -m slow         # end to end on a local clip (see docs/RUNBOOK.md)
ruff check src tools tests
pyright src tools tests
```

Settings for all three tools are in `pyproject.toml`. Tests that need models or data skip
themselves when those are absent.

## Design decisions

- **Read the whole plate with CTC.** No digit segmentation is needed. In the first version,
  segmentation (not digit classification) was the main source of errors.
- **Synthetic pretraining, real fine-tuning.**
  - Real labelled plates are scarce.
  - Rendered plates follow the real Israeli layouts and numbering rules, and their labels are exact
    by construction.
- **Synthetic scenes for the detector.**
  - Image-generation models draw realistic vehicles with **blank** plates, since they cannot be
    trusted to print exact digits.
  - Realistic numbers are then printed onto those plates in the right perspective and light.
  - Validation and test data stay real only.
- **Train the detector at its pretraining and inference size (640).** It scored higher than the
  same model trained at 960, when both were scored at 640.
- **No rotation or shear augmentation for the detector.**
  - Ultralytics replaces a rotated box with the box around it. For a long, thin plate, that box is
    much taller than the plate itself, and training on it measurably loosened the predicted boxes.
  - The reader has its own plate-level augmentation instead: rotation up to ±12° without cutting
    digits, viewing angle, blur, low resolution, compression, lighting, glare and shadow.
- **Cross-validation by plate** for the reader, because real plates are too few for a meaningful
  fixed test split.
- **ByteTrack boxes are used as they are.** A measured extra smoothing step added more lag than the
  jitter it removed.

## Known limitations

- **Two-row plates** (motorcycles) are detected but not read: the reader reads a single row.
- **The tall plate layout,** with the IL badge under the number, is not read reliably yet.
- **A vehicle that is hidden and re-appears** gets a new track id, so it is logged twice.
- **The read gate is in pixels** (90 px wide). It was tuned on 1080p video, so a lower-resolution
  camera needs the plate to fill more of the frame.

## License

[MIT](LICENSE)
