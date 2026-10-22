# Runbook

Every command, in order: install, train both models, evaluate, run. Run them from the project
root, the folder that contains `src/`. Building the datasets is covered in
[DATASETS.md](DATASETS.md).

Scripts live in a flat `src/` and import each other directly, so they are run as
`python src/<name>.py`. The venv must be active in every new terminal, or `python` resolves to
the system Python. Instead of activating it, `.venv\Scripts\python.exe` can be called directly.

---

## 0. Install

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux / macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt   # tests, ruff, pyright (optional)
python -c "import torch; print(torch.cuda.is_available())"
```

**Devices.**
- `app.py` and `train_reader.py` use the GPU automatically when there is one.
- `predict.py` and `auto_label.py` take `--device 0` (GPU) or `--device cpu`.
- `train_detector.py` sets `device` at the top of `main()`.

---

## 1. Train the detector

```bash
python tools/dataset_check.py --split all     # must report Errors: 0
python src/train_detector.py
```

- **Settings.** The training settings (run name, image size, epochs, batch, device) are at the top
  of `main()` in `src/train_detector.py`.
- **Starting point.** The run starts from the COCO-pretrained YOLO26m checkpoint, which is
  downloaded to `weights/pretrained/` on first use. Ultralytics gets a copy of
  `configs/dataset.yaml` with an absolute dataset path, so the project works from any folder.
- **Augmentation.** It is online: Ultralytics' defaults (mosaic, shift and scale, colour, mirror)
  plus camera effects (blur, motion blur, noise, JPEG, exposure). There is no rotation or shear;
  the README explains why.
- **Output.** The run is written to `runs/train/<run_name>/`. At the end, `best.pt` is scored once
  on the held-out **test** split.

**Putting a model into service.** The app loads the promoted copy, not the latest run:

```bash
cp runs/train/<run_name>/weights/best.pt weights/detector/best.pt
```

Every run also appears in the app's **Detector** list, so it can be tried before it is promoted.

**Detector only, without tracking or reading:**

```bash
python src/predict.py --weights weights/detector/best.pt --source <image | folder | video | camera index> --imgsz 640
```

The output (annotated media, `detections.csv`, `metadata.json`) goes to `runs/predict/<timestamp>/`.

---

## 2. Train the plate reader

```bash
python src/train_reader.py             # pretrain on synthetic plates, fine-tune on real crops
python src/train_reader.py --promote   # the same, then copy the result into service
python src/train_reader.py --cv        # 5-fold cross-validation by plate instead (no model saved)
python src/train_reader.py --preview   # a sheet of augmented training samples
```

**Stage 1: pretraining.** The network trains on `data/synthetic/plates/v1/`. It keeps 2,000 plates
for synthetic validation. The pretrained weights are stored under `runs/reader/pretrained/` and
reused by later runs with the same settings.

**Stage 2: fine-tuning.** The network trains on the real crops in `data/ocr/plates/`, with
synthetic plates mixed into every batch.

**Output.** A run writes `runs/reader/<timestamp>/plate_reader.keras` and `metadata.json`. The
model in service, `weights/reader/plate_reader.keras`, changes only with `--promote`. Every run also
appears in the app's **Reader** list.

**Cross-validation.** `--cv` is the honest score while real plates are few. Each fold's model is
fine-tuned without the plates it is tested on. All crops of one plate stay in the same fold.

---

## 3. Run

```bash
python src/app.py
```

Pick a **Source** (camera or video file), the **Detector** and **Reader**, and **Votes** (agreeing
reads needed to lock; default 5), then **Start**.

**What the screen shows.**
- **Over the video:** each plate gets a box that reads `reading n/5`, then a red reticle once its
  number locks.
- **In the side panel:** one card per plate in view, up to four, as the window height allows. Each
  card shows:
  - the camera's crop next to the number read;
  - every digit with its confidence (unsure digits marked);
  - the country and the read and detector scores.

  Cards keep their order while their plates stay in view.
- **Under the video:**
  - the time to process one frame: **FRAME**, split into DETECT+TRACK / READ / CONVERT;
  - **CAPTURE**, **FPS** and **TRACKS**;
  - a graph against the source's frame budget.

**Timing.** Video files are played at their own frame rate, so a fast GPU does not play them in
fast-forward. A slower machine runs below that rate. On a weak GPU, set `DETECT_EVERY_N_FRAMES = 2`
in `src/pipeline.py`.

**The log.** Every locked plate is appended to `reports/gate_log.jsonl`.

---

## 4. Verify

```bash
python tools/dataset_check.py --split all
pytest
pytest -m slow
ruff check src tools tests
pyright src tools tests
```

`pytest -m slow` runs the real models end to end on a local clip. It reads which clip to play and
which number it must lock from `data/raw/videos/expected_plates.json`:

```json
{"my_clip.mp4": "12345601"}
```

Without that file, or without the models, the slow tests skip themselves.

---

## At a glance

```text
Build (once):
  label -> pool -> split_dataset -> dataset_check -> train_detector.py   => weights/detector/best.pt
  synth_plates (40,000) + real crops -> train_reader.py --promote        => weights/reader/plate_reader.keras

Run:
  app.py  =>  detect -> track -> read all plates in one batch -> vote -> locked number
```
