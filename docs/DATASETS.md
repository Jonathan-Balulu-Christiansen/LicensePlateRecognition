# Building the datasets from scratch

Visus trains two models, each on its own dataset:

| Model | Learns | Data | Label |
|---|---|---|---|
| Detector (YOLO26m) | where the plates are | full frames | one box per plate (YOLO format) |
| Reader (CNN+CTC) | what the number is | plate crops | the number, as the file name |

Both follow the same rules:
- Validation and test data are **real images only**. Synthetic data is for training only.
- Nothing is split by image if the same vehicle can appear in more than one image (frames of one
  video, crops of one plate). Split by vehicle or plate instead, or the scores will be inflated.

The commands below are run from the project root. Every tool prints its options with `-h`.

---

## 1. Detector dataset

### Layout

```text
data/detector/
  pool/images/      every labelled image   (the source of truth; never modified by a split)
  pool/labels/      one .txt per image, same name
  images/{train,val,test}/   derived by split_dataset.py
  labels/{train,val,test}/
  images/synth/  labels/synth/   optional synthetic scenes (training only, see section 3)
```

`configs/dataset.yaml` points at this folder. A label file has one line per plate, in YOLO
format, with values normalised to 0–1:

```text
0 <x_center> <y_center> <width> <height>
```

Class `0` is `license_plate`. An image with no plate has an empty label file.

### Collecting images

- **Coverage.** Use photos and video frames of vehicles, front and rear, at the distances, angles,
  times of day and weather the system will see.
- **Video frames.** `tools/extract_frames_from_video.py` pulls frames at a fixed interval. Prefer
  frames a few seconds apart: neighbouring frames are near-duplicates and add little.
- **Difficult examples.** Include them: night, rain, glare, small and partly hidden plates,
  motorcycles.

### Labelling

- **What to box.** Label **every** visible plate in an image, including small ones and those of
  background vehicles. An unlabelled plate teaches the detector that it is not a plate. Box the
  plate itself, tight to its edges, not the area around it.
- **The first labels.** Any tool that exports YOLO format works (for example Label Studio, CVAT or
  Roboflow).
- **After the first model.** Once a first model exists, `tools/auto_label.py` pre-labels new images
  and sorts them into piles for a quick human review:
  - `auto_labeled/`: confident
  - `review/`: uncertain
  - `needs_manual/`: nothing found

  ```bash
  python tools/auto_label.py --weights weights/detector/best.pt --source <new images> --output <review folder>
  ```

  Check every pile before moving images and labels into `data/detector/pool/`.

### Splitting and checking

```bash
python tools/split_dataset.py --seed 42             # 80/10/10 from the pool, reproducible
python tools/dataset_check.py --split all           # format, normalisation, cross-split leakage
```

`split_dataset.py` rebuilds `images/` and `labels/` from the pool on every run and writes
`split_manifest.csv` as the record of the split. Re-splitting never touches the pool.
`dataset_check.py` must report `Errors: 0` before training.

---

## 2. Reader dataset

### Real plate crops

```text
data/ocr/plates/
  12345601.jpg                       the file name is the label: the plate number
  1234567_clip03_f00180.png          several crops of one plate: number, "_", anything unique
```

- **Where crops come from.** Crop each plate from the detector images or from video, using the
  detector's own boxes. Real crops should look like what the detector hands the reader at run time.
- **Size.** Keep crops at least ~90 px wide; the app does not read smaller plates.
- **What to leave out.** Leave out plates whose number you cannot read with certainty yourself.
  One wrong label teaches the reader a wrong digit.
- **Many crops of one plate.** Name them all with the same number. Cross-validation
  (`train_reader.py --cv`) then keeps all of them in the same fold.

### Synthetic plates (pretraining)

```bash
python tools/synth_plates.py --count 40000 --name v1 --preview
```

This writes `data/synthetic/plates/v1/`: the plate images, `labels.csv`, and a preview sheet. The
rendered plates follow the Israeli layouts:
- long plates, with and without the blue IL strip;
- short plates;
- two-row plates.

They carry 7- or 8-digit numbers, mild camera and wear effects, and labels that are exact by
construction. `train_reader.py` reads `data/synthetic/plates/v1/` and leaves two-row plates out,
since the reader reads one row.

During training, every plate is also augmented online, so each epoch sees new versions:
- rotation up to ±12° on an enlarged canvas, so no digit is cut;
- viewing angle and camera pitch;
- loose or tight detector boxes;
- blur, motion blur, low resolution, noise, JPEG compression;
- exposure, haze, night colour, glare, shadow;
- dirt on the border.

Each sample gets 1–3 of these groups, and 15% stay clean. To see a sheet of examples:

```bash
python src/train_reader.py --preview
```

---

## 3. Synthetic scenes (optional, detector training)

Full synthetic frames add variety the real images may lack: places, cameras, weather, vehicle
makes and colours. They are made in two steps.

**1. Draw the scenes with blank plates.** An image-generation model draws each scene with the
plates left **blank**: image models cannot be trusted to print exact digits.
- `tools/gen_scenes.py` plans a balanced batch and writes the request to send:
  - camera, place, time and weather;
  - the vehicles' make, model, body and colour;
  - the plate type.
- The plates in the returned images are measured (boxes and corners, YOLO labels plus a
  `manifest.jsonl`) and checked against the detector. Plate positions reported by the image model
  itself are not reliable.

**2. Print numbers onto the plates.**

```bash
python tools/number_scenes.py data/synthetic/scenes/<batch> --preview preview.png   # check first
python tools/number_scenes.py data/synthetic/scenes/<batch>
```

Each blank long plate gets a number in the real format:
- **8 digits:** `123:45-601`, where the 7th digit is 0 and the last is 1–4.
- **7 digits:** `12:345-67`.
- **The strip:** IL with or without the flag, or empty.

The digits are printed by darkening the blank plate, so they take on the scene's light, blur and
colour. The output folder `<batch>_numbered/` holds:
- the images and their labels;
- reader crops of plates at least 90 px wide;
- the manifest, with each plate's number and its vehicle's make, model and colour.

**3. Use them for training only.** Copy the numbered images and labels to
`data/detector/images/synth/` and `labels/synth/`. `configs/dataset.yaml` lists that folder under
`train_extra`, which is added to the training set only. Validation and test stay real.

---

## 4. Before training

- `dataset_check.py` reports no errors, and no image or plate appears in more than one split.
- Validation and test contain real data only, and enough of it to be meaningful: at least a few
  dozen images, covering day and night.
- The real reader crops were read by a person, and doubtful ones were removed.

Training itself is in [RUNBOOK.md](RUNBOOK.md).
