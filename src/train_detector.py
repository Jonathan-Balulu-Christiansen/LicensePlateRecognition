"""Train the license-plate DETECTOR (YOLO26m).

Fine-tunes the pretrained YOLO on our plate dataset so it learns to find plates,
then measures it once on the held-out test set. Edit the parameters at the top of
main(), then run `python src/train_detector.py`.

YOLO26m (Ultralytics, Jan 2026): an NMS-free end-to-end head and a label assignment tuned
for small objects — our plates are often only 50-250 px wide in a 1920x1080 frame. Trained
at 640, the same size it was pretrained at on COCO and the size the app runs at. The
pretrained COCO checkpoint lives in weights/pretrained/ and is downloaded there on first use.

Augmentation is online, inside Ultralytics' training loader: its defaults (mosaic, shift and
scale, colour, mirror images), plus CameraEffects — what a real camera does to a frame: blur,
motion blur, sensor noise, JPEG artefacts, exposure — at modest rates, without a new dependency
(Ultralytics would need albumentations, whose OpenCV wheel clashes with the one installed).
No rotation, shear or perspective: Ultralytics turns a rotated box into the box around it, and
for a plate (~4.7:1) 10 degrees already makes that box ~80% taller than the plate, and training
on such loose boxes teaches loose predictions.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

from common import PROJECT_ROOT, check_cuda_or_fail, resolve_project_path, resolved_dataset_yaml

PRETRAINED_DIR = PROJECT_ROOT / "weights" / "pretrained"
APP_IMGSZ = 640  # the size the app runs inference at (pipeline.INFER_IMGSZ)


class CameraEffects:
    """One step in Ultralytics' augmentation chain: what a real camera does to a frame.

    Each effect is applied independently with its own probability and a mild strength, so most
    frames get none or one. Pixel values only — the plate boxes are untouched.
    """

    def __init__(self, p_blur=0.10, p_motion=0.10, p_noise=0.15, p_jpeg=0.20, p_exposure=0.20) -> None:
        """Probabilities of each effect (per training image)."""
        self.p = {"blur": p_blur, "motion": p_motion, "noise": p_noise, "jpeg": p_jpeg, "exposure": p_exposure}

    def __call__(self, labels: dict) -> dict:
        """Ultralytics transforms take and return the labels dict; the image is labels["img"] (BGR)."""
        img = labels["img"]
        rng = random.Random()
        if rng.random() < self.p["blur"]:
            img = cv2.GaussianBlur(img, (rng.choice((3, 5)),) * 2, 0)
        if rng.random() < self.p["motion"]:  # horizontal: cars move sideways through the frame
            length = rng.choice((3, 5, 7))
            kernel = np.zeros((length, length), np.float32)
            kernel[length // 2, :] = 1.0 / length
            img = cv2.filter2D(img, -1, kernel)
        if rng.random() < self.p["exposure"]:
            exposed = img.astype(np.float32) * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20)
            img = np.clip(exposed, 0, 255).astype(np.uint8)
        if rng.random() < self.p["noise"]:
            grain = np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(3, 8), img.shape)
            img = np.clip(img.astype(np.float32) + grain, 0, 255).astype(np.uint8)
        if rng.random() < self.p["jpeg"]:
            _, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(40, 90)])
            img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        labels["img"] = img
        return labels


def add_camera_effects(dataset) -> None:
    """Put CameraEffects into a training dataset's chain, just before the final Format step."""
    transforms = dataset.transforms
    if not any(isinstance(t, CameraEffects) for t in transforms.transforms):
        transforms.insert(len(transforms.transforms) - 1, CameraEffects())


class PlateTrainer(DetectionTrainer):
    """Ultralytics' detection trainer, with CameraEffects in the training chain.

    Added when the training dataset is built, and again after the mosaic switch-off for the last
    epochs, because that rebuilds the chain (the trainer resets the loader's workers right after,
    so they pick it up).
    """

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        dataset = super().build_dataset(img_path, mode, batch)
        if mode == "train":
            add_camera_effects(dataset)
        return dataset

    def _close_dataloader_mosaic(self):
        super()._close_dataloader_mosaic()
        add_camera_effects(self.train_loader.dataset)


def train_detector(
    base_model: str,
    run_name: str,
    train_params: dict[str, Any],
    device: str = "0",
    data_yaml: Path = PROJECT_ROOT / "configs" / "dataset.yaml",
    camera_effects: bool = True,
) -> Path:
    """Fine-tune `base_model` on the plate dataset, then score best.pt on the test split.

    `base_model` is a checkpoint name in weights/pretrained/ (e.g. "yolo26m.pt"), or a
    model YAML such as "yolo26m-p2.yaml" with train_params["pretrained"] pointing at the
    checkpoint whose weights it should start from. Returns the run folder.
    """
    check_cuda_or_fail(device)
    runs_train_dir = (PROJECT_ROOT / "runs" / "train").resolve()
    data = resolved_dataset_yaml(data_yaml, runs_train_dir)
    model_source = base_model if base_model.endswith(".yaml") else str(PRETRAINED_DIR / base_model)

    # --- Train ---
    model = YOLO(model_source)
    model.train(
        trainer=PlateTrainer if camera_effects else None,
        data=str(data),
        device=device,
        project=str(runs_train_dir),
        name=run_name,
        **train_params,
    )

    # --- Evaluate best.pt once on the held-out test split ---
    # Ultralytics may auto-increment run_name if the folder exists, so read the
    # real save_dir back from the trainer instead of guessing it.
    trainer = model.trainer
    assert trainer is not None
    save_dir = Path(trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    print(f"\nTraining finished. Best weights: {best}")

    # Scored at the training size and at the app's inference size: the app runs at 640
    # even when training used 960, so the 640 number is the one the demo actually gets.
    for imgsz in dict.fromkeys((int(train_params["imgsz"]), APP_IMGSZ)):  # each size once
        metrics = YOLO(str(best)).val(
            data=str(data),
            split="test",
            imgsz=imgsz,
            device=device,
            project=str(save_dir),
            name=f"test_eval_{imgsz}",
        )
        print(f"Held-out TEST metrics at imgsz={imgsz}:")
        print(f"  precision: {metrics.box.mp:.4f}")
        print(f"  recall:    {metrics.box.mr:.4f}")
        print(f"  mAP50:     {metrics.box.map50:.4f}")
        print(f"  mAP50-95:  {metrics.box.map:.4f}")
    return save_dir


def main() -> None:
    """One full run: fine-tune the detector, then score `best.pt` on the test split."""
    # --- Parameters (edit these directly) ---
    base_model = "yolo26m.pt"  # checkpoint in weights/pretrained/ to fine-tune from
    # Output folder under runs/train/. To put a newly trained model into service, copy its
    # weights/best.pt to weights/detector/best.pt — that promoted copy is what the app loads
    # by default (the app's Detector list also offers every run directly).
    run_name = "plate_yolo26m_img640_synth_cam"
    device = "0"  # "0" = GPU, "cpu" = CPU
    train_params: dict[str, Any] = {
        "imgsz": 640,
        "epochs": 100,
        "batch": 8,
        "workers": 4,
        # No early stop: mAP50-95 (how tight the boxes are) keeps improving late in training,
        # and a small validation split is noisy.
        "patience": 100,
        "amp": True,  # mixed precision (faster on GPU)
        "seed": 0,
    }
    train_detector(base_model, run_name, train_params, device, resolve_project_path("configs/dataset.yaml"))


if __name__ == "__main__":
    main()
