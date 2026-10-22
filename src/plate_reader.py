"""Read an Israeli licence-plate crop in one pass: a CNN + CTC network, no digit segmentation.

The whole plate crop goes in; the network shrinks the height to 1 but keeps the width, so it
emits 48 columns, each saying "blank" or "digit d" for one 4-pixel slice of the plate. CTC
training only needs the full number as a string — the network learns by itself where each
digit starts and ends. Decoding merges repeated columns, drops blanks, and is constrained to
return exactly 7 or 8 digits (the two Israeli civilian formats). Temporal voting (PlateVoter)
then locks a number once enough reads of the same tracked plate agree.

Architecture ("tcn"): a 2D CNN trunk (4 conv blocks, 2x2 pooling twice then 2x1 twice, so the
width stays at 48) collapses the height, then a residual stack of dilated 1D convolutions
(dilation 1..32) runs along the plate, so every output column sees the whole plate and can use
its 3-2-3 / 2-3-2 layout. 1.85M parameters. Measured with 5-fold cross-validation by plate on
the 107 real crops: 98.1% of whole plates exact (the segmentation + digit-CNN reader it replaced:
64.5-67.3%).

Training lives in train_reader.py; this module is what the app needs at run time.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common import PROJECT_ROOT

# Keras picks its backend when it is first imported: set it before anything can import it, or
# Keras looks for TensorFlow (not installed). The project runs Keras on PyTorch.
os.environ.setdefault("KERAS_BACKEND", "torch")
cv2.setNumThreads(0)  # OpenCV's thread pool crashes intermittently next to PyTorch in one process

READER_PATH = PROJECT_ROOT / "weights" / "reader" / "plate_reader.keras"
H, W = 48, 192  # input: height 48, aspect kept, width padded (or squashed) to 192
T = W // 4  # two horizontal /2 poolings -> 48 time steps (CTC needs >= 2 * 8 + 1 = 17)
CLASSES = 11  # 0 = CTC blank, digit d -> class d + 1
MAX_LEN = 8
PLATE_LENGTHS = (7, 8)
TCN_DILATIONS = (1, 2, 4, 8, 16, 32)  # receptive field 127 steps >= 2 * T - 1: the whole plate


# ----------------------------------------------------------------------------- input


def prep(bgr: np.ndarray) -> np.ndarray:
    """A plate crop -> the network input (48, 192, 1) uint8. Identical for training and reading.

    Grey, resized to height 48 keeping the aspect ratio; wider than 4:1 is squashed to 192,
    narrower is centred on a 192-wide canvas. Standardised over the plate pixels only (so
    lighting and contrast do not matter) and stored as z * 32 + 128 in uint8 — padding is 128,
    i.e. z = 0. The model's first layer undoes that mapping.
    """
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    h, w = grey.shape
    new_w = int(min(W, max(8, round(w * H / h))))
    resized = cv2.resize(grey, (new_w, H), interpolation=cv2.INTER_AREA if h >= H else cv2.INTER_LINEAR)
    pixels = resized.astype(np.float32)
    z = (pixels - pixels.mean()) / (pixels.std() + 1e-6)
    out = np.full((H, W), 128, np.uint8)
    x0 = (W - new_w) // 2
    out[:, x0: x0 + new_w] = np.clip(np.round(z * 32 + 128), 0, 255).astype(np.uint8)
    return out[..., None]


def encode(numbers: list[str]) -> np.ndarray:
    """Plate numbers -> CTC targets (N, 8): digit d -> d + 1, padded with 0 (the blank)."""
    targets = np.zeros((len(numbers), MAX_LEN), np.int32)
    for i, number in enumerate(numbers):
        targets[i, : len(number)] = [int(c) + 1 for c in number]
    return targets


# ----------------------------------------------------------------------------- network


def build_model():
    """The tcn reader network (Keras): CNN trunk -> 48 columns -> dilated residual Conv1D -> 11 logits."""
    import keras
    from keras import layers

    x = inputs = keras.Input((H, W, 1))
    x = layers.Rescaling(1 / 32, offset=-4)(x)  # uint8 (z * 32 + 128) -> z
    for filters, pool in ((32, (2, 2)), (64, (2, 2)), (128, (2, 1)), (192, (2, 1))):
        for _ in range(2 if filters < 192 else 1):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        x = layers.MaxPooling2D(pool)(x)  # 2x2 early, then 2x1: the height shrinks, the width stays 48
    x = layers.Conv2D(256, (H // 16, 1), padding="valid", activation="relu")(x)  # 3 x 48 -> 1 x 48
    x = layers.Reshape((T, 256))(x)
    for dilation in TCN_DILATIONS:  # context along the plate; residual so it trains easily
        y = layers.Conv1D(256, 3, padding="same", dilation_rate=dilation, use_bias=False)(x)
        y = layers.BatchNormalization()(y)
        y = layers.ReLU()(y)
        x = layers.Add()([x, y])
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(CLASSES)(x)  # logits: blank + digits 0-9, one row per 4-pixel column
    return keras.Model(inputs, outputs, name="plate_reader")


# ----------------------------------------------------------------------------- decoding


def softmax(logits: np.ndarray) -> np.ndarray:
    """Per-column class probabilities."""
    e = np.exp(logits - logits.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def collapse(path, probs: np.ndarray) -> tuple[str, tuple[float, ...]]:
    """Column labels -> (number, per-digit confidence): merge repeats, drop blanks. A digit's
    confidence is the highest probability over the columns that emitted it."""
    text, confs, previous = "", [], 0
    for t, k in enumerate(path):
        k = int(k)
        if k != 0 and k != previous:
            text += str(k - 1)
            confs.append(float(probs[t, k]))
        elif k != 0:
            confs[-1] = max(confs[-1], float(probs[t, k]))
        previous = k
    return text, tuple(confs)


def best_path_constrained(probs: np.ndarray, lengths=PLATE_LENGTHS) -> list[int] | None:
    """The most probable single CTC alignment whose collapsed number has one of `lengths` digits.

    Dynamic programming over (time, digits emitted so far, label of the previous column), with
    CTC's blank and repeat-merging rules. When the unconstrained best path (greedy decoding)
    already has an allowed length, this returns that same path; otherwise it returns the best
    path that does — so a 6- or 9-digit read becomes the most likely 7- or 8-digit one.
    """
    logp = np.log(probs + 1e-30)
    steps, classes = logp.shape
    longest = max(lengths)
    score = np.full((longest + 1, classes), -np.inf)
    score[0, 0] = logp[0, 0]
    score[1, 1:] = logp[0, 1:]
    back = np.zeros((steps, longest + 1, classes, 2), np.int16)
    rows = np.arange(longest + 1)
    same_class = np.eye(classes, dtype=bool)
    for t in range(1, steps):
        new = np.full((longest + 1, classes), -np.inf)
        b = np.zeros((longest + 1, classes, 2), np.int16)
        best_prev = score.argmax(1)  # a blank: from any previous label, same digit count
        new[:, 0] = score[rows, best_prev] + logp[t, 0]
        b[:, 0, 0], b[:, 0, 1] = rows, best_prev
        new[:, 1:] = score[:, 1:] + logp[t, 1:]  # the same digit repeated: same count
        b[:, 1:, 0] = rows[:, None]
        b[:, 1:, 1] = np.arange(1, classes)[None, :]
        emit = np.where(same_class[None], -np.inf, score[:-1, :, None])  # a new digit: count + 1
        emit_from = emit.argmax(1)
        emitted = np.take_along_axis(emit, emit_from[:, None, :], 1)[:, 0, :] + logp[t][None, :]
        better = emitted[:, 1:] > new[1:, 1:]
        new[1:, 1:] = np.where(better, emitted[:, 1:], new[1:, 1:])
        b[1:, 1:, 0] = np.where(better, rows[:-1, None], b[1:, 1:, 0])
        b[1:, 1:, 1] = np.where(better, emit_from[:, 1:], b[1:, 1:, 1])
        score, back[t] = new, b
    best, count, label = max((score[n, c], n, c) for n in lengths for c in range(classes))
    if not np.isfinite(best):
        return None
    path = [int(label)]
    for t in range(steps - 1, 0, -1):
        count, label = back[t, count, label]
        path.append(int(label))
    return path[::-1]


def decode(logits: np.ndarray, constrained: bool = True) -> list[tuple[str, tuple[float, ...]]]:
    """Logits (N, 48, 11) -> [(number, per-digit confidences)] — constrained to 7 or 8 digits by default."""
    reads = []
    for row in logits:
        probs = softmax(row)
        path = best_path_constrained(probs) if constrained else None
        reads.append(collapse(probs.argmax(-1) if path is None else path, probs))
    return reads


# ----------------------------------------------------------------------------- reading

_readers: dict[Path, Any] = {}


def load_reader(path: Path = READER_PATH):
    """Load (once per file) and warm up a trained reader, so the first plate does not pay the init cost."""
    path = Path(path).resolve()
    if path not in _readers:
        if not path.is_file():
            raise FileNotFoundError(f"Plate reader not found: {path}")
        import keras

        model: Any = keras.models.load_model(path, compile=False)
        model(np.full((1, H, W, 1), 128, np.float32), training=False)
        _readers[path] = model
    return _readers[path]


def read_plates(crops: list[np.ndarray], model: Any = None) -> list[tuple[str, tuple[float, ...]]]:
    """Read several plate crops in one batch -> [(number, per-digit confidences)].

    One forward pass for all the plates of a frame (~1 ms per plate on the RTX 4070 Ti at
    batch 8). The model is called directly rather than through predict(), which adds several
    milliseconds of per-call overhead. Empty crops read as ("", ()).
    """
    import keras

    reads: list[tuple[str, tuple[float, ...]]] = [("", ())] * len(crops)
    usable = [i for i, crop in enumerate(crops) if crop.size and min(crop.shape[:2]) >= 4]
    if not usable:
        return reads
    batch = np.stack([prep(crops[i]) for i in usable]).astype(np.float32)
    network: Any = model if model is not None else load_reader()
    logits = np.asarray(keras.ops.convert_to_numpy(network(batch, training=False)))
    for i, read in zip(usable, decode(logits)):
        reads[i] = read
    return reads


class PlateVoter:
    """Temporal voting: accumulate per-track reads and lock the number once one whole-string
    read reaches ``min_votes`` agreeing frames.
    """

    def __init__(self, min_votes: int = 5, min_conf: float = 0.5,
                 min_len: int = min(PLATE_LENGTHS), max_len: int = max(PLATE_LENGTHS)) -> None:
        """Set the voting thresholds: agreeing reads needed, minimum digit confidence, and the
        allowed plate length (7-8 digits)."""
        self.min_votes = min_votes
        self.min_conf = min_conf
        self.min_len = min_len
        self.max_len = max_len
        self._votes: dict[int, Counter] = {}
        self._locked: dict[int, str] = {}
        self._reads: Counter = Counter()  # every read offered per track, accepted or not

    def update(self, track_id: int, string: str, confs: tuple[float, ...]) -> str | None:
        """Add one read; return the locked number for this track, or None yet.

        Gates on the *weakest* digit: one shaky digit fails the whole read. (The reader's
        confidences are informative: across cross-validation, right reads averaged 0.99 and
        wrong ones 0.76 for the product of digit confidences.)
        """
        self._reads[track_id] += 1
        if string and confs and min(confs) >= self.min_conf and self.min_len <= len(string) <= self.max_len:
            votes = self._votes.setdefault(track_id, Counter())
            votes[string] += 1
            best, count = votes.most_common(1)[0]
            if count >= self.min_votes:
                self._locked[track_id] = best
        return self._locked.get(track_id)

    def locked(self, track_id: int) -> str | None:
        """Return the already-locked number for a track (no new vote)."""
        return self._locked.get(track_id)

    def tally(self, track_id: int) -> tuple[str, int, int]:
        """(leading string, its votes, all reads so far) for a track — what the UI shows while a
        plate is still being read. ("", 0, reads) before any read was accepted."""
        reads = self._reads[track_id]
        votes = self._votes.get(track_id)
        if not votes:
            return "", 0, reads
        best, count = votes.most_common(1)[0]
        return best, count, reads
