"""CPU-only regression tests for synthetic plate labels, layouts, sampling and safe output."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import replace

import cv2
import numpy as np
import pytest
import synth_plates as synth


@pytest.fixture
def output_root(tmp_path, monkeypatch):
    monkeypatch.setattr(synth, "OUT_ROOT", tmp_path)
    return tmp_path


def _labels(path):
    with (path / "labels.csv").open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


@pytest.mark.parametrize("plate_type", synth.PLATE_SIZES)
@pytest.mark.parametrize("number", ("0123456", "01234567"))
def test_each_layout_renders_both_lengths_without_losing_digits(plate_type, number, monkeypatch):
    drawn = []
    original = synth._draw_number

    def record(img, text, *args, **kwargs):
        drawn.append(text)
        return original(img, text, *args, **kwargs)

    monkeypatch.setattr(synth, "_draw_number", record)
    image = synth.render_plate(number, synth.Rng(7), plate_type)
    width, height = synth.PLATE_SIZES[plate_type]
    assert image.shape == (height, width, 3)
    assert image.dtype == np.uint8
    assert len(drawn) == (2 if plate_type == "two_row" else 1)
    assert "".join(char for line in drawn for char in line if char.isdigit()) == number
    assert image.std() > 20


@pytest.mark.parametrize("number", ("123456", "123456789", "ABC1234", "123 567", "１２３４５６７"))
def test_rejects_non_israeli_registration_labels(number):
    with pytest.raises(ValueError, match="ASCII digits"):
        synth.render_plate(number, synth.Rng(0))


def test_exact_labels_filenames_and_metadata(output_root, monkeypatch):
    monkeypatch.setattr(synth, "random_number", lambda rng, eight_share: "01234567")
    path = synth.generate(24, "labels", 4, 0.25, True)
    rows = _labels(path)
    assert len(rows) == 24
    assert tuple(rows[0]) == synth.LABEL_COLUMNS
    for index, row in enumerate(rows):
        assert row["file"] == f"{row['number']}_{index:06d}.png"
        assert row["number"] == "01234567"
        assert row["digits"] == "8"
        assert int(row["rows"]) == (2 if row["plate_type"] == "two_row" else 1)
        assert row["separator"] in synth.SEPARATORS
        assert row["clean"] in ("0", "1")
        if row["plate_type"].endswith("_plain"):
            assert row["has_flag"] == row["has_il"] == "0"
        assert cv2.imread(str(path / row["file"])) is not None
    assert cv2.imread(str(path / "preview.png")) is not None


def test_same_seed_produces_identical_images_labels_and_preview(output_root):
    first = synth.generate(16, "first", 17, 0.25, True)
    second = synth.generate(16, "second", 17, 0.25, True)
    assert sorted(p.name for p in first.iterdir()) == sorted(p.name for p in second.iterdir())
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()
    different = synth.generate(16, "different", 18, 0.25, False)
    assert (first / "labels.csv").read_bytes() != (different / "labels.csv").read_bytes()


@pytest.mark.parametrize("populated", (False, True))
def test_refuses_to_overwrite_even_an_empty_run(output_root, populated):
    existing = output_root / "existing"
    existing.mkdir()
    if populated:
        (existing / "keep.txt").write_text("keep", encoding="utf-8")
    before = {p.name: p.read_bytes() for p in existing.iterdir()}
    with pytest.raises(FileExistsError):
        synth.generate(1, "existing", 1, 0.1, False)
    assert {p.name: p.read_bytes() for p in existing.iterdir()} == before


@pytest.mark.parametrize("name", ("../escape", "..", ".", "a/b", r"a\b", "D:escape", ""))
def test_run_name_cannot_escape_output_root(output_root, name):
    with pytest.raises(ValueError):
        synth.generate(1, name, 1, 0.1, False)
    assert list(output_root.iterdir()) == []


@pytest.mark.parametrize("weights", (
    {"unknown": 1}, {"long_plain": -1}, {"long_plain": 0},
    {"long_plain": float("nan")}, {"long_plain": float("inf")},
))
def test_rejects_invalid_mixes_before_writing(output_root, weights):
    with pytest.raises(ValueError):
        synth.generate(1, "invalid", 0, 0.1, False, weights=weights)
    assert list(output_root.iterdir()) == []


def test_requested_type_mix_and_digit_share_in_written_labels(output_root):
    weights = {"long_plain": 1.0, "short_badge": 3.0}
    path = synth.generate(240, "mix", 42, 1, False, weights=weights, eight_share=0.65)
    rows = _labels(path)
    counts = Counter(row["plate_type"] for row in rows)
    assert set(counts) == set(weights)
    assert abs(counts["long_plain"] / len(rows) - 0.25) < 0.10
    assert abs(sum(row["digits"] == "8" for row in rows) / len(rows) - 0.65) < 0.10


def test_default_mix_and_independent_decorations():
    rng = synth.Rng(8)
    mix = synth.type_mix()
    counts = Counter(synth.choose_type(rng, mix) for _ in range(12000))
    for name, weight in synth.DEFAULT_MIX.items():
        assert abs(counts[name] / 12000 - weight / 95) < 0.015
    styles = [synth.choose_style("long_strip", rng) for _ in range(4000)]
    assert {(s.has_flag, s.has_il) for s in styles} == {(False, False), (False, True), (True, False), (True, True)}
    assert len({(s.triangle, s.small_print, s.has_flag) for s in styles}) == 8


def test_selected_types_use_renormalized_defaults():
    assert synth.type_mix(["short_strip", "short_badge"]) == {"short_strip": 7 / 15, "short_badge": 8 / 15}
    with pytest.raises(ValueError):
        synth.type_mix(["long_strip"], {"long_plain": 1})


def test_flag_and_il_independently_change_pixels():
    rng = synth.Rng(7)
    style = synth.choose_style("long_strip", rng)
    full = replace(style, has_flag=True, has_il=True)
    no_flag = replace(full, has_flag=False)
    no_il = replace(full, has_il=False)
    image = synth.render_plate("01234567", synth.Rng(1), style=full)
    flagless = synth.render_plate("01234567", synth.Rng(1), style=no_flag)
    illess = synth.render_plate("01234567", synth.Rng(1), style=no_il)
    assert np.any(image[:40, :50] != flagless[:40, :50])
    assert np.array_equal(image[45:, :50], flagless[45:, :50])
    assert np.array_equal(image[:35, :50], illess[:35, :50])
    assert np.any(image[45:, :50] != illess[45:, :50])


@pytest.mark.parametrize("plate_type", synth.PLATE_SIZES)
def test_augmentation_is_deterministic_and_keeps_resolution(plate_type):
    plate = synth.render_plate("01234567", synth.Rng(1), plate_type)
    assert np.array_equal(plate, synth.augment(plate, synth.Rng(1), 0))
    for seed in range(8):
        image = synth.augment(plate, synth.Rng(seed))
        assert image.shape[0] >= 28 and image.shape[1] >= 180
        assert np.array_equal(image, synth.augment(plate, synth.Rng(seed)))


def test_cli_mix_options(monkeypatch):
    monkeypatch.setattr("sys.argv", ["synth_plates.py", "--name", "custom", "--types", "short_badge",
                                    "--type-mix", "short_badge=1", "--eight-share", "0.4"])
    args = synth.parse_args()
    assert args.types == ["short_badge"]
    assert args.weights == {"short_badge": 1}
    assert args.eight_share == 0.4
