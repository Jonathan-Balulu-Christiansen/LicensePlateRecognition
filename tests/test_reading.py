"""Reading a plate: the CTC reader's input and decoding, voting, country and number format."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from plate_origin import format_plate, origin_from_samples, plate_color, plate_format_label
from plate_reader import CLASSES, PLATE_LENGTHS, H, PlateVoter, T, W, collapse, decode, encode, prep, softmax

GOOD = (0.99,) * 8


# --------------------------------------------------------------------- reader input


def test_prep_keeps_short_plates_and_squashes_only_very_wide_ones():
    short = prep(np.full((60, 120, 3), 200, np.uint8))  # 2:1 -> 96 px wide, centred on 192
    assert short.shape == (H, W, 1) and short.dtype == np.uint8
    assert (short[:, :40] == 128).all() and (short[:, -40:] == 128).all()  # padding = z 0
    wide = prep(np.random.default_rng(0).integers(0, 255, (40, 400, 3), dtype=np.uint8))
    assert (wide != 128).any(axis=0).sum() > W - 4  # 10:1 fills the whole width


def test_encode_maps_digits_after_the_blank():
    assert encode(["1234567"]).tolist() == [[2, 3, 4, 5, 6, 7, 8, 0]]


# ------------------------------------------------------------------------- decoding


def _logits_for(frames: list[int]) -> np.ndarray:
    """Sharp logits that put one class per column (T columns, padded with blanks)."""
    logits = np.full((1, T, CLASSES), -8.0)
    labels = frames + [0] * (T - len(frames))
    logits[0, np.arange(T), labels] = 8.0
    return logits


def test_collapse_merges_repeats_and_drops_blanks():
    path = [4, 4, 0, 6, 0, 0, 3, 3]  # "3 3 _ 5 _ _ 2 2" -> 352
    probs = softmax(np.eye(CLASSES)[path] * 5)
    text, confs = collapse(path, probs)
    assert text == "352" and len(confs) == 3


def test_a_blank_separates_a_real_double_digit():
    assert collapse([3, 0, 3], softmax(np.eye(CLASSES)[[3, 0, 3]] * 5))[0] == "22"
    assert collapse([3, 3, 3], softmax(np.eye(CLASSES)[[3, 3, 3]] * 5))[0] == "2"


def test_decoding_reads_a_clean_eight_digit_plate():
    digits = "12345601"
    frames = []
    for d in digits:
        frames += [int(d) + 1, 0]
    (text, confs), = decode(_logits_for(frames))
    assert text == digits and min(confs) > 0.99


@pytest.mark.parametrize("n_digits", [5, 6, 9, 10])
def test_constrained_decoding_always_returns_seven_or_eight_digits(n_digits):
    rng = np.random.default_rng(n_digits)
    frames = []
    for d in rng.integers(0, 10, n_digits):
        frames += [int(d) + 1, 0]
    logits = _logits_for(frames) + rng.normal(0, 1.0, (1, T, CLASSES))
    greedy, = decode(logits, constrained=False)
    constrained, = decode(logits)
    assert len(constrained[0]) in PLATE_LENGTHS and constrained[0].isdigit()
    if len(greedy[0]) in PLATE_LENGTHS:
        assert constrained == greedy


# --------------------------------------------------------------------------- voting


def test_locks_after_min_votes_agreeing_reads():
    voter = PlateVoter(min_votes=3)
    assert voter.update(7, "12345601", GOOD) is None
    assert voter.update(7, "12345601", GOOD) is None
    assert voter.update(7, "12345601", GOOD) == "12345601"
    assert voter.locked(7) == "12345601"


def test_rejects_short_long_and_shaky_reads():
    voter = PlateVoter(min_votes=1)
    assert voter.update(1, "123456", (0.99,) * 6) is None  # 6 digits: a partial read
    assert voter.update(1, "123456011", (0.99,) * 9) is None  # 9 digits: too long
    assert voter.update(1, "12345601", (0.99,) * 7 + (0.3,)) is None  # one shaky digit fails it
    assert voter.update(1, "", ()) is None
    assert voter.tally(1) == ("", 0, 4)  # four reads offered, none accepted


def test_tally_reports_the_leading_candidate():
    voter = PlateVoter(min_votes=5)
    for text in ("3456781", "34567801", "34567801"):
        voter.update(9, text, (0.99,) * len(text))
    assert voter.tally(9) == ("34567801", 2, 3)


def test_tracks_are_independent():
    voter = PlateVoter(min_votes=1)
    voter.update(1, "1234567", (0.99,) * 7)
    assert voter.locked(2) is None
    assert voter.tally(2) == ("", 0, 0)


# ------------------------------------------------------------------------- country


def _plate(bgr: tuple[int, int, int]) -> np.ndarray:
    image = np.full((40, 160, 3), bgr, np.uint8)
    cv2.putText(image, "1234567", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
    return image


def test_yellow_plate_is_israeli():
    share, coverage = plate_color(_plate((28, 194, 246)))  # BGR of plate yellow #F6C21C
    assert share > 0.9 and coverage > 0.5
    origin = origin_from_samples([(share, coverage)])
    assert (origin.code, origin.country) == ("IL", "Israel")


def test_blue_plate_is_not_yellow():
    share, coverage = plate_color(_plate((200, 120, 40)))
    assert share == 0.0 and coverage > 0.5
    assert origin_from_samples([(share, coverage)]).method == "plate color · not yellow"


def test_grey_plate_carries_no_color_evidence():
    # A plate still dark at night (or a white one) has no hue: "can't tell", not "not yellow".
    share, coverage = plate_color(_plate((235, 235, 235)))
    assert coverage == 0.0
    assert origin_from_samples([(share, coverage)]).method == "plate color · too dark to judge"


def test_dark_first_reads_do_not_outvote_the_lit_ones():
    # night1, measured: the first read is 25% colored, the rest become yellow as the car nears.
    samples = [(0.03, 0.25), (0.32, 0.36), (0.53, 0.43), (0.67, 0.51), (0.81, 0.51), (0.86, 0.48)]
    assert origin_from_samples(samples).code == "IL"
    assert origin_from_samples([]).code == ""


def test_israeli_number_format():
    assert format_plate("12345601", "IL") == "123-45-601"
    assert format_plate("1234567", "IL") == "12-345-67"
    assert format_plate("12345601", "") == "12345601"
    assert plate_format_label("12345601", "IL") == "8 digits · 3-2-3"
    assert plate_format_label("1234567", "IL") == "7 digits · 2-3-2"
