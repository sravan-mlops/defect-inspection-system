"""
test_convert_to_yolo.py

Unit tests for the pure box-conversion math (Pascal VOC -> YOLO format).
No files, no I/O - exactly the kind of function that should always have
a fast unit test, since it's easy to get an off-by-one or normalization
error wrong and silently corrupt every label in the dataset.
"""

from src.data.convert_to_yolo import convert_box


def test_convert_box_full_image():
    # A box covering the ENTIRE 100x100 image should center at (0.5, 0.5)
    # with width/height of 1.0 (100% of the image).
    xc, yc, w, h = convert_box(100, 100, 0, 0, 100, 100)
    assert xc == 0.5
    assert yc == 0.5
    assert w == 1.0
    assert h == 1.0


def test_convert_box_top_left_quarter():
    # A box covering the top-left quarter of a 200x200 image should
    # center at (0.25, 0.25) with width/height of 0.5.
    xc, yc, w, h = convert_box(200, 200, 0, 0, 100, 100)
    assert xc == 0.25
    assert yc == 0.25
    assert w == 0.5
    assert h == 0.5