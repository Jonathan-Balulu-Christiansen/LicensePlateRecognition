"""Shared test fixtures. src/ and tools/ are put on the import path by pyproject.toml."""

from __future__ import annotations

import os

import pytest

# Qt widgets need a display; "offscreen" lets the tests run anywhere (CI, a locked screen).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp():
    """One QApplication for every test that touches Qt."""
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])
