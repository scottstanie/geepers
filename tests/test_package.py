from __future__ import annotations

import importlib.metadata

import geepers as m


def test_version():
    assert importlib.metadata.version("geepers") == m.__version__
