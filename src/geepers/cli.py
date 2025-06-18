from __future__ import annotations

import logging

import tyro

from .core import main

logger = logging.getLogger("geepers")
logger.setLevel(logging.INFO)
h = logging.StreamHandler()
f = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
h.setFormatter(f)
logger.addHandler(h)


def cli():  # noqa: D103
    tyro.cli(main)


cli.__doc__ = main.__doc__


if __name__ == "__main__":
    cli()
