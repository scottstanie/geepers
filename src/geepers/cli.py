from __future__ import annotations

import tyro

from .core import main


def cli():  # noqa: D103
    tyro.cli(main)


cli.__doc__ = main.__doc__


if __name__ == "__main__":
    cli()
