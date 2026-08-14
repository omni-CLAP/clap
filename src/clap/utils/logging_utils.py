"""Rich-formatted console logging, shared by every clap entrypoint (training/eval/rollout)."""

import logging

from rich.logging import RichHandler


def setup_logging(level=logging.INFO):
    """Configure the root logger with a single Rich console handler.

    Safe to call more than once (e.g. once per entrypoint import) — clears any
    handlers a previous call added first, so messages aren't printed twice.
    """
    root = logging.getLogger()
    root.handlers.clear()
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False, markup=True)],
    )
