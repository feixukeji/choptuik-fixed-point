"""Canonical repository paths.

Everything writes through here, so the layout of ``out/`` is defined in one
place: ``out/<package>`` mirrors ``src/<package>``.

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from paths import out
    OUT = out("fixedpoint")      # -> <repo>/out/fixedpoint, created if absent

Scripts therefore do not care about the working directory they are run from.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
DATA = os.path.join(ROOT, "data")          # the tabulated solution and its modes


def out(*parts, mkdir=True):
    """``<repo>/out/<parts...>``, created on demand."""
    p = os.path.join(ROOT, "out", *parts)
    if mkdir:
        os.makedirs(p, exist_ok=True)
    return p


def logs():
    """``<repo>/out/logs`` -- batch-job and long-run logs, all packages."""
    return out("logs")
