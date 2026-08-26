"""Tiny shared verbose-logging helper for the pipeline steps.

Each step exposes a ``--verbose`` flag. When it is off (the default) only the
step's boundary / summary / error lines print; per-view, per-iteration and
debug chatter is suppressed. ``set_verbose()`` flips a process-global flag that
``vprint()`` checks, so helper functions anywhere in the step module stay quiet
without threading a ``verbose`` argument through every call.
"""

import os

# Default from the environment so the orchestrator can enable verbose for every
# step subprocess at once by exporting MILO_VERBOSE=1 (run_pipeline.py --verbose).
_VERBOSE = os.environ.get("MILO_VERBOSE") == "1"


def set_verbose(value: bool) -> None:
    """Enable/disable verbose (per-view / per-iteration) printing for this process.
    The MILO_VERBOSE env var (set by run_pipeline --verbose) also forces it on."""
    global _VERBOSE
    _VERBOSE = bool(value) or os.environ.get("MILO_VERBOSE") == "1"


def is_verbose() -> bool:
    return _VERBOSE


def vprint(*args, **kwargs) -> None:
    """``print`` only when ``--verbose`` is set."""
    if _VERBOSE:
        print(*args, **kwargs)
