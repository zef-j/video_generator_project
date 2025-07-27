"""
Utilities and pipeline logic for the automated 60‑second vertical video
generator.

This package exposes a single function, :func:`run_video_generation`, which
performs all steps necessary to transform a text script into a finished video.
The pipeline is separated into helper functions for clarity and ease of
testing.  See ``pipeline.py`` for the implementation details.
"""

from .pipeline import run_video_generation  # noqa: F401  re-export for convenience