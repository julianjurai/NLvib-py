"""
NLvib — Python port of the NLvib MATLAB toolbox.

Nonlinear vibration analysis via harmonic balance, shooting method,
and arc-length continuation.

Original MATLAB toolbox by Malte Krack & Johann Gross (University of Stuttgart).
Python port — GPL-3.0.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("nlvib")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"
