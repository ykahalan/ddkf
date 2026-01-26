"""ddkf package — compact export of TwoDKF."""

from .core import TwoDKF, denoise

ddkf = TwoDKF

__all__ = ["TwoDKF", "ddkf", "denoise"]
__version__ = "0.1.0"
