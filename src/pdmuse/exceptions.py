"""Custom exceptions used by PD-MUSE."""


class PDMuseError(Exception):
    """Base class for package-specific errors."""


class NotFittedError(PDMuseError):
    """Raised when prediction is requested before fitting."""


class DataValidationError(PDMuseError, ValueError):
    """Raised when choice data are malformed."""


class OptimizationError(PDMuseError, RuntimeError):
    """Raised when an optimizer cannot make numerical progress."""
