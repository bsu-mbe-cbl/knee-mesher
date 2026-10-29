"""Expose public facing exceptions and warnings."""


class Error(Exception):
    """Base class for other exceptions."""


class MeshPreparationError(Error):
    """Raised when mesh preparation fails."""


class OptimizationError(Error):
    """Raised when mesh optimization fails."""


class BlendingError(Error):
    """Raised when mesh blending fails."""


class ElementQualityError(Error):
    """Raised when mesh quality check fails."""



class MenscusNodePlacementError(Error):
    """Raised when placing nodes on a slice of a meniscus mesh occurs."""


class GridError(Error):
    """Raised when our UnstructuredGrid subclass has errors."""


class CartilageMeshError(Error):
    """Raised when a CartilageMesh error occurs."""


class RaySectorError(Error):
    """Raised when raytracing fails for the sector/web pattern projections."""


class DegenerateError(Error):
    """Raised when identifying or repairing degenerate hex elements fails."""


class LimitReachedError(Error):
    """Raised when a user-passed limit is reached."""


class MesherWarning(Warning):
    """Base class for other warnings."""


class BlendingFailureWarning(MesherWarning):
    """Raised when blending algorithm returns unblended result."""


class GuiExitError(Error):
    """Raised when the GUI is closed."""
