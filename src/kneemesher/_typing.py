from __future__ import annotations

import sys
from collections.abc import (
    Hashable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
from datetime import (
    date,
    datetime,
    timedelta,
    tzinfo,
)
from os import PathLike
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    overload,
)
from typing import (
    Type as type_t,
)

import numpy.typing as npt
from pyvista.core import pyvista_ndarray

# PyVista
from pyvista.core.composite import MultiBlock
from pyvista.core.grid import Grid, ImageData, RectilinearGrid
from pyvista.core.pointset import (
    ExplicitStructuredGrid,
    PointGrid,
    PointSet,
    PolyData,
    StructuredGrid,
    UnstructuredGrid,
)
from trimesh import Trimesh

# To prevent import cycles place any internal imports in the branch below
# and use a string literal forward reference to it in subsequent types
# https://mypy.readthedocs.io/en/latest/common_issues.html#import-cycles
if TYPE_CHECKING:
    from typing import SupportsIndex

    # Numpy

    ArrayLike = npt.ArrayLike

    # Kneemesher
    from kneemesher.core_api import (
        CartilageMesh,
        GridCartilageMesh,
        MeniscusMesh,
        SectorCartilageMesh,
        TissueMesh,
    )

    if sys.version_info >= (3, 10):
        from typing import TypeGuard  # pyright: ignore[reportUnusedImport]
    else:
        from typing_extensions import TypeGuard  # pyright: ignore[reportUnusedImport]

    if sys.version_info >= (3, 11):
        from typing import Self  # pyright: ignore[reportUnusedImport]
    else:
        from typing_extensions import Self  # pyright: ignore[reportUnusedImport]
else:
    Self: Any = None
    TypeGuard: Any = None

# array-like

Mesh = Union[PolyData, UnstructuredGrid, StructuredGrid, RectilinearGrid]
TriangleMesh = Union[PolyData, UnstructuredGrid, Trimesh]
HexahedralMesh = Union[StructuredGrid, RectilinearGrid, UnstructuredGrid]
QuadMesh = Union[PolyData, UnstructuredGrid]
SurfaceMesh = Union[TriangleMesh, QuadMesh]
VolumeMesh = HexahedralMesh
PyvistaArray = pyvista_ndarray
ArrayLike = npt.ArrayLike
AnyArrayLike = Union[pyvista_ndarray, ArrayLike]

# list-like

# from https://github.com/hauntsaninja/useful_types
# includes Sequence-like objects but excludes str and bytes
