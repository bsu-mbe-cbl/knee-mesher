"""Dummy conftest.py for kneemesher.

If you don't know what this is for, just leave it empty.
Read more about conftest.py under:
- https://docs.pytest.org/en/stable/fixture.html
- https://docs.pytest.org/en/stable/writing_plugins.html
"""

from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pytest

# New module-level constants for file mappings
MESH_FILES = {
    "femur_bone": "FEM_BONE_G.inp",
    "femur_cartilage": "FEM_CART_G.inp",
    "fibula_bone": "FIB_BONE_G.inp",
    "meniscus_lateral": "MEN_LAT_G.inp",
    "meniscus_medial": "MEN_MED_G.inp",
    "patella_bone": "PAT_BONE_G.inp",
    "patella_cartilage": "PAT_CART_G.inp",
    "tibia_bone": "TIB_BONE_G.inp",
    "tibia_cartilage_lateral": "TIB_CART_LAT_G.inp",
    "tibia_cartilage_medial": "TIB_CART_MED_G.inp",
}
RECONSTRUCTION_FILES = {
    "femur_bone": "9028904_M00_femur_bone.stl",
    "femur_cartilage": "9028904_M00_femur_cartilage.stl",
    "fibula_bone": "9028904_M00_fibula_bone.stl",
    "meniscus_lateral": "9028904_M00_meniscus_lateral.stl",
    "meniscus_medial": "9028904_M00_meniscus_medial.stl",
    "patella_bone": "9028904_M00_patella_bone.stl",
    "patella_cartilage": "9028904_M00_patella_cartilage.stl",
    "tibia_bone": "9028904_M00_tibia_bone.stl",
    "tibia_cartilage_lateral": "9028904_M00_tibia_cartilage_lateral.stl",
    "tibia_cartilage_medial": "9028904_M00_tibia_cartilage_medial.stl",
}
MESH_KEYS = list(MESH_FILES.keys())
MeshPaths = namedtuple("MeshPaths", MESH_KEYS)
ReconstructionPaths = namedtuple("ReconstructionPaths", MESH_KEYS)

RESOURCES_DIR = Path(__file__).resolve().parent / "resources"
TEST_MESH_BASE_DIR = RESOURCES_DIR / "test_mesh"
TEST_RECONSTRUCTION_BASE_DIR = RESOURCES_DIR / "test_reconstruction"


@dataclass
class GeometryPaths:
    mesh: Dict[str, Path]
    reconstruction: Dict[str, Path]


@pytest.fixture(scope="session", params=["right"])
def geometry_paths(request):
    """Fixture to provide geometry data for testing."""
    selected_side = request.param
    selected_folder = f"sag_3d_dess_{selected_side}"
    TEST_MESH_DIR = TEST_MESH_BASE_DIR / selected_folder
    TEST_RECONSTRUCTION_DIR = TEST_RECONSTRUCTION_BASE_DIR / selected_folder

    # Build file paths from module-level constants
    mesh_paths = MeshPaths(*[TEST_MESH_DIR / MESH_FILES[key] for key in MESH_KEYS])
    reconstruction_paths = ReconstructionPaths(
        *[TEST_RECONSTRUCTION_DIR / RECONSTRUCTION_FILES[key] for key in MESH_KEYS]
    )

    return GeometryPaths(mesh=mesh_paths, reconstruction=reconstruction_paths)
