# License

## Core software

Project-authored code is licensed under the **PolyForm Noncommercial License
1.0.0**. Copyright (c) 2025 Kalin Gibbons. Third-party code and dependencies
retain their own licenses.

The license permits noncommercial purposes and specified personal uses without
anticipated commercial application. Its Noncommercial Organizations section
expressly permits use by educational institutions, public research organizations,
and the other listed organizations, regardless of funding source. The full terms
below control; this summary does not add restrictions or grants.

Commercial uses outside these permissions, including commercial corporate R&D,
product prototyping, or consulting, require a separate agreement. Contact
[Kalin Gibbons](mailto:jointmodels@protonmail.com) to discuss commercial
licensing. Any agreement covers only rights the project licensor can grant,
not rights in third-party software. PolyForm Noncommercial is source-available,
not an OSI-approved open-source license.

## Optional third-party dependencies

The optional `meshfix` extra installs **PyMeshFix**, a wrapper around MeshFix,
for surface repair and hole filling. The project does not vendor its implementation
in its source package. PyMeshFix is installed separately and imported on demand;
`repair_backend="auto"` selects it automatically when it is installed.

[PyMeshFix's license](https://github.com/pyvista/pymeshfix/blob/v0.18.1/LICENSE)
is GNU GPLv3. Its bundled MeshFix implementation has
[upstream licensing statements](https://github.com/pyvista/pymeshfix/blob/v0.18.1/README.rst#copyright)
describing GPLv3-or-later and a separate commercial licensing route involving
the authors and IMATI-GE/CNR. The upstream README also states that commercial
use requires a licensing contract. Review those statements with the upstream
rights holders; this documentation does not resolve them or grant permission.

GPLv3 itself permits commercial use. Its copyleft, source, and notice obligations
depend on modification and conveyance of covered software; it does not impose
a noncommercial-only restriction. Conditional imports and separate installation
do not establish legal separation, license compatibility, or exemption from
those obligations. A combined work subject to GPLv3 cannot simply carry
PolyForm's additional noncommercial restriction.

Users and redistributors must assess and satisfy all applicable dependency
licenses, including when they hold a separate commercial license for this
project. That agreement does not relicense PyMeshFix, MeshFix, or other
third-party components. A MeshFix agreement does not automatically cover the
PyMeshFix wrapper or its other components. Obtain any necessary permissions
from the relevant rights holders and seek legal advice for the intended
integration and distribution.

Omit the `meshfix` extra and explicitly select
`repair_backend="trimesh_open3d"` to avoid selecting MeshFix. Trimesh, Open3D,
and all other dependencies still retain their own licenses. This project does
not guarantee compliance of a particular combination or distribution. See the
[repair guide](user-guide.md) for backend behavior and limitations.

## Full project license

```{literalinclude} ../LICENSE
:language: text
```
