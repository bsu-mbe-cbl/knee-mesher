"""Provide functions for reading and writing mesh data in the Abaqus INP format.

This module provides functions for reading and writing mesh data in the Abaqus INP
format.  It includes a regular expression manager class for parsing input files, and a
function for writing mesh or unstructured grid to an Abaqus input file.

Functions:
- write_inp_file: Write mesh or unstructured grid to abaqus input file.

Classes:
- _InpRegexManager: A class that manages regular expressions used for parsing input
    files.

Constants:
- KEYWORDS: A list of keywords used in the Abaqus INP format.

"""

from __future__ import annotations

import re
from io import StringIO
from pathlib import Path
from typing import IO, Dict, List, Optional, Tuple, Union

import meshio
import numpy as np
import pyvista as pv
import vtk
from numpy.typing import ArrayLike

from kneemesher.meshing.cells import HexMesh

KEYWORDS = ["heading", "node", "element", "nset", "elset"]


def _get_node_format(n_values: int, indents: int = 0) -> str:
    """Return a string format for a node with the given number of values and indents.

    Args:
        n_values (int): The number of values in the node.
        indents (int, optional): The number of indents to include in the format.
            Defaults to 0.

    Returns:
        str: The string format for a node with the given number of values and indents.
    """
    n_dims = n_values - 1
    indent_str = "".join(["    "] * indents)
    num_str = ",".join(("%9d", *([" %- 16.14g"] * n_dims)))
    return "".join((indent_str, num_str))


def _get_integer_line_format(
    n_values,
    indents: int = 0,
    needs_continuation: bool = False,
) -> str:
    """Return a string format for a line of integers.

    Args:
        n_values (int): The number of integer values in the line.
        indents (int, optional): The number of indents to add before the line. Defaults
            to 0.
        needs_continuation (bool, optional): Whether the line needs to be continued on
            the next line. Defaults to False.

    Returns:
        str: A string format for a line of integers.
    """
    indent_str = "".join(["    "] * indents)
    num_str = ",".join((["%9d"] * n_values))
    fmt_str = "".join((indent_str, num_str))
    return ",".join((fmt_str, "")) if needs_continuation else fmt_str


class _InpRegexManager:
    """A class that manages regular expressions used for parsing input files.

    Attributes:
        regex_dict (dict): A dictionary containing regular expressions for various
            components of the input file format.
    """

    def __init__(self) -> None:
        self.regex_dict = dict(
            keyline=r"(?P<keyline>[ ]*\*\w+(?:,.+|)\n)",
            blocks=r"\*\w+(?:,.+\n|\n)",
            keyword=r"\*(?P<keyword>\w+)",
            comment=r"(?P<comment>\*{2}.*\n)",
            continuation=r"(?P<continuation>(?:.+,\n)+.+)(?:\n|)",
        )

    def __len__(self):
        return len(self.regex_dict)

    def __getitem__(self, keys):
        if isinstance(keys, tuple):
            regex = r"|".join([self.regex_dict[k] for k in keys])
        else:
            regex = self.regex_dict[keys]
        return regex

    def __setitem__(self, key, value):
        self.regex_dict[key] = value

    def __contains__(self, name):
        return name in self.regex_dict


inp_regex_manager = _InpRegexManager()


def write_inp_file(
    hex_mesh: Union[HexMesh, pv.UnstructuredGrid],
    filename: Union[str, Path],
    hexahedron_type: Optional[str] = None,
    triangle_type: Optional[str] = None,
    node_num_begins: int = 1,
    elem_num_begins: int = 1,
    elem_sets: Optional[Dict[str, ArrayLike]] = None,
    node_sets: Optional[Dict[str, ArrayLike]] = None,
    reflect_normal: Optional[Tuple[Union[int, float]]] = None,
    reflect_point: Optional[Tuple[Union[int, float]]] = None,
    **write_point_cells_kwargs,
) -> None:
    """Write mesh or unstructured grid to abaqus input file.

    Input file text is initially written to memory using the `meshio` library, then
    processed to have consistent numeric fieldwidths and optionally to offset the
    initial node and element IDs.

    Args:
        hex_mesh (Union[HexMesh, pv.UnstructuredGrid]): The mesh to be exported to
            abaqus.
        filename (Union[str, Path]): The INP filename to be exported.
        hexahedron_type (str, optional): The abaqus element code for hexahedrons.
            Defaults to C3D8R.
        node_num_begins (int, optional): What node number to begin the file with.
            Defaults to 1.
        elem_num_begins (int, optional): The number to begin sequential element
            numbering. Defaults to 1.
        elem_set (str, optional): The element set name that elements can be assigned.
            Defaults to None.
        node_set (str, optional): The node set name for nodes to be assigned to.
            Defaults to None.
        reflect_normal (tuple, optional): a 3-component tuple representing the normal
            vector of the reflection plane. Defaults to None.
        reflect_point (tuple, optioanl): a 3-component tuple representing a point on the
            reflection plane. Defaults to None.
        write_point_cells_kwargs: See `meshio.write_point_cells` or 'meshio.Mesh.write'
            for additional keyword arguments.
    """
    if isinstance(hex_mesh, HexMesh):
        hex_mesh = hex_mesh.mesh

    if reflect_normal is not None and reflect_point is not None:
        hex_mesh = hex_mesh.reflect(reflect_normal, point=reflect_point)
        reflected = True
    else:
        reflected = False

    filename = Path(filename)

    points = hex_mesh.points
    if isinstance(hex_mesh, pv.PolyData):
        cells = {vtk.VTK_TRIANGLE: hex_mesh.faces.reshape(-1, 4)[:, 1:]}
    else:
        cells = hex_mesh.cells_dict

    if hexahedron_type:
        cells[hexahedron_type] = cells.pop(vtk.VTK_HEXAHEDRON)
    else:
        hexahedron_type = "C3D8R"
        cells[hexahedron_type] = cells.pop(vtk.VTK_HEXAHEDRON, None)
        cells.pop(hexahedron_type) if cells[hexahedron_type] is None else None

    if triangle_type:
        cells[triangle_type] = cells.pop(vtk.VTK_TRIANGLE)
    else:
        triangle_type = "R3D3"
        cells[triangle_type] = cells.pop(vtk.VTK_TRIANGLE, None)
        cells.pop(triangle_type) if cells[triangle_type] is None else None

    if reflected:
        hex_elems = cells.get(hexahedron_type)
        tri_elems = cells.get(triangle_type)
        if hex_elems is not None:
            hex_elems[:, 0:4] = np.fliplr(hex_elems[:, 0:4])
            hex_elems[:, 4:] = np.fliplr(hex_elems[:, 4:])
            cells[hexahedron_type] = hex_elems
        if tri_elems is not None:
            cells[triangle_type] = np.fliplr(tri_elems)

    translate_cell_names = False
    if "translate_cell_names" not in write_point_cells_kwargs.keys():
        write_point_cells_kwargs["translate_cell_names"] = translate_cell_names


    # Fix for meshio 5.1.0 changes
    if hexahedron_type in cells:
        cells["hexahedron"] = cells.pop(hexahedron_type)
    if triangle_type in cells:
        cells["triangle"] = cells.pop(triangle_type)

    if elem_sets is not None:
        elem_sets = {
            k: [
                v,
            ]
            for k, v in elem_sets.items()
        }
    msh = meshio.Mesh(points, cells, cell_sets=elem_sets, point_sets=node_sets)
    for idx, cell_block in enumerate(msh.cells):
        if "triangle" in cell_block.type:
            msh.cells[idx].type = triangle_type
        if "hexahedron" in cell_block.type:
            msh.cells[idx].type = hexahedron_type

    with StringIO() as mem_file:
        msh.write(mem_file, **write_point_cells_kwargs, file_format="abaqus")
        txt = mem_file.getvalue()

    with open(filename, "w") as save_file:
        _prettify_meshio_output(
            save_file,
            txt,
            node_num_begins=node_num_begins,
            elem_num_begins=elem_num_begins,
        )


def _prettify_meshio_output(
    filename: Union[str, Path, IO],
    meshio_text: str,
    node_num_begins: int = 1,
    elem_num_begins: int = 1,
) -> None:
    """Prettify meshio output and write them to a file or IO object.

    Args:
        filename (Union[str, Path, IO]): The name of the file to write to, or an IO
            object.
        meshio_text (str): The meshio output text to prettify.
        node_num_begins (int, optional): The starting index for node numbers. Defaults
            to 1.
        elem_num_begins (int, optional): The starting index for element numbers.
            Defaults to 1.
    """
    # This could probably be handled using non-greedy matching with groups.
    keylines = re.findall(inp_regex_manager["keyline"], meshio_text)
    blocks = re.split(inp_regex_manager["blocks"], meshio_text)
    if isinstance(filename, str) or isinstance(filename, Path):
        with open(filename, "w") as save_file:
            _process_inp_keys(
                save_file,
                keylines,
                blocks,
                node_num_begins=node_num_begins,
                elem_num_begins=elem_num_begins,
            )
    else:
        _process_inp_keys(
            filename,
            keylines,
            blocks,
            node_num_begins=node_num_begins,
            elem_num_begins=elem_num_begins,
        )


def _process_inp_keys(
    file_obj: IO,
    keylines: List[str],
    blocks: List[str],
    node_num_begins: int = 1,
    elem_num_begins: int = 1,
) -> None:
    """Process the keywords and blocks in an input file and write to a file object.

    Args:
        file_obj (IO): The file object to write the processed content to.
        keylines (List[str]): A list of strings representing the lines containing
            keywords in the input file.
        blocks (List[str]): A list of strings representing the blocks of content
            associated with each keyword in the input file.
        node_num_begins (int, optional): The starting node number. Defaults to 1.
        elem_num_begins (int, optional): The starting element number. Defaults to 1.

    Returns:
        None
    """
    current_indent = 0
    indent_keywords = []  # ["nset", "elset"]
    file_obj.write(blocks[0])
    for keyline, block in zip(keylines, blocks[1:]):
        file_obj.write(keyline)
        match = re.match(inp_regex_manager["keyword"], keyline)
        keyword = match["keyword"].lower()
        if keyword in indent_keywords:
            current_indent += 1

            # split on comments
        sub_blocks = re.split(inp_regex_manager["comment"], block)
        for sb in sub_blocks:
            if "**" in sb:
                file_obj.write("".join(current_indent * ["    "]))
                file_obj.write(sb)
            elif "" == sb:  # extra sub-blocks when capturing comments in groups
                file_obj.write(sb)
            elif keyword == "node":
                file_obj.write(
                    _process_node_definition(
                        sb, indent=current_indent, node_ids_begin=node_num_begins
                    )
                )
            elif keyword == "element":
                file_obj.write(
                    _process_elem_definition(
                        sb, indent=current_indent, elem_ids_begin=elem_num_begins, node_ids_begin=node_num_begins
                    )
                )
            elif keyword == "nset":
                file_obj.write(
                    _process_continuation(
                        sb,
                        indent=current_indent,
                        integer_offset=node_num_begins - 1,
                    )
                )
            elif keyword == "elset":
                file_obj.write(
                    _process_continuation(
                        sb,
                        indent=current_indent,
                        integer_offset=elem_num_begins - 1,
                    )
                )
            else:
                file_obj.write(sb)
        if keyword in indent_keywords:
            current_indent -= 1


def _process_node_definition(
    node_def_string: str,
    node_ids_begin: int = 1,
    indent: int = 0,
) -> str:
    """Process a node definition string and return a formatted string of node IDs.

    Args:
        node_def_string (str): A string containing node IDs separated by commas.
        node_ids_begin (int, optional): The starting ID number for the nodes. Defaults
            to 1.
        indent (int, optional): The number of spaces to indent the formatted string.
            Defaults to 0.

    Returns:
        str: A formatted string of node IDs.

    """
    with StringIO(node_def_string) as ndef, StringIO() as save_buffer:
        node_ids = np.atleast_2d(np.loadtxt(ndef, delimiter=","))
        node_ids[:, 0] += node_ids_begin - 1
        n_values = node_ids.shape[1]
        
        np.savetxt(
            save_buffer,
            node_ids,
            fmt=_get_node_format(n_values, indents=indent),
        )
        return save_buffer.getvalue()


def _process_elem_definition(
    elem_def_str: str,
    elem_ids_begin: int = 1,
    node_ids_begin: int = 1,
    indent: int = 0,
) -> str:
    """Process the element definition string and return a string of element IDs.

    Args:
        elem_def_str (str): The element definition string.
        elem_ids_begin (int, optional): The starting ID for the element IDs. Defaults to
            1.
        indent (int, optional): The number of spaces to indent the output string.
            Defaults to 0.

    Returns:
        str: A string of element IDs.
    """
    with StringIO(elem_def_str) as elem_def, StringIO() as save_buffer:
        elem_ids = np.atleast_2d(np.loadtxt(elem_def, delimiter=","))
        elem_ids[:, 0] += elem_ids_begin - 1
        elem_ids[:, 1:] += node_ids_begin - 1
        n_values = elem_ids.shape[1]
        np.savetxt(
            save_buffer,
            elem_ids,
            fmt=_get_integer_line_format(n_values, indents=indent),
        )
        return save_buffer.getvalue()


def _process_continuation(
    continue_string: str,
    integer_offset=0,
    indent: int = 0,
) -> str:
    """Process a continuation string.

    Args:
        continue_string (str): The continuation string to process.
        integer_offset (int, optional): The integer offset to add to the continuation
            values. Defaults to 0.
        indent (int, optional): The number of spaces to indent the continuation values.
            Defaults to 0.

    Returns:
        str: The processed continuation string.
    """
    with StringIO(continue_string) as continue_file, StringIO() as save_buffer:
        try:  # Read blocks with numpy
            continuation = np.atleast_2d(
                np.genfromtxt(continue_file, delimiter=",", skip_footer=1)
            )[:, :-1]
            n_values = continuation.shape[1]

            if integer_offset != 0:
                continuation += integer_offset
            np.savetxt(
                save_buffer,
                continuation,
                fmt=_get_integer_line_format(
                    n_values, needs_continuation=True, indents=indent
                ),
            )
            continue_file.seek(0)
            continuation = np.atleast_2d(
                np.genfromtxt(
                    continue_file,
                    delimiter=",",
                    skip_header=continuation.shape[0],
                )
            )
            n_values = continuation.shape[1]

            if integer_offset != 0:
                continuation += integer_offset
            np.savetxt(
                save_buffer,
                np.atleast_2d(continuation[-1]),
                fmt=_get_integer_line_format(n_values, indents=indent),
            )
        except None:  # process line-by-line
            continue_file.seek(0)
            raise NotImplementedError("Need to process continuation line-by-line.")
            for line in continue_file.getlines():
                pass
        return save_buffer.getvalue()
