"""A module for creating docstrings for sphinx ``data`` domains."""

import re
import textwrap

_docstrings_list = []


def add_newdoc(name: str, value: str | None, doc: str) -> None:
    """Append ``_docstrings_list`` with a docstring for `name`.

    Parameters
    ----------
    name : str
        The name of the object.
    value : str or None
        A string-representation of the object, or ``None`` to omit it.
    doc : str
        The docstring of the object.

    """
    _docstrings_list.append((name, value, doc))


def _parse_docstrings() -> str:
    """Convert all docstrings in ``_docstrings_list`` into a single
    sphinx-legible text block.

    """
    type_list_ret = []
    for name, value, doc in _docstrings_list:
        s = textwrap.dedent(doc).replace("\n", "\n    ")

        # Replace sections by rubrics
        lines = s.split("\n")
        new_lines = []
        indent = ""
        for line in lines:
            m = re.match(r'^(\s+)[-=]+\s*$', line)
            if m and new_lines:
                prev = textwrap.dedent(new_lines.pop())
                if prev == "Examples":
                    indent = ""
                    new_lines.append(f'{m.group(1)}.. rubric:: {prev}')
                else:
                    indent = 4 * " "
                    new_lines.append(f'{m.group(1)}.. admonition:: {prev}')
                new_lines.append("")
            else:
                new_lines.append(f"{indent}{line}")

        s = "\n".join(new_lines)
        s_value = "" if value is None else f"\n    :value: {value}"
        s_block = f""".. data:: {name}{s_value}\n    {s}"""
        type_list_ret.append(s_block)
    return "\n".join(type_list_ret)


add_newdoc('ArrayLike', 'typing.Union[...]',
    """
    A `~typing.Union` representing objects that can be coerced
    into an `~numpy.ndarray`.

    Among others this includes the likes of:

    * Scalars.
    * (Nested) sequences.
    * Objects implementing the `~class.__array__` protocol.

    .. versionadded:: 1.20

    See Also
    --------
    :term:`array_like`:
        Any scalar or sequence that can be interpreted as an ndarray.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import numpy.typing as npt

        >>> def as_array(a: npt.ArrayLike) -> np.ndarray:
        ...     return np.array(a)

    """)

add_newdoc('DTypeLike', 'typing.Union[...]',
    """
    A `~typing.Union` representing objects that can be coerced
    into a `~numpy.dtype`.

    Among others this includes the likes of:

    * :class:`type` objects.
    * Character codes or the names of :class:`type` objects.
    * Objects with the ``.dtype`` attribute.

    .. versionadded:: 1.20

    See Also
    --------
    :ref:`Specifying and constructing data types <arrays.dtypes.constructing>`
        A comprehensive overview of all objects that can be coerced
        into data types.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import numpy.typing as npt

        >>> def as_dtype(d: npt.DTypeLike) -> np.dtype:
        ...     return np.dtype(d)

    """)

add_newdoc('NDArray[ST: generic]', None,
    """
    A :term:`generic <generic type>` type alias for arrays with a given
    dtype and unspecified shape.

    .. code-block:: python

        type NDArray[ST: generic] = ndarray[tuple[Any, ...], dtype[ST]]

    .. versionadded:: 1.21

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import numpy.typing as npt

        >>> print(npt.NDArray)
        NDArray

        >>> print(npt.NDArray[np.float64])
        NDArray[numpy.float64]

        >>> NDArrayInt = npt.NDArray[np.int_]
        >>> a: NDArrayInt = np.arange(10)

        >>> def func(a: npt.ArrayLike) -> npt.NDArray[Any]:
        ...     return np.array(a)

    """)

add_newdoc('Array0D[ST: generic]', None,
    """
    A 0-d `NDArray` generic type alias.

    .. code-block:: python

        type Array0D[ST: generic] = ndarray[tuple[()], dtype[ST]]

    .. versionadded:: 2.6

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import numpy.typing as npt

        >>> x: npt.Array0D[np.float64] = np.array(3.14)

    """)

add_newdoc('Array1D[ST: generic]', None,
    """
    A 1-d `NDArray` generic type alias.

    .. code-block:: python

        type Array1D[ST: generic] = ndarray[tuple[int], dtype[ST]]

    .. versionadded:: 2.6

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import numpy.typing as npt

        >>> t: npt.Array1D[np.float64] = np.linspace(0, 1, 5)

    """)

add_newdoc('Array2D[ST: generic]', None,
    """
    A 2-d `NDArray` generic type alias.

    .. code-block:: python

        type Array2D[ST: generic] = ndarray[tuple[int, int], dtype[ST]]

    .. versionadded:: 2.6

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import numpy.typing as npt

        >>> m: npt.Array2D[np.float64] = np.eye(3)

    """)

add_newdoc('Array3D[ST: generic]', None,
    """
    A 3-d `NDArray` generic type alias.

    .. code-block:: python

        type Array3D[ST: generic] = ndarray[tuple[int, int, int], dtype[ST]]

    .. versionadded:: 2.6

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import numpy.typing as npt

        >>> rgb: npt.Array3D[np.uint8] = np.zeros((480, 640, 3), dtype=np.uint8)

    """)

add_newdoc('Array4D[ST: generic]', None,
    """
    A 4-d `NDArray` generic type alias.

    .. code-block:: python

        type Array4D[ST: generic] = ndarray[tuple[int, int, int, int], dtype[ST]]

    .. versionadded:: 2.6

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import numpy.typing as npt

        >>> rgb_gif: npt.Array4D[np.uint8] = np.zeros((8, 480, 640, 3), dtype=np.uint8)

    """)

_docstrings = _parse_docstrings()
