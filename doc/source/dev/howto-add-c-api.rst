.. _adding-c-api:

*****************************
Adding or extending the C API
*****************************

What this guide helps you do
============================

This guide explains how to add or extend NumPy's public C API.

It is intended for NumPy contributors who are implementing or modifying
functionality in the C API, including the classic array API, ufunc API, and
DType and ArrayMethod APIs.

By following this guide, you will be able to:

* decide whether functionality should be part of the public C API;
* choose the appropriate C API mechanism;
* implement and expose a new API;
* preserve API and ABI compatibility;
* update C API versioning and generated metadata;
* document and test the new API; and
* prepare the change for review.


When a public C API is appropriate
==================================

A public C API should be added when downstream C or C++ extensions need a
stable, supported way to access functionality that cannot reasonably be
implemented using the existing public API.

Before adding a new entry, check whether an existing public API already
provides the required functionality.

Consider keeping functionality internal when:

* it is only an implementation detail;
* its design is still expected to change;
* downstream extensions do not need direct access to it; or
* exposing it would unnecessarily constrain future implementation changes.

Public API additions create compatibility obligations. Once exposed, an API
must be maintained according to NumPy's API and ABI compatibility guarantees.

For experimental functionality, it can be preferable to stabilize the
implementation and usage pattern internally before exposing a public API.


Choose the API mechanism
========================

NumPy currently has several mechanisms for exposing C-level functionality.
Determine which mechanism applies before implementing the change.


Classic MultiArray C API
------------------------

The classic array C API exposes functions through the ``PyArray_API``
function-pointer table.

Functions are assigned stable indices in
``numpy/_core/code_generators/numpy_api.py``.

For example::

    'PyArray_Sort':         (129,),
    'PyArray_ArgSort':      (130,),
    'PyArray_SearchSorted': (131,),

The index of an existing API entry must never be changed because it would
change the layout of the API table and break ABI compatibility.

A new API entry normally uses the next available index, subject to the
reserved and unused ranges maintained by the API generator.


UFunc C API
-----------

The ufunc API has a separate API table and uses the same general compatibility
model.

For example, ``PyUFunc_FromFuncAndDataAndSignatureAndIdentity`` is exposed
through the ufunc API table::

    'PyUFunc_FromFuncAndDataAndSignatureAndIdentity': (42, MinVersion("1.16")),

The public declaration and implementation are combined with the API metadata
by the code-generation machinery.


DType and ArrayMethod APIs
--------------------------

The DType and ArrayMethod APIs provide extensible mechanisms for defining
DType behavior and implementing operations for specific DTypes.

For example, an ArrayMethod specification uses public structures and slot
identifiers such as::

    PyArrayMethod_Spec
    NPY_METH_resolve_descriptors
    NPY_METH_get_loop
    NPY_METH_strided_loop

A conceptual specification can look like::

    PyType_Slot slots[] = {
        {NPY_METH_resolve_descriptors, resolve_descriptors},
        {NPY_METH_get_loop, get_loop},
        {0, NULL},
    };

    PyArrayMethod_Spec spec = {
        .name = "example_method",
        .nin = 1,
        .nout = 1,
        .dtypes = dtypes,
        .slots = slots,
    };

The ArrayMethod implementation is selected through the DType machinery.
Public DType objects use the public DType API and indirection through
``PyArray_DTypeMeta`` and its implementation slots.

Internal implementation helpers that operate on DType objects are not
necessarily public C API merely because they take a public struct as an
argument. Always check the declaration, export marker, and documentation
before treating an internal helper as part of the public API.


Header-only and internal functionality
--------------------------------------

Not every C-level addition requires an entry in an API table.

Header-only functionality, such as a new macro, may not require a function
pointer in ``PyArray_API``. Internal implementation helpers should generally
remain unexported unless there is a specific reason to make them public.

The naming of a declaration alone does not determine whether it is public.
Check how the functionality is declared, exported, documented, versioned,
and registered before treating it as public.


Design the compatibility contract
=================================

Before implementing the API, determine what compatibility guarantees the new
functionality requires.


API compatibility
------------------

NumPy's public C API is versioned independently from its ABI.

Adding a new public API can therefore change the C API without breaking the
ABI of existing extensions.

The C API version is maintained in:

``numpy/_core/meson.build``

The API version must be increased when the public C API changes.

For example, find the existing ``C_API_VERSION`` assignment and increment its
value for the API change::

    C_API_VERSION = '0x00000012'

becomes::

    C_API_VERSION = '0x00000013'

Use the actual current value from the file; do not copy a version number from
an older change.


Minimum API version
-------------------

If a function is only available starting with a particular NumPy version,
use the ``MinVersion`` annotation in the API table.

For example::

    'PyDataMem_SetHandler': (304, MinVersion("1.22")),

This allows generated headers to expose the API only when the requested
feature/API version is sufficiently recent.


Feature-version guards
----------------------

The public headers use ``NPY_FEATURE_VERSION`` to determine which API
features are available to code being compiled.

For example, a function introduced in a later NumPy release can be guarded
using the minimum version recorded in the API table.

Keep in mind that the NumPy version installed at runtime and the API version
against which an extension is compiled are not necessarily the same.

In particular, ``NPY_TARGET_VERSION`` allows downstream extensions to target
a specific NumPy C API version rather than automatically targeting the newest
API available from the headers.

This distinction is important when testing compatibility across NumPy
versions.


ABI compatibility
-----------------

The NumPy C API is designed to remain ABI compatible across compatible
releases.

Do not change the position of an existing function in an API table.

When an existing ABI-visible structure needs to be extended, preserve the
layout expected by older binaries. New fields may need feature-version
guards.

For example, ``PyUnicodeScalarObject`` has a field guarded by an API-version
check because the field was added while preserving the existing layout::

    #if NPY_FEATURE_VERSION >= NPY_1_20_API_VERSION
        char *buffer_fmt;
    #endif

If a change is not ABI compatible, it requires the appropriate ABI-version
handling and must be reviewed as an ABI-breaking change.

See also :ref:`reviewer-guidelines` for the compatibility requirements used
during API review.


Implement and register the API
==============================

Once the compatibility contract is understood, implement the functionality
and expose it through the appropriate API mechanism.


Classic API workflow
--------------------

For a classic MultiArray C API function:

#. Implement the function in the appropriate NumPy source file.
#. Mark the implementation with the appropriate ``NUMPY_API`` declaration
   marker when required by the API generator.
#. Add the public function to
   ``numpy/_core/code_generators/numpy_api.py``.
#. Assign a new API-table index without changing existing indices.
#. Add ``MinVersion`` if the API is introduced after the oldest supported
   feature version.
#. Update the C API version in ``numpy/_core/meson.build``.
#. Regenerate or verify the generated API files.
#. Add documentation and tests.

The code-generation machinery uses the declaration markers and the API
dictionary to generate entries in the public API table.

A generated declaration has the general form::

    #define PyArray_Foo \
        (*(return_type (*)(argument_types *))PyArray_API[N])

The generated implementation table contains the corresponding function
pointer at the same index.

Never manually change generated API-table positions to make a new function
fit. The index assignment belongs in the API metadata.


UFunc API workflow
------------------

For a ufunc API addition:

#. Implement the functionality in the appropriate ufunc source files.
#. Add the required public declaration and API marker.
#. Add the function to the ufunc API metadata in
   ``numpy/_core/code_generators/numpy_api.py``.
#. Assign a new ufunc API-table index.
#. Add ``MinVersion`` when appropriate.
#. Update the C API version.
#. Regenerate or verify the generated API files.
#. Add C API reference documentation.
#. Add tests covering the public behavior and compatibility requirements.

The public declaration and implementation are combined with the API metadata
by the code-generation machinery.


ArrayMethod workflow
--------------------

For a change involving the DType and ArrayMethod APIs:

#. Identify the public structure, slot, callback, or API function being
   extended.
#. Confirm that the functionality is intended to be public rather than
   internal implementation machinery.
#. Preserve the existing public structure layout.
#. Add new functionality through the appropriate slot or indirection
   mechanism.
#. Consider how older NumPy versions will interact with the new fields or
   slots.
#. Add the required API-table exposure when the functionality is exposed
   through the classic C API.
#. Add documentation and tests.

The public DType API deliberately uses indirection so that implementation
details can evolve without continually exposing new fields in stable public
structures.

For example, ``PyArray_DTypeMeta`` contains a ``dt_slots`` pointer that leads
to DType-specific implementation slots.

Similarly, ArrayMethod behavior is described through
``PyArrayMethod_Spec`` and its slots rather than by exposing implementation
details directly in the public DType object.


Update compatibility bookkeeping
================================

Changes to the public API must be reflected in NumPy's compatibility
metadata.


API version
-----------

Update the C API version in:

``numpy/_core/meson.build``

The C API version changes whenever the public C API changes.

For example, find the existing ``C_API_VERSION`` assignment and increment its
value for the API change::

    C_API_VERSION = '0x00000012'

becomes::

    C_API_VERSION = '0x00000013'

Use the actual current value from the file; do not copy a version number from
an older change.

This is distinct from the ABI version. A new API-table entry can be a C API
change without being an ABI-breaking change.


API hash and ``cversions.txt``
------------------------------

NumPy verifies the generated C API against historical API hashes stored in
``cversions.txt``.

If the API dictionary changes, run::

    python numpy/_core/code_generators/cversions.py

Use the resulting information to update the appropriate version/hash
bookkeeping.

Do not edit ``cversions.txt`` merely to silence a verification failure.
First determine why the API hash changed and confirm that the API change is
intentional.

If an API entry was added, removed, or otherwise changed, the corresponding
C API version and generated metadata must remain consistent.


Generated files
---------------

The API-generation machinery produces generated headers and source files,
including the public API table.

Do not make manual edits to generated files when the source metadata or
generator is the correct place to make the change.

After modifying the API metadata, run the relevant generation and verification
steps and inspect the generated diff.


Document the API
================

A public API needs documentation describing its supported contract.

At minimum, document:

* the function, type, macro, or slot;
* its arguments and return value;
* ownership and reference-counting requirements;
* error behavior;
* version information;
* compatibility requirements; and
* any restrictions on how the API may be used.

Use the C API reference documentation for the public API itself.

For example, the classic C API reference is organized under:

``doc/source/reference/c-api/``

The contributor workflow and compatibility requirements belong in the
Developer Guide, including this document.


Version-added information
-------------------------

When introducing a new public API, include the appropriate ``versionadded``
information in the API reference documentation.

For example::

    .. versionadded:: 2.4

Use the NumPy version in which the public API becomes available, not the
version in which an internal implementation first appeared.


Test the API
============

Test both the implementation and the public compatibility contract.

At minimum:

#. Test the new functionality directly.
#. Test error paths and reference-counting behavior where applicable.
#. Test the public header/API exposure.
#. Verify the generated API metadata.
#. Run the relevant C and Python test suites.
#. Run documentation and lint checks for documentation changes.
#. Where compatibility matters, test an extension built against an older
   supported API target.

When testing C extensions, remember that ``NPY_TARGET_VERSION`` controls the
API version an extension targets when compiling against newer NumPy headers.
This is useful for testing that a new NumPy runtime remains compatible with
extensions targeting older API versions.

For API changes involving DTypes or ArrayMethods, include tests that exercise
the public path rather than testing only private implementation functions.


Naming conventions
==================

Follow the established NumPy C naming conventions.

Common public prefixes include:

``PyArray_*``
    Public array C API functions.

``PyUFunc_*``
    Public ufunc C API functions.

``NPY_*``
    Public NumPy C macros, constants, flags, and related identifiers.

``npy_*``
    Public or semi-public low-level NumPy C utilities, depending on the
    specific API.

An underscore in a name does not by itself determine whether an API is
private. For example, some underscored functions are intentionally exposed
through the generated API tables.

Always verify whether the specific declaration is part of the supported
public API before using it as a model for a new API.


Review the C style guide
========================

C and C++ implementation changes should follow NumPy's C coding conventions.

See the
`NumPy C style guide <https://numpy.org/neps/nep-0045-c_style_guide.html>`_
for the detailed conventions.


A complete workflow
===================

For a typical new public C API, use the following sequence:

#. Determine whether a public API is necessary.
#. Search the existing C API for equivalent functionality.
#. Identify whether the change belongs to the classic MultiArray API, ufunc
   API, DType/ArrayMethod API, or another public mechanism.
#. Design the API so that future implementation changes remain possible.
#. Implement the functionality.
#. Add the public API declaration and registration.
#. Assign a new API-table index when required.
#. Add ``MinVersion`` when the API requires a minimum feature version.
#. Update the C API version in ``numpy/_core/meson.build``.
#. Regenerate and verify the generated API files.
#. Run ``python numpy/_core/code_generators/cversions.py`` when API hash
   bookkeeping needs to be updated.
#. Add C API reference documentation.
#. Add ``versionadded`` information.
#. Add tests for the implementation and public API.
#. Test compatibility with appropriate ``NPY_TARGET_VERSION`` settings.
#. Update release notes when required.
#. Review the complete generated diff.
#. Check the change against the reviewer guidelines before opening the PR.


Historical examples
===================

Looking at previous API additions is useful when designing a new API.


``PyUFunc_AddLoopsFromSpecs``
-----------------------------

The addition of ``PyUFunc_AddLoopsFromSpecs`` demonstrates a complete public
API addition involving the ufunc and ArrayMethod infrastructure.

The change involved:

* implementation work;
* public API registration;
* API version metadata;
* minimum API-version handling;
* C API hash/version bookkeeping;
* documentation;
* tests; and
* integration with the ArrayMethod machinery.

When using historical changes as examples, check the corresponding revision
of ``numpy/_core/code_generators/numpy_api.py`` rather than assuming that an
API-table index remains unchanged across NumPy versions.


DType API indirection
---------------------

The DType API demonstrates another important design pattern.

Public DType objects use stable outer structures and indirection through
implementation slots. This allows additional behavior to be introduced
without exposing every internal implementation detail as part of the stable
public structure layout.

When adding new DType functionality, prefer the established slot and
indirection mechanisms rather than exposing internal structures directly.


Public API versus implementation
---------------------------------

NumPy has also had changes where an implementation existed before the
corresponding functionality was exposed as public C API.

This distinction is important: implementing a function and committing to it
as public API are separate decisions.

Before exposing an internal helper, verify that downstream users actually
need the API and that NumPy can support its compatibility contract over time.


Checklist
=========

Before opening a PR that adds or changes the public C API, verify:

* [ ] An existing public API cannot provide the required functionality.
* [ ] The functionality genuinely needs to be public.
* [ ] The appropriate API mechanism has been selected.
* [ ] Existing API-table indices have not changed.
* [ ] A new API-table index is used where required.
* [ ] ``MinVersion`` is present when required.
* [ ] The C API version has been updated when required.
* [ ] ABI compatibility has been considered.
* [ ] Public structure layouts remain compatible.
* [ ] Generated API files have been regenerated or verified.
* [ ] ``cversions.py`` has been run when API hash bookkeeping changed.
* [ ] The API is documented in the C API reference.
* [ ] ``versionadded`` information is present.
* [ ] Tests cover the public API.
* [ ] Compatibility with ``NPY_TARGET_VERSION`` has been considered where
      relevant.
* [ ] Relevant release notes have been updated.
* [ ] The NumPy C style guide has been followed.
* [ ] The reviewer guidelines have been checked.


.. seealso::

   :ref:`c-api`
       NumPy C-API reference documentation.

   :ref:`reviewer-guidelines`
       Reviewer guidelines, including the API changes section.

   `NumPy C style guide <https://numpy.org/neps/nep-0045-c_style_guide.html>`_
       C coding conventions for NumPy.

   `numpy-user-dtypes <https://github.com/numpy/numpy-user-dtypes>`_
       Example user-defined dtypes consuming the public DType API.