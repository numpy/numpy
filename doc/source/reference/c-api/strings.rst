NpyString API
=============

.. sectionauthor:: Nathan Goldbaum

.. versionadded:: 2.0

This API allows access to the UTF-8 string data stored in NumPy StringDType
arrays. See :ref:`NEP-55 <NEP55>` for
more in-depth details into the design of StringDType.

Examples
--------

Loading a String
^^^^^^^^^^^^^^^^

Say we are writing a ufunc implementation for ``StringDType``. If we are given
``const char *buf`` pointer to the beginning of a ``StringDType`` array entry, and a
``PyArray_Descr *`` pointer to the array descriptor, one can
access the underlying string data like so:

.. code-block:: C

   npy_string_allocator *allocator = NpyString_acquire_allocator(
           (PyArray_StringDTypeObject *)descr);

   npy_static_string sdata = {0, NULL};
   npy_packed_static_string *packed_string = (npy_packed_static_string *)buf;
   int is_null = 0;

   is_null = NpyString_load(allocator, packed_string, &sdata);

   if (is_null == -1) {
       // failed to load string, set error
       return -1;
   }
   else if (is_null) {
       // handle missing string
       // sdata->buf is NULL
       // sdata->size is 0
   }
   else {
       // sdata->buf is a pointer to the beginning of a string
       // sdata->size is the size of the string
   }
   NpyString_release_allocator(allocator);

Packing a String
^^^^^^^^^^^^^^^^

This example shows how to pack a new string entry into an array:

.. code-block:: C

   char *str = "Hello world";
   size_t size = 11;
   npy_packed_static_string *packed_string = (npy_packed_static_string *)buf;

   npy_string_allocator *allocator = NpyString_acquire_allocator(
           (PyArray_StringDTypeObject *)descr);

   // copy contents of str into packed_string
   if (NpyString_pack(allocator, packed_string, str, size) == -1) {
       // string packing failed, set error
       return -1;
   }

   // packed_string contains a copy of "Hello world"

   NpyString_release_allocator(allocator);

Types
-----

.. c:type:: npy_packed_static_string

    An opaque struct that represents "packed" encoded strings. Individual
    entries in array buffers are instances of this struct. Direct access
    to the data in the struct is undefined and future version of the library may
    change the packed representation of strings.

.. c:type:: npy_static_string

    An unpacked string allowing access to the UTF-8 string data.

    .. code-block:: c

      typedef struct npy_unpacked_static_string {
          size_t size;
          const char *buf;
      } npy_static_string;

    .. c:member:: size_t size

        The size of the string, in bytes.

    .. c:member:: const char *buf

        The string buffer. Holds UTF-8-encoded bytes. Does not currently end in
        a null string but we may decide to add null termination in the
        future, so do not rely on the presence or absence of null-termination.

        Note that this is a ``const`` buffer. If you want to alter an
        entry in an array, you should create a new string and pack it
        into the array entry.

.. c:type:: npy_string_allocator

    An opaque pointer to an object that handles string allocation.
    Before using the allocator, you must acquire the allocator lock and release
    the lock after you are done interacting with strings managed by the
    allocator.

.. c:type:: PyArray_StringDTypeObject

    The C struct backing instances of StringDType in Python. Attributes store
    the settings the object was created with, an instance of
    ``npy_string_allocator`` that manages string allocations for arrays
    associated with the DType instance, and several attributes caching
    information about the missing string object that is commonly needed in cast
    and ufunc loop implementations.

    .. code-block:: c

        typedef struct {
            PyArray_Descr base;
            PyObject *na_object;
            char coerce;
            char has_nan_na;
            char has_string_na;
            char array_owned;
            npy_static_string default_string;
            npy_static_string na_name;
            npy_string_allocator *allocator;
        } PyArray_StringDTypeObject;

    .. c:member:: PyArray_Descr base

        The base object. Use this member to access fields common to all
        descriptor objects.

    .. c:member:: PyObject *na_object

        A reference to the object representing the null value. If there is no
        null value (the default) this will be NULL.

    .. c:member:: char coerce

        1 if string coercion is enabled, 0 otherwise.

    .. c:member:: char has_nan_na

        1 if the missing string object (if any) is NaN-like, 0 otherwise.

    .. c:member:: char has_string_na

        1 if the missing string object (if any) is a string, 0 otherwise.

    .. c:member:: char array_owned

        1 if an array owns the StringDType instance, 0 otherwise.

    .. c:member:: npy_static_string default_string

        The default string to use in operations. If the missing string object
        is a string, this will contain the string data for the missing string.

    .. c:member:: npy_static_string na_name

        The name of the missing string object, if any. An empty string
        otherwise.

    .. c:member:: npy_string_allocator allocator

        The allocator instance associated with the array that owns this
        descriptor instance. The allocator should only be directly accessed
        after acquiring the allocator_lock and the lock should be released
        immediately after the allocator is no longer needed


Functions
---------

.. c:function:: npy_string_allocator *NpyString_acquire_allocator( \
        const PyArray_StringDTypeObject *descr)

     Acquire the mutex locking the allocator attached to
     ``descr``. ``NpyString_release_allocator`` must be called on the allocator
     returned by this function exactly once. Note that functions requiring the
     GIL should not be called while the allocator mutex is held, as doing so may
     cause deadlocks.

.. c:function:: void NpyString_acquire_allocators( \
        size_t n_descriptors, PyArray_Descr *const descrs[], \
        npy_string_allocator *allocators[])

     Simultaneously acquire the mutexes locking the allocators attached to
     multiple descriptors. Writes a pointer to the associated allocator in the
     allocators array for each StringDType descriptor in the array. If any of
     the descriptors are not StringDType instances, write NULL to the allocators
     array for that entry.

     ``n_descriptors`` is the number of descriptors in the descrs array that
     should be examined. Any descriptor after ``n_descriptors`` elements is
     ignored. A buffer overflow will happen if the ``descrs`` array does not
     contain n_descriptors elements.

     If pointers to the same descriptor are passed multiple times, only acquires
     the allocator mutex once but sets identical allocator pointers appropriately.
     The allocator mutexes must be released after this function returns, see
     ``NpyString_release_allocators``.

     Note that functions requiring the GIL should not be called while the
     allocator mutex is held, as doing so may cause deadlocks.

.. c:function:: void NpyString_release_allocator( \
        npy_string_allocator *allocator)

     Release the mutex locking an allocator. This must be called exactly once
     after acquiring the allocator mutex and all operations requiring the
     allocator are done.

     If you need to release multiple allocators, see
     NpyString_release_allocators, which can correctly handle releasing the
     allocator once when given several references to the same allocator.

.. c:function:: void NpyString_release_allocators( \
        size_t length, npy_string_allocator *allocators[])

     Release the mutexes locking N allocators. ``length`` is the length of the
     allocators array. NULL entries are ignored.

     If pointers to the same allocator are passed multiple times, only releases
     the allocator mutex once.

.. c:function:: int NpyString_load(npy_string_allocator *allocator, \
               const npy_packed_static_string *packed_string, \
               npy_static_string *unpacked_string)

     Extract the packed contents of ``packed_string`` into ``unpacked_string``.

     The ``unpacked_string`` is a read-only view onto the ``packed_string`` data
     and should not be used to modify the string data. If ``packed_string`` is
     the null string, sets ``unpacked_string.buf`` to the NULL
     pointer. Returns -1 if unpacking the string fails, returns 1 if
     ``packed_string`` is the null string, and returns 0 otherwise.

     A useful pattern is to define a stack-allocated npy_static_string instance
     initialized to ``{0, NULL}`` and pass a pointer to the stack-allocated
     unpacked string to this function.  This function can be used to
     simultaneously unpack a string and determine if it is a null string.

.. c:function:: int NpyString_pack_null( \
        npy_string_allocator *allocator, \
        npy_packed_static_string *packed_string)

   Pack the null string into ``packed_string``. Returns 0 on success and -1 on
   failure.

.. c:function:: int NpyString_pack( \
        npy_string_allocator *allocator, \
        npy_packed_static_string *packed_string, \
        const char *buf, \
        size_t size)

   Copy and pack the first ``size`` entries of the buffer pointed to by ``buf``
   into the ``packed_string``. Returns 0 on success and -1 on failure.
