stringlib uses the same design idea (and "templating" mechanism) with the stringlib
implementation in CPython (https://github.com/python/cpython/tree/main/Objects/stringlib).

The implementations in the header files are included twice: once after including
stringlib.h (for bytes) and once after including ucs4lib.h (for ucs-4 unicode strings).

