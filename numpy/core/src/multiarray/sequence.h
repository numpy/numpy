#ifndef _NPY_ARRAY_SEQUENCE_H_
#define _NPY_ARRAY_SEQUENCE_H_

#ifdef NPY_ENABLE_SEPARATE_COMPILATION
extern NPY_NO_EXPORT PySequenceMethods array_as_sequence;
#else
NPY_NO_EXPORT PySequenceMethods array_as_sequence;
#endif

#endif
