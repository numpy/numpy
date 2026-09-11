#include "npy_fenv.h"

#ifdef __EMSCRIPTEN__

__attribute__((visibility("default"))) int npy__wasm_fenv_flags = 0;

#endif  /* __EMSCRIPTEN__ */
