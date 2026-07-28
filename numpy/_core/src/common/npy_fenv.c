#include "npy_fenv.h"

#ifdef __EMSCRIPTEN__

__attribute__((visibility("default"))) int npy_wasm_fenv_flags = 0;

#endif  /* __EMSCRIPTEN__ */
