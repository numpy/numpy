import asyncio
import pytest
import numpy as np
import threading
from numpy.testing import extbuild


@pytest.fixture
def get_module(tmp_path):
    """ Add a memory policy that returns a false pointer 64 bytes into the
    actual allocation, and fill the prefix with some text. Then check at each
    memory manipulation that the prefix exists, to make sure all alloc/realloc/
    free/calloc go via the functions here.
    """
    functions = [(
        "set_secret_data_policy", "METH_NOARGS",
         """
             PyDataMem_Handler *old = (PyDataMem_Handler *) PyDataMem_SetHandler(&secret_data_handler);
             return PyCapsule_New(old, NULL, NULL);
         """),
        ("set_old_policy", "METH_O",
         """
             PyDataMem_Handler *old = NULL;
             if (args != NULL && PyCapsule_CheckExact(args)) {
                 old = (PyDataMem_Handler *) PyCapsule_GetPointer(args, NULL);
             }
             PyDataMem_SetHandler(old);
             Py_RETURN_NONE;
         """),
        ]
    prologue = '''
        #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
        #include <numpy/arrayobject.h>
        /*
         * This struct allows the dynamic configuration of the allocator funcs
         * of the `secret_data_allocator`. It is provided here for
         * demonstration purposes, as a valid `ctx` use-case scenario.
         */
        typedef struct {
            void *(*malloc)(size_t);
            void *(*calloc)(size_t, size_t);
            void *(*realloc)(void *, size_t);
            void (*free)(void *);
        } SecretDataAllocatorFuncs;
        NPY_NO_EXPORT void *
        shift_alloc(void *ctx, size_t sz) {
            SecretDataAllocatorFuncs *funcs = (SecretDataAllocatorFuncs *) ctx;
            char *real = (char *)funcs->malloc(sz + 64);
            if (real == NULL) {
                return NULL;
            }
            snprintf(real, 64, "originally allocated %ld", (unsigned long)sz);
            return (void *)(real + 64);
        }
        NPY_NO_EXPORT void *
        shift_zero(void *ctx, size_t sz, size_t cnt) {
            SecretDataAllocatorFuncs *funcs = (SecretDataAllocatorFuncs *) ctx;
            char *real = (char *)funcs->calloc(sz + 64, cnt);
            if (real == NULL) {
                return NULL;
            }
            snprintf(real, 64, "originally allocated %ld via zero",
                     (unsigned long)sz);
            return (void *)(real + 64);
        }
        NPY_NO_EXPORT void
        shift_free(void *ctx, void * p, npy_uintp sz) {
            SecretDataAllocatorFuncs *funcs = (SecretDataAllocatorFuncs *) ctx;
            if (p == NULL) {
                return ;
            }
            char *real = (char *)p - 64;
            if (strncmp(real, "originally allocated", 20) != 0) {
                fprintf(stdout, "uh-oh, unmatched shift_free, "
                        "no appropriate prefix\\n");
                /* Make C runtime crash by calling free on the wrong address */
                funcs->free((char *)p + 10);
                /* funcs->free(real); */
            }
            else {
                npy_uintp i = (npy_uintp)atoi(real +20);
                if (i != sz) {
                    fprintf(stderr, "uh-oh, unmatched shift_free"
                            "(ptr, %ld) but allocated %ld\\n", sz, i);
                    /* This happens in some places, only print */
                    funcs->free(real);
                }
                else {
                    funcs->free(real);
                }
            }
        }
        NPY_NO_EXPORT void *
        shift_realloc(void *ctx, void * p, npy_uintp sz) {
            SecretDataAllocatorFuncs *funcs = (SecretDataAllocatorFuncs *) ctx;
            if (p != NULL) {
                char *real = (char *)p - 64;
                if (strncmp(real, "originally allocated", 20) != 0) {
                    fprintf(stdout, "uh-oh, unmatched shift_realloc\\n");
                    return realloc(p, sz);
                }
                return (void *)((char *)funcs->realloc(real, sz + 64) + 64);
            }
            else {
                char *real = (char *)funcs->realloc(p, sz + 64);
                if (real == NULL) {
                    return NULL;
                }
                snprintf(real, 64, "originally allocated "
                         "%ld  via realloc", (unsigned long)sz);
                return (void *)(real + 64);
            }
        }
        /* As an example, we use the standard {m|c|re}alloc/free funcs. */
        static SecretDataAllocatorFuncs secret_data_handler_ctx = {
            malloc,
            calloc,
            realloc,
            free
        };
        static PyDataMem_Handler secret_data_handler = {
            "secret_data_allocator",
            {
                &secret_data_handler_ctx, /* ctx */
                shift_alloc,              /* malloc */
                shift_zero,               /* calloc */
                shift_realloc,            /* realloc */
                shift_free                /* free */
            }
        };
        '''
    more_init = "import_array();"
    try:
        import mem_policy
        return mem_policy
    except ImportError:
        pass
    # if it does not exist, build and load it
    return extbuild.build_and_import_extension(
        'mem_policy',
        functions, prologue=prologue, include_dirs=[np.get_include()],
        build_dir=tmp_path, more_init=more_init
        )


def test_set_policy(get_module):
    orig_policy_name = np.core.multiarray.get_handler_name()

    a = np.arange(10).reshape((2, 5)) # a doesn't own its own data
    assert np.core.multiarray.get_handler_name(a) == orig_policy_name

    orig_policy = get_module.set_secret_data_policy()

    b = np.arange(10).reshape((2, 5)) # b doesn't own its own data
    assert np.core.multiarray.get_handler_name(b) == 'secret_data_allocator'

    if orig_policy_name == 'default_allocator':
        get_module.set_old_policy(None)

        assert np.core.multiarray.get_handler_name() == 'default_allocator'
    else:
        get_module.set_old_policy(orig_policy)

        assert np.core.multiarray.get_handler_name() == orig_policy_name


async def concurrent_context1(get_module, event):
    get_module.set_secret_data_policy()

    assert np.core.multiarray.get_handler_name() == 'secret_data_allocator'

    event.set()


async def concurrent_context2(get_module, orig_policy_name, event):
    await event.wait()

    assert np.core.multiarray.get_handler_name() == orig_policy_name


async def secret_data_context(get_module):
    assert np.core.multiarray.get_handler_name() == 'secret_data_allocator'

    get_module.set_old_policy(None)


async def async_test_context_locality(get_module):
    orig_policy_name = np.core.multiarray.get_handler_name()

    event = asyncio.Event()
    concurrent_task1 = asyncio.create_task(concurrent_context1(get_module, event))
    concurrent_task2 = asyncio.create_task(concurrent_context2(get_module, orig_policy_name, event))
    await concurrent_task1
    await concurrent_task2

    assert np.core.multiarray.get_handler_name() == orig_policy_name

    orig_policy = get_module.set_secret_data_policy()

    await asyncio.create_task(secret_data_context(get_module))

    assert np.core.multiarray.get_handler_name() == 'secret_data_allocator'

    get_module.set_old_policy(orig_policy)


def test_context_locality(get_module):
    asyncio.run(async_test_context_locality(get_module))


def concurrent_thread1(get_module, event):
    assert np.core.multiarray.get_handler_name() == 'default_allocator'

    get_module.set_secret_data_policy()

    assert np.core.multiarray.get_handler_name() == 'secret_data_allocator'

    event.set()


def concurrent_thread2(get_module, event):
    event.wait()

    assert np.core.multiarray.get_handler_name() == 'default_allocator'


def test_thread_locality(get_module):
    orig_policy_name = np.core.multiarray.get_handler_name()

    event = threading.Event()
    concurrent_task1 = threading.Thread(target=concurrent_thread1, args=(get_module, event))
    concurrent_task2 = threading.Thread(target=concurrent_thread2, args=(get_module, event))
    concurrent_task1.start()
    concurrent_task2.start()
    concurrent_task1.join()
    concurrent_task2.join()

    assert np.core.multiarray.get_handler_name() == orig_policy_name


@pytest.mark.slow
def test_new_policy(get_module):
    a = np.arange(10)
    orig_policy_name = np.core.multiarray.get_handler_name(a)

    orig_policy = get_module.set_secret_data_policy()

    b = np.arange(10)
    assert np.core.multiarray.get_handler_name(b) == 'secret_data_allocator'

    # test array manipulation. This is slow
    if orig_policy_name == 'default_allocator':
        # when the np.core.test tests recurse into this test, the
        # policy will be set so this "if" will be false, preventing
        # infinite recursion
        #
        # if needed, debug this by
        # - running tests with -- -s (to not capture stdout/stderr
        # - setting extra_argv=['-vv'] here
        assert np.core.test('full', verbose=2, extra_argv=['-vv'])
        # also try the ma tests, the pickling test is quite tricky
        assert np.ma.test('full', verbose=2, extra_argv=['-vv'])

    get_module.set_old_policy(orig_policy)

    c = np.arange(10)
    assert np.core.multiarray.get_handler_name(c) == orig_policy_name
