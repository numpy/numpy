_cython_hash = [
  fs.copyfile('khash.pxd'),
  fs.copyfile('hashtable.pxd'),
]

tempita = files('generate_pxi.py')

_khash_primitive_helper = custom_target('khash_primitive_helper_pxi',
    output: 'khash_for_primitive_helper.pxi',
    input: 'khash_for_primitive_helper.pxi.in',
    command: [
        py, tempita, '@INPUT@', '-o', '@OUTDIR@'
    ]
)
_hashtable_class_helper = custom_target('hashtable_class_helper_pxi',
    output: 'hashtable_class_helper.pxi',
    input: 'hashtable_class_helper.pxi.in',
    command: [
        py, tempita, '@INPUT@', '-o', '@OUTDIR@'
    ]
)
_hashtable_func_helper = custom_target('hashtable_func_helper_pxi',
    output: 'hashtable_func_helper.pxi',
    input: 'hashtable_func_helper.pxi.in',
    command: [
        py, tempita, '@INPUT@', '-o', '@OUTDIR@'
    ]
)
_khash_primitive_helper_dep = declare_dependency(sources: _khash_primitive_helper)


libs_sources = {
    # Dict of extension name -> dict of {sources, include_dirs, and deps}
    # numpy include dir is implicitly included
    'hashing': {'sources': ['hashing.pyx']},
    'hashtable': {
        'sources': ['hashtable.pyx', _hashtable_class_helper, _hashtable_func_helper],
        'deps': _khash_primitive_helper_dep
    },
}

cython_args = [
    '--include-dir',
    meson.current_build_dir(),
    '-X always_allow_keywords=true'
]
if get_option('buildtype') == 'debug'
    cython_args += ['--gdb']
endif

foreach ext_name, ext_dict : libs_sources
    py.extension_module(
        ext_name,
        [ext_dict.get('sources'), _cython_hash],
        cython_args: cython_args,
        # include_directories: 'src',
        # include_directories: [inc_np],
        dependencies: np_core_dep,
        subdir: 'numpy/_core/_hashing',
        install: true
    )
endforeach

# Basically just __init__.py and the .pyi files
sources_to_install = [
    'hashing.pyi',
    'hashtable.pyi',
]

foreach source: sources_to_install
    py.install_sources(
        source,
        subdir: 'numpy/_core/_hashing'
    )
endforeach
