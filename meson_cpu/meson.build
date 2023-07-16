# The CPU Dispatcher implmentation.
#
# Notes:
#  - This script must runs before any other subdir scripts to avoid override
#    any variables.
#
#  - Any variables with upercase must not be overrided,
#    there are other scripts depend on them.
#
#  - This script requires Meson module 'features'.
#
# This script is responsible of three things:
#  - Define the enabled baseline and dispatched features,
#    by parsing build options or compiler arguments in case
#    of detecting native flags
#  - Specfiy the baseline arguments and definitions through
#    all sources.
#  - Generate main configuration file which contains
#    information about the enabled features, also
#    contains collection of C macros required for
#    runtime dispatching not nesscery to part of this
#    header but we thought its better to gather everything
#    in one place.
#
#
# I. Configure the required variables
#
# CPU_CONF_CPREFIX: str, The prefix of C/C++ #definitions,
# that will be used for defining the enabled CPU features,
# C Macros that required by the runtime dispatching.
# e.g. {CPU_CONF_CPREFIX}_HAVE_AVX2
CPU_PREFIX = 'NPY_'
# The name of the main configuration file which contains
# information about the enabled features, the headers
# of the CPU features, also contains collection of
# C macros required for runtime dispatching not nesscery
# to part of this header but we thought its better to
# gather everything in one place.
CPU_CONFIG_NAME = 'npy_cpu_dispatch_config.h'

if get_option('disable-optimization')
  add_project_arguments('-DNPY_DISABLE_OPTIMIZATION', language: ['c', 'cpp'])
  CPU_REQ_BASELINE = 'none'
  CPU_REQ_DISPATCH = 'none'
else
  # The required minimal set of required CPU features.
  CPU_REQ_BASELINE = get_option('cpu-baseline')
  # The required dispatched set of additional CPU features.
  CPU_REQ_DISPATCH = get_option('cpu-dispatch')
endif
# II. Initlize the CPU features
# Export the X86 features objects 'SSE', 'AVX', etc
# and a dictionary "X86_FEATURES" which maps to each
# object by its name
subdir('x86')
subdir('ppc64')
subdir('s390x')
subdir('arm')

CPU_FEATURES = {}
CPU_FEATURES += X86_FEATURES
CPU_FEATURES += PPC64_FEATURES
CPU_FEATURES += S390X_FEATURES
CPU_FEATURES += ARM_FEATURES

# III. Parse the requsted baseline(CPU_REQ_BASELINE)
# and dispatch features(CPU_REQ_DISPATCH).
cpu_family = host_machine.cpu_family()
# Used by build option 'min'
min_features = {
  'x86': [SSE2],
  'x86_64': [SSE3],
  'ppc64': [],
  's390x': [],
  'arm': [],
  'aarch64': [ASIMD]
}.get(cpu_family, [])
if host_machine.endian() == 'little' and cpu_family == 'ppc64'
  min_features = [VSX2]
endif
# Used by build option 'max'
max_features_dict = {
  'x86': X86_FEATURES,
  'x86_64': X86_FEATURES,
  'ppc64': PPC64_FEATURES,
  's390x': S390X_FEATURES,
  'arm': ARM_FEATURES,
  'aarch64': ARM_FEATURES,
}.get(cpu_family, [])
max_features = []
foreach fet_name, fet_obj : max_features_dict
  max_features += [fet_obj]
endforeach

parse_options = {
  'cpu-baseline': CPU_REQ_BASELINE,
  'cpu-dispatch': CPU_REQ_DISPATCH
}
parse_result = {
  'cpu-baseline': [],
  'cpu-dispatch': []
}
mod_features = import('feature')
foreach opt_name, conf : parse_options
  # no support for regex :(?
  tokens = conf.replace(',', ' ').replace('+', ' + ').replace('-', ' - ').strip().to_upper().split()
  result = []
  ignored = []
  # append is the default
  append = true
  foreach tok : tokens
    if tok == '+'
      append = true
      continue
    elif tok == '-'
      append = false
      continue
    elif tok == 'NATIVE'
      if not is_variable('cpu_native_features')
        compiler_id = meson.get_compiler('c').get_id()
        native_flags = {
          'intel': '-xHost',
          'intel-cl': '/QxHost',
          # FIXME: Add support for fcc(-mcpu=a64fx) compiler
        }.get(compiler_id, '-march=native')
        test_native = mod_features.test(
          max_features, anyfet: true,
          force_args: [native_flags] + '-DDETECT_FEATURES'
        )
        if not test_native[0]
          error('Option "native" doesn\'t support compiler', compiler_id)
        endif
        cpu_native_features = []
        foreach fet_name : test_native[1].get('features')
          cpu_native_features += CPU_FEATURES[fet_name]
        endforeach
      endif
      accumulate = cpu_native_features
    elif tok == 'DETECT'
      if not is_variable('cpu_detect_features')
        test_detect = mod_features.test(
          max_features, anyfet: true,
          force_args: ['-DDETECT_FEATURES']
        )
        cpu_detect_features = []
        foreach fet_name : test_detect[1].get('features')
          cpu_detect_features += CPU_FEATURES[fet_name]
        endforeach
      endif
      accumulate = cpu_detect_features
    elif tok == 'MIN'
      accumulate = min_features
    elif tok == 'MAX'
      accumulate = max_features
    elif tok in CPU_FEATURES
      tokobj = CPU_FEATURES[tok]
      if tokobj not in max_features
        #error('Unsupported feature "'+tok+'" within option --'+opt_name)
        ignored += tok
        continue
      endif
      accumulate = [tokobj]
    else
      error('Invalid token "'+tok+'" within option --'+opt_name)
    endif
    if append
      foreach fet : accumulate
        if fet not in result
          result += fet
        endif
      endforeach
    else
      filterd = []
      foreach fet : result
        if fet not in accumulate
          filterd += fet
        endif
      endforeach
      result = filterd
    endif # append
  endforeach # tok : tokens
  if ignored.length() > 0
    message(
      'During parsing', opt_name,
      ': The following CPU features were ignored due to',
      'platform incompatibility or not being supported:',
      '\n"'+' '.join(ignored)+'"'
    )
  endif
  if result.length() > 0
    parse_result += {opt_name: mod_features.implicit_c(result)}
  endif
endforeach # opt_name, conf : parse_options
# IV. Test the baseline and dispatch features
# And set its flags and #definitions a cross all the sources.
#
# Its important to know that test enables the maxmuim
# supported features by the platform depend on the
# required features.
#
# For example, if the user sepecified `--cpu-baseline=avx512_skx` and
# the compiler doesn't support it but still support the implied features
# then we enable the maxmuim supported of the implied features e.g. AVX2
# which can be done by specify `any: true` to the test function.
if parse_result['cpu-baseline'].length() > 0
  baseline = mod_features.test(parse_result['cpu-baseline'], anyfet: true)[1]
  baseline_args = baseline['args']
  foreach baseline_fet : baseline['defines']
    baseline_args += ['-D'+CPU_PREFIX+'HAVE_' + baseline_fet]
  endforeach
  add_project_arguments(baseline_args, language: ['c', 'cpp'])
else
  baseline = {}
endif
# The name of the baseline features including
# its implied features.
CPU_BASELINE_NAMES = baseline['features']
CPU_BASELINE = []
foreach fet_name : CPU_BASELINE_NAMES
  CPU_BASELINE += [CPU_FEATURES[fet_name]]
endforeach
# Loop all initlized features and disable any
# feature not part of requsted baseline and dispatch features
# to avoid been enabled by import('feature').multi_targets
foreach fet_name, fet_obj : CPU_FEATURES
  if fet_obj in parse_result['cpu-dispatch'] or fet_name in CPU_BASELINE_NAMES
    continue
  endif
  fet_obj.update(disable: 'Not part of the requsted features')
endforeach

CPU_DISPATCH_NAMES = []
foreach fet_obj : parse_result['cpu-dispatch']
  # skip baseline features
  if fet_obj.get('name') in CPU_BASELINE_NAMES
    continue
  endif
  fet_test = mod_features.test(fet_obj)
  if not fet_test[0]
    continue
  endif
  CPU_DISPATCH_NAMES += [fet_obj.get('name')]
endforeach
# V. Generate main configuration header
# see 'main_config.h.in' for more clearfiction
main_config = {
  'PREFIX': CPU_PREFIX,
  'WITH_CPU_BASELINE': ' '.join(CPU_BASELINE_NAMES),
  'WITH_CPU_BASELINE_N': CPU_BASELINE_NAMES.length(),
  'WITH_CPU_DISPATCH': ' '.join(CPU_DISPATCH_NAMES),
  'WITH_CPU_DISPATCH_N': CPU_DISPATCH_NAMES.length(),
}
clines = []
macro_tpl = '    @0@_CPU_EXPAND(EXEC_CB(@1@, __VA_ARGS__)) \\'
foreach fet : CPU_BASELINE_NAMES
  clines += macro_tpl.format(CPU_PREFIX, fet)
endforeach
main_config += {'WITH_CPU_BASELINE_CALL': '\n'.join(clines)}
clines = []
foreach fet : CPU_DISPATCH_NAMES
  clines += macro_tpl.format(CPU_PREFIX, fet)
endforeach
main_config += {'WITH_CPU_DISPATCH_CALL': '\n'.join(clines)}

configure_file(
  input : 'main_config.h.in',
  output : CPU_CONFIG_NAME,
  configuration : configuration_data(main_config)
)
add_project_arguments(
  '-I' + meson.current_build_dir(),
  language: ['c', 'cpp']
)

CPU_MTARGETS_LOG = {}
