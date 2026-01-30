# The CPU Dispatcher implementation.
#
# This script handles the CPU dispatcher and requires the Meson module
# 'features'.
#
# The CPU dispatcher script is responsible for three main tasks:
#
# 1. Defining the enabled baseline and dispatched features by parsing build
#    options or compiler arguments, including detection of native flags.
#
# 2. Specifying the baseline arguments and definitions across all sources.
#
# 3. Generating the main configuration file, which contains information about
#    the enabled features, along with a collection of C macros necessary for
#    runtime dispatching. For more details, see the template file
#    `main_config.h.in`.
#
# This script exposes the following variables:
#
# - `CPU_BASELINE`: A set of CPU feature objects obtained from
#                   `mod_features.new()`, representing the minimum CPU features
#                   specified by the build option `-Dcpu-baseline`.
#
# - `CPU_BASELINE_NAMES`: A set of enabled CPU feature names, representing the
#                         minimum CPU features specified by the build option
#                         `-Dcpu-baseline`.
#
# - `CPU_DISPATCH_NAMES`: A set of enabled CPU feature names, representing the
#                         additional CPU features that can be dispatched at
#                         runtime, specified by the build option
#                         `-Dcpu-dispatch`.
#
# - `CPU_FEATURES`: A dictionary containing all supported CPU feature objects.
#
# Additionally, this script exposes a set of variables that represent each
# supported feature to be used within the Meson function
# `mod_features.multi_targets()`.

# Prefix used by all macros and features definitions
CPU_CONF_PREFIX = 'NPY_'
# main configuration name
CPU_CONF_CONFIG = 'npy_cpu_dispatch_config.h'

if get_option('disable-optimization')
  add_project_arguments('-D' + CPU_CONF_PREFIX + 'DISABLE_OPTIMIZATION', language: ['c', 'cpp'])
  CPU_CONF_BASELINE = 'none'
  CPU_CONF_DISPATCH = 'none'
else
  baseline_detect = get_option('cpu-baseline-detect').enabled()
  c_args = get_option('c_args')
  if get_option('cpu-baseline-detect').auto()
    foreach arg : c_args
      foreach carch : ['-march', '-mcpu', '-xhost', '/QxHost']
        if arg.contains(carch)
          message('Appending option "detect" to "cpu-baseline" due to detecting global architecture c_arg "' + arg + '"')
          baseline_detect = true
          break
        endif
      endforeach
      if baseline_detect
        break
      endif
    endforeach
  endif
  # The required minimal set of required CPU features.
  CPU_CONF_BASELINE = get_option('cpu-baseline')
  if baseline_detect
    CPU_CONF_BASELINE += '+detect'
  endif
  # The required dispatched set of additional CPU features.
  CPU_CONF_DISPATCH = get_option('cpu-dispatch')
endif

# Initialize the CPU features Export the X86 features objects 'SSE', 'AVX',
# etc. plus a dictionary "X86_FEATURES" which maps to each object by its name
subdir('x86')
subdir('ppc64')
subdir('s390x')
subdir('arm')
subdir('riscv64')
subdir('loongarch64')

CPU_FEATURES = {}
CPU_FEATURES += ARM_FEATURES
CPU_FEATURES += X86_FEATURES
CPU_FEATURES += PPC64_FEATURES
CPU_FEATURES += S390X_FEATURES
CPU_FEATURES += RV64_FEATURES
CPU_FEATURES += LOONGARCH64_FEATURES

CPU_FEATURES_REDIRECT = {}
CPU_FEATURES_REDIRECT += X86_REDIRECT

# Parse the requested baseline (CPU_CONF_BASELINE) and dispatch features
# (CPU_CONF_DISPATCH).
cpu_family = host_machine.cpu_family()
# Used by build option 'min'
min_features = {
  'x86': [X86_V2],
  'x86_64': [X86_V2],
  'ppc64': [],
  's390x': [],
  'arm': [],
  'aarch64': [ASIMD],
  'riscv64': [],
  'wasm32': [],
  'loongarch64': [LSX],
}.get(cpu_family, [])
if host_machine.endian() == 'little' and cpu_family == 'ppc64'
  min_features = [VSX2]
endif

# Used by build option 'max/native/detect'
max_features_dict = {
  'x86': X86_FEATURES,
  'x86_64': X86_FEATURES,
  'ppc64': PPC64_FEATURES,
  's390x': S390X_FEATURES,
  'arm': ARM_FEATURES,
  'aarch64': ARM_FEATURES,
  'riscv64': RV64_FEATURES,
  'wasm32': {},
  'loongarch64': LOONGARCH64_FEATURES,
}.get(cpu_family, {})
max_features = []
foreach fet_name, fet_obj : max_features_dict
  max_features += [fet_obj]
endforeach
if max_features.length() == 0
  message('Disabling CPU feature detection due to unsupported architecture: "' + cpu_family + '"')
  CPU_CONF_BASELINE = 'none'
  CPU_CONF_DISPATCH = 'none'
endif

parse_options = {
  'cpu-baseline': CPU_CONF_BASELINE,
  'cpu-dispatch': CPU_CONF_DISPATCH
}
parse_result = {
  'cpu-baseline': [],
  'cpu-dispatch': []
}
mod_features = import('features')
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
    elif tok == 'NONE'
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
          force_args: ['-DDETECT_FEATURES'] + get_option('c_args')
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
    else
      if tok in CPU_FEATURES_REDIRECT
        ntok = CPU_FEATURES_REDIRECT[tok]
        if ntok == ''
          warning('Ignoring CPU feature "@0@" in --@1@ option - feature is no longer supported.'.format(tok, opt_name))
        else
          warning('CPU Feature "@0@" is no longer explicitly supported, redirecting to "@1@".'.format(tok, ntok))
        endif
        warning('Please check the latest documentation for build options.')
        if ntok == '' or not append # redirected features not safe to be execluded
          continue
        endif
        tok = ntok
      endif
      if tok not in CPU_FEATURES
        error('Invalid token "'+tok+'" within option --'+opt_name)
      endif
      if tok in CPU_FEATURES
        tokobj = CPU_FEATURES[tok]
        if tokobj not in max_features
          ignored += tok
          continue
        endif
        accumulate = [tokobj]
      endif
    endif
    if append
      foreach fet : accumulate
        if fet not in result
          result += fet
        endif
      endforeach
    else
      filterd = []
      # filter out the features that are in the accumulate list 
      # including any successor features
      foreach fet : result
        escape = false
        foreach fet2 : accumulate
          if fet2 in mod_features.implicit_c(fet)
            escape = true
            break
          endif
        endforeach
        if not escape
          filterd += fet
        endif
      endforeach
      result = filterd
    endif # append
  endforeach # tok : tokens
  if ignored.length() > 0
    message(
      'During parsing ' + opt_name +
      ': The following CPU features were ignored due to platform ' +
      'incompatibility or lack of support:\n"' + ' '.join(ignored) + '"'
    )
  endif
  if result.length() > 0
    parse_result += {opt_name: mod_features.implicit_c(result)}
  endif
endforeach # opt_name, conf : parse_options

# Test the baseline and dispatch features and set their flags and #definitions
# across all sources.
#
# It is important to know that this test enables the maximum supported features
# by the platform depending on the required features.
#
# For example, if the user specified `--cpu-baseline=avx512_skx`, and the
# compiler doesn't support it, but still supports any of the implied features,
# then we enable the maximum supported implied features, e.g., AVX2, which can
# be done by specifying `anyfet: true` to the test function.
if parse_result['cpu-baseline'].length() > 0
    baseline = mod_features.test(parse_result['cpu-baseline'], anyfet: true)[1]
    baseline_args = baseline['args']
    foreach baseline_fet : baseline['defines']
        baseline_args += ['-D' + CPU_CONF_PREFIX + 'HAVE_' + baseline_fet]
    endforeach
    add_project_arguments(baseline_args, language: ['c', 'cpp'])
else
    baseline = {}
endif
# The name of the baseline features including its implied features.
CPU_BASELINE_NAMES = baseline.get('features', [])
CPU_BASELINE = []
foreach fet_name : CPU_BASELINE_NAMES
  CPU_BASELINE += [CPU_FEATURES[fet_name]]
endforeach
# Loop over all initialized features and disable any feature that is not part
# of the requested baseline and dispatch features to avoid it enabled by
# import('feature').multi_targets
foreach fet_name, fet_obj : CPU_FEATURES
  if fet_obj in parse_result['cpu-dispatch'] or fet_name in CPU_BASELINE_NAMES
    continue
  endif
  fet_obj.update(disable: 'Not part of the requested features')
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
# Generate main configuration header see 'main_config.h.in' for more
# clarification.
main_config = {
  'P': CPU_CONF_PREFIX,
  'WITH_CPU_BASELINE': ' '.join(CPU_BASELINE_NAMES),
  'WITH_CPU_BASELINE_N': CPU_BASELINE_NAMES.length(),
  'WITH_CPU_DISPATCH': ' '.join(CPU_DISPATCH_NAMES),
  'WITH_CPU_DISPATCH_N': CPU_DISPATCH_NAMES.length(),
}
clines = []
macro_tpl = '@0@_CPU_EXPAND(EXEC_CB(@1@, __VA_ARGS__)) \\'
foreach fet : CPU_BASELINE_NAMES
  clines += macro_tpl.format(CPU_CONF_PREFIX, fet)
endforeach
main_config += {'WITH_CPU_BASELINE_CALL': '\n'.join(clines)}
clines = []
foreach fet : CPU_DISPATCH_NAMES
  clines += macro_tpl.format(CPU_CONF_PREFIX, fet)
endforeach
main_config += {'WITH_CPU_DISPATCH_CALL': '\n'.join(clines)}

configure_file(
  input : 'main_config.h.in',
  output : CPU_CONF_CONFIG,
  configuration : configuration_data(main_config)
)
add_project_arguments(
  '-I' + meson.current_build_dir(),
  language: ['c', 'cpp']
)

message(
'''
CPU Optimization Options
  baseline:
    Requested : @0@
    Enabled   : @1@
  dispatch:
    Requested : @2@
    Enabled   : @3@
'''.format(
    CPU_CONF_BASELINE, ' '.join(CPU_BASELINE_NAMES),
    CPU_CONF_DISPATCH, ' '.join(CPU_DISPATCH_NAMES)
  )
)
