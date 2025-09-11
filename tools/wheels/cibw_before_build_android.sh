#!/bin/bash
set -e
set -x

# This script is designed to be called by cibuildwheecho "--- END WORKAROUND ---"

# Get the target architecture from cibuildwheel's environment variables
# (Now set by our workaround above if cibuildwheel didn't set it)
echo "--- Architecture Configuration ---"
echo "CIBW_ARCHS: ${CIBW_ARCHS:-'(not set)'}"
echo "CIBW_ARCHS_ANDROID: ${CIBW_ARCHS_ANDROID:-'(not set)'}"

# Check for architecture in standard cibuildwheel environment variablesbuilding for Android.
# Since cibuildwheel 3.x has built-in Android support, we mainly need to generate 
# the correct Meson cross file.

# WORKAROUND FOR CIBUILDWHEEL ISSUE:
# cibuildwheel documentation states that CIBW_ARCHS should be available for Android builds,
# but cibuildwheel 3.1.4 does NOT set these environment variables for Android.
# This appears to be a bug or incomplete implementation in cibuildwheel.
# 
# References:
# - Documentation: https://cibuildwheel.pypa.io/en/stable/options/#archs
# - Issue should be reported to: https://github.com/pypa/cibuildwheel/issues
#
# As a workaround, we detect the architecture from various sources:
# 1. Check if CIBW_ARCHS is already set (future-proofing)
# 2. Look for architecture hints in environment variables
# 3. Check process command line for architecture indicators

echo "--- CIBUILDWHEEL ANDROID ARCHITECTURE DETECTION WORKAROUND ---"
echo "Note: Working around missing CIBW_ARCHS environment variables in cibuildwheel Android builds"

# Don't override if already set (future-proofing for when cibuildwheel fixes this)
if [ -z "$CIBW_ARCHS" ]; then
    # Try multiple detection methods
    DETECTED_ARCH=""
    
    # Method 1: Check current process command line
    if [ -f /proc/self/cmdline ]; then
        CMDLINE=$(tr '\0' ' ' < /proc/self/cmdline)
        if [[ "$CMDLINE" == *"x86_64"* ]]; then
            DETECTED_ARCH="x86_64"
            echo "Detected x86_64 from process command line"
        elif [[ "$CMDLINE" == *"arm64"* ]] || [[ "$CMDLINE" == *"aarch64"* ]]; then
            DETECTED_ARCH="arm64_v8a"
            echo "Detected arm64_v8a from process command line"
        fi
    fi
    
    # Method 2: Check parent process command line (cibuildwheel)
    if [ -z "$DETECTED_ARCH" ] && [ -n "$PPID" ] && [ -f "/proc/$PPID/cmdline" ]; then
        PARENT_CMDLINE=$(tr '\0' ' ' < "/proc/$PPID/cmdline" 2>/dev/null || echo "")
        if [[ "$PARENT_CMDLINE" == *"x86_64"* ]]; then
            DETECTED_ARCH="x86_64"
            echo "Detected x86_64 from parent process command line"
        elif [[ "$PARENT_CMDLINE" == *"arm64"* ]] || [[ "$PARENT_CMDLINE" == *"aarch64"* ]]; then
            DETECTED_ARCH="arm64_v8a"
            echo "Detected arm64_v8a from parent process command line"
        fi
    fi
    
    # Method 3: Check environment variables for hints
    if [ -z "$DETECTED_ARCH" ]; then
        # Look for architecture in any environment variable
        ENV_ARCH_HINTS=$(env | grep -i "x86_64\|aarch64\|arm64" | head -1 || echo "")
        if [[ "$ENV_ARCH_HINTS" == *"x86_64"* ]]; then
            DETECTED_ARCH="x86_64"
            echo "Detected x86_64 from environment variables"
        elif [[ "$ENV_ARCH_HINTS" == *"aarch64"* ]] || [[ "$ENV_ARCH_HINTS" == *"arm64"* ]]; then
            DETECTED_ARCH="arm64_v8a"
            echo "Detected arm64_v8a from environment variables"
        fi
    fi
    
    # Fallback: Use arm64_v8a as it's the most common Android architecture
    if [ -z "$DETECTED_ARCH" ]; then
        DETECTED_ARCH="arm64_v8a"
        echo "No architecture detected, defaulting to arm64_v8a (most common Android architecture)"
    fi
    
    export CIBW_ARCHS="$DETECTED_ARCH"
    echo "Set CIBW_ARCHS=${CIBW_ARCHS} (via workaround)"
else
    echo "CIBW_ARCHS already set: ${CIBW_ARCHS}"
fi

echo "--- END WORKAROUND ---"

# Get the target architecture from cibuildwheel's environment variables
# According to cibuildwheel docs, CIBW_ARCHS should be available for Android builds
echo "--- Debug: Checking for cibuildwheel architecture environment variables ---"
echo "CIBW_ARCHS: ${CIBW_ARCHS:-'(not set)'}"
echo "CIBW_ARCHS_ANDROID: ${CIBW_ARCHS_ANDROID:-'(not set)'}"
echo "--- End debug ---"

# Get the target architecture from cibuildwheel's environment variables
# According to cibuildwheel docs, CIBW_ARCHS should be available for Android builds
echo "--- Debug: Checking for cibuildwheel architecture environment variables ---"
echo "CIBW_ARCHS: ${CIBW_ARCHS:-'(not set)'}"
echo "CIBW_ARCHS_ANDROID: ${CIBW_ARCHS_ANDROID:-'(not set)'}"
echo "--- End debug ---"

# Check for architecture in standard cibuildwheel environment variables
if [ -n "$CIBW_ARCHS" ]; then
    TARGET_ARCH="$CIBW_ARCHS"
    echo "Using CIBW_ARCHS: $TARGET_ARCH"
elif [ -n "$CIBW_ARCHS_ANDROID" ]; then
    TARGET_ARCH="$CIBW_ARCHS_ANDROID"
    echo "Using CIBW_ARCHS_ANDROID: $TARGET_ARCH"
else
    # This should not happen now that we set CIBW_ARCHS in the workaround above
    echo "ERROR: Architecture detection failed even after workaround!"
    echo "This indicates a serious issue with the detection logic."
    exit 1
fi

echo "--- Building for architecture: $TARGET_ARCH ---"

# Set the target API level from cibuildwheel's environment variable  
if [ -n "$ANDROID_API_LEVEL" ]; then
    API_LEVEL="$ANDROID_API_LEVEL"
    echo "Using ANDROID_API_LEVEL: $API_LEVEL"
else
    echo "ERROR: ANDROID_API_LEVEL environment variable is not set!"
    echo "This should be set by your cibuildwheel configuration."
    echo ""
    echo "Example in pyproject.toml:"
    echo '[tool.cibuildwheel.android.environment]'
    echo 'ANDROID_API_LEVEL = "30"'
    echo ""
    echo "Or as environment variable:"
    echo 'export ANDROID_API_LEVEL=30'
    exit 1
fi

echo "Using TARGET_ARCH: $TARGET_ARCH"
echo "Using API_LEVEL: $API_LEVEL"

# Map cibuildwheel architecture names to Android NDK architecture names
case "$TARGET_ARCH" in
    "arm64_v8a")
        ANDROID_ARCH="arm64-v8a"
        MESON_CPU_FAMILY="aarch64"
        MESON_CPU="aarch64"
        ;;
    "x86_64")
        ANDROID_ARCH="x86_64"
        MESON_CPU_FAMILY="x86_64"
        MESON_CPU="x86_64"
        ;;
    *)
        echo "Unsupported architecture: $TARGET_ARCH"
        exit 1
        ;;
esac

echo "Mapped to Android: $ANDROID_ARCH, Meson: $MESON_CPU_FAMILY/$MESON_CPU"

# Generate a dynamic cross file for the current architecture
CROSS_FILE="/tmp/android_${ANDROID_ARCH}.meson.cross"

# Since cibuildwheel sets up the Android toolchain environment variables 
# after our before_build script runs, we create a minimal cross file that 
# lets Meson find the compilers from the environment when it runs
cat > "$CROSS_FILE" << EOF
#
# Meson cross file for Android ($ANDROID_ARCH)
#
# This file defines the toolchain for Android cross-compilation.
# Generated dynamically by cibw_before_build_android.sh
#
# Note: The Android NDK toolchain binaries are set by cibuildwheel
# via environment variables after this script runs, but we override
# them here to use API $API_LEVEL toolchain.
#

[binaries]
# Use wrapper scripts that filter out \$(BLDLIBRARY) from linking commands
c = '/tmp/android-clang-wrapper.sh'
cpp = '/tmp/android-clang++-wrapper.sh'
ar = '/home/kfchou/.cache/briefcase/tools/android_sdk/ndk/27.2.12479018/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar'
strip = '/home/kfchou/.cache/briefcase/tools/android_sdk/ndk/27.2.12479018/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip'

[host_machine]
system = 'android'
cpu_family = '$MESON_CPU_FAMILY'
cpu = '$MESON_CPU'
endian = 'little'

[properties]
# This tells Meson not to try running compiled binaries on the host.
needs_exe_wrapper = true

# Android-specific configuration
android_api_level = '$API_LEVEL'

# Tell Meson that math library is not needed on Android (functions in libc)
has_function_sin = true
has_function_cos = true 
has_function_tan = true
has_function_sinh = true
has_function_cosh = true
has_function_tanh = true
has_function_asin = true
has_function_acos = true
has_function_atan = true
has_function_atan2 = true
has_function_exp = true
has_function_log = true
has_function_log10 = true
has_function_sqrt = true
has_function_floor = true
has_function_ceil = true
has_function_fabs = true
has_function_pow = true
has_function_fmod = true
has_function_ldexp = true
has_function_frexp = true
has_function_modf = true
has_function_copysign = true
has_function_finite = true
has_function_isinf = true
has_function_isnan = true
has_function_isfinite = true
has_function_expm1 = true
has_function_log1p = true
has_function_hypot = true
has_function_rint = true
has_function_trunc = true
has_function_exp2 = true
has_function_log2 = true
has_function_round = true
has_function_nearbyint = true
has_function_remainder = true
has_function_acosh = true
has_function_asinh = true
has_function_atanh = true
has_function_erf = true
has_function_erfc = true
has_function_lgamma = true
has_function_tgamma = true
has_function_j0 = true
has_function_j1 = true
has_function_jn = true
has_function_y0 = true
has_function_y1 = true
has_function_yn = true
has_function_cbrt = true
has_function_nextafter = true
has_function_csin = true
has_function_ccos = true
has_function_ctan = true
has_function_csinh = true
has_function_ccosh = true
has_function_ctanh = true
has_function_casin = true
has_function_cacos = true
has_function_catan = true
has_function_casinh = true
has_function_cacosh = true
has_function_catanh = true
has_function_cexp = true
has_function_clog = true
has_function_cpow = true
has_function_csqrt = true
has_function_cabs = true
has_function_carg = true
has_function_cimag = true
has_function_creal = true
has_function_conj = true
has_function_strtoll = true
has_function_strtoull = true
has_function_strtold = true

# Skip BLAS/LAPACK for Android to avoid complex math library dependencies
allow-noblas = true

# Specify long double format for ARM64 Android to avoid cross-compilation test run
# Android ARM64 uses IEEE 754 quad precision (128-bit)
longdouble_format = 'IEEE_QUAD_LE'

[built-in options]
# Android Python extensions need proper library linking for runtime resolution
# This ensures symbols like PyExc_ValueError and math functions are available
# Note: On Android, math functions are part of libc, not a separate libm
c_link_args = ['-lpython3.13']
cpp_link_args = ['-lpython3.13']

# Enable Python library linking for Android to resolve Python API symbols
python_link_args = ['-lpython3.13']

# Override system library detection for Android - math functions are in libc
c_args = ['-DHAVE_SIN=1', '-DHAVE_COS=1', '-DHAVE_TAN=1', '-DHAVE_SINH=1', '-DHAVE_COSH=1', '-DHAVE_TANH=1', '-DHAVE_ASIN=1', '-DHAVE_ACOS=1', '-DHAVE_ATAN=1', '-DHAVE_ATAN2=1', '-DHAVE_EXP=1', '-DHAVE_LOG=1', '-DHAVE_LOG10=1', '-DHAVE_SQRT=1', '-DHAVE_FLOOR=1', '-DHAVE_CEIL=1', '-DHAVE_FABS=1', '-DHAVE_POW=1', '-DHAVE_FMOD=1', '-DHAVE_LDEXP=1', '-DHAVE_FREXP=1', '-DHAVE_MODF=1', '-DHAVE_COPYSIGN=1', '-DHAVE_FINITE=1', '-DHAVE_ISINF=1', '-DHAVE_ISNAN=1', '-DHAVE_ISFINITE=1', '-DHAVE_EXPM1=1', '-DHAVE_LOG1P=1', '-DHAVE_ASINH=1', '-DHAVE_ACOSH=1', '-DHAVE_ATANH=1', '-DHAVE_RINT=1', '-DHAVE_TRUNC=1', '-DHAVE_EXP2=1', '-DHAVE_LOG2=1', '-DHAVE_HYPOT=1', '-DHAVE_CBRT=1', '-DHAVE_NEXTAFTER=1', '-DHAVE_CSIN=1', '-DHAVE_CCOS=1', '-DHAVE_CTAN=1', '-DHAVE_CSINH=1', '-DHAVE_CCOSH=1', '-DHAVE_CTANH=1', '-DHAVE_CASIN=1', '-DHAVE_CACOS=1', '-DHAVE_CATAN=1', '-DHAVE_CASINH=1', '-DHAVE_CACOSH=1', '-DHAVE_CATANH=1', '-DHAVE_CEXP=1', '-DHAVE_CLOG=1', '-DHAVE_CPOW=1', '-DHAVE_CSQRT=1', '-DHAVE_CABS=1', '-DHAVE_CARG=1', '-DHAVE_CIMAG=1', '-DHAVE_CREAL=1', '-DHAVE_CONJ=1', '-DHAVE_STRTOLL=1', '-DHAVE_STRTOULL=1', '-DHAVE_STRTOLD=1']
cpp_args = ['-DHAVE_SIN=1', '-DHAVE_COS=1', '-DHAVE_TAN=1', '-DHAVE_SINH=1', '-DHAVE_COSH=1', '-DHAVE_TANH=1', '-DHAVE_ASIN=1', '-DHAVE_ACOS=1', '-DHAVE_ATAN=1', '-DHAVE_ATAN2=1', '-DHAVE_EXP=1', '-DHAVE_LOG=1', '-DHAVE_LOG10=1', '-DHAVE_SQRT=1', '-DHAVE_FLOOR=1', '-DHAVE_CEIL=1', '-DHAVE_FABS=1', '-DHAVE_POW=1', '-DHAVE_FMOD=1', '-DHAVE_LDEXP=1', '-DHAVE_FREXP=1', '-DHAVE_MODF=1', '-DHAVE_COPYSIGN=1', '-DHAVE_FINITE=1', '-DHAVE_ISINF=1', '-DHAVE_ISNAN=1', '-DHAVE_ISFINITE=1', '-DHAVE_EXPM1=1', '-DHAVE_LOG1P=1', '-DHAVE_ASINH=1', '-DHAVE_ACOSH=1', '-DHAVE_ATANH=1', '-DHAVE_RINT=1', '-DHAVE_TRUNC=1', '-DHAVE_EXP2=1', '-DHAVE_LOG2=1', '-DHAVE_HYPOT=1', '-DHAVE_CBRT=1', '-DHAVE_NEXTAFTER=1', '-DHAVE_CSIN=1', '-DHAVE_CCOS=1', '-DHAVE_CTAN=1', '-DHAVE_CSINH=1', '-DHAVE_CCOSH=1', '-DHAVE_CTANH=1', '-DHAVE_CASIN=1', '-DHAVE_CACOS=1', '-DHAVE_CATAN=1', '-DHAVE_CASINH=1', '-DHAVE_CACOSH=1', '-DHAVE_CATANH=1', '-DHAVE_CEXP=1', '-DHAVE_CLOG=1', '-DHAVE_CPOW=1', '-DHAVE_CSQRT=1', '-DHAVE_CABS=1', '-DHAVE_CARG=1', '-DHAVE_CIMAG=1', '-DHAVE_CREAL=1', '-DHAVE_CONJ=1', '-DHAVE_STRTOLL=1', '-DHAVE_STRTOULL=1', '-DHAVE_STRTOLD=1']
EOF

echo "Generated cross file: $CROSS_FILE"
echo "Cross file contents:"
cat "$CROSS_FILE"

# Create a symlink with a predictable name for the config
ln -sf "$CROSS_FILE" "/tmp/android_current.meson.cross"

echo "--- Android build environment configured ---"
echo "Using cibuildwheel's built-in Android toolchain"
echo "Cross file: /tmp/android_current.meson.cross"

# Debug Python configuration to understand the BLDLIBRARY issue
echo "--- Debugging Python configuration ---"
python -c "
import sysconfig
import os

# Check BLDLIBRARY configuration
bldlibrary = sysconfig.get_config_var('BLDLIBRARY')
print('BLDLIBRARY from sysconfig:', repr(bldlibrary))

# Check if it's in environment
bldlib_env = os.environ.get('BLDLIBRARY', 'NOT_SET')
print('BLDLIBRARY from env:', repr(bldlib_env))

# Get other Python config vars that might be relevant
ldlibrary = sysconfig.get_config_var('LDLIBRARY') 
print('LDLIBRARY:', repr(ldlibrary))

libs = sysconfig.get_config_var('LIBS')
print('LIBS:', repr(libs))

# Check library directory
libdir = sysconfig.get_config_var('LIBDIR')
print('LIBDIR:', repr(libdir))
"

# Try different approaches to fix BLDLIBRARY issue
# The issue is meson-python is inserting literal '$(BLDLIBRARY)' in link commands
# For Android, we don't want to link Python extensions to libpython at all

# Create wrapper scripts to filter out BLDLIBRARY from linker commands
echo "Creating linker wrapper scripts..."

# Determine the correct compiler prefix based on architecture
if [ "$ANDROID_ARCH" = "arm64-v8a" ]; then
    COMPILER_PREFIX="aarch64-linux-android${API_LEVEL}"
elif [ "$ANDROID_ARCH" = "x86_64" ]; then
    COMPILER_PREFIX="x86_64-linux-android${API_LEVEL}"
else
    echo "Error: Unsupported architecture: $ANDROID_ARCH"
    exit 1
fi

echo "Using compiler prefix: $COMPILER_PREFIX"

# Create C compiler wrapper
cat > /tmp/android-clang-wrapper.sh << WRAPPER_EOF
#!/bin/bash
# Filter out \$(BLDLIBRARY) from arguments and add proper linking for Android
args=()

for arg in "\$@"; do
    if [ "\$arg" != '\$(BLDLIBRARY)' ]; then
        args+=("\$arg")
    fi
done

# For shared library builds, add proper library linking for Android
if [[ "\${args[*]}" == *"-shared"* ]]; then
    # Add static linking for C++ standard library to avoid libc++_shared.so dependency
    args+=("-static-libstdc++")
    
    # Add math library explicitly for Android
    args+=("-lm")
    
    # Extract Python library path from LDFLAGS environment variable
    # LDFLAGS contains -L/path/to/python/lib which we need for linking
    if [ -n "\$LDFLAGS" ]; then
        python_lib_path=\$(echo "\$LDFLAGS" | grep -o -- '-L[^[:space:]]*' | head -1)
        if [ -n "\$python_lib_path" ]; then
            args+=("\$python_lib_path")
        fi
    fi
    
    # Always add Python library linking
    args+=("-lpython3.13")
fi

# Use API $API_LEVEL compiler explicitly instead of the API 21 one set by cibuildwheel
exec /home/kfchou/.cache/briefcase/tools/android_sdk/ndk/27.2.12479018/toolchains/llvm/prebuilt/linux-x86_64/bin/${COMPILER_PREFIX}-clang "\${args[@]}"
WRAPPER_EOF
chmod +x /tmp/android-clang-wrapper.sh

# Create C++ compiler wrapper  
cat > /tmp/android-clang++-wrapper.sh << WRAPPER_EOF
#!/bin/bash
# Filter out \$(BLDLIBRARY) from arguments and add proper linking for Android
args=()
for arg in "\$@"; do
    if [ "\$arg" != '\$(BLDLIBRARY)' ]; then
        args+=("\$arg")
    fi
done

# For shared library builds, add proper library linking for Android
if [[ "\${args[*]}" == *"-shared"* ]]; then
    # Add static linking for C++ standard library to avoid libc++_shared.so dependency
    args+=("-static-libstdc++")
    
    # Add math library explicitly for Android
    args+=("-lm")
    
    # Extract Python library path from LDFLAGS environment variable
    # LDFLAGS contains -L/path/to/python/lib which we need for linking
    if [ -n "\$LDFLAGS" ]; then
        python_lib_path=\$(echo "\$LDFLAGS" | grep -o -- '-L[^[:space:]]*' | head -1)
        if [ -n "\$python_lib_path" ]; then
            args+=("\$python_lib_path")
        fi
    fi
    
    # Always add Python library linking
    args+=("-lpython3.13")
fi

# Use API $API_LEVEL compiler explicitly instead of the API 21 one set by cibuildwheel
exec /home/kfchou/.cache/briefcase/tools/android_sdk/ndk/27.2.12479018/toolchains/llvm/prebuilt/linux-x86_64/bin/${COMPILER_PREFIX}-clang++ "\${args[@]}"
WRAPPER_EOF
chmod +x /tmp/android-clang++-wrapper.sh

# Final environment setup to force math function availability
# Set environment variables that tell build system math functions are available
export HAVE_SIN=1
export HAVE_COS=1 
export HAVE_TAN=1
export MATHLIB_AVAILABLE=1

# Force skip of math library detection
export SKIP_MATH_LIB_CHECK=1

echo "--- Set up environment to skip math library detection ---"
echo "Android math functions are available in libc, no separate libm needed"

# Override the environment variables that cibuildwheel sets to ensure 
# the wheel naming and repair process uses API 30
export CC="/tmp/android-clang-wrapper.sh"
export CXX="/tmp/android-clang++-wrapper.sh"

echo "Created compiler wrappers and environment overrides for API 30 Android"

# Debug: Show what compiler toolchain cibuildwheel is trying to use
echo "--- Debug: Checking cibuildwheel's compiler settings ---"
echo "Original CC: $CC"
echo "Original CXX: $CXX"
echo "Our wrapper CC: /tmp/android-clang-wrapper.sh"
echo "Our wrapper CXX: /tmp/android-clang++-wrapper.sh"

# Test our wrapper to make sure it's using the right compiler
echo "Testing wrapper script:"
/tmp/android-clang-wrapper.sh --version 2>/dev/null | head -1 || echo "Wrapper test failed"