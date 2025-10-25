#!/bin/bash
set -e

# Android wheel repair script for cibuildwheel
# This script ensures proper wheel tagging and basic validation

WHEEL_PATH="$1"
DEST_DIR="$2"

if [ -z "$WHEEL_PATH" ] || [ -z "$DEST_DIR" ]; then
    echo "Usage: $0 <wheel_path> <dest_dir>"
    exit 1
fi

echo "--- Repairing Android wheel: $WHEEL_PATH ---"

# Use Python from virtual environment, fallback to python3
PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

# Extract wheel filename and prepare directories
WHEEL_NAME=$(basename "$WHEEL_PATH")
mkdir -p "$DEST_DIR"
DEST_DIR="$(cd "$DEST_DIR" && pwd)"

# Create temporary directory for wheel manipulation
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT
cd "$TEMP_DIR"

# Extract the wheel
$PYTHON_BIN -m zipfile -e "$WHEEL_PATH" .

# Find the .dist-info directory
DIST_INFO_DIR=$(find . -name "*.dist-info" -type d | head -1)
if [ -z "$DIST_INFO_DIR" ]; then
    echo "Error: Could not find .dist-info directory in wheel"
    exit 1
fi

# Parse platform and API level from wheel name
if [[ "$WHEEL_NAME" =~ android_([0-9]+)_(arm64_v8a|x86_64) ]]; then
    ACTUAL_API_LEVEL="${BASH_REMATCH[1]}"
    ARCH_SUFFIX="${BASH_REMATCH[2]}"
    PLATFORM_TAG="android_${ACTUAL_API_LEVEL}_${ARCH_SUFFIX}"
    
    # Set architecture tag for validation
    if [[ "$ARCH_SUFFIX" == "arm64_v8a" ]]; then
        ARCH_TAG="aarch64"
    else
        ARCH_TAG="x86_64"
    fi
    
    # Warn if API level doesn't match expectation
    EXPECTED_API_LEVEL="30"
    if [ "$ACTUAL_API_LEVEL" != "$EXPECTED_API_LEVEL" ]; then
        echo "WARNING: Expected API level $EXPECTED_API_LEVEL but wheel was compiled for API level $ACTUAL_API_LEVEL"
        echo "This may indicate a toolchain configuration issue."
    fi
else
    echo "Error: Cannot parse API level and architecture from wheel name: $WHEEL_NAME"
    exit 1
fi

# Extract Python tag from wheel name and construct full wheel tag
if [[ "$WHEEL_NAME" =~ (cp[0-9]+)-(cp[0-9]+) ]]; then
    PYTHON_TAG="${BASH_REMATCH[1]}"
    ABI_TAG="${BASH_REMATCH[2]}"
    WHEEL_TAG="${PYTHON_TAG}-${ABI_TAG}-${PLATFORM_TAG}"
else
    echo "Error: Cannot parse Python version from wheel name: $WHEEL_NAME"
    exit 1
fi

# Update WHEEL metadata file
WHEEL_FILE="$DIST_INFO_DIR/WHEEL"
if [ -f "$WHEEL_FILE" ]; then
    # Check if Tag already exists and update if needed
    if grep -q "^Tag:" "$WHEEL_FILE"; then
        EXISTING_TAG=$(grep "^Tag:" "$WHEEL_FILE" | cut -d' ' -f2-)
        if [ "$EXISTING_TAG" != "$WHEEL_TAG" ]; then
            echo "Updating wheel tag from '$EXISTING_TAG' to '$WHEEL_TAG'"
            sed -i "s/^Tag:.*/Tag: $WHEEL_TAG/" "$WHEEL_FILE"
        fi
    else
        echo "Tag: $WHEEL_TAG" >> "$WHEEL_FILE"
        echo "Added Tag: $WHEEL_TAG to WHEEL metadata"
    fi
else
    echo "Warning: WHEEL file not found at $WHEEL_FILE"
fi

# Basic library validation
SHARED_LIBS=$(find . -name "*.so" -type f)
if [ -n "$SHARED_LIBS" ]; then
    echo "Validating $(echo "$SHARED_LIBS" | wc -l) shared libraries..."
    
    # Check for problematic dependencies
    if command -v readelf >/dev/null 2>&1; then
        for lib in $SHARED_LIBS; do
            # Check architecture compatibility
            file_output=$(file "$lib")
            if [[ "$ARCH_TAG" == "aarch64" ]] && [[ ! "$file_output" =~ "ARM aarch64" ]]; then
                echo "Warning: Library $lib may not be ARM64 compatible"
            elif [[ "$ARCH_TAG" == "x86_64" ]] && [[ ! "$file_output" =~ "x86-64" ]]; then
                echo "Warning: Library $lib may not be x86_64 compatible"
            fi
            
            # Check for problematic dependencies
            deps=$(readelf -d "$lib" 2>/dev/null | grep NEEDED | awk '{print $5}' | tr -d '[]' || true)
            for dep in $deps; do
                case "$dep" in
                    "libc++_shared.so")
                        echo "Warning: $lib depends on libc++_shared.so which may not be available on all Android devices"
                        ;;
                    "libpython"*)
                        echo "Warning: $lib links to libpython which is unusual for Android"
                        ;;
                esac
            done
        done
    fi
fi

# Repackage the wheel
echo "Repackaging wheel..."
$PYTHON_BIN -c "
import zipfile
import os

with zipfile.ZipFile('$DEST_DIR/$WHEEL_NAME', 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk('.'):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, '.')
            zf.write(file_path, arcname)
"

echo "--- Android wheel repair completed ---"
echo "Repaired wheel: $DEST_DIR/$WHEEL_NAME"
