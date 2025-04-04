#!/usr/bin/env bash
set -e

# List of suspicious function calls:
SUSPICIOUS_FUNCS=(
    "PyList_GetItem"
    "PyList_GET_ITEM"
    "PyDict_GetItem"
    "PyDict_GetItemWithError"
    "PyDict_Next"
    "PyDict_GetItemString"
    "_PyDict_GetItemStringWithError"
)

# Gather changed files in PR; allow grep to return 1 without killing the script
CHANGED_FILES=$(git diff --name-only origin/main...HEAD | grep -E '\.c$|\.h$' || true)

# For debugging: print out files, i.e., confirm it's empty
echo "Changed C/headers:"
echo "$CHANGED_FILES"

# Prepare a result file
OUTPUT="c_api_usage_report.txt"
echo "Running Suspicious C API usage report workflow..." > $OUTPUT

FAIL=0

# Loop over changed files
for file in $CHANGED_FILES; do
    # Skip non-C, non-header files
    if [[ ! $file =~ \.(c|h)$ ]]; then
        continue
    fi
    
    for func in "${SUSPICIOUS_FUNCS[@]}"; do
        # Run grep, if found, record and flag
        if grep -n "$func" "$file" > /dev/null 2>&1; then

            # If not whitelisted...
            echo "Found suspicious call $func in $file" >> $OUTPUT
            FAIL=1
        fi
    done
done

cat $OUTPUT
exit $FAIL
