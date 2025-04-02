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
    "PySequence_Fast"
)

# Find all C/C++ source files in the repo
# ALL_FILES=$(find . -type f \( -name "*.c" -o -name "*.h" -o -name "*.c.src" -o -name "*.cpp" \))
ALL_FILES=$(find numpy -type f \( -name "*.c" -o -name "*.h" -o -name "*.c.src" -o -name "*.cpp" \))

# For debugging: print out file count
echo "Scanning $(echo "$ALL_FILES" | wc -l) C/C++ source files..."

# Prepare a result file
OUTPUT="c_api_usage_report.txt"
echo -e "Running Suspicious C API usage report workflow...\n" > $OUTPUT

FAIL=0

# Scan each changed file
for file in $ALL_FILES; do
    
    for func in "${SUSPICIOUS_FUNCS[@]}"; do
        # -n     : show line number
        # -P     : perl-style boundaries
        # (?<!\w): check - no letter/number/underscore before
        # (?!\w) : check - no letter/number/underscore after
        matches=$(grep -n -P "(?<!\w)$func(?!\w)" "$file" || true)

        # Check each match for 'noqa'
        if [[ -n "$matches" ]]; then
            while IFS= read -r line; do
                if [[ "$line" != *"noqa: borrowed-ref OK"* ]]; then
                    echo "Found suspcious call to $func in file: $file" >> "$OUTPUT"
                    echo " -> $line" >> "$OUTPUT"
                    echo "Recommendation:" >> "$OUTPUT"
                    echo "If this use is intentional and safe, add '// noqa: borrowed-ref OK' on the same line to silence this warning." >> "$OUTPUT"
                    echo "Otherwise, consider replacing $func with a thread-safe API function." >> "$OUTPUT"
                    echo "" >> "$OUTPUT"
                    FAIL=1
                fi 
            done <<< "$matches"
        fi
    done
done

cat $OUTPUT
exit $FAIL
