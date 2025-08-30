#!/usr/bin/env bash
set -e

# List of suspicious function calls:
SUSPICIOUS_FUNCS=(
    "PyList_GetItem"
    "PyDict_GetItem"
    "PyDict_GetItemWithError"
    "PyDict_GetItemString"
    "PyDict_SetDefault"
    "PyDict_Next"
    "PyWeakref_GetObject"
    "PyWeakref_GET_OBJECT"
    "PyList_GET_ITEM"
    "_PyDict_GetItemStringWithError"
    "PySequence_Fast"
)

# Find all C/C++ source files in the repo
ALL_FILES=$(find numpy -type f \( -name "*.c" -o -name "*.h" -o -name "*.c.src" -o -name "*.cpp" \) ! -path "*/pythoncapi-compat/*")

# For debugging: print out file count
echo "Scanning $(echo "$ALL_FILES" | wc -l) C/C++ source files..."

# Prepare a result file
mkdir -p .tmp
OUTPUT=$(mktemp .tmp/c_api_usage_report.XXXXXX)
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

        # Check each match for 'noqa' and comment filtering
        if [[ -n "$matches" ]]; then
            while IFS= read -r line; do

                line_number=$(cut -d: -f1 <<< "$line")
                code_line=$(cut -d: -f2- <<< "$line")

                # Skip if line contains noqa
                if [[ "$code_line" == *"noqa: borrowed-ref OK"* || "$code_line" == *"noqa: borrowed-ref - manual fix needed"* ]]; then
                    continue
                fi

                # Skip if line is fully commented with //
                if [[ "$code_line" =~ ^[[:space:]]*// ]]; then
                    continue
                fi

                # Skip if match is inside a block comment
                # Check previous lines up to this one for an opening /* without closing */
                in_block_comment=0
                while IFS= read -r prev_line && [[ $((--line_number)) -gt 0 ]]; do
                    [[ "$prev_line" =~ /\* ]] && in_block_comment=1
                    [[ "$prev_line" =~ \*/ ]] && in_block_comment=0
                done < "$file"

                if [[ "$in_block_comment" -eq 1 ]]; then
                    continue
                fi

                
                echo "Found suspcious call to $func in file: $file" >> "$OUTPUT"
                echo " -> $line" >> "$OUTPUT"
                echo "Recommendation:" >> "$OUTPUT"
                echo "If this use is intentional and safe, add '// noqa: borrowed-ref OK' on the same line to silence this warning." >> "$OUTPUT"
                echo "Otherwise, consider replacing $func with a thread-safe API function." >> "$OUTPUT"
                echo "" >> "$OUTPUT"
                FAIL=1
            done <<< "$matches"
        fi
    done
done

if [[ $FAIL -eq 1 ]]; then
    echo "C API borrow-ref linter found issues."
else
    echo "C API borrow-ref linter found no issues." > $OUTPUT
fi

cat "$OUTPUT"
exit "$FAIL"
