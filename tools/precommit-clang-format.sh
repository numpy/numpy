#!/bin/bash

# This script wraps git-clang-format (installed via a python wheel in
# pre-commit) and runs it with our desired arguments, with desired exit codes.
# git-clang-format only formats the changed blocks between two commits.
#
# We need to provide the tool with the proper parameters to apply the diff if
# possible (i.e. it is comparing to our HEAD), and if not, just display the diff
# and exit error if it exists (when not pointing at HEAD, i.e. uneditable)
#
# This script usage allows for comparing two diffs from arbritrary points in
# time easily.
#
# It can also be fake run without pre-commit: PRE_COMMIT_FROM_REF=someref
# PRE_COMMIT_TO_REF=someref ./tools/precommit-clang-format.sh file1.c file2.c
# PRE_COMMIT_TO_REF can be omitted to use HEAD
#
# This is known to work with clang-format==14.0.3 available on PyPi. If their
# "no modified files to format" message or return code scheme changes, this will
# need to be updated.

set -o errexit
set -o nounset
set -o pipefail

FORMAT_ARGS="--style file"

# nounset for sanity checks of our variables, but we want these to be unset
PRE_COMMIT_TO_REF=${PRE_COMMIT_TO_REF:-}
PRE_COMMIT_FROM_REF=${PRE_COMMIT_FROM_REF:-}

RED_LIGHT='\033[1;31m'
CYAN_LIGHT='\033[1;36m'
GREEN_LIGHT='\033[1;32m'
YELLOW='\033[1;33m'
END_C='\033[0m'

echo "Starting git-clang-format launcher"

# Check if we have a TO_REF target and if so, enter this block
if [[ -n "${PRE_COMMIT_TO_REF}" ]]; then

    # Get the hashes of the specified revs for easy comparison
    # Need a FROM_REF in this case so it's OK if the rev-parse fails
    to_ref=$(git rev-parse --short "$PRE_COMMIT_TO_REF")
    head_ref=$(git rev-parse --short HEAD)


    # If we are not comparing to HEAD, changes cannot be safely applied; all we
    # can do is print the diff and exit appropriately
    if [[ "$to_ref" != "$head_ref" ]]; then
        run_cmd="git-clang-format $FORMAT_ARGS --diff $PRE_COMMIT_FROM_REF $PRE_COMMIT_TO_REF -- $*"
        from_ref=$(git rev-parse --short "$PRE_COMMIT_FROM_REF")

        echo -e "\n${CYAN_LIGHT}Comparing \"${from_ref}\" to \"${to_ref}\"${END_C}"
        echo -e "${YELLOW}Cannot safely apply changes; \"to\" ref \"${to_ref}\" is not HEAD.${END_C}"
        echo -e "${YELLOW}I'll just show you the diff instead. Running... \n> ${run_cmd}${END_C}\n"

        output=$(eval "$run_cmd")
        exitcode=$?

        # Success! No diff
        if [[ $output = "no modified files to format" ]]; then
            echo -e "${GREEN_LIGHT}Everything looks OK; no work for me here${END_C}"
            exit 0
        fi

        echo "$output"
        echo -e "\n"

        if [[ exitcode -eq 0 ]]; then
            # git-clang-format exits 0 even if a diff is created, so we need to
            # make sure this script fails
            echo -e "${RED_LIGHT}Changes between the two revisions are not properely formatted, but I'm too"
            echo -e "scared to apply formatting to target refs that are not HEAD. I quit.${END_C}"
            exit 1
        fi

        # If we're here, git-clang-format exited nonzero which is some kind of error
        echo -e "${RED_LIGHT}Call the doctor! Something went wrong with git-clang-format${END_C}"
        exit $exitcode
    fi
fi

# Otherwise, our target ref is HEAD and the tool can edit it
run_cmd="git-clang-format $FORMAT_ARGS $PRE_COMMIT_FROM_REF -- $*"

echo -e "${CYAN_LIGHT}Target ref is HEAD, running...\n> ${run_cmd}${END_C}\n"

# Collect our output, print it and act upon it
output=$(eval "$run_cmd")
exitcode=$?

echo -e "${output}\n"

if [[ $output = "no modified files to format" ]]; then
    echo -e "${GREEN_LIGHT}Everything looks OK; no work for me here${END_C}"
else
    echo -e "${YELLOW}OK, I formatted that nice for you. Retry your commit now.${END_C}"
fi

exit $exitcode
