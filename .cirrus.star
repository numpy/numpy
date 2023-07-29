# The guide to programming cirrus-ci tasks using starlark is found at
# https://cirrus-ci.org/guide/programming-tasks/
#
# In this simple starlark script we simply check conditions for whether
# a CI run should go ahead. If the conditions are met, then we just
# return the yaml containing the tasks to be run.

load("cirrus", "env", "fs", "http")

def main(ctx):
    ######################################################################
    # Should wheels be built?
    # Only test on the numpy/numpy repository
    ######################################################################

    if env.get("CIRRUS_REPO_FULL_NAME") != "numpy/numpy":
        return []

    # only run the wheels entry on a cron job
    if env.get("CIRRUS_CRON", "") == "nightly":
        return fs.read("tools/ci/cirrus_wheels.yml")

    # Obtain commit message for the event. Unfortunately CIRRUS_CHANGE_MESSAGE
    # only contains the actual commit message on a non-PR trigger event.
    # For a PR event it contains the PR title and description.
    SHA = env.get("CIRRUS_CHANGE_IN_REPO")
    url = "https://api.github.com/repos/numpy/numpy/git/commits/" + SHA
    dct = http.get(url).json()
    # if "[wheel build]" in dct["message"]:
    #     return fs.read("ci/cirrus_wheels.yml")

    if "[skip cirrus]" in dct["message"] or "[skip ci]" in dct["message"]:
        return []

    # add extra jobs to the cirrus run by += adding to config
    config = fs.read("tools/ci/cirrus_wheels.yml")
    config += fs.read("tools/ci/cirrus_macosx_arm64.yml")

    return config
