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

    commit_msg = dct["message"]
    if "[skip cirrus]" in  commit_msg or "[skip ci]" in commit_msg:
        return []

    wheel = False
    labels = env.get("CIRRUS_PR_LABELS", "")
    pr_number = env.get("CIRRUS_PR", "-1")
    tag = env.get("CIRRUS_TAG", "")

    if "[wheel build]" in commit_msg:
        wheel = True

    # if int(pr_number) > 0 and ("14 - Release" in labels or "36 - Build" in labels):
    #     wheel = True

    if tag.startswith("v") and "dev0" not in tag:
        wheel = True

    if wheel:
        return fs.read("tools/ci/cirrus_wheels.yml")

    if int(pr_number) < 0:
        return []

    return fs.read("tools/ci/cirrus_arm.yml")
