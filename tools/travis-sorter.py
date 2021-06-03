#!/usr/bin/env python3
"""
Run with a repo/build number or list of Travis CI build times to show the optimal build
order to run faster and make full use of all available parallel build jobs.

Requires the Travis Client CLI

https://github.com/travis-ci/travis.rb#installation

# Example

$ # Check build 22 of hugovk/numpy, and skip the first job (it's a single stage)
$ travis-sorter.py hugovk/numpy 22 --skip 1
travis show -r hugovk/numpy 22
[8, 7, 8, 10, 9, 18, 8, 11, 8, 10, 8, 8, 17, 8, 26]
[7, 8, 10, 9, 18, 8, 11, 8, 10, 8, 8, 17, 8, 26]
Before:

ID Duration in mins
 1 *******
 2 ********
 3 **********
 4 *********
 5 ******************
 6        ********
 7         ***********
 8          ********
 9           **********
10                ********
11                  ********
12                   *****************
13                    ********
14                     **************************
End: 46
   ----------------------------------------------

After:

ID Duration in mins
14 **************************
 5 ******************
12 *****************
 7 ***********
 3 **********
 9           **********
 4            *********
 2                  ********
 6                   ********
 8                     ********
10                     ********
11                          ********
13                           ********
 1                           *******
End: 34
   ----------------------------------

# Example

$ python travis-sorter.py 4 4 4 4 4 12 19

Before:

****
****
****
****
****
    ************
    *******************
12345678901234567890123 = 23 minutes

After:

*******************
************
****
****
****
    ****
    ****
1234567890123456789 = 19 minutes
"""
import argparse
import re
import subprocess
import sys

count = 1


def summarise(jobs):
    end = 0
    print("ID Duration in mins")
    for job in jobs:
        before = " " * job.started
        active = "*" * job.length
        print("{:2d} {}{}".format(job.id, before, active))
        if job.started + job.length > end:
            end = job.started + job.length
    # for job in jobs:
    #     print(job)
    print("End:", end)
    print("   " + "-" * end)


class Job:
    def __init__(self, length):
        global count
        self.id = count
        count += 1
        self.length = length
        self.started = -1
        self.status = "not started"
        self.ended = False

    def __str__(self):
        return "{}\tLength: {}\tStarted: {}\tEnded: {}".format(
            self.id, self.length, self.started, self.ended
        )


def count_status(jobs, status):
    number = 0
    for job in jobs:
        if job.status == status:
            number += 1
    return number


def simulate(jobs, limit):

    time = 0

    # summarise(jobs)

    while True:
        # Check if any have ended
        for job in jobs:
            if job.status == "active":
                if time >= job.started + job.length:
                    # print("{}/{} Finished:".format(count_status(jobs, "active"), limit))
                    job.ended = time
                    job.status = "finished"
                    # print(job)

        # Check if any can start
        for job in jobs:
            if job.status == "not started":
                if count_status(jobs, "active") < limit:
                    # print("{}/{} Starting:".format(count_status(jobs, "active"), limit))
                    job.started = time
                    job.status = "active"
                    # print(job)

        time += 1

        # Exit loop?
        if count_status(jobs, "finished") == len(jobs):
            break

    summarise(jobs)


def do_thing(repo, number):
    cmd = f"travis show -r {repo} {number or ''}"
    # cmd = f"travis show --com -r {repo} {number or ''}"
    print(cmd)

    exitcode = 0
    # For offline testing
    output = """Build #4:  Upgrade Python syntax with pyupgrade https://github.com/asottile/pyupgrade
State:         passed
Type:          push
Branch:        add-3.7
Compare URL:   https://github.com/hugovk/diff-cover/compare/4ae7cf97c6fa...7eeddb300175
Duration:      16 min 7 sec
Started:       2018-10-17 19:03:01
Finished:      2018-10-17 19:09:53

#4.1 passed:     1 min          os: linux, env: TOXENV=py27, python: 2.7
#4.2 passed:     1 min 43 sec   os: linux, env: TOXENV=py34, python: 3.4
#4.3 passed:     1 min 52 sec   os: linux, env: TOXENV=py35, python: 3.5
#4.4 passed:     1 min 38 sec   os: linux, env: TOXENV=py36, python: 3.6
#4.5 passed:     1 min 47 sec   os: linux, env: TOXENV=py37, python: 3.7
#4.6 passed:     4 min 35 sec   os: linux, env: TOXENV=pypy, python: pypy
#4.7 passed:     3 min 17 sec   os: linux, env: TOXENV=pypy3, python: pypy3"""

    # For offline testing
    output = """Build #9:  :arrows_clockwise: [EngCom] Public Pull Requests - 2.3-develop
State:         errored
Type:          push
Branch:        2.3-develop
Compare URL:   https://github.com/hugovk/magento2/compare/80469a61e061...77af5d65ef4f
Duration:      4 hrs 12 min 13 sec
Started:       2018-10-27 17:50:51
Finished:      2018-10-27 18:54:14

#9.1 passed:     3 min 30 sec   os: linux, env: TEST_SUITE=unit, php: 7.1
#9.2 passed:     3 min 35 sec   os: linux, env: TEST_SUITE=unit, php: 7.2
#9.3 passed:     3 min 41 sec   os: linux, env: TEST_SUITE=static, php: 7.2
#9.4 passed:     8 min 48 sec   os: linux, env: TEST_SUITE=js GRUNT_COMMAND=spec, php: 7.2
#9.5 passed:     3 min 24 sec   os: linux, env: TEST_SUITE=js GRUNT_COMMAND=static, php: 7.2
#9.6 errored:    50 min         os: linux, env: TEST_SUITE=integration INTEGRATION_INDEX=1, php: 7.1
#9.7 passed:     49 min 25 sec  os: linux, env: TEST_SUITE=integration INTEGRATION_INDEX=1, php: 7.2
#9.8 passed:     31 min 54 sec  os: linux, env: TEST_SUITE=integration INTEGRATION_INDEX=2, php: 7.1
#9.9 passed:     31 min 24 sec  os: linux, env: TEST_SUITE=integration INTEGRATION_INDEX=2, php: 7.2
#9.10 passed:    27 min 23 sec  os: linux, env: TEST_SUITE=integration INTEGRATION_INDEX=3, php: 7.1
#9.11 passed:    26 min 9 sec   os: linux, env: TEST_SUITE=integration INTEGRATION_INDEX=3, php: 7.2
#9.12 passed:    13 min         os: linux, env: TEST_SUITE=functional, php: 7.2"""

    # Real use
    exitcode, output = subprocess.getstatusoutput(cmd)

    # print(exitcode)
    # print(output)
    if exitcode != 0:
        print(output)
        sys.exit(exitcode)

    minutes = []
    matches = re.findall(r"(pass|fail|error)ed.* (\d+) min (\d+)? ", output)
    for match in matches:
        status, m, s = match
        s = 0 if s == "" else int(s)
        s += int(m) * 60
        minutes.append(round(s / 60))

    # print(minutes)
    return minutes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Either give minutes for --jobs (3 5 3 2 5), "
        "or --repo slug (hugovk/test) and build --number (5)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        nargs="+",
        help="Either: times for each build job (minutes), "
        "or an org/repo slug and optionally build number",
    )
    parser.add_argument(
        "-l", "--limit", type=int, default=5, help="Concurrent jobs limit"
    )
    parser.add_argument(
        "-s", "--skip", type=int, default=0, help="Skip X jobs at the start"
    )
    args = parser.parse_args()

    # If all ints
    try:
        for x in args.input:
            int(x)
        job_times = args.input
    except ValueError:
        try:
            number = args.input[1]
        except IndexError:
            number = None
        job_times = do_thing(args.input[0], number)

    job_times = job_times[args.skip :]
    # print(job_times)

    print("Before:")
    print()

    jobs = []
    for job_time in job_times:
        job = Job(job_time)
        jobs.append(job)

    simulate(jobs, args.limit)

    print()
    print("After:")
    print()

    # Sort with longest first
    jobs.sort(key=lambda job: job.length, reverse=True)
    # Reset status
    for job in jobs:
        job.status = "not started"

    simulate(jobs, args.limit)
