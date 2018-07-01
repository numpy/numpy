=============================================
NEP 14 — Plan for dropping Python 2.7 support
=============================================

:Status: Accepted
:Resolution: https://mail.python.org/pipermail/numpy-discussion/2017-November/077419.html

The Python core team plans to stop supporting Python 2 in 2020. The NumPy
project has supported both Python 2 and Python 3 in parallel since 2010, and
has found that supporting Python 2 is an increasing burden on our limited
resources; thus, we plan to eventually drop Python 2 support as well. Now that
we're entering the final years of community-supported Python 2, the NumPy
project wants to clarify our plans, with the goal of to helping our downstream
ecosystem make plans and accomplish the transition with as little disruption as
possible.

Our current plan is as follows.

Until **December 31, 2018**, all NumPy releases will fully support both
Python2 and Python3.

Starting on **January 1, 2019**, any new feature releases will support only
Python3.

The last Python2 supporting release will be designated as a long term support
(LTS) release, meaning that we will continue to merge bug fixes and make bug
fix releases for a longer period than usual.  Specifically, it will be
supported by the community until **December 31, 2019**.

On **January 1, 2020** we will raise a toast to Python2, and community support
for the last Python2 supporting release will come to an end. However, it will
continue to be available on PyPI indefinitely, and if any commercial vendors
wish to extend the LTS support past this point then we are open to letting them
use the LTS branch in the official NumPy repository to coordinate that.

If you are a NumPy user who requires ongoing Python2 support in 2020 or later,
then please contact your vendor. If you are a vendor who wishes to continue to
support NumPy on Python2 in 2020+, please get in touch; ideally we'd like you
to get involved in maintaining the LTS before it actually hits end of life so
that we can make a clean handoff.

To minimize disruption, running ``pip install numpy`` on Python 2 will continue
to give the last working release in perpetuity, but after January 1, 2019 it
may not contain the latest features, and after January 1, 2020 it may not
contain the latest bug fixes.

For more information on the scientific Python ecosystem's transition
to Python3 only, see the python3-statement_.

For more information on porting your code to run on Python 3, see the
python3-howto_.

.. _python3-statement: https://python3statement.org/

.. _python3-howto: https://docs.python.org/3/howto/pyporting.html
