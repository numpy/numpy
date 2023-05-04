NumPy security
==============

Security issues can be reported privately as described in the project README
and when opening a `new issue on the issue tracker <https://github.com/numpy/numpy/issues/new/choose>`_.
The `Python security reporting guidelines <https://www.python.org/dev/security/>`_
are a good resource and its notes apply also to NumPy.

NumPy's maintainers are not security experts.  However, we are conscientious
about security and experts of both the NumPy codebase and how it's used.
Please do notify us before creating security advisories against NumPy as
we are happy to prioritize issues or help with assessing the severity of a bug.
A security advisory we are not aware of beforehand can lead to a lot of work
for all involved parties.


Advice for using NumPy on untrusted data
----------------------------------------

A user who can freely execute NumPy (or Python) functions must be considered
to have the same privilege as the process/Python interpreter.

That said, NumPy should be generally safe to use on *data* provided by
unprivileged users and read through safe API functions (e.g. loaded from a
text file or ``.npy`` file without pickle support).
Malicious *values* or *data sizes* should never lead to privilege escalation. 

The following points may be useful or should be noted when working with
untrusted data:

* Exhausting memory can result in an out-of-memory kill, which is a possible
  denial of service attack.  Possible causes could be:

  * Functions reading text files, which may require much more memory than
    the original input file size.
  * If users can create arbitrarily shaped arrays, NumPy's broadcasting means
    that intermediate or result arrays can be much larger than the inputs.

* NumPy structured dtypes allow for a large amount of complexity.  Fortunately,
  most code fails gracefully when a structured dtype is provided unexpectedly.
  However, code should either disallow untrusted users to provide these
  (e.g. via ``.npy`` files) or carefully check the fields included for
  nested structured/subarray dtypes.

* Passing on user input should generally be considered unsafe
  (except for the data being read).
  An example would be ``np.dtype(user_string)`` or ``dtype=user_string``.

* The speed of operations can depend on values and memory order can lead to
  larger temporary memory use and slower execution.
  This means that operations may be significantly slower or use more memory
  compared to simple test cases.

* When reading data, consider enforcing a specific shape (e.g. one dimensional)
  or dtype such as ``float64``, ``float32``, or ``int64`` to reduce complexity.

When working with non-trivial untrusted data, it is advisable to sandbox the
analysis to guard against potential privilege escalation.
This is especially advisable if further libraries based on NumPy are used since
these add additional complexity and potential security issues.

