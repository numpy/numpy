====================================================
NEP 26 — Summary of Missing Data NEPs and discussion
====================================================

:Author: Mark Wiebe <mwwiebe@gmail.com>, Nathaniel J. Smith <njs@pobox.com>
:Status: Deferred
:Type: Standards Track
:Created: 2012-04-22

*Context*: this NEP was written as summary of the large number of discussions
and proposals (`NEP 12`_, `NEP 24`_, `NEP 25`_), regarding missing data
functionality.

The debate about how NumPy should handle missing data, a subject with
many preexisting approaches, requirements, and conventions, has been long and
contentious. There has been more than one proposal for how to implement
support into NumPy, and there is a testable implementation which is
merged into NumPy's current master. The vast number of emails and differing
points of view has made it difficult for interested parties to understand
the issues and be comfortable with the direction NumPy is going.

Here is our (Mark and Nathaniel's) attempt to summarize the
problem, proposals, and points of agreement/disagreement in a single
place, to help the community move towards consensus.

The NumPy developers' problem
=============================

For this discussion, "missing data" means array elements
which can be indexed (e.g. A[3] in an array A with shape (5,)),
but have, in some sense, no value.

It does not refer to compressed or sparse storage techniques where
the value for A[3] is not actually stored in memory, but still has a
well-defined value like 0.

This is still vague, and to create an actual implementation,
it is necessary to answer such questions as:

* What values are computed when doing element-wise ufuncs.
* What values are computed when doing reductions.
* Whether the storage for an element gets overwritten when marking
  that value missing.
* Whether computations resulting in NaN automatically treat in the
  same way as a missing value.
* Whether one interacts with missing values using a placeholder object
  (e.g. called "NA" or "masked"), or through a separate boolean array.
* Whether there is such a thing as an array object that cannot hold
  missing array elements.
* How the (C and Python) API is expressed, in terms of dtypes,
  masks, and other constructs.
* If we decide to answer some of these questions in multiple ways,
  then that creates the question of whether that requires multiple
  systems, and if so how they should interact.

There's clearly a very large space of missing-data APIs that *could*
be implemented. There is likely at least one user, somewhere, who
would find any possible implementation to be just the thing they
need to solve some problem. On the other hand, much of NumPy's power
and clarity comes from having a small number of orthogonal concepts,
such as strided arrays, flexible indexing, broadcasting, and ufuncs,
and we'd like to preserve that simplicity.

There has been dissatisfaction among several major groups of NumPy users
about the existing status quo of missing data support. In particular,
neither the numpy.ma component nor use of floating-point NaNs as a
missing data signal fully satisfy the performance requirements and
ease of use for these users. The example of R, where missing data
is treated via an NA placeholder and is deeply integrated into all
computation, is where many of these users point to indicate what
functionality they would like. Doing a deep integration of missing
data like in R must be considered carefully, it must be clear it
is not being done in a way which sacrifices existing performance
or functionality.

Our problem is, how can we choose some incremental additions to
NumPy that will make a large class of users happy, be
reasonably elegant, complement the existing design, and that we're
comfortable we won't regret being stuck with in the long term.

Prior art
=========

So a major (maybe *the* major) problem is figuring out how ambitious
the project to add missing data support to NumPy should be, and which
kinds of problems are in scope. Let's start with the
best understood situation where "missing data" comes into play:

"Statistical missing data"
--------------------------

In statistics, social science, etc., "missing data" is a term of art
referring to a specific (but extremely common and important)
situation: we have tried to gather some measurements according to some
scheme, but some of these measurements are missing. For example, if we
have a table listing the height, age, and income of a number of
individuals, but one person did not provide their income, then we need
some way to represent this::

  Person | Height | Age | Income
  ------------------------------
     1   |   63   | 25  | 15000
     2   |   58   | 32  | <missing>
     3   |   71   | 45  | 30000

The traditional way is to record that income as, say, "-99", and
document this in the README along with the data set. Then, you have to
remember to check for and handle such incomes specially; if you
forget, you'll get superficially reasonable but completely incorrect
results, like calculating the average income on this data set as
14967. If you're in one of these fields, then such missing-ness is
routine and inescapable, and if you use the "-99" approach then it's a
pitfall you have to remember to check for explicitly on literally
*every* calculation you ever do. This is, obviously, an unpleasant way
to live.

Let's call this situation the "statistical missing data" situation,
just to have a convenient handle for it. (As mentioned, practitioners
just call this "missing data", and what to do about it is literally an
entire sub-field of statistics; if you google "missing data" then
every reference is on how to handle it.) NumPy isn't going to do
automatic imputation or anything like that, but it could help a great
deal by providing some standard way to at least represent data which
is missing in this sense.

The main prior art for how this could be done comes from the S/S+/R
family of languages. Their strategy is, for each type they support,
to define a special value called "NA". (For ints this is INT_MAX,
for floats it's a special NaN value that's distinguishable from
other NaNs, ...) Then, they arrange that in computations, this
value has a special semantics that we will call "NA semantics".

NA Semantics
------------

The idea of NA semantics is that any computations involving NA
values should be consistent with what would have happened if we
had known the correct value.

For example, let's say we want to compute the mean income, how might
we do this? One way would be to just ignore the missing entry, and
compute the mean of the remaining entries. This gives us (15000 +
30000)/2, or 22500.

Is this result consistent with discovering the income of person 2?
Let's say we find out that person 2's income is 50000. This means
the correct answer is (15000 + 50000 + 30000)/3, or 31666.67,
indicating clearly that it is not consistent. Therefore, the mean
income is NA, i.e. a specific number whose value we are unable
to compute.

This motivates the following rules, which are how R implements NA:

Assignment:
  NA values are understood to represent specific
  unknown values, and thus should have value-like semantics with
  respect to assignment and other basic data manipulation
  operations. Code which does not actually look at the values involved
  should work the same regardless of whether some of them are
  missing. For example, one might write::

    income[:] = income[np.argsort(height)]

  to perform an in-place sort of the ``income`` array, and know that
  the shortest person's income would end up being first. It turns out
  that the shortest person's income is not known, so the array should
  end up being ``[NA, 15000, 30000]``, but there's nothing
  special about NAness here.

Propagation:
  In the example above, we concluded that an operation like ``mean``
  should produce NA when one of its data values was NA.
  If you ask me, "what is 3 plus x?", then my only possible answer is
  "I don't know what x is, so I don't know what 3 + x is either". NA
  means "I don't know", so 3 + NA is NA.

  This is important for safety when analyzing data: missing data often
  requires special handling for correctness -- the fact that you are
  missing information might mean that something you wanted to compute
  cannot actually be computed, and there are whole books written on
  how to compensate in various situations. Plus, it's easy to not
  realize that you have missing data, and write code that assumes you
  have all the data. Such code should not silently produce the wrong
  answer.

  There is an important exception to characterizing this as propagation,
  in the case of boolean values. Consider the calculation::

    v = np.any([False, False, NA, True])

  If we strictly propagate, ``v`` will become NA. However, no
  matter whether we place True or False into the third array position,
  ``v`` will then get the value True. The answer to the question
  "Is the result True consistent with later discovering the value
  that was missing?" is yes, so it is reasonable to not propagate here,
  and instead return the value True. This is what R does::

    > any(c(F, F, NA, T))
    [1] TRUE
    > any(c(F, F, NA, F))
    [1] NA

Other:
  NaN and NA are conceptually distinct. 0.0/0.0 is not a mysterious,
  unknown value -- it's defined to be NaN by IEEE floating point, Not
  a Number. NAs are numbers (or strings, or whatever), just unknown
  ones. Another small but important difference is that in Python, ``if
  NaN: ...`` treats NaN as True (NaN is "truthy"); but ``if NA: ...``
  would be an error.

  In R, all reduction operations implement an alternative semantics,
  activated by passing a special argument (``na.rm=TRUE`` in R).
  ``sum(a)`` means "give me the sum of all the
  values" (which is NA if some of the values are NA);
  ``sum(a, na.rm=True)`` means "give me the sum of all the non-NA
  values".

Other prior art
---------------

Once we move beyond the "statistical missing data" case, the correct
behavior for missing data becomes less clearly defined. There are many
cases where specific elements are singled out to be treated specially
or excluded from computations, and these could often be conceptualized
as involving 'missing data' in some sense.

In image processing, it's common to use a single image together with
one or more boolean masks to e.g. composite subsets of an image. As
Joe Harrington pointed out on the list, in the context of processing
astronomical images, it's also common to generalize to a
floating-point valued mask, or alpha channel, to indicate degrees of
"missingness". We think this is out of scope for the present design,
but it is an important use case, and ideally NumPy should support
natural ways of manipulating such data.

After R, numpy.ma is probably the most mature source of
experience on missing-data-related APIs. Its design is quite different
from R; it uses different semantics -- reductions skip masked values
by default and NaNs convert to masked -- and it uses a different
storage strategy via a separate mask. While it seems to be generally
considered sub-optimal for general use, it's hard to pin down whether
this is because the API is immature but basically good, or the API
is fundamentally broken, or the API is great but the code should be
faster, or what. We looked at some of those users to try and get a
better idea.

Matplotlib is perhaps the best known package to rely on numpy.ma. It
seems to use it in two ways. One is as a way for users to indicate
what data is missing when passing it to be graphed. (Other ways are
also supported, e.g., passing in NaN values gives the same result.) In
this regard, matplotlib treats np.ma.masked and NaN values in the same way
that R's plotting routines handle NA and NaN values. For these purposes,
matplotlib doesn't really care what semantics or storage strategy is
used for missing data.

Internally, matplotlib uses numpy.ma arrays to store and pass around
separately computed boolean masks containing 'validity' information
for each input array in a cheap and non-destructive fashion. Mark's
impression from some shallow code review is that mostly it works
directly with the data and mask attributes of the masked arrays,
not extensively using the particular computational semantics of
numpy.ma. So, for this usage they do rely on the non-destructive
mask-based storage, but this doesn't say much about what semantics
are needed.

Paul Hobson `posted some code`__ on the list that uses numpy.ma for
storing arrays of contaminant concentration measurements. Here the
mask indicates whether the corresponding number represents an actual
measurement, or just the estimated detection limit for a concentration
which was too small to detect. Nathaniel's impression from reading
through this code is that it also mostly uses the .data and .mask
attributes in preference to performing operations on the MaskedArray
directly.

__ https://mail.scipy.org/pipermail/numpy-discussion/2012-April/061743.html

So, these examples make it clear that there is demand for a convenient
way to keep a data array and a mask array (or even a floating point
array) bundled up together and "aligned". But they don't tell us much
about what semantics the resulting object should have with respect to
ufuncs and friends.

Semantics, storage, API, oh my!
===============================

We think it's useful to draw a clear line between use cases,
semantics, and storage. Use cases are situations that users encounter,
regardless of what NumPy does; they're the focus of the previous
section. When we say *semantics*, we mean the result of different
operations as viewed from the Python level without regard to the
underlying implementation.

*NA semantics* are the ones described above and used by R::

  1 + NA = NA
  sum([1, 2, NA]) = NA
  NA | False = NA
  NA | True = True

With ``na.rm=TRUE`` or ``skipNA=True``, this switches to::

  1 + NA = illegal # in R, only reductions take na.rm argument
  sum([1, 2, NA], skipNA=True) = 3

There's also been discussion of what we'll call *ignore
semantics*. These are somewhat underdefined::

  sum([1, 2, IGNORED]) = 3
  # Several options here:
  1 + IGNORED = 1
  #  or
  1 + IGNORED = <leaves output array untouched>
  #  or
  1 + IGNORED = IGNORED

The numpy.ma semantics are::

  sum([1, 2, masked]) = 3
  1 + masked = masked

If either NA or ignore semantics are implemented with masks, then there
is a choice of what should be done to the value in the storage
for an array element which gets assigned a missing value. Three
possibilities are:

* Leave that memory untouched (the choice made in the NEP).
* Do the calculation with the values independently of the mask
  (perhaps the most useful option for Paul Hobson's use-case above).
* Copy whatever value is stored behind the input missing value into
  the output (this is what numpy.ma does. Even that is ambiguous in
  the case of ``masked + masked`` -- in this case numpy.ma copies the
  value stored behind the leftmost masked value).

When we talk about *storage*, we mean the debate about whether missing
values should be represented by designating a particular value of the
underlying data-type (the *bitpattern dtype* option, as used in R), or
by using a separate *mask* stored alongside the data itself.

For mask-based storage, there is also an important question about what
the API looks like for accessing the mask, modifying the mask, and
"peeking behind" the mask.

Designs that have been proposed
===============================

One option is to just copy R, by implementing a mechanism whereby
dtypes can arrange for certain bitpatterns to be given NA semantics.

One option is to copy numpy.ma closely, but with a more optimized
implementation. (Or to simply optimize the existing implementation.)

One option is that described in `NEP 12`_, for which an implementation
of mask-based missing data exists. This system is roughly:

* There is both bitpattern and mask-based missing data, and both
  have identical interoperable NA semantics.
* Masks are modified by assigning np.NA or values to array elements.
  The way to peek behind the mask or to unmask values is to keep a
  view of the array that shares the data pointer but not the mask pointer.
* Mark would like to add a way to access and manipulate the mask more
  directly, to be used in addition to this view-based API.
* If an array has both a bitpattern dtype and a mask, then assigning
  np.NA writes to the mask, rather than to the array itself. Writing
  a bitpattern NA to an array which supports both requires accessing
  the data by "peeking under the mask".

Another option is that described in `NEP 24`_, which is to implement
bitpattern dtypes with NA semantics for the "statistical missing data"
use case, and to also implement a totally independent API for masked
arrays with ignore semantics and all mask manipulation done explicitly
through a .mask attribute.

Another option would be to define a minimalist aligned array container
that holds multiple arrays and that can be used to pass them around
together. It would support indexing (to help with the common problem
of wanting to subset several arrays together without their becoming
unaligned), but all arithmetic etc. would be done by accessing the
underlying arrays directly via attributes. The "prior art" discussion
above suggests that something like this holding a .data and a .mask
array might actually be solve a number of people's problems without
requiring any major architectural changes to NumPy. This is similar to
a structured array, but with each field in a separately stored array
instead of packed together.

Several people have suggested that there should be a single system
that has multiple missing values that each have different semantics,
e.g., a MISSING value that has NA semantics, and a separate IGNORED
value that has ignored semantics.

None of these options are necessarily exclusive.

The debate
==========

We both are dubious of using ignored semantics as a default missing
data behavior. **Nathaniel** likes NA semantics because he is most
interested in the "statistical missing data" use case, and NA semantics
are exactly right for that. **Mark** isn't as interested in that use
case in particular, but he likes the NA computational abstraction
because it is unambiguous and well-defined in all cases, and has a
lot of existing experience to draw from.

What **Nathaniel** thinks, overall:

* The "statistical missing data" use case is clear and compelling; the
  other use cases certainly deserve our attention, but it's hard to say what
  they *are* exactly yet, or even if the best way to support them is
  by extending the ndarray object.
* The "statistical missing data" use case is best served by an R-style
  system that uses bitpattern storage to implement NA semantics. The
  main advantage of bitpattern storage for this use case is that it
  avoids the extra memory and speed overhead of storing and checking a
  mask (especially for the common case of floating point data, where
  some tricks with NaNs allow us to effectively hardware-accelerate
  most NA operations). These concerns alone appears to make a
  mask-based implementation unacceptable to many NA users,
  particularly in areas like neuroscience (where memory is tight) or
  financial modeling (where milliseconds are critical). In addition,
  the bit-pattern approach is less confusing conceptually (e.g.,
  assignment really is just assignment, no magic going on behind the
  curtain), and it's possible to have in-memory compatibility with R
  for inter-language calls via rpy2.  The main disadvantage of the
  bitpattern approach is the need to give up a value to represent NA,
  but this is not an issue for the most important data types (float,
  bool, strings, enums, objects); really, only integers are
  affected. And even for integers, giving up a value doesn't really
  matter for statistical problems. (Occupy Wall Street
  notwithstanding, no-one's income is 2**63 - 1. And if it were, we'd
  be switching to floats anyway to avoid overflow.)
* Adding new dtypes requires some cooperation with the ufunc and
  casting machinery, but doesn't require any architectural changes or
  violations of NumPy's current orthogonality.
* His impression from the mailing list discussion, esp. the `"what can
  we agree on?" thread`__, is that many numpy.ma users specifically
  like the combination of masked storage, the mask being easily
  accessible through the API, and ignored semantics. He could be
  wrong, of course. But he cannot remember seeing anybody besides Mark
  advocate for the specific combination of masked storage and NA
  semantics, which makes him nervous.

  __ http://thread.gmane.org/gmane.comp.python.numeric.general/46704
* Also, he personally is not very happy with the idea of having two
  storage implementations that are almost-but-not-quite identical at
  the Python level. While there likely are people who would like to
  temporarily pretend that certain data is "statistically missing
  data" without making a copy of their array, it's not at all clear
  that they outnumber the people who would like to use bitpatterns and
  masks simultaneously for distinct purposes. And honestly he'd like
  to be able to just ignore masks if he wants and stick to
  bitpatterns, which isn't possible if they're coupled together
  tightly in the API.  So he would say the jury is still very much out
  on whether this aspect of the NEP design is an advantage or a
  disadvantage. (Certainly he's never heard of any R users complaining
  that they really wish they had an option of making a different
  trade-off here.)
* R's NA support is a `headline feature`__ and its target audience
  consider it a compelling advantage over other platforms like Matlab
  or Python. Working with statistical missing data is very painful
  without platform support.

  __ http://www.sr.bham.ac.uk/~ajrs/R/why_R.html
* By comparison, we clearly have much more uncertainty about the use
  cases that require a mask-based implementation, and it doesn't seem
  like people will suffer too badly if they are forced for now to
  settle for using NumPy's excellent mask-based indexing, the new
  where= support, and even numpy.ma.
* Therefore, bitpatterns with NA semantics seem to meet the criteria
  of making a large class of users happy, in an elegant way, that fits
  into the original design, and where we can have reasonable certainty
  that we understand the problem and use cases well enough that we'll
  be happy with them in the long run. But no mask-based storage
  proposal does, yet.

What **Mark** thinks, overall:

* The idea of using NA semantics by default for missing data, inspired
  by the "statistical missing data" problem, is better than all the
  other default behaviors which were considered. This applies equally
  to the bitpattern and the masked approach.

* For NA-style functionality to get proper support by all NumPy
  features and eventually all third-party libraries, it needs to be
  in the core. How to correctly and efficiently handle missing data
  differs by algorithm, and if thinking about it is required to fully
  support NumPy, NA support will be broader and higher quality.

* At the same time, providing two different missing data interfaces,
  one for masks and one for bitpatterns, requires NumPy developers
  and third-party NumPy plugin developers to separately consider the
  question of what to do in either case, and do two additional
  implementations of their code. This complicates their job,
  and could lead to inconsistent support for missing data.

* Providing the ability to work with both masks and bitpatterns through
  the same C and Python programming interface makes missing data support
  cleanly orthogonal with all other NumPy features.

* There are many trade-offs of memory usage, performance, correctness, and
  flexibility between masks and bitpatterns. Providing support for both
  approaches allows users of NumPy to choose the approach which is
  most compatible with their way of thinking, or has characteristics
  which best match their use-case. Providing them through the same
  interface further allows them to try both with minimal effort, and
  choose the one which performs better or uses the least memory for
  their programs.

* Memory Usage

  * With bitpatterns, less memory is used for storing a single array
    containing some NAs.

  * With masks, less memory is used for storing multiple arrays that
    are identical except for the location of their NAs. (In this case a
    single data array can be re-used with multiple mask arrays;
    bitpattern NAs would need to copy the whole data array.)

* Performance

  * With bitpatterns, the floating point type can use native hardware
    operations, with nearly correct behavior. For fully correct floating
    point behavior and with other types, code must be written which
    specially tests for equality with the missing-data bitpattern.

  * With masks, there is always the overhead of accessing mask memory
    and testing its truth value. The implementation that currently exists
    has no performance tuning, so it is only good to judge a minimum
    performance level. Optimal mask-based code is in general going to
    be slower than optimal bitpattern-based code.

* Correctness

  * Bitpattern integer types must sacrifice a valid value to represent NA.
    For larger integer types, there are arguments that this is ok, but for
    8-bit types there is no reasonable choice. In the floating point case,
    if the performance of native floating point operations is chosen,
    there is a small inconsistency that NaN+NA and NA+NaN are different.
  * With masks, it works correctly in all cases.

* Generality

  * The bitpattern approach can work in a fully general way only when
    there is a specific value which can be given up from the
    data type. For IEEE floating point, a NaN is an obvious choice,
    and for booleans represented as a byte, there are plenty of choices.
    For integers, a valid value must be sacrificed to use this approach.
    Third-party dtypes which plug into NumPy will also have to
    make a bitpattern choice to support this system, something which
    may not always be possible.

  * The mask approach works universally with all data types.

Recommendations for Moving Forward
==================================

**Nathaniel** thinks we should:

* Go ahead and implement bitpattern NAs.
* *Don't* implement masked arrays in the core -- or at least, not
  yet. Instead, we should focus on figuring out how to implement them
  out-of-core, so that people can try out different approaches without
  us committing to any one approach. And so new prototypes can be
  released more quickly than the NumPy release cycle. And anyway,
  we're going to have to figure out how to experiment with such
  changes out-of-core if NumPy is to continue to evolve without
  forking -- might as well do it now. The existing code can live in
  master, disabled, or it can live in a branch -- it'll still be there
  once we know what we're doing.

**Mark** thinks we should:

* The existing code should remain as is, with a global run-time experimental
  flag added which disables NA support by default.

A more detailed rationale for this recommendation is:

* A solid preliminary NA-mask implementation is currently in NumPy
  master. This implementation has been extensively tested
  against scipy and other third-party packages, and has been in master
  in a stable state for a significant amount of time.
* This implementation integrates deeply with the core, providing an
  interface which is usable in the same way R's NA support is. It
  provides a compelling, user-friendly answer to R's NA support.
* The missing data NEP provides a plan for adding bitpattern-based
  dtype support of NAs, which will operate through the same interface
  but allow for the same performance/correctness tradeoffs that R has made.
* Making it very easy for users to try out this implementation, which
  has reasonable feature coverage and performance characteristics, is
  the best way to get more concrete feedback about how NumPy's missing
  data support should look.

Because of its preliminary state, the existing implementation is marked
as experimental in the NumPy documentation. It would be good for this
to remain marked as experimental until it is more fleshed out, for
example supporting struct and array dtypes and with a fuller set of
NumPy operations.

I think the code should stay as it is, except to add a run-time global
NumPy flag, perhaps numpy.experimental.maskna, which defaults to
False and can be toggled to True. In its default state, any NA feature
usage would raise an "ExperimentalError" exception, a measure which
would prevent it from being accidentally used and communicate its
experimental status very clearly.

The `ABI issues`__ seem very tricky to deal with effectively in the 1.x
series of releases, but I believe that with proper implementation-hiding
in a 2.0 release, evolving the software to support various other
ABI ideas that have been discussed is feasible. This is the approach
I like best.

__ http://thread.gmane.org/gmane.comp.python.numeric.general/49485>

**Nathaniel** notes in response that he doesn't really have any
objection to shipping experimental APIs in the main numpy distribution
*if* we're careful to make sure that they don't "leak out" in a way
that leaves us stuck with them. And in principle some sort of "this
violates your warranty" global flag could be a way to do that. (In
fact, this might also be a useful strategy for the kinds of changes
that he favors, of adding minimal hooks to enable us to build
prototypes more easily -- we could have some "rapid prototyping only"
hooks that let prototype hacks get deeper access to NumPy's internals
than we were otherwise ready to support.)

But, he wants to point out two things. First, it seems like we still
have fundamental questions to answer about the NEP design, like
whether masks should have NA semantics or ignore semantics, and there
are already plans to majorly change how NEP masks are exposed and
accessed. So he isn't sure what we'll learn by asking for feedback on
the NEP code in its current state.

And second, given the concerns about their causing (minor) ABI issues,
it's not clear that we could really prevent them from leaking out. (He
looks forward to 2.0 too, but we're not there yet.) So maybe it would
be better if they weren't present in the C API at all, and the hoops
required for testers were instead something like, 'we have included a
hacky pure-Python prototype accessible by typing "import
numpy.experimental.donttrythisathome.NEP" and would welcome feedback'?

If so, then he should mention that he did implement a horribly klugy,
pure Python implementation of the NEP API that works with NumPy
1.6.1. This was mostly as an experiment to see how possible such
prototyping was and to test out a possible ufunc override mechanism,
but if there's interest, the module is available here:
https://github.com/njsmith/numpyNEP

It passes the maskna test-suite, with some minor issues described
in a big comment at the top.

**Mark** responds:

I agree that it's important to be careful when adding new
features to NumPy, but I also believe it is essential that the project
have forward development momentum. A project like NumPy requires
developers to write code for advancement to occur, and obstacles
that impede the writing of code discourage existing developers
from contributing more, and potentially scare away developers
who are thinking about joining in.

All software projects, both open source and closed source, must
balance between short-term practicality and long-term planning.
In the case of the missing data development, there was a short-term
resource commitment to tackle this problem, which is quite immense
in scope. If there isn't a high likelihood of getting a contribution
into NumPy that concretely advances towards a solution, I expect
that individuals and companies interested in doing such work will
have a much harder time justifying a commitment of their resources.
For a project which is core to so many other libraries, only
relying on the good will of selfless volunteers would mean that
NumPy could more easily be overtaken by another project.

In the case of the existing NA contribution at issue, how we resolve
this disagreement represents a decision about how NumPy's
developers, contributers, and users should interact. If we create
a document describing a dispute resolution process, how do we
design it so that it doesn't introduce a large burden and excessive
uncertainty on developers that could prevent them from productively
contributing code?

If we go this route of writing up a decision process which includes
such a dispute resolution mechanism, I think the meat of it should
be a roadmap that potential contributers and developers can follow
to gain influence over NumPy. NumPy development needs broad support
beyond code contributions, and tying influence in the project to
contributions seems to me like it would be a good way to encourage
people to take on tasks like bug triaging/management, continuous
integration/build server administration, and the myriad other
tasks that help satisfy the project's needs. No specific meritocratic,
democratic, consensus-striving system will satisfy everyone, but the
vigour of the discussions around governance and process indicate that
something at least a little bit more formal than the current status
quo is necessary.

In conclusion, I would like the NumPy project to prioritize movement
towards a more flexible and modular ABI/API, balanced with strong
backwards-compatibility constraints and feature additions that
individuals, universities, and companies want to contribute.
I do not believe keeping the NA code in 1.7 as it is, with the small
additional measure of requiring it to be enabled by an experimental
flag, poses a risk of long-term ABI troubles. The greater risk I see
is a continuing lack of developers contributing to the project,
and I believe backing out this code because these worries would create a
risk of reducing developer contribution.


References and Footnotes
------------------------

`NEP 12`_ describes Mark's NA-semantics/mask implementation/view based mask
handling API.

`NEP 24`_ ("the alterNEP") was Nathaniel's initial attempt at separating MISSING
and IGNORED handling into bit-patterns versus masks, though there's a bunch
he would change about the proposal at this point.

`NEP 25`_ ("miniNEP 2") was a later attempt by Nathaniel to sketch out an
implementation strategy for NA dtypes.

A further discussion overview page can be found at:
https://github.com/njsmith/numpy/wiki/NA-discussion-status


Copyright
---------

This document has been placed in the public domain.

.. _NEP 12: http://www.numpy.org/neps/nep-0012-missing-data.html

.. _NEP 24: http://www.numpy.org/neps/nep-0024-missing-data-2.html

.. _NEP 25: http://www.numpy.org/neps/nep-0025-missing-data-3.html
