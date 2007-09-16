# It is desireable that both NumPy and SciPy follow a convention for docstrings
# that provide for some consistency while also allowing epydoc to produce
# nicely-formatted reference guides. However, such a convention has not yet
# been decided on. This is my current thinking on the topic.  If you have
# suggestions for improvements, post them on the numpy-dev list together with
# the epydoc output so they may be discussed.
#
# The docstring format uses reST syntax as interpreted by epydoc. The markup
# in this proposal is as basic as possible and in particular avoids the use of
# epydoc consolidated fields. This is both because there are a limited number
# of such fields, inadequate to our current needs, and because epydoc moves
# the fields to the end of the documentation, messing up the ordering. So here
# standard definition lists are used instead.  Likewise, epydoc moves headings
# and have an unwelcome size in the default style sheet, hence they have also
# been avoided.
#
# A maximum line width of 79 is suggested, as this will allow the docstrings to
# display on standard terminals. This convention is a bit old and traces back
# to IBM punchcard days, but still seems to be the standard.
#
# Comments:
#
# 1) You can run epydoc on this file like so:
#
# $ epydoc HOWTO_DOCUMENT.txt
#
# The output will be in a directory named html in the same directory as this
# document and may be viewed by loading the index.html file into your browser.
#
# 2) The developmental version of epydoc, version 3.0 beta or later, is
# suggested as it is faster and produces better looking output. Epydoc can be
# downloaded from http://epydoc.sourceforge.net/
#
# 3) The appearance of some elements can be changed in the epydoc.css
# style sheet. The list headings, i.e. *Parameters*:, are emphasized text, so
# their appearance is controlled by the definition of the <em>
# tag. For instance, to make them bold, insert
#
# em     {font-weight: bold;}
#
# The variables' types are in a span of class rst-classifier, hence can be
# changed by inserting something like:
#
# span.rst-classifier     {font-weight: normal;}
#
# 4) The first line of the signature should **not** copy the signature unless
# the function is written in C, in which case it is mandatory.  If the function
# signature is generic (uses *args or **kwds), then a function signature may be
# included
#
# 5) Use optional in the "type" field for parameters that are non-keyword
# optional for C-functions.
#
# 6) The Other Parameters section is for functions taking a lot of keywords
# which are not always used or neeeded and whose description would clutter then
# main purpose of the function. (Comment by Chuck : I think this should be
# rarely used, if at all)
#
# 7) The See Also section can list additional related functions.  The purpose
# of this section is to direct users to other functions they may not be aware
# of or have easy means to discover (i.e. by looking at the docstring of the
# module).  Thus, repeating functions that are in the same module is not useful
# and can create a cluttered document.  Please use judgement when listing
# additional functions.  Routines that provide additional information in their
# docstrings for this function may be useful to include here.
#
# 8) The Notes section can contain algorithmic information if that is useful.
#
# 9) The Examples section is strongly encouraged.  The examples can provide a
# mini-tutorial as well as additional regression testing. (Comment by Chuck:
# blank lines in the numpy output, for instance in multidimensional arrays,
# will break doctest.) You can run the tests by doing
#
# >>> import doctest
# >>> doctest.testfile('HOWTO_DOCUMENT.txt')
#
#
# Common reST concepts:

# A reST-documented module should define
#
#   __docformat__ = 'restructuredtext en'
#
# at the top level in accordance with PEP 258.  Note that the __docformat__
# variable in a package's __init__.py file does not apply to objects defined in
# subpackages and submodules.
#
# For paragraphs, indentation is significant and indicates indentation in the
# output. New paragraphs are marked with blank line.
#
# Use *italics*, **bold**, and ``courier`` if needed in any explanations (but
# not for variable names and doctest code or multi-line code)
#
# Use :lm:`eqn` for in-line math in latex format (remember to use the
# raw-format for your text string or escape any '\' symbols). Use :m:`eqn` for
# non-latex math.
#
# A more extensive example of reST markup can be found here:
# http://docutils.sourceforge.net/docs/user/rst/demo.txt
# An example follows. Line spacing and indentation are significant and should
# be carefully followed.

__docformat__ = "restructuredtext en"


def foo(var1, var2, long_var_name='hi') :
    """One-line summary or signature.

    Several sentences providing an extended description. You can put
    text in mono-spaced type like so: ``var``.

    *Parameters*:

        var1 : {array_like}
            Array_like means all those objects -- lists, nested lists, etc. --
            that can be converted to an array.
        var2 : {integer}
            Write out the full type
        long_variable_name : {'hi', 'ho'}, optional
            Choices in brackets, default first when optional.

    *Returns*:

        named : {type}
            Explanation
        list
            Explanation
        of
            Explanation
        outputs
            even more explaining

    *Other Parameters*:

        only_seldom_used_keywords : type
            Explanation
        common_parametrs_listed_above : type
            Explanation

    *See Also*:

        `otherfunc` : relationship (optional)

        `newfunc` : relationship (optional)

    *Notes*

        Notes about the implementation algorithm (if needed).

        This can have multiple paragraphs as can all sections.

    *Examples*

        examples in doctest format

        >>> a=[1,2,3]
        >>> [x + 3 for x in a]
        [4, 5, 6]

    """
    pass


def newfunc() :
    """Do nothing.

    I never saw a purple cow.

    """
    pass


def otherfunc() :
    """Do nothing.

    I never hope to see one.

    """
    pass

