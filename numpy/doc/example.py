
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

