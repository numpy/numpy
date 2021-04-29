.. _dot2-dot3:

========================================
 Two and three dots in difference specs
========================================

Thanks to Yarik Halchenko for this explanation.

Imagine a series of commits A, B, C, D...  Imagine that there are two
branches, *topic* and *main*.  You branched *topic* off *main* when
*main* was at commit 'E'.  The graph of the commits looks like this::


        A---B---C topic
        /
   D---E---F---G main

Then::

   git diff main..topic

will output the difference from G to C (i.e. with effects of F and G),
while::

   git diff main...topic

would output just differences in the topic branch (i.e. only A, B, and
C).
