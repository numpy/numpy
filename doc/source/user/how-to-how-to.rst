.. _how-to-how-to:

##############################################################################
How to write a NumPy how-to
##############################################################################

How-tos get straight to the point -- they

  - answer a focused question, or
  - narrow a broad question into focused questions that the user can
    choose among.

******************************************************************************
A stranger has asked for directions...
******************************************************************************

**"I need to refuel my car."**

******************************************************************************
Give a brief but explicit answer
******************************************************************************

  - `"Three kilometers/miles, take a right at Hayseed Road, it's on your left."`

Add helpful details for newcomers ("Hayseed Road", even though it's the only
turnoff at three km/mi). But not irrelevant ones:

  - Don't also give directions from Route 7.
  - Don't explain why the town has only one filling station.

If there's related background (tutorial, explanation, reference, alternative
approach), bring it to the user's attention with a link ("Directions from Route 7,"
"Why so few filling stations?").


******************************************************************************
Delegate
******************************************************************************

  - `"Three km/mi, take a right at Hayseed Road, follow the signs."`

If the information is already documented and succinct enough for a how-to,
just link to it, possibly after an introduction ("Three km/mi, take a right").

******************************************************************************
If the question is broad, narrow and redirect it
******************************************************************************

 **"I want to see the sights."**

The `See the sights` how-to should link to a set of narrower how-tos:

- Find historic buildings
- Find scenic lookouts
- Find the town center

and these might in turn link to still narrower how-tos -- so the town center
page might link to

   - Find the court house
   - Find city hall

By organizing how-tos this way, you not only display the options for people
who need to narrow their question, you also have provided answers for users
who start with narrower questions ("I want to see historic buildings," "Which
way to city hall?").

******************************************************************************
If there are many steps, break them up
******************************************************************************

If a how-to has many steps:

  - Consider breaking a step out into an individual how-to and linking to it.
  - Include subheadings. They help readers grasp what's coming and return
    where they left off.

******************************************************************************
Why write how-tos when there's Stack Overflow, Reddit, Gitter...?
******************************************************************************

 - We have authoritative answers.
 - How-tos make the site less forbidding to non-experts.
 - How-tos bring people into the site and help them discover other information
   that's here .
 - Creating how-tos helps us see NumPy usability through new eyes.

******************************************************************************
Aren't how-tos and tutorials the same thing?
******************************************************************************

People use the terms "how-to" and "tutorial" interchangeably, but we draw a
distinction, following Daniele Procida's `taxonomy of documentation`_.

 .. _`taxonomy of documentation`: https://documentation.divio.com/

Documentation needs to meet users where they are.  `How-tos` offer get-it-done
information; the user wants steps to copy and doesn't necessarily want to
understand NumPy. `Tutorials` are warm-fuzzy information; the user wants a
feel for some aspect of NumPy (and again, may or may not care about deeper
knowledge).

We distinguish both tutorials and how-tos from `Explanations`, which are
deep dives intended to give understanding rather than immediate assistance,
and `References`, which give complete, authoritative data on some concrete
part of NumPy (like its API) but aren't obligated to paint a broader picture.

For more on tutorials, see :doc:`numpy-tutorials:content/tutorial-style-guide`

******************************************************************************
Is this page an example of a how-to?
******************************************************************************

Yes -- until the sections with question-mark headings; they explain rather
than giving directions. In a how-to, those would be links.
