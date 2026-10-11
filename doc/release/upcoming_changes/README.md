# Changelog

This directory contains "news fragments" which are short files that contain a
small **MyST Markdown**-formatted text that will be added to the next what's new
page.

Make sure to use full sentences with correct case and punctuation, and please
try to use Sphinx cross-references (see below). The fragment should start with a
level three heading (`###`).

Each file should be named like `<PULL REQUEST>.<TYPE>.md`, where
`<PULL REQUEST>` is a pull request number, and `<TYPE>` is one of:

* `new_function`: New user facing functions.
* `deprecation`: Changes existing code to emit a DeprecationWarning.
* `future`: Changes existing code to emit a FutureWarning.
* `expired`: Removal of a deprecated part of the API.
* `compatibility`: A change which requires users to change code and is not
  backwards compatible. (Not to be used for removal of deprecated features.)
* `c_api`: Changes in the Numpy C-API exported functions
* `new_feature`: New user facing features like `kwargs`.
* `improvement`: General improvements and edge-case changes which are
  not new features or compatibility related.
* `performance`: Performance changes that should not affect other behaviour.
* `typing`: Improvements and changes related to static typing.
* `change`: Other changes
* `highlight`: Adds a highlight bullet point to use as a possibly highlight
  of the release.

It is possible to add two files with different categories (and text) if both
are relevant. For example a change may improve performance but have some
compatibility concerns.

Most categories should be formatted as paragraphs with a heading.
So for example: `123.new_feature.md` would have the content:

```md
### `my_new_feature` option for `my_favorite_function`

The `my_new_feature` option is now available for {func}`my_favorite_function`.
To use it, write `np.my_favorite_function(..., my_new_feature=True)`.
```

`highlight` is usually formatted as bullet points making the fragment
`* This is a highlight`.

Note that single backticks are literal text in Markdown; they do not create a
link. To cross-reference the NumPy API, use an explicit role, such as
``{func}`numpy.sum` ``, ``{class}`numpy.ndarray` ``,
``{attr}`numpy.ndarray.shape` ``, or ``{data}`numpy.newaxis` ``. Use the role
that matches the object type. The `np.` alias also works, as in
``{func}`np.sum` ``.

If you are unsure what pull request type to use, don't hesitate to ask in your
PR.

`towncrier` is required to build the docs; it will be automatically run when
you build the docs locally with `spin docs`. You can also run `towncrier
build --draft --version 1.18` if you want to get a preview of how your change
will look in the final release notes.

```{note}
This README was adapted from the pytest changelog readme under the terms of
the MIT licence.
```
