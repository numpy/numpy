===============================================================
Tutorial: Do linear algebra on n-dimensional arrays
===============================================================

.. currentmodule:: numpy

.. testsetup::

   import numpy as np
   np.random.seed(1)

**What you'll do**

Use NumPy to compress an image using singular value decomposition.


**What you'll learn**

You'll work with arrays of one, two, and three dimensions -- multiplying,
scaling, transposing, and reordering them. You'll see NumPy broadcasting in
action, and put some of NumPy's linear algebra tools to use. By the end,
you'll be able to repeat the steps on an image of your own.


**What you'll need**

Some linear algebra knowledge and basic familiarity with NumPy's ndarrays.

To run the examples, you'll need `Matplotlib
<https://matplotlib.org/>`_ and `SciPy <https://scipy.org>`_ installed.


**First step: Loading and examining the image**

We'll use the ``face`` image from the `scipy.misc` module:

    >>> from scipy import misc
    >>> img = misc.face()

``img`` is a NumPy array::

    >>> type(img)
    <class 'numpy.ndarray'>

We can view the image using `matplotlib.pyplot.imshow`::

    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(img)

.. plot:: user/plot_face.py
    :align: center
    :include-source: 0

.. note::

   If you're using the IPython shell, type ``plt.show()`` if the image fails
   to appear. You'll need `ctrl-C` to return to IPython.

**Shape, axis and array properties**

In linear algebra, a vector can have any number of dimensions, depending on
its length. In programming, that vector -- viewed as an array -- has just one,
because one index retrieves any of its values. To avoid ambiguity, Numpy
uses the term `axis` to mean a programming-style dimension. We'll use both
words, always with the programming meaning.

First, let's check the shape of our data. Since an image is two-dimensional,
we might expect to see a two-dimensional array -- a matrix.  But the array's
``shape`` property says something different:

    >>> img.shape
    (768, 1024, 3)

The output is a :ref:`tuple <python:tut-tuples>` with three elements, which
means this is a three-dimensional array -- or equivalently, an array of 3
matrices, each 768x1024. The extra dimension comes from working with a color
image; these matrices are color channels. Conventions vary, but here the
colors are red, green and blue, at indexes 0, 1, and 2 respectively.

The ``ndim`` property of this array confirms there are three dimensions:

::

    >>> img.ndim
    3

Using slice notation, we can view just the red pixels:

::

    >>> img[:, :, 0]
    array([[121, 138, 153, ..., 119, 131, 139],
           [ 89, 110, 130, ..., 118, 134, 146],
           [ 73,  94, 115, ..., 117, 133, 144],
           ...,
           [ 87,  94, 107, ..., 120, 119, 119],
           [ 85,  95, 112, ..., 121, 120, 120],
           [ 85,  97, 111, ..., 120, 119, 118]], dtype=uint8)

Every value is an integer from 0 to 255, representing the level of red in each
of the 768 * 1024 = 786,432 red pixels.

As expected, this is a 768x1024 matrix::

    >>> img[:, :, 0].shape
    (768, 1024)

To prepare the array for SVD, let's rescale these values into real numbers
between 0 and 1:

    >>> img_array = img / 255

You can check that the rescaling worked by inquiring about maximum and minimum values::

    >>> img_array.max(), img_array.min()
    (1.0, 0.0)

or checking the type::

    >>> img_array.dtype
    dtype('float64')

Using slice syntax, we can assign the channels to separate matrices::

    >>> red_array = img_array[:, :, 0]
    >>> green_array = img_array[:, :, 1]
    >>> blue_array = img_array[:, :, 2]

**Operations on an axis**

Methods from linear algebra can reduce data into a smaller approximate version
that retains the most important features of the original. We'll do that using
`SVD (Singular Value Decomposition)
<https://web.archive.org/web/20200102142016/https://www.cs.princeton.edu/picasso/mats/PCA-Tutorial-Intuition_jp.pdf>`_/

Start by importing the linear algebra submodule from NumPy::

    >>> from numpy import linalg

Given a matrix :math:`A`, SVD returns three arrays that equal :math:`A` when
multiplied together:

.. math::

   U \Sigma V^T = A

:math:`U` and :math:`V^T` are square and :math:`\Sigma` is the same size
as :math:`A`. :math:`\Sigma` is a diagonal matrix and contains the
`singular values <https://en.wikipedia.org/wiki/Singular_value>`_ of :math:`A`,
organized from largest to smallest. These values are always non-negative and can
be used as an indicator of the "importance" of some features in
matrix :math:`A`.

Let's see how this works in practice, starting with a single matrix -- a
grasyscale version of ``img_array``. According to
`colorimetry <https://en.wikipedia.org/wiki/Grayscale#Colorimetric_(perceptual_luminance-preserving)_conversion_to_grayscale>`_,
a color image can be turned into a grayscale version using the formula

.. math::

   Y = 0.2126 R + 0.7152 G + 0.0722 B

where :math:`Y` is the array representing the grayscale image, and :math:`R,
G` and :math:`B` are the red, green and blue channel arrays we had originally.
We can use the ``@`` operator (the matrix multiplication operator for NumPy
arrays, see `numpy.matmul`):

::

   >>> img_gray = img_array @ [0.2126, 0.7152, 0.0722]

As expected, the grayscale image has two dimensions:

::

   >>> img_gray.shape
   (768, 1024)

We'll take a look at the image using the ``Matplotlib`` gray colormap::

   >>> plt.imshow(img_gray, cmap="gray")

.. plot:: user/plot_gray.py
    :align: center
    :include-source: 0

Now, applying the `linalg.svd` function to this matrix, we obtain the
following SVD decomposition:
::

    >>> U, s, Vt = linalg.svd(img_gray)

Checking the shapes:
::

    >>> U.shape, s.shape, Vt.shape
    ((768, 768), (768,), (1024, 1024))

Unexpectedly, ``s`` has just one dimension. Under matrix algebra rules, it
can't be multipled with ``Vt``. Executing

::

    >>> s @ Vt
    Traceback (most recent call last):
      ...
    ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0,
    with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 1024 is different from
    768)

results in a ``ValueError``.

The one-dimensional ``s`` is an economized version of the diagonal matrix
:math:`\Sigma` that we're looking for. We need to construct :math:`\Sigma` from ``s``:

::

    >>> import numpy as np
    >>> Sigma = np.zeros((768, 1024))
    >>> for i in range(768):
    ...     Sigma[i, i] = s[i]

:math:`\Sigma` is 768x1024 since ``U`` is 768x768 and ``Vt`` is
1024x1024.

**Approximation**

Theory says that  ``U @ Sigma @ Vt`` should equal the original
``img_gray`` matrix. Let's see how close it comes.

The `linalg` module includes a ``norm`` function, which
computes the norm of a vector or matrix represented in a NumPy array. For
example, from the SVD explanation above, we would expect the norm of the
difference between ``img_gray`` and the reconstructed SVD product to be small.
As expected, you should see something like

::

    >>> linalg.norm(img_gray - U @ Sigma @ Vt)
    1.3926466851808837e-12

(Your result may be different depending on your architecture and linear
algebra setup. Regardless, you should see a small number.)

We can also use the `numpy.allclose` function to make sure the
reconstructed product is, in fact, *close* to our original matrix (the
difference of the two arrays is small)::

    >>> np.allclose(img_gray, U @ Sigma @ Vt)
    True

Now recall that :math:`\Sigma` orders the singular values starting with
the most important. Can we find "unimportant" values to eliminate?

Let's plot ``s``::

    >>> plt.plot(s)

.. plot:: user/plot_gray_svd.py
    :align: center
    :include-source: 0

Although ``s`` has 768 singular values, most of them are pretty small. To
build a more economical approximation, it might make sense to use only the
first 50 or so.

We'll zero out all but the first ``k`` singular values in ``Sigma``, keeping
``U`` and ``Vt`` intact, and compute the product. Instead of picking the first
50 singular values, let's see what happens if we use just the first 10:

::

    >>> k = 10

We can build the approximation with

::

    >>> approx = U @ Sigma[:, :k] @ Vt[:k, :]

Note that the slices use only the first ``k`` rows of ``Vt``, since the rest
would be multiplied by zeros.

The result:

::

    >>> plt.imshow(approx, cmap="gray")

.. plot:: user/plot_approx.py
    :align: center
    :include-source: 0

Try repeating this experiment with other values of ``k``; each should give you a
slightly better or worse image.

**Applying to all colors**

Now we want to do SVD on all three colors. Our first instinct might be to
repeat the the steps for each color matrix individually. But NumPy's
`broadcasting` takes care of this for us.

If our array has more than two dimensions, SVD can be applied to all axes at
once. An adjustment is needed first: The linear algebra functions in NumPy
expect an array where the first axis represents the number of matrices.

Since our shape is this

::

    >>> img_array.shape
    (768, 1024, 3)

we need to permute the axes. The function

::

   np.transpose(x, axes=(i, j, k))

reorders  ``x`` such that axis 0 will be the current axis ``i``, axis 1 will
be the current ``j``, and so on.

Let's see how this goes for our array::

    >>> img_array_transposed = np.transpose(img_array, (2, 0, 1))
    >>> img_array_transposed.shape
    (3, 768, 1024)

Now we are ready to apply the SVD::

    >>> U, s, Vt = linalg.svd(img_array_transposed)

**Products with n-dimensional arrays**

Note the shapes of the result:
::

    >>> U.shape, s.shape, Vt.shape
    ((3, 768, 768), (3, 768), (3, 1024, 1024))

To do the multiplications you'll need to understand how multiplication across
different axes works.

If you have worked with only one- or two-dimensional arrays in NumPy, you
might use `numpy.dot` and `numpy.matmul` (or the ``@`` operator)
interchangeably. However, n-dimensional arrays work very differently. For
details, check the documentation `numpy.matmul`.

Again, a ``Sigma`` matrix must be built from ``s``. ``Sigma`` must have
dimensions ``(3, 768, 1024)``. We'll call the ``fill_diagonal`` function,
using each of the 3 rows in ``s`` as the diagonal for each of the 3 matrices
in ``Sigma``:

::

    >>> Sigma = np.zeros((3, 768, 1024))
    >>> for j in range(3):
    ...     np.fill_diagonal(Sigma[j, :, :], s[j, :])

To rebuild using the full SVD with no approximation, we can do

::

    >>> reconstructed = U @ Sigma @ Vt

The shape of the result means that to view it we'll again need ``np.transpose``:

::

    >>> reconstructed.shape
    (3, 768, 1024)

Let's see the reconstructed image:

    >>> plt.imshow(np.transpose(reconstructed, (1, 2, 0)))

.. plot:: user/plot_reconstructed.py
    :align: center
    :include-source: 0

It should be indistinguishable from the original image. You
may see the warning `"Clipping input data to the valid range for imshow with
RGB data ([0..1] for floats or [0..255] for integers)."` These reflect small
floating-point errors expected from the manipulation we just did.

Now, for the approximation, we must choose only the first ``k`` singular
values for each color channel. This can be done as ::

    >>> approx_img = U @ Sigma[..., :k] @ Vt[..., :k, :]

Let's examine the indexes carefully. We have selected only the first ``k``
components of the last axis for ``Sigma``.  This means that we have used only
the first ``k`` columns of each of the three matrices in the stack.

Similarly we have selected only the first ``k`` components in the
second-to-last axis of ``Vt``. This means we have selected only the first
``k`` rows from every matrix in the stack ``Vt`` and all columns.

The ellipsis syntax is a placeholder meaning ``:`` on all other axes. For
details, see the documentation on :ref:`Indexing <basics.indexing>`.

As with ``reconstructed``, the shape needs to change before we can view it.

::

    >>> approx_img.shape
    (3, 768, 1024)

So at last, reordering the axes back to the original ``(768, 1024, 3)``, we
see our color approximation::

    >>> plt.imshow(np.transpose(approx_img, (1, 2, 0)))

.. plot:: user/plot_final.py
    :align: center
    :include-source: 0

Though the image is less sharp, using a small number of ``k`` singular values
(compared to the original set of 768) has allowed us to recover many
distinguishing features.

**On your own**

Use your own image and  work through this tutorial. To transform your image
into a NumPy array that can be manipulated, use the ``imread`` function from
the `matplotlib.pyplot` submodule or the :func:`imageio.imread` function from
the ``imageio`` library. For more information on how images are treated when
converted to NumPy arrays, see :std:doc:`user_guide/numpy_images` from the
``scikit-image`` documentation.

``linalg.svd`` might take a while to run, depending on the size of your image
and your hardware. Don't worry, this is normal! The SVD can be a pretty
intensive computation.

**In practice...**

- We used NumPy's linear algebra module, `numpy.linalg`. Most of its linear
  algebra functions are also in `scipy.linalg`, and for real-world
  applications users are encouraged to use the `scipy` module. `scipy.linalg`
  is currently unable to apply linear algebra operations to n-dimensional
  arrays; see :doc:`scipy.linalg Reference<scipy:tutorial/linalg>`.

- We rescaled the integer pixels by simply using ``/``.  In real-world
  applications, it would be better to use, for example, the
  :func:`img_as_float <skimage.img_as_float>` utility function from
  ``scikit-image``.

- Numerically, SVD compression is unbeatable: a result in linear algebra says
  that the approximation we built above is the best we can get in terms of the
  norm of the difference between the original and reduced matrices. (See *G.
  H. Golub and C. F. Van Loan, Matrix Computations, Baltimore, MD, Johns
  Hopkins University  Press, 1985*.) But human eyes disagree; lossy image
  compression techniques used in practice, like jpeg, are built around
  discarding data that the eye won't notice is missing.

**Further reading**

-  :doc:`Python tutorial <python:tutorial/index>`
-  :ref:`reference`
-  :doc:`SciPy Tutorial <scipy:tutorial/index>`
-  `SciPy Lecture Notes <https://scipy-lectures.org>`__
-  `A matlab, R, IDL, NumPy/SciPy dictionary <http://mathesaurus.sf.net/>`__
