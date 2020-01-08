================================================
Tutorial: Linear algebra on n-dimensional arrays
================================================

.. currentmodule:: numpy

.. testsetup::

   import numpy as np
   np.random.seed(1)

Prerequisites
=============

Before reading this tutorial you should know a bit of Python. If you
would like to refresh your memory, take a look at the `Python
tutorial <https://docs.python.org/tutorial/>`__.

If you wish to work the examples in this tutorial, you must also have
some software installed on your computer. Please see
https://scipy.org/install.html for instructions.

Learner profile
===============

This tutorial is aimed at people who have some basic prior knowledge of linear
algebra and arrays in NumPy and want to understand how n-dimensional
(:math:`n>=2`) arrays are represented and can be manipulated. In particular, if
you don't know how to apply common functions to n-dimensional arrays in stacked
format (without using for-loops), or if you wish to understand axis and shape
properties for n-dimensional arrays, this tutorial might be of help.

Learning Objectives
===================

After this tutorial, you should be able to:

- Understand the difference between one-, two- and n-dimensional arrays in NumPy;
- Understand how to apply some linear algebra operations in stacked format;
- Understand axis and shape properties for n-dimensional arrays.

Content
=======

In this tutorial, we will use linear algebra to generate an approximation of an
image. We'll use the ``face`` image from the `scipy.misc` module:

    >>> from scipy import misc
    >>> img = misc.face()

If you would rather use your own image, in order to transform this image into a
numpy array that can be manipulated, we will use the ``imread`` function from the
`matplotlib.image` submodule.

Now, ``img`` is a numpy array, as we can see when using the ``type`` function:

::

    >>> type(img)
    numpy.ndarray

We can see the image using the `matplotlib.pyplot.imshow` function:

    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(img)

.. image:: https://docs.scipy.org/doc/scipy/reference/_images/scipy-misc-face-1.png

Shape, axis and array properties
--------------------------------

One particular important observation to be made is that, if you are at all
familiar with linear algebra, you might expect `dimensions` to refer to the
number of coordinates in each axis of your coordinate system; thus, a
two-dimensional vector would be a list of two coordinates. In NumPy, however,
a `dimension` is one of the axes in the array, so a two-dimensional vector
is the equivalent of a matrix.

First, let's check for the shape of the data in our array. Since this image is
a two-dimensional set of data (the pixels in the image form a rectangle), we
might expect a two-dimensional array to represent it (a matrix). However, using
the ``shape`` property of this NumPy array gives us a different result:

::

    >>> img.shape
    (768, 1024, 3)

The output is a tuple, with three elements, which means that this is a
three-dimensional array. In fact, since this is a color image, we have one set
of pixel data per color (in this case, red, green and blue - RGB). You can see
this by looking at the shape above: it indicates that we have 3 sets of 768x1024
matrices.

Furthermore, using the ``ndim`` property of this array, we can see that

::

    >>> img.ndim
    3

Each dimension is also referred to as an `axis` on NumPy. Because of how \
color images work, the *first index in the 3rd axis* is the red pixel data for
our image. We can access this by using the syntax

::

    >>> img[:, :, 0]
    array([[121, 138, 153, ..., 119, 131, 139],
           [ 89, 110, 130, ..., 118, 134, 146],
           [ 73,  94, 115, ..., 117, 133, 144],
           ...,
           [ 87,  94, 107, ..., 120, 119, 119],
           [ 85,  95, 112, ..., 121, 120, 120],
           [ 85,  97, 111, ..., 120, 119, 118]], dtype=uint8)

From the output above, we can see that every value in ``img[:,:,0]`` is an
integer value between 0 and 255, representing the level of red in each
corresponding image pixel.

As expected, this is a 768x1024 matrix:

::

    >>> img[:, :, 0].shape
    (768, 1024)

Since we are going to perform linear algebra on this data, it might be more
interesting to have real numbers between 0 and 1 in each entry of the matrices
to represent the RGB values. We can do that by setting

    >>> img_array = img/255

You can check if this worked by doing some tests; for example, inquiring about
maximum and minimum values for this array:

::

    >>> img_array.max(), img_array.min()
    (1.0, 0.0)

or checking the type of data in the array:

::

    >>> img_array.dtype
    dtype('float64')

Linear algebra on an axis
-------------------------

It is possible to use methods from linear algebra to approximate an existing set
of data. Here, we will use the SVD (Singular Value Decomposition) to try to
extract the most important information from this image.

Keep in mind that most linear algebra operations (functions in the
`numpy.linalg` module) can also be found in `scipy.linalg`. For more information
on this, check the `numpy.linalg` reference.

To proceed, import the linear algebra submodule from either NumPy or SciPy:

    >>> import numpy.linalg as LA

To simplify, let's assign each color channel to a separate matrix:

::

    >>> red_array = img_array[:,:,0]
    >>> green_array = img_array[:,:,1]
    >>> blue_array = img_array[:,:,2]

In order to extract information from each matrix, we can use the SVD to obtain
3 arrays representing a decomposition. From the theory of linear algebra, given
a matrix :math:`A`, the following decomposition can be computed:

.. math::

   U \Sigma V^T = A

where :math:`U` and :math:`V^T` are square and :math:`\Sigma` is the same size
as :math:`A` (so that the multiplication between :math:`U`, :math:`\Sigma` and
:math:`V^T` works). :math:`\Sigma` is a diagonal matrix and contains the
`singular values` of :math:`A`, organized from highest to smallest. These values
are always non-negative, and can be used as an indicator of the "importance" of
some features represented by this data matrix.

Let's see how this works in practice with ``blue_array``.

::

    >>> U_blue, s_blue, Vt_blue = LA.svd(blue_array)

.. note::

    This command might take a while to run, depending on your hardware; don't
    worry, as this is expected! The SVD is a pretty intensive computation.

Let's check that this is what we expected:

::

    >>> U_blue.shape, s_blue.shape, Vt_blue.shape
    ((768, 768), (768,), (1024, 1024))

Note that ``s_blue`` has a particular shape: it has only one dimension. This
means that some functions might work differently than expected for this array.
For example, from the theory one might expect ``s_blue`` and ``Vt_blue`` to be
compatible for multiplication. However, this is not true as ``s_blue`` does not
have a second axis. Executing

::

    >>> s_blue @ Vt_blue

results in a ``ValueError``. This happens because having a one-dimensional array
for ``s``, in this case, is much more economic in practice than building a
diagonal matrix with the same data. To reconstruct the original matrix, we can
rebuild the diagonal matrix :math:`\Sigma` with the elements of ``s_blue`` in
its diagonal and with the appropriate dimensions for multiplying: in our case,
``Sigma`` should be 768x1024 since ``U`` is 768x768 and ``Vt`` is 1024x1024.

::

    >>> import numpy as np
    >>> Sigma_blue = np.zeros((768, 1024))
    >>> for i in range(768):
    ...     Sigma_blue[i,i] = s_blue[i]

Now, we want to check if the reconstructed ``U_blue @ Sigma_blue @ Vt_blue`` is
close to the original ``blue_array`` matrix.

Approximation
-------------

The `linalg` module includes a ``norm`` function, which
computes the norm of a vector or matrix represented in a NumPy array. For
example, from the SVD explanation above, we would expect the norm of the
difference between ``blue_array`` and the reconstructed SVD product to be small.
As expected, you should see something like

::

    >>> LA.norm(blue_array - U_blue @ Sigma_blue @ Vt_blue)
    9.183739693484683e-13

(keep in mind that the actual result of this operation might be different
depending on your architecture and linear algebra setup; however, you should
see a small number.)

We could also have used the `numpy.allclose` function to make sure the
reconstructed product is, in fact, *close* to our original matrix (that is, the
difference between the two arrays is small):

::

    >>> np.allclose(blue_array, U_blue @ Sigma_blue @ Vt_blue)
    True

To see if an approximation is reasonable, we can check the values in ``s_blue``:

::

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(s_blue)

In the graph, we can see that although we have 768 singular values in
``s_blue``, most of those (after the 150th entry or so) are pretty small. So it
might make sense to use only the information related to the first (say, 50)
*singular values* to build a more economic approximation to our image.

The idea is to consider all but the first ``k`` singular values in
``Sigma_blue`` (which are the same as in ``s_blue``) as zeros, keeping
``U_blue`` and ``Vt_blue`` intact, and computing the product of these matrices
as the approximation.

For example, if we choose 

::

    >>> k = 10

we can build the approximation by doing

::

    >>> approx_blue = U_blue @ Sigma_blue[:, :k] @ Vt_blue[:k,:]

Note that we had to use only the first ``k`` rows of ``Vt_blue``, since all
other rows would be multiplied by the zeros corresponding to the singular
values we eliminated from this approximation.

To see if this approximation makes sense in our image, we must use a colormap
from `matplotlib` corresponding to the color we wish to see in each
individual image (otherwise, `matplotlib` will default to a colormap that does
not correspond to the real data).

In our case, we are approximating the blue portion of the image, so we will use
the colormap ``Blues``:

::

    >>> plt.imshow(approx_blue, cmap=plt.cm.Blues)

Now, you can go ahead and repeat this experiment with other values of `k`, and
each of those should give you a slightly better (or worse) image depending on
the value you choose.

Just for good measure, to see how the original blue array looks like, you can
do

::

    >>> plt.imshow(blue_array, cmap=plt.cm.Blues)

Applying to all colors
----------------------

Now, we want to do the same kind of operation, but to all three colors. Our
first instinct might be to repeat the same operation we did above to each color
matrix individually. However, numpy's `broadcasting` takes care of this
for us.

If our array has more than two dimensions, then the SVD can be applied in
“stacked” mode. However, the linear algebra functions in numpy expect to see a
stacking of the form ``(N, M, M)``, where the first axis represents the number
of stacked matrices (i.e. ``(N, M, M)`` is interpreted as a “stack” of ``N``
matrices, each of size ``(M, M)``.)

In our case,

::

    >>> img_array.shape
    (768, 1024, 3)

so we need to permutate the axis on this array to make get a shape like
``(3, 768, 1024)``. Fortunately, the `numpy.transpose` function can do that for
us:

``np.transpose(x, axes=(i, j, k))`` 

indicates that the axis will be reordered such that the final shape of the
transposed array will be reordered according to the indices ``(i, j, k)``.

Let's see how this goes for our array:

::

    >>> img_array_transposed = np.transpose(img_array, (2, 0, 1))
    >>> img_array.shape
    (3, 768, 1024)

Now we are ready to apply the SVD:

::

    >>> U, s, Vt = LA.svd(img_array_transposed)

Finally, to obtain the full approximated image, we need to reassemble these
matrices into the approximation. Now, note that

::

    >>> U.shape, s.shape, Vt.shape
    ((3, 768, 768), (3, 768), (3, 1024, 1024))

To build the final approximation matrix, we must understand how multiplication
across different axes works.

Products with n-dimensional arrays
----------------------------------

If you have worked before with only one- or two-dimensional arrays in NumPy,
you might use `numpy.dot` and `numpy.matmul` (or the ``@`` operator)
interchangeably. However, for n-dimensional arrays, they work in very different
ways. For more details, check the documentation `numpy.matmul`.

Now, to build our approximation, we need first to make sure that our singular
values are ready for multiplication, so we build our ``Sigma`` matrix similarly
to what we did before. The ``Sigma`` array must have dimensions
``(3, 768, 1024)``.

::

    >>> Sigma = np.zeros((3, 768, 1024))
    >>> for j in range(3):
    ...     for i in range(768):
    ...         Sigma[j, i, i] = s[j, i]

Now, if we wish to rebuild the full SVD (with no approximation), we could do

::

    >>> reconstructed = U @ Sigma @ Vt

Note that

::

    >>> reconstructed.shape
    (3, 768, 1024)

and

::

    >>> plt.imshow(np.transpose(reconstructed, (1, 2, 0)))

Should give you an image indistinguishable from the original one (although we
do possibly introduce floating point errors for this reconstruction; in fact, it
is possible that you see a warning message saying "Clipping input data to the
valid range for imshow with RGB data ([0..1] for floats or [0..255] for
integers)." This is expected from the manipulation we just did on the original
image.

Now, to do the approximation, we must choose only the first ``k`` singular
values for each color channel. This can be done by the following syntax:

::

    >>> approx_img = U @ Sigma[..., :k] @ Vt[..., :k, :]

You can see that we have selected only the first ``k`` components of the last
axis for ``Sigma`` (this means that we have used only the first ``k`` columns
of each of the three matrices in the stack), and that we have selected only the
first ``k`` components in the second-to-last axis of ``Vt`` (this means we have
selected, from every matrix in the stack ``Vt``, the first ``k`` rows only, and
all columns). If you are unfamiliar with the ellipsis syntax, it is a
placeholder for other axes; for more details, see the documentation on
:ref:`Indexing <basics.indexing>`.

Now,

::

    >>> approx_img.shape
    (3, 768, 1024)

which is not the right shape for showing the image. So, finally, reordering the
axes back to our original shape of (768, 1024, 3), we can see our approximation:

::

    >>> plt.imshow(np.transpose(approx_img, (1, 2, 0)))

Even though the image is not as sharp, using a small number of ``k`` singular
values (compared to the original set of 768 values) we can recover many of the
distinguishing features from this image.

Final words
-----------

Of course, this is not the best method to *approximate* or *compress* an image.
However, there is in fact a result in linear algebra that says that the
approximation we built above is the best we can get to the original matrix in
terms of the norm of the difference. For more information, see G. H. Golub and
C. F. Van Loan, Matrix Computations, Baltimore, MD, Johns Hopkins University
Press, 1985.

Further reading
===============

-  The `Python tutorial <https://docs.python.org/tutorial/>`__
-  :ref:`reference`
-  `SciPy Tutorial <https://docs.scipy.org/doc/scipy/reference/tutorial/index.html>`__
-  `SciPy Lecture Notes <https://scipy-lectures.org>`__
-  A `matlab, R, IDL, NumPy/SciPy dictionary <http://mathesaurus.sf.net/>`__

