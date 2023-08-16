# Learn NumPy


### For the official NumPy documentation visit [numpy.org/doc/stable](https://numpy.org/doc/stable/).

**NumPy** is an open source project that enables numerical computing with Python. It was created in 2005 building on the early work of the Numeric and Numarray libraries. NumPy will always be 100% open source software and free for all to use. It is released under the liberal terms of the [modified BSD license](https://github.com/numpy/numpy/blob/main/LICENSE.txt).

## Table of Contents

- [Install](#install)
- [Beginners](#beginners)
- [Advanced](#advanced)

## Install
The only prerequisite for installing NumPy is Python itself. If you don’t have Python yet and want the simplest way to get started, we recommend you use the [Anaconda Distribution](https://www.anaconda.com/data-science-platform) - it includes Python, NumPy, and many other commonly used packages for scientific computing and data science.

NumPy can be installed with conda, with pip, with a package manager on macOS and Linux, or from [source](https://numpy.org/devdocs/user/building.html). For more detailed instructions, consult our [Python and NumPy installation guide below](https://numpy.org/install/#python-numpy-install-guide).

### `Conda`

```sh
# Best practice, use an environment rather than install in the base env
conda create -n my-env
conda activate my-env
# If you want to install from conda-forge
conda config --env --add channels conda-forge
# The actual install command
conda install numpy
```

### `PIP`

If you use pip, you can install NumPy with:

```sh
pip install numpy
```

## Beginners
There’s a ton of information about NumPy out there. If you are just starting, we’d strongly recommend the following:

**Tutorials**
- [NumPy Quickstart Tutorial](https://numpy.org/devdocs/user/quickstart.html)
- [NumPy Tutorials](https://numpy.org/numpy-tutorials/) A collection of tutorials and educational materials in the format of Jupyter Notebooks developed and maintained by the NumPy Documentation team.
- [NumPy Illustrated: The Visual Guide to NumPy](https://betterprogramming.pub/numpy-illustrated-the-visual-guide-to-numpy-3b1d4976de1d?sk=57b908a77aa44075a49293fa1631dd9b) by Lev Maximov
- [SciPy Lectures](https://lectures.scientific-python.org/) Besides covering NumPy, these lectures offer a broader introduction to the scientific Python ecosystem.
- [NumPy: the absolute basics for beginners](https://numpy.org/devdocs/user/absolute_beginners.html)
- [NumPy tutorial](https://github.com/rougier/numpy-tutorial) by Nicolas Rougier
- [Stanford CS231](https://cs231n.github.io/python-numpy-tutorial/) by Justin Johnson
- [NumPy User Guide](https://numpy.org/devdocs/)

**Books**
- [Guide to NumPy](http://web.mit.edu/dvp/Public/numpybook.pdf) by Travis E. Oliphant This is a free version 1 from 2006. For the latest copy (2015) see [here](https://www.youtube.com/watch?v=ZB7BZMhfPgk).
- [From Python to NumPy](https://www.labri.fr/perso/nrougier/from-python-to-numpy/) by Nicolas P. Rougier
- [Elegant SciPy](https://www.amazon.com/Elegant-SciPy-Art-Scientific-Python/dp/1491922877) by Juan Nunez-Iglesias, Stefan van der Walt, and Harriet Dashnow

**Videos**
- [Introduction to Numerical Computing with NumPy](https://www.youtube.com/watch?v=ZB7BZMhfPgk) by Alex Chabot-Leclerc


## Advanced
Try these advanced resources for a better understanding of NumPy concepts like advanced indexing, splitting, stacking, linear algebra, and more.

**Tutorials**
- [100 NumPy Exercises](https://github.com/rougier/numpy-100) by Nicolas P. Rougier
- [An Introduction to NumPy and Scipy](https://sites.engineering.ucsb.edu/~shell/che210d/numpy.pdf) by M. Scott Shell
- [Numpy Medkits](https://mentat.za.net/numpy/numpy_advanced_slides/) by Stéfan van der Walt
- [NumPy Tutorials](https://numpy.org/numpy-tutorials/) A collection of tutorials and educational materials in the format of Jupyter Notebooks developed and maintained by the NumPy Documentation team.
- 
**Books**
- [Python Data Science Handbook](https://www.amazon.com/Python-Data-Science-Handbook-Essential/dp/1491912057) by Jake Vanderplas
- [Python for Data Analysis by Wes McKinney](https://www.amazon.com/Python-Data-Analysis-Wrangling-IPython/dp/1491957662) by Wes McKinney
- [Numerical Python: Scientific Computing and Data Science Applications with Numpy, SciPy, and Matplotlib](https://www.amazon.com/Numerical-Python-Scientific-Applications-Matplotlib/dp/1484242459)

**Videos**
- [Advanced NumPy - broadcasting rules, strides, and advanced indexing](https://www.youtube.com/watch?v=cYugp9IN1-Q) by Juan Nunez-Iglesias

[Go to Top](#learn-numpy)