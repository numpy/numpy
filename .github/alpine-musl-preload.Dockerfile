FROM alpine:3.20

# Build dependencies and system BLAS/LAPACK
RUN apk add --no-cache python3 py3-pip python3-dev build-base openblas-dev lapack-dev gfortran musl-dev git

# Virtualenv to keep pip installs contained
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Copy source and install NumPy + test deps
WORKDIR /src
COPY . /src
RUN pip install -U pip setuptools wheel \
    && pip install -U cython meson-python pytest pytest-xdist \
    && pip install -r requirements/test_requirements.txt \
    && pip install .

# Run tests from outside the source tree to avoid importing the checkout
WORKDIR /tmp
