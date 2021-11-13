# Doing a local shallow clone - keeps the container secure
# and much slimmer than using COPY directly or making a 
# remote clone
ARG BASE_CONTAINER="numpy/numpy-dev:latest"
FROM gitpod/workspace-base:latest as clone

COPY --chown=gitpod . /tmp/numpy_repo

# the clone should be deep enough for versioneer to work
RUN git clone --shallow-since=2021-05-22 file:////tmp/numpy_repo /tmp/numpy

# -----------------------------------------------------------------------------
# Using the numpy-dev Docker image as a base
# This way, we ensure we have all the needed compilers and dependencies
# while reducing the build time
FROM ${BASE_CONTAINER} as build

# -----------------------------------------------------------------------------
USER root

# -----------------------------------------------------------------------------
# ---- ENV variables ----
# ---- Directories needed ----
ENV WORKSPACE=/workspace/numpy/ \
    CONDA_ENV=numpy-dev

# Allows this Dockerfile to activate conda environments
SHELL ["/bin/bash", "--login", "-o", "pipefail", "-c"]

# Copy over the shallow clone
COPY --from=clone --chown=gitpod /tmp/numpy ${WORKSPACE}

# Everything happens in the /workspace/numpy directory
WORKDIR ${WORKSPACE}

# Build numpy to populate the cache used by ccache
RUN git submodule update --init --depth=1 -- numpy/core/src/umath/svml
RUN conda activate ${CONDA_ENV} && \ 
    python setup.py build_ext --inplace && \
    ccache -s

# Gitpod will load the repository into /workspace/numpy. We remove the
# directory from the image to prevent conflicts
RUN rm -rf ${WORKSPACE}

# -----------------------------------------------------------------------------
# Always return to non privileged user
USER gitpod
