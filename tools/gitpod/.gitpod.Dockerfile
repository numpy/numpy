# Doing a local shallow clone - keeps the container secure
# and much slimmer than usins COPY directly
ARG BASE_CONTAINER="trallard/numpy-dev:latest"
FROM gitpod/workspace-base:latest as clone

COPY --chown=gitpod . /tmp/numpy_repo
RUN git clone --depth 1 file:////tmp/numpy_repo /tmp/numpy

# -----------------------------------------------------------------------------
# Using the numpy-dev Docker image as a base
# This way, we ensure we have all the needed compilers and dependencies
# while reducing the build time
FROM ${BASE_CONTAINER} as build

USER root

# -----------------------------------------------------------------------------
# ---- ENV variables ----
# ---- Directories needed ----
ENV WORKSPACE=/workspace/numpy/ \
    CONDA_ENV=numpy-dev

# Allows this Dockerfile to activate conda environments
SHELL ["/bin/bash", "--login", "-o", "pipefail", "-c"]

# Install numpy dev dependencies
COPY --from=clone --chown=gitpod /tmp/numpy ${WORKSPACE}

WORKDIR ${WORKSPACE}

# Build numpy to populate the cache used by ccache
RUN conda activate ${CONDA_ENV} && \ 
    python setup.py build_ext --inplace && \
    ccache -s

# gitpod will load the repository into /workspace/numpy. We remove the
# directoy from the image to prevent conflicts
RUN rm -rf ${WORKSPACE}

# -----------------------------------------------------------------------------
USER gitpod
