# Builds a development environment for gitpod by building numpy with
# ccache enabled. When gitpod is prebuilding or starting up, it should
# build numpy faster because of ccache.
FROM gitpod/workspace-base as clone

COPY --chown=gitpod . /tmp/numpy_repo

# We use a multistage build to create a shallow clone of the repo to avoid
# having the complete git history in the build stage and reducing the image
# size. During the build stage, the shallow clone is used to install the
# dependencies and build numpy to populate the cache used by ccache. Building
# numpy with setup.py uses versioneer.py which requires a git history.
RUN git clone --depth 1 file:////tmp/numpy_repo /tmp/numpy

FROM gitpod/workspace-base as build

# gitpod/workspace-base needs at least one file here
RUN touch /home/gitpod/.bashrc.d/empty

ARG MAMBAFORGE_VERSION="4.10.0-0"
ARG CONDA_ENV=numpy-dev

ENV CONDA_DIR=/home/gitpod/mambaforge3
ENV PATH=$CONDA_DIR/bin:$PATH

# Allows this Dockerfile to activate conda environments
SHELL ["/bin/bash", "--login", "-o", "pipefail", "-c"]

# Install mambaforge3
RUN wget -q -O mambaforge3.sh \
    https://github.com/conda-forge/miniforge/releases/download/$MAMBAFORGE_VERSION/Mambaforge-$MAMBAFORGE_VERSION-Linux-x86_64.sh && \
    bash mambaforge3.sh -p $CONDA_DIR -b && \
    rm mambaforge3.sh

# makes conda activate command for this Dockerfile
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile
# enables conda for interactive sessions
RUN conda init bash

# Install numpy dev dependencies
COPY --from=clone --chown=gitpod /tmp/numpy /workspace/numpy
RUN mamba env create -f /workspace/numpy/environment.yml -n $CONDA_ENV && \
    conda activate $CONDA_ENV && \
    mamba install ccache -y && \
    conda clean --all -f -y

# Set up ccache for compilers for this Dockerfile and interactino sessions
# Using `conda env config vars set` does not work with Docker
# REF: https://github.com/conda-forge/compilers-feedstock/issues/31
RUN echo "conda activate $CONDA_ENV" >> ~/.startuprc && \
    echo "export CC=\"ccache \$CC\"" >> ~/.startuprc && \
    echo "export CXX=\"ccache \$CXX\"" >> ~/.startuprc && \
    echo "export F77=\"ccache \$F77\"" >> ~/.startuprc && \
    echo "export F90=\"ccache \$F90\"" >> ~/.startuprc && \
    echo "export GFORTRAN=\"ccache \$GFORTRAN\"" >> ~/.startuprc && \
    echo "export FC=\"ccache \$FC\"" >> ~/.startuprc && \
    echo "source ~/.startuprc" >> ~/.profile && \
    echo "source ~/.startuprc" >> ~/.bashrc

# Build numpy to populate the cache used by ccache
RUN python /workspace/numpy/setup.py build_ext -i && \
    ccache -s

# gitpod will load the repository into /workspace/numpy. We remove the
# directoy from the image to prevent conflicts
RUN sudo rm -rf /workspace/numpy
