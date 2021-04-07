FROM gitpod/workspace-full

ARG miniforge_version="4.10.0-0"
ARG conda_env=numpy-dev

ENV CONDA_DIR=/home/gitpod/mambaforge3
ENV PATH=$CONDA_DIR/bin:$PATH

SHELL ["/bin/bash", "--login", "-o", "pipefail", "-c"]

# Install mambaforge3
RUN wget -q -O mambaforge3.sh \
    https://github.com/conda-forge/miniforge/releases/download/${miniforge_version}/Mambaforge-${miniforge_version}-Linux-x86_64.sh && \
    bash mambaforge3.sh -p $CONDA_DIR -b && \
    rm mambaforge3.sh

# makes conda activate command for this Dockerfile
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile
# enables conda for interactive sessions
RUN conda init bash

# Install numpy dev dependencies
COPY environment.yml /tmp/environment.yml
RUN mamba env create -f /tmp/environment.yml -n ${conda_env} && \
    conda activate ${conda_env} && \
    mamba install ccache -y && \
    conda clean --all -f -y

# Set up ccache for compilers for this Dockerfile and interactino sessions
# Using `conda env config vars set` does not work with Docker
# REF: https://github.com/conda-forge/compilers-feedstock/issues/31
RUN conda activate ${conda_env} && \
    echo "conda activate ${conda_env}" >> ~/.startuprc && \
    echo "export CC=\"ccache $CC\"" >> ~/.startuprc && \
    echo "export CXX=\"ccache $CXX\"" >> ~/.startuprc && \
    echo "export F77=\"ccache $F77\"" >> ~/.startuprc && \
    echo "export F90=\"ccache $F90\"" >> ~/.startuprc && \
    echo "export GFORTRAN=\"ccache $GFORTRAN\"" >> ~/.startuprc && \
    echo "export FC=\"ccache $FC\"" >> ~/.startuprc && \
    echo "source $HOME/.startuprc" >> ~/.profile && \
    echo "source $HOME/.startuprc" >> ~/.bashrc

COPY --chown=gitpod . /workspace/numpy

RUN python /workspace/numpy/setup.py build_ext -i && \
    ccache -s

# gitpod will load the repository into /workspace/numpy. We remove the
# directoy from the image to prevent conflicts
RUN sudo rm -rf /workspace/numpy
