FROM gitpod/workspace-full

RUN sudo apt-get update && \
    sudo apt-get install -y ccache gfortran && \
    sudo rm -rf /var/lib/apt/lists/*

RUN sudo /usr/sbin/update-ccache-symlinks && \
    echo 'export PATH="/usr/lib/ccache:$PATH"' | tee -a ~/.bashrc

COPY --chown=gitpod . /workspace/numpy

RUN pip install --no-cache-dir -r /workspace/numpy/test_requirements.txt

# Build numpy to populate ccache
RUN cd /workspace/numpy && \
    PATH="/usr/lib/ccache:$PATH" python setup.py build_ext -i && \
    ccache -s

# gitpod will load the repository into /workspace/numpy. We remove the
# directoy from the image to prevent conflicts.
RUN sudo rm -rf /workspace/numpy
