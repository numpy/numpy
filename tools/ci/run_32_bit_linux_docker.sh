set -xe

git config --global --add safe.directory /numpy
cd /numpy
/opt/python/cp39-cp39/bin/python -mvenv venv
source venv/bin/activate
pip install -r ci_requirements.txt
python3 -m pip install -r test_requirements.txt
echo CFLAGS \$CFLAGS
spin config-openblas --with-scipy-openblas=32
export PKG_CONFIG_PATH=/numpy/.openblas
python3 -m pip install .
cd tools
python3 -m pytest --pyargs numpy
