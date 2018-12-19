wget http://buildbot.pypy.org/nightly/py3.5/pypy-c-jit-latest-linux64.tar.bz2 -O pypy.tar.bz2
mkdir -p pypy3.5-latest
(cd pypy3.5-latest; tar --strip-components=1 -xf ../pypy.tar.bz2)
pypy3.5-latest/bin/pypy3 -mensurepip
pypy3.5-latest/bin/pypy3 -m pip install --upgrade pip setuptools
pypy3.5-latest/bin/pypy3 -m pip install --user cython==0.29.0 pytest pytz
pypy3.5-latest/bin/pypy3 runtests.py -- -rsx --junitxml=junit/test-results.xml --durations 10
# do not fail the CI run
echo ''
 
