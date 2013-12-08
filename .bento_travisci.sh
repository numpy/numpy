export CI_ROOT=$PWD
cd ..

# Waf
wget http://waf.googlecode.com/files/waf-1.7.13.tar.bz2
tar xjvf waf-1.7.13.tar.bz2
cd waf-1.7.13
python waf-light
export WAFDIR=$PWD
cd ..

# Bento
wget https://github.com/cournape/Bento/archive/master.zip
unzip master.zip
cd Bento-master
python bootstrap.py
export BENTO_ROOT=$PWD
cd ..

cd $CI_ROOT

# In-place numpy build
$BENTO_ROOT/bentomaker build -i -j
# Prepend to PYTHONPATH so tests can be run
export PYTHONPATH=$PWD:$PYTHONPATH

