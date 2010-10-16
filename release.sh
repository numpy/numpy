#! /bin/sh
# script to build tarballs, mac os x and windows installers on mac os x
paver bootstrap
source bootstrap/bin/activate
python setupsconsegg.py install
paver sdist
paver dmg -p 2.6
paver dmg -p 2.5
export MACOSX_DEPLOYMENT_TARGET=10.5
paver dmg -p 2.7

paver bdist_superpack -p 3.1
paver bdist_superpack -p 2.7
paver bdist_superpack -p 2.6
paver bdist_superpack -p 2.5
paver write_release_and_log
