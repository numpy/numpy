#! /bin/sh
# script to build tarballs, mac os x and windows installers on mac os x
paver bootstrap
source bootstrap/bin/activate
CFLAGS="-arch x86_64" FFLAGS="-arch x86_64" python setupsconsegg.py install
paver sdist
paver dmg -p 2.5
paver dmg -p 2.6
paver bdist_superpack -p 2.5
paver bdist_superpack -p 2.6
paver write_release_and_log
