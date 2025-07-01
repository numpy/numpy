set -xe

WHEEL="$1"
DEST_DIR="$2"

cwd=$PWD
cd $DEST_DIR

# the libopenblas.dll is placed into this directory in the cibw_before_build
# script.
delvewheel repair --add-path $cwd/.openblas/lib -w $DEST_DIR $WHEEL
