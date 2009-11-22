This is a set of scripts used to build the new numpy .dmg installer with
documentation.

The actual content of the dmg is to be put in content: documentation go into
the Documentation subdir, and the .mpkg installer for numpuy itself in the
content directory. The name of the installer should match exactly the one in
the numpy script (otherwise, the background will not appear correctly).

The artwork is done in inkscape.

The main script (new-create-dmg) was taken from stackoverflow.
