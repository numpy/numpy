import numpy as np
import re

def new_pyfile(txtfile, nparray):
    filename = re.sub(r"\.txt", ".py", txtfile)
    f = open(filename, 'w')
    f.write('np.')
    f.write(np.array_repr(nparray))
    f.close()


def new_pyfile_masked(txtfile, masked_nparray):
    filename = re.sub(r"\.txt", "_masked.py", txtfile)
    f = open(filename, 'w')
    f.write('np.ma.')
    f.write(np.array_repr(masked_nparray.data))
    f.write(masked_nparray.mask)
    f.close()


#This was just used to add the masked tests originally.
#It shouldn't be used for anything else, and will be removed
#from this file before the pull request to the main
#repository is made.
def generate_masked_tests(test_file):

    tf = open(test_file)
    for l in tf:
        if re.match(r'\s*check_datafile\(\"', l):
            if ')' not in l:
                l = l.strip() + next(tf).strip()
            com = re.sub(r'check_datafile', 'np.loadtable',l)
            com = re.sub(r', NA_re = None', '', com)
            x = eval(com)
            print com
            txtfile = re.findall(r'\"([a-z._1-9]*)\"',com)[0]
            new_pyfile_masked(txtfile,x)





