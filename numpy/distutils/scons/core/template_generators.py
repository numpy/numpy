import re
from os.path import basename as pbasename, splitext, join as pjoin, dirname as pdirname

from numpy.distutils.conv_template import process_file as process_c_file
from numpy.distutils.from_template import process_file as process_f_file

def do_generate_from_c_template(targetfile, sourcefile, env):
    t = open(targetfile, 'w')
    writestr = process_c_file(sourcefile)
    t.write(writestr)
    t.close()
    return 0

def do_generate_from_f_template(targetfile, sourcefile, env):
    t = open(targetfile, 'w')
    writestr = process_f_file(sourcefile)
    t.write(writestr)
    t.close()
    return 0

def generate_from_c_template(target, source, env):
    for t, s in zip(target, source):
        do_generate_from_c_template(str(t), str(s), env)
    return 0

def generate_from_f_template(target, source, env):
    for t, s in zip(target, source):
        do_generate_from_f_template(str(t), str(s), env)
    return 0

def generate_from_template_emitter(target, source, env):
    base, ext = splitext(pbasename(str(source[0])))
    t = pjoin(pdirname(str(target[0])), base)
    return ([t], source)
    
_INCLUDE_RE = re.compile(r"include\s*['\"](\S+)['\"]", re.M)

def generate_from_template_scanner(node, env, path, arg):
    print "SCANNING, YO !"
    cnt = node.get_contents()
    return _INCLUDE_RE.findall(cnt)
