from os.path import basename as pbasename, splitext, join as pjoin, dirname as pdirname

from numpy.distutils.conv_template import process_str as c_process_str
from numpy.distutils.from_template import process_str as f_process_str

def do_generate_from_c_template(targetfile, sourcefile, env):
    t = open(targetfile, 'w')
    s = open(sourcefile, 'r')
    allstr = s.read()
    s.close()
    writestr = c_process_str(allstr)
    t.write(writestr)
    t.close()
    return 0

def do_generate_from_f_template(targetfile, sourcefile, env):
    t = open(targetfile, 'w')
    s = open(sourcefile, 'r')
    allstr = s.read()
    s.close()
    writestr = f_process_str(allstr)
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
    
