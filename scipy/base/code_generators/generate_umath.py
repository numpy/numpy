
import string
import re

Zero = "PyUFunc_Zero"
One = "PyUFunc_One"
None_ = "PyUFunc_None"
#each entry in defdict is 

#name: [string of chars for which it is defined,
#	string of characters using func interface,
#	tuple of strings giving funcs for data,
#       (in, out), or (instr, outstr) giving the signature as character codes,
#       identity,
#	docstring,
#       output specification (optional)
#       ]

all = '?bBhHiIlLqQfdgFDGO'
ints = 'bBhHiIlLqQ'
intsO = ints + 'O'
bintsO = '?'+ints+'O'
flts = 'fdg'
fltsO = flts+'O'
fltsM = flts+'M'
cmplx = 'FDG'
cmplxO = cmplx+'O'
cmplxM = cmplx+'M'
noint = flts+cmplx+'O'
nointM = flts+cmplx+'M'
all = '?'+ints+flts+cmplxO
nobool = all[1:]
nobool_or_obj = all[1:-1]
intflt = ints+flts
nocmplx = '?'+ints+flts
nocmplxO = nocmplx+'O'
nocmplxM = nocmplx+'M'
noobj = all[:-1]

defdict = {
'add': [all,'O',("PyNumber_Add",),
        (2,1), Zero,
        "Add the arguments elementwise."
        ],
'subtract' : [all,'O',("PyNumber_Subtract",),
              (2,1), Zero,
              "Subtract the arguments elementwise."
              ],
'multiply' : [all,cmplxO,
              ("prod,"*3,"PyNumber_Multiply",),
              (2,1), One,
              "Multiply the arguments elementwise."
              ],
'divide' : [nobool,cmplxO,
            ("quot,"*3,"PyNumber_Divide",),
            (2,1), One,
            "Divide the arguments elementwise."
            ],
'floor_divide' : [nobool, cmplxO,
                  ("floor_quot,"*3,
                   "PyNumber_FloorDivide"),
                  (2,1), One,
                  "Floor divide the arguments elementwise."
                  ],
'true_divide' : [nobool, cmplxO,
                 ("quot,"*3,"PyNumber_TrueDivide"),
                 (2,1), One,
                 "True divide the arguments elementwise.",
                 'f'*4+'d'*6+flts+cmplxO
                 ],
'conjugate' : [nobool_or_obj, 'M',
               ('"conjugate"',),
               (1,1), None,
               "take the conjugate elementwise."
               ],
'remainder' : [intflt,fltsO,
               ("fmod,"*3, "PyNumber_Remainder"),
               (2,1), Zero,
               "remainder(x,y) returns x % y elementwise."
               ],
'power' : [nobool,noint,
           ("pow,"*6,
            "PyNumber_Power"),
           (2,1), One,
           "power(x,y) == x**y elementwise."
           ],
'absolute' : [all,'O',
              ("PyNumber_Absolute",),
              (1,1), None,
              "absolute value elementwise."
              ],
'negative' : [all,cmplxO,
              ("neg,"*3,"PyNumber_Negative"),
              (1,1), None,
              "negative elements",
              ],
'greater' : [all,'',(),(2,1), None,
             "greater(x,y) is the bool array x > y.",
             '?'*len(all)
             ],
'greater_equal' : [all,'',(),(2,1), None,
                   "greater_equal(x,y) is the bool array x >= y.",
                   '?'*len(all)
             ],
'less' : [all,'',(),(2,1), None,
             "less(x,y) is the bool array x < y.",
          '?'*len(all)
             ],
'less_equal' : [all,'',(),(2,1), None,
                "less_equal(x,y) is the bool array x <= y.",
                '?'*len(all)
             ],
'equal' : [all, '', (), (2,1), None,
           "equal(x,y) is the bool array x == y.",
           '?'*len(all)
           ],
'not_equal' : [all, '', (), (2,1), None,
               "not_equal(x,y) is the bool array x != y.",
               '?'*len(all)
               ],
'logical_and': [nocmplxM,'M',('"logical_and"',),
                (2,1), One,               
                "logical and of arguments elementwise.",
                '?'*len(nocmplxM)
                ],
'logical_or': [nocmplxM,'M',('"logical_or"',),
                (2,1), Zero,               
                "logical or of arguments elementwise.",
               '?'*len(nocmplxM)
               ],
'logical_xor': [nocmplxM, 'M', ('"logical_xor"',),
                (2,1), None,
                "logical xor of arguments elementwise.",
                '?'*len(nocmplxM)
                ],
'logical_not' : [nocmplxM, 'M', ('"logical_not"',),
                 (1,1), None,
                 "logical not of argument elementwise.",
                 '?'*len(nocmplxM)
                 ],
'maximum' : [noobj,'',(),
             (2,1), None,
             "maximum of two arrays elementwise."],
'minimum' : [noobj,'',(),
             (2,1), None,
             "minimum of two arrays elementwise."],
'bitwise_and' : [bintsO,'O',("PyNumber_And",),
                 (2,1), One,
                 "bitwise and of two arrays elementwise."],
'bitwise_or' : [bintsO, 'O', ("PyNumber_Or",),
                (2,1), Zero,
                "bitwise or of two arrays elementwise."],
'bitwise_xor' : [bintsO, 'O', ("PyNumber_Xor",),
                 (2,1), None,
                 "bitwise xor of two arrays elementwise."],
'invert' : [bintsO,'O', ("PyNumber_Invert",),
            (1,1), None,
            "bit inversion elementwise."
            ],
'left_shift' : [intsO, 'O', ("PyNumber_Lshift",),
                (2,1), None,
                "left_shift(n,m) is n << m (n shifted to left by m bits) elementwise."
                ],
'right_shift' : [intsO, 'O', ("PyNumber_Rshift",),
                (2,1), None,
                "right_shift(n,m) is n >> m (n shifted to right by m bits) elementwise."
                ],
'arccos' : [nointM, nointM,
            ("acos,"*6, '"arccos"'),
            (1, 1), None,
            "inverse cosine elementwise."
            ],
'arcsin': [nointM, nointM,
            ("asin,"*6, '"arcsin"'),
            (1, 1), None,
            "inverse sine elementwise."
            ],
'arctan': [nointM, nointM,
            ("atan,"*6, '"arctan"'),
            (1, 1), None,
            "inverse tangent elementwise."
            ],
'arccosh' : [nointM, nointM,
            ("acosh,"*6, '"arccosh"'),
            (1, 1), None,
            "inverse hyperbolic cosine elementwise."
            ],
'arcsinh': [nointM, nointM,
            ("asinh,"*6, '"arcsinh"'),
            (1, 1), None,
            "inverse hyperbolic sine elementwise."
            ],
'arctanh': [nointM, nointM,
            ("atanh,"*6, '"arctanh"'),
            (1, 1), None,
            "inverse hyperbolic tangent elementwise."
            ],
'cos': [nointM, nointM,
        ("cos,"*6, '"cos"'),
        (1, 1), None,
        "cosine elementwise."
        ],
'sin': [nointM, nointM,
        ("sin,"*6, '"sin"'),
        (1, 1), None,
        "sine elementwise."
        ],
'tan': [nointM, nointM,
        ("tan,"*6, '"tan"'),
        (1, 1), None,
        "tangent elementwise."
        ],
'cosh': [nointM, nointM,
        ("cosh,"*6, '"cosh"'),
        (1, 1), None,
        "hyperbolic cosine elementwise."
        ],
'sinh': [nointM, nointM,
        ("sinh,"*6, '"sinh"'),
        (1, 1), None,
        "hyperbolic sine elementwise."
        ],
'tanh': [nointM, nointM,
        ("tanh,"*6, '"tanh"'),
        (1, 1), None,
        "hyperbolic tangent elementwise."
        ],
'exp' : [nointM, nointM,
         ("exp,"*6, '"exp"'),
         (1, 1), None,
         "e**x elementwise."
         ],
'log' : [nointM, nointM,
         ("log,"*6, '"log"'),
         (1, 1), None,
         "log(x) elementwise."
         ],
'log10' : [nointM, nointM,
         ("log10,"*6, '"log10"'),
         (1, 1), None,
         "log10(x) elementwise."
         ],
'sqrt' : [nointM, nointM,
          ("sqrt,"*6, '"sqrt"'),
          (1,1), None,
          "sqrt(x) elementwise."
          ],
'ceil' : [fltsM, fltsM,
          ("ceil,"*3, '"ceil"'),
          (1,1), None,
          "elementwise smallest integer >= x."
    ],
'fabs' : [fltsM, fltsM,
          ("fabs,"*3, '"fabs"'),
          (1,1), None,
          "absolute values."
          ],
'floor' : [fltsM, fltsM,
           ("floor,"*3, '"floor"'),
           (1,1), None,
           "elementwise largest integer <= x"
           ],
'arctan2' : [fltsM, fltsM,
             ("atan2,"*3, '"arctan2"'),
             (2,1), None,
             "arctan2(x,y) is a safe and corrent tan(x/y)"
             ],
'fmod' : [fltsM, fltsM,
          ("fmod,"*3,'"fmod"'),
          (2,1), None,
          "fmod(x,y) is remainder(x,y)"],
'hypot' : [fltsM, fltsM,
           ("hypot,"*3, '"hypot"'),
           (2,1), None,
           "sqrt(x**2 + y**2) elementwise"
           ],

'isnan' : [flts+cmplx, '',
           (), (1,1), None,
           "isnan(x) returns True where x is Not-A-Number",
           '?'*len(flts+cmplx)
           ],

'isinf' : [flts+cmplx, '',
           (), (1,1), None,
           "isinf(x) returns True where x is + or - infinity",
           '?'*len(flts+cmplx)
           ],

'isfinite' : [flts+cmplx, '',
           (), (1,1), None,
           "isfinite(x) returns True where x is finite",
           '?'*len(flts+cmplx)
           ],

'signbit' : [flts,'',
             (),(1,1),None,
             "signbit(x) returns True where signbit of x is set (x<0).",
             '?'*len(flts)
             ],

'modf' : [flts,'',
          (),(1,2),None,
          "modf breaks argument into integral and fractional parts (each with the same sign as input)"
          ]
}


def indent(st,spaces):
    indention = ' '*spaces
    indented = indention + string.replace(st,'\n','\n'+indention)
    # trim off any trailing spaces
    indented = re.sub(r' +$',r'',indented)
    return indented

chartoname = {'?': 'bool',
              'b': 'byte',
              'B': 'ubyte',
              'h': 'short',
              'H': 'ushort',
              'i': 'int',
              'I': 'uint',
              'l': 'long',
              'L': 'ulong',
              'q': 'longlong',
              'Q': 'ulonglong',
              'f': 'float',
              'd': 'double',
              'g': 'longdouble',
              'F': 'cfloat',
              'D': 'cdouble',
              'G': 'clongdouble',
              'O': 'OBJECT',
              'M': 'OBJECT',
              }

chartotype1 = {'f': 'f_f',
               'd': 'd_d',
               'g': 'g_g',
               'F': 'F_F',
               'D': 'D_D',
               'G': 'G_G',
               'O': 'O_O',
               'M': 'O_O_method'}

chartotype2 = {'f': 'ff_f',
               'd': 'dd_d',
               'g': 'gg_g',
               'F': 'FF_F',
               'D': 'DD_D',
               'G': 'GG_G',
               'O': 'OO_O',
               'M': 'O_O_method'}
#for each name
# 1) create functions, data, and signature
# 2) fill in functions and data in InitOperators
# 3) add function. 

def make_arrays(funcdict):
    # functions array contains an entry for every type implemented
    #   NULL should be placed where PyUfunc_ style function will be filled in later
    #
    code1list = []
    code2list = []
    for name, vals in funcdict.iteritems():
        funclist = []
        datalist = []
        siglist = []
        k=0;
        sub=0;
        numin, numout = vals[3]

        if numin > 1:
            thedict = chartotype2  # two inputs and one output
        else:                                                   
            thedict = chartotype1  # one input and one output

        instr = ''.join([x*numin for x in list(vals[0])])
        if len(vals) > 6:
            if isinstance(vals[6],type('')):
                outstr = vals[6]
            else:                # a tuple specifying input signature, output signature
                instr, outstr = vals[6]
        else:
            outstr = ''.join([x*numout for x in list(vals[0])])

        _valslen = len(vals[0])
        assert _valslen*numout == len(outstr), "input/output signature doesn't match"
        assert len(instr) == _valslen*numin, "input/output signature doesn't match"

        for char in vals[0]:
            if char in vals[1]:                # use generic function-based interface
                funclist.append('NULL')
                astr = '%s_functions[%d] = PyUFunc_%s;' % \
                       (name, k, thedict[char])
                code2list.append(astr)
                thisfunc = vals[2][sub]                
                if len(thisfunc) > 8 and thisfunc[:8] == "PyNumber":                    
                    astr = '%s_data[%d] = (void *) %s;' % \
                           (name, k, thisfunc)
                    code2list.append(astr)
                    datalist.append('(void *)NULL');
                else:
                    datalist.append('(void *)%s' % thisfunc)
                sub += 1
            else:                              # individual wrapper interface
                datalist.append('(void *)NULL');                
                funclist.append('%s_%s' % (chartoname[char].upper(), name))

            insubstr = instr[numin*k:numin*(k+1)]
            outsubstr = outstr[numout*k:numout*(k+1)]
            siglist.extend(['PyArray_%s' % chartoname[x].upper() for x in insubstr])
            siglist.extend(['PyArray_%s' % chartoname[x].upper() for x in outsubstr])
            k += 1
        funcnames = ', '.join(funclist)
        signames = ', '.join(siglist)
        datanames = ', '.join(datalist)
        code1list.append("static PyUFuncGenericFunction %s_functions[] = { %s };" \
                         % (name, funcnames))
        code1list.append("static void * %s_data[] = { %s };" \
                         % (name, datanames))
        code1list.append("static char %s_signatures[] = { %s };" \
                         % (name, signames))                        
    return "\n".join(code1list),"\n".join(code2list)

def make_ufuncs(funcdict):
    code3list = []
    for name, vals in funcdict.items():
        mlist = []
        mlist.append(\
r"""f = PyUFunc_FromFuncAndData(%s_functions, %s_data, %s_signatures, %d,
                                %d, %d, %s, "%s",
                                "%s", 0);""" % (name,name,name,len(vals[0]),
                                                vals[3][0], vals[3][1], vals[4],
                                                name, vals[5]))
        mlist.append(r"""PyDict_SetItemString(dictionary, "%s", f);"""%name)
        mlist.append(r"""Py_DECREF(f);""")
        code3list.append('\n'.join(mlist))        
    return '\n'.join(code3list)
        

def convert_vals(funcdict):
    for name, vals in funcdict.iteritems():
        if vals[4] is None:
            vals[4] = None_
        vals2 = vals[2]
        if len(vals2) > 0:
            alist = vals2[0].split(',')
            if len(alist) == 4:
                a = alist[0]
                if 'f' in vals[1]:
                    newlist = [ a+'f', a, a+'l']
                else:
                    newlist = ['nc_'+a+'f', 'nc_'+a, 'nc_'+a+'l']
            elif len(alist) == 7:
                a = alist[0]
                newlist = [a+'f', a, a+'l','nc_'+a+'f', 'nc_'+a, 'nc_'+a+'l']
            else:
                newlist = alist
            newlist = newlist + list(vals2[1:])
            vals[2] = tuple(newlist)
            funcdict[name] = vals


def make_code(funcdict,filename):
    convert_vals(funcdict)
    code1, code2 = make_arrays(funcdict)
    code3 = make_ufuncs(funcdict)
    code2 = indent(code2,4)
    code3 = indent(code3,4)
    code = r"""

/** Warning this file is autogenerated!!!

    Please make changes to the code generator program (%s)
**/

%s

static void
InitOperators(PyObject *dictionary) {
    PyObject *f;

%s
%s
}
""" % (filename, code1, code2, code3)
    return code;


if __name__ == "__main__":
    filename = __file__
    fid = open('__umath_generated.c','w')
    code = make_code(defdict, filename)
    fid.write(code)
    fid.close()
    










