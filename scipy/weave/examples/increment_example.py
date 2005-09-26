# examples/increment_example.py

#from weave import ext_tools

# use the following so that development version is used.
import sys
sys.path.insert(0,'..')
import ext_tools

def build_increment_ext():
    """ Build a simple extension with functions that increment numbers.
        The extension will be built in the local directory.
    """        
    mod = ext_tools.ext_module('increment_ext')

    a = 1 # effectively a type declaration for 'a' in the 
          # following functions.

    ext_code = "return_val = PyInt_FromLong(a+1);"    
    func = ext_tools.ext_function('increment',ext_code,['a'])
    mod.add_function(func)
    
    ext_code = "return_val = PyInt_FromLong(a+2);"    
    func = ext_tools.ext_function('increment_by_2',ext_code,['a'])
    mod.add_function(func)
            
    mod.compile()

if __name__ == "__main__":
    try:
        import increment_ext
    except ImportError:
        build_increment_ext()
        import increment_ext
    a = 1
    print 'a, a+1:', a, increment_ext.increment(a)
    print 'a, a+2:', a, increment_ext.increment_by_2(a)    
