def test_compile():
    
    import numpy.f2py
    source = '''
        function foo(input)
            implicit none
            integer :: foo
            integer, intent(in) :: input

            foo = input + 3
        end function
    '''
    numpy.f2py.compile(source, modulename='bar')
    
    import bar
    a = bar.foo(10)
    assert a == 13
