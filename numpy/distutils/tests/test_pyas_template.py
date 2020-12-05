import os, pytest
from numpy.distutils.pyas_template import process_str, process_file

def pyas_file(tmpdir, astr, filename="dummy.test.pyas"):
    source = tmpdir.join(filename)
    source.write(astr)
    out = tmpdir.join('.'.join(filename.split('.')[:-1]))
    process_file(source, out)
    return out.read(), source

def pyas(astr, filename=__file__):
    return process_str(astr, filename).getvalue()

def test_inline():
    hw = pyas('inline <?p("py")?> !')
    assert hw == 'inline py !'
    hw = pyas('inline <%py%> !')
    assert hw == 'inline py !'

def test_multi():
    txt = ''' <%

        hi f-string

    %> !

    <%
        bla bla
    %>
    '''
    assert pyas(txt) == txt.replace('<%', '').replace('%>', '')

    txt = '''
        <?<%

            hi f-string

        %>?>

        xxxxxx

        <?<%
            xxxxxxxxxx
        %>?>
    !'''
    assert pyas(txt) == txt.replace('<?<%', '').replace('%>?>', '')

def test_ffstring_trans():
    assert pyas("<?Hi='hi'?><%$Hi f-string $Hi%>") == "hi f-string hi"
    assert pyas("<?Hi='hi'; p(<%%$Hi f-string $Hi%%>) ?>") == "hi f-string hi"
    assert pyas("<?Hi='hi'?><%$hi f-string $hi%>") == "$hi f-string $hi"
    assert pyas("<?tr(Hi='hi')?><%Hi f-string Hi%>") == "hi f-string hi"
    assert pyas("<?Hi='hi'?><%{{Hi}} f-string {{Hi}}%>") == "hi f-string hi"

@pytest.mark.parametrize('tabs', range(0, 10))
def test_ffstring_indent(tabs):
    assert len(pyas(f"<%{tabs}    x%>")) == tabs*4 + len('x')

def test_missing_tag():
    pytest.raises(SyntaxError, pyas, "<?")
    pytest.raises(SyntaxError, pyas, "<? <%% ?>")
    pytest.raises(SyntaxError, pyas, "<? <% ?>")
    pytest.raises(SyntaxError, pyas, "<? <% %>")
    pytest.raises(SyntaxError, pyas, "<? <%\n %> ?>\n <? <%% ?>")

def test_lineno():
    lineno = int(pyas("x:<%{{lineno()}}%>").split(':')[1])
    assert lineno == 1
    lineno = int(pyas("x\n:<%{{lineno()}}%>").split(':')[1])
    assert lineno == 2
    lineno = int(pyas("x\n:<%\n{{lineno()}}%>").split(':')[1])
    assert lineno == 3
    lineno = int(pyas("x\n:<?\np(\nlineno())?>").split(':')[1])
    assert lineno == 4
    lineno = int(pyas("x\n<?\n?><%%%%>:<?\np(\nlineno())?>").split(':')[1])
    assert lineno == 5

def test_filename(tmpdir):
    filename, source = pyas_file(tmpdir, "<%{{filename()}}%>")
    assert filename == source
    filename = pyas("<%{{filename()}}%>")
    assert filename == __file__
    filename = pyas("<%{{filename()}}%>", "../setup.py")
    assert filename == "../setup.py"

def test_include(tmpdir):
    file = tmpdir.join("test_include.pyas")
    file.write(r"test include")
    out, _ = pyas_file(tmpdir, f"<?include(r'{file}')?>")
    assert out == 'test include'
    multi_include = f"<?for _ in range(10): include(r'{file}')?>"
    out = pyas(multi_include)
    assert out == 'test include'*10
