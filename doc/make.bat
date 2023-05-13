@echo off

pushd %~dp0

:: Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=LANG=C sphinx-build
)
set SOURCEDIR=source
set BUILDDIR=build
if defined SPHINXOPTS goto skipopts
set SPHINXOPTS=-W --keep-going -d build/doctrees %SPHINXOPTS% source
set DOXYGEN=doxygen
set FILES=
:skipopts

if "%1" == "" goto help
if "%1" == "clean" goto clean
if "%1" == "docenv" goto docenv
if "%1" == "html" goto html
if "%1" == "linkcheck" goto linkcheck
if "%1" == "show" goto show

:help
	echo.
	echo Please use "make.bat <target>" where ^<target^> is one of
	echo.
	echo    clean     to remove generated doc files and start fresh
	echo    docenv    make a virtual environment in which to build docs
	echo    html      to make standalone HTML files
	echo    linkcheck to check all external links for integrity
	echo    show      to show the html output in a browser
goto end

:clean
if exist "%SOURCEDIR%\build\" (
	rmdir /s /q "%SOURCEDIR%\build"
	:: TODO
	:: find . -name generated -type d -prune -exec rm -rf "{}" ";"
)
goto end

:docenv
echo Not implemented
Rem 	python -mvenv docenv
Rem 	( \
Rem             . docenv/bin/activate; \
Rem             pip install -q --upgrade pip; \
Rem             pip install -q  -r ../test_requirements.txt; \
Rem             pip install -q  -r ../doc_requirements.txt; \
Rem             pip install -q ..; \
Rem 	)
goto end

:html
%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:linkcheck
	md build
	md build\linkcheck
	md build\doctrees
	%SPHINXBUILD% -b linkcheck %SOURCEDIR% build\linkcheck
	echo.
	echo Link check complete; look for any errors in the above output
	echo    or in build\linkcheck\output.txt.
goto end

:show
python -m webbrowser -t "%~dp0\build\html\index.html"

:end
popd
