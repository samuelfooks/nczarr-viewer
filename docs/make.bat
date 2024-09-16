@ECHO OFF

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
    set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=.
set BUILDDIR=_build
set ALLSPHINXOPTS=-d %BUILDDIR%/doctrees %SPHINXOPTS% %SOURCEDIR%

if "%1" == "" goto help

%SPHINXBUILD% -M %1 %ALLSPHINXOPTS% %BUILDDIR% %2
goto end

:help
%SPHINXBUILD% -M help %ALLSPHINXOPTS% %BUILDDIR% %2

:end