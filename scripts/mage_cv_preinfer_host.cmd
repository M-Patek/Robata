@echo off
setlocal
python "%~dp0mage_cv_preinfer_host.py" %*
exit /b %ERRORLEVEL%
