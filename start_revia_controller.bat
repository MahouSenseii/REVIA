@echo off
setlocal
set "ROOT=%~dp0"
set "VENV_PY=%ROOT%.venv\Scripts\python.exe"
set "MAIN=%ROOT%revia_controller_py\main.py"

if exist "%VENV_PY%" (
  "%VENV_PY%" "%MAIN%" %*
) else (
  python "%MAIN%" %*
)

