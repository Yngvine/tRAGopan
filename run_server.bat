@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

set "HOST=127.0.0.1"
set "PORT=8001"
set "LLM_PROVIDER=ollama"
set "OLLAMA_BASE_URL=https://eim-alu-83071.tail3405b4.ts.net/"
set "OLLAMA_MODEL=llama3-groq-tool-use"
set "AUTO_PORT=1"

:parse_args
if "%~1"=="" goto args_done

if /I "%~1"=="--host" (
  if "%~2"=="" goto missing_value
  set "HOST=%~2"
  shift
  shift
  goto parse_args
)

if /I "%~1"=="--port" (
  if "%~2"=="" goto missing_value
  set "PORT=%~2"
  shift
  shift
  goto parse_args
)

if /I "%~1"=="--provider" (
  if "%~2"=="" goto missing_value
  set "LLM_PROVIDER=%~2"
  shift
  shift
  goto parse_args
)

if /I "%~1"=="--ollama-url" (
  if "%~2"=="" goto missing_value
  set "OLLAMA_BASE_URL=%~2"
  shift
  shift
  goto parse_args
)

if /I "%~1"=="--ollama-model" (
  if "%~2"=="" goto missing_value
  set "OLLAMA_MODEL=%~2"
  shift
  shift
  goto parse_args
)

if /I "%~1"=="--no-auto-port" (
  set "AUTO_PORT=0"
  shift
  goto parse_args
)

if /I "%~1"=="-h" goto usage_ok
if /I "%~1"=="--help" goto usage_ok

echo Unknown option: %~1 1>&2
call :usage
exit /b 1

:missing_value
echo Missing value for option: %~1 1>&2
call :usage
exit /b 1

:usage_ok
call :usage
exit /b 0

:args_done
if not exist ".venv\Scripts\activate.bat" (
  echo Error: .venv not found. Create it first: python -m venv .venv 1>&2
  exit /b 1
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
  echo Error: failed to activate .venv 1>&2
  exit /b 1
)

if "%AUTO_PORT%"=="1" (
  call :is_port_in_use "%PORT%"
  if "!PORT_IN_USE!"=="1" (
    set "ORIGINAL_PORT=%PORT%"
    :find_next_free
    call :is_port_in_use "%PORT%"
    if "!PORT_IN_USE!"=="1" (
      set /a PORT+=1
      goto find_next_free
    )
    echo Port !ORIGINAL_PORT! is busy. Using !PORT! instead.
  )
) else (
  call :is_port_in_use "%PORT%"
  if "!PORT_IN_USE!"=="1" (
    echo Error: Port %PORT% is already in use.
    exit /b 1
  )
)

set "HOST=%HOST%"
set "PORT=%PORT%"
set "LLM_PROVIDER=%LLM_PROVIDER%"
set "OLLAMA_BASE_URL=%OLLAMA_BASE_URL%"
set "OLLAMA_MODEL=%OLLAMA_MODEL%"

echo Starting server with:
echo   HOST=%HOST%
echo   PORT=%PORT%
echo   LLM_PROVIDER=%LLM_PROVIDER%
if /I "%LLM_PROVIDER%"=="ollama" (
  echo   OLLAMA_BASE_URL=%OLLAMA_BASE_URL%
  echo   OLLAMA_MODEL=%OLLAMA_MODEL%
)

set "ACTION=from idena_service.app import app; import os; app.run(host=os.environ['HOST'], port=int(os.environ['PORT']), debug=False)"
python -c "%ACTION%"
set "EXIT_CODE=%ERRORLEVEL%"
exit /b %EXIT_CODE%

:is_port_in_use
set "PORT_IN_USE=0"
for /f "delims=" %%L in ('netstat -ano ^| findstr /R /C:":%~1 .*LISTENING"') do (
  set "PORT_IN_USE=1"
  goto :eof
)
goto :eof

:usage
echo Usage: run_server.bat [options]
echo.
echo Options:
echo   --host ^<host^>             Host to bind ^(default: 127.0.0.1^)
echo   --port ^<port^>             Preferred port ^(default: 8001^)
echo   --provider ^<provider^>     LLM provider: ollama^|rules ^(default: ollama^)
echo   --ollama-url ^<url^>        Ollama base URL ^(default: https://eim-alu-83071.tail3405b4.ts.net/^)
echo   --ollama-model ^<model^>    Ollama model ^(default: llama3-groq-tool-use^)
echo   --no-auto-port              Do not search for next free port if occupied
echo   -h, --help                  Show this help
echo.
echo Examples:
echo   run_server.bat
echo   run_server.bat --provider rules
echo   run_server.bat --ollama-url https://myhost:11434 --port 8002
goto :eof