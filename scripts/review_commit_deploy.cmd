@echo off
setlocal

set "SCRIPT_DIR=%~dp0"

where pwsh.exe >nul 2>nul
if %ERRORLEVEL%==0 (
    pwsh.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%commit-deploy-helper.ps1" %*
) else (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%commit-deploy-helper.ps1" %*
)

set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo Script failed with exit code %EXIT_CODE%.
    pause
)

exit /b %EXIT_CODE%
