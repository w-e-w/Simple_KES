@echo off
:: Check if Python is installed and available in PATH
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not added to PATH.
    exit /b 1
)

:: Run the Python script
echo Running install_requirements.py...
python install_requirements.py

:: Check if the script executed successfully
IF %ERRORLEVEL% EQU 0 (
    echo Requirements installation completed successfully.
) ELSE (
    echo An error occurred while installing requirements.
    exit /b 1
)

pause
