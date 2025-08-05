@echo off
title Elliott Wave Bot - Offline Launch

echo.
echo ============================================================
echo    ELLIOTT WAVE BOT - OFFLINE MODE LAUNCHER
echo ============================================================
echo.
echo 🌊 Professional Elliott Wave Analysis System
echo 🔌 NO NETWORK REQUIRED - Fully Offline Operation
echo 📈 Built-in sample data for immediate analysis
echo 🎯 Zero external dependencies for data
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.7+ first.
    echo 💡 Download from: https://python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python installation verified
echo.

REM Check if we're in the right directory
if not exist "web\app_offline.py" (
    echo ❌ Error: app_offline.py not found!
    echo 📁 Current directory: %cd%
    echo 🔧 Please run this from the Elliott Bot project directory.
    pause
    exit /b 1
)

echo ✅ Offline application found
echo.

REM Check for basic Python packages
echo 🔍 Checking required packages...
python -c "import flask, pandas, numpy, plotly" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Some packages missing. Installing basic requirements...
    python -m pip install flask pandas numpy plotly >nul 2>&1
    if errorlevel 1 (
        echo ❌ Failed to install packages. Manual installation required:
        echo    pip install flask pandas numpy plotly
        pause
        exit /b 1
    )
    echo ✅ Packages installed successfully
) else (
    echo ✅ All required packages available
)

echo.

REM Kill any existing processes on port 5000
echo 🔧 Checking port 5000...
netstat -ano | findstr :5000 >nul 2>&1
if not errorlevel 1 (
    echo ⚠️  Port 5000 in use. Clearing...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5000') do (
        taskkill /F /PID %%a >nul 2>&1
    )
    timeout /t 2 >nul
)

echo ✅ Port 5000 ready
echo.

REM Start the offline application
echo 🚀 Starting Elliott Wave Bot (Offline Mode)...
echo ⏰ Please wait for the server to initialize...
echo.
echo ============================================================
echo 📊 FEATURES AVAILABLE:
echo    • Professional Elliott Wave Analysis
echo    • Built-in Sample Data (AAPL, BTC-USD, EURUSD, TSLA)
echo    • Interactive TradingView-style Charts
echo    • Pattern Recognition & Validation
echo    • Fibonacci Level Analysis
echo    • Future Price Projections
echo    • Complete Offline Operation
echo ============================================================
echo.

REM Try to start the offline version
python web\app_offline.py

if errorlevel 1 (
    echo.
    echo ❌ Failed to start offline version!
    echo.
    echo 🔧 TROUBLESHOOTING:
    echo    1. Check Python installation: python --version
    echo    2. Install packages: pip install flask pandas numpy plotly
    echo    3. Run as Administrator if needed
    echo    4. Check antivirus/firewall settings
    echo.
    echo 💡 MANUAL START COMMAND:
    echo    python web\app_offline.py
    echo.
) else (
    echo.
    echo ✅ Elliott Wave Bot started successfully!
    echo 🌐 Access at: http://localhost:5000
)

pause
