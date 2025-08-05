@echo off
echo ===============================================
echo    Elliott Wave Technical Analysis System
echo    High-Probability Target Zone Detection
echo ===============================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if requirements are installed
echo Checking dependencies...
python -c "import numpy, pandas, requests, flask" >nul 2>&1
if errorlevel 1 (
    echo Installing required dependencies...
    pip install numpy pandas requests flask
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

echo Dependencies OK!
echo.

REM Show menu
:menu
echo ===============================================
echo            Choose Analysis Mode
echo ===============================================
echo 1. Quick Analysis (Command Line)
echo 2. Start API Server (Web Interface)
echo 3. Run Test Suite
echo 4. List Available Symbols
echo 5. Exit
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto quick_analysis
if "%choice%"=="2" goto start_api
if "%choice%"=="3" goto run_tests
if "%choice%"=="4" goto list_symbols
if "%choice%"=="5" goto exit
echo Invalid choice. Please try again.
goto menu

:quick_analysis
echo.
echo ===============================================
echo           Quick Analysis Mode
echo ===============================================
echo.
echo Examples:
echo   BTC/USDT
echo   ETH/USDT
echo   SOL/USDT
echo.
set /p symbol="Enter symbol to analyze (or press Enter for BTC/USDT): "
if "%symbol%"=="" set symbol=BTC/USDT

echo.
echo Timeframes: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w
set /p timeframe="Enter timeframe (or press Enter for 1h): "
if "%timeframe%"=="" set timeframe=1h

echo.
echo Exchanges: binance, bybit
set /p exchange="Enter exchange (or press Enter for binance): "
if "%exchange%"=="" set exchange=binance

echo.
echo Starting analysis of %symbol% on %exchange% (%timeframe%)...
echo.
python analyze.py %symbol% -t %timeframe% -e %exchange%

echo.
pause
goto menu

:start_api
echo.
echo ===============================================
echo          Starting API Server
echo ===============================================
echo.
echo API will be available at: http://localhost:5000
echo.
echo Endpoints:
echo   GET  /api/analyze?symbol=BTC/USDT^&timeframe=1h^&exchange=binance
echo   GET  /api/multi-timeframe?symbol=BTC/USDT^&exchange=binance
echo   GET  /api/confluence-details?symbol=BTC/USDT^&price=50000
echo.
echo Press Ctrl+C to stop the server
echo.
python src/api/technical_analysis_api.py
pause
goto menu

:run_tests
echo.
echo ===============================================
echo            Running Test Suite
echo ===============================================
echo.
python test_technical_analysis.py
echo.
pause
goto menu

:list_symbols
echo.
echo ===============================================
echo          Available Trading Symbols
echo ===============================================
echo.
echo Exchanges: binance, bybit
set /p exchange="Enter exchange (or press Enter for binance): "
if "%exchange%"=="" set exchange=binance

echo.
python analyze.py --list-symbols -e %exchange%
echo.
pause
goto menu

:exit
echo.
echo Thanks for using Elliott Wave Technical Analysis System!
echo.
pause
exit /b 0
