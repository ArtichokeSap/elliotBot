@echo off
echo ===============================================
echo    Elliott Wave Web App with Technical Analysis
echo    Enhanced Elliott Wave + Technical Confluence
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

echo Starting Enhanced Elliott Wave Web Application...
echo.
echo Features:
echo   🌊 Elliott Wave Analysis
echo   🧩 Technical Confluence (6 methods)
echo   🎯 High-Probability Target Zones
echo   📊 Multi-Exchange Data (Binance, Bybit)
echo   📈 Data-Only Mode (Charts disabled)
echo   🌐 REST API Endpoints
echo.
echo Web Interface: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.

cd /d "c:\Users\Emre Yılmaz\projects\elliottBot"
python web/app.py

pause
