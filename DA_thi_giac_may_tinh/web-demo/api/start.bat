@echo off
echo.
echo  ====================================
echo   DefectDetect OBB — FastAPI Server
echo  ====================================
echo.
echo  Dang khoi dong FastAPI tai http://localhost:8000
echo  Nhan Ctrl+C de dung
echo.

cd /d "%~dp0"

REM Cai dependencies neu chua co
pip show fastapi >nul 2>&1 || pip install -r requirements.txt

REM Chay server
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
