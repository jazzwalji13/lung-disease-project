@echo off
echo ========================================
echo 🫁 LUNG DISEASE DETECTION SYSTEM
echo ========================================
echo.

cd /d D:\lung_disease_detector

echo 🔧 Checking Python installation...
python --version
if errorlevel 1 (
    echo ❌ Python not found! Please install Python first.
    pause
    exit
)

echo 📦 Installing Flask...
pip install flask==2.3.3

echo 🚀 Starting Web Server...
echo 📱 The website will open automatically...
echo 💡 If it doesn't open, go to: http://localhost:5000
echo.

timeout /t 3 /nobreak

start http://localhost:5000

python app.py

pause