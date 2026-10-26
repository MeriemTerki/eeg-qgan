@echo off
cd /d "%~dp0\.."
echo Preparing samples if needed...
if not exist demo\samples\catalog.json python demo\prepare.py
echo Starting demo at http://127.0.0.1:8000
python demo\server.py
