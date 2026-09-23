@echo off
cd /d "%~dp0"
echo ===================================================
echo Starting Semantic Search Engine (Govt Scheme Finder)
echo Using isolated virtual environment: .\venv
echo ===================================================
.\venv\Scripts\python.exe -m streamlit run app.py
pause
