@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
call D:\anaconda3\Scripts\activate.bat MWDCsAI
python -m streamlit run app_unified.py
pause
