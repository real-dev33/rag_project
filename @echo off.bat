@echo off
:: Navigate to your project folder
cd /d E:\rag_project

:: Open Terminal 1 (The Server)
start "FastAPI Server" cmd /k "venv\Scripts\activate && uvicorn app.main:app --reload"

:: Open Terminal 2 (The Dev Environment)
start "Dev Terminal" cmd /k "venv\Scripts\activate"