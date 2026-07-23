@echo off
REM Start Movie Flow API (Windows)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
