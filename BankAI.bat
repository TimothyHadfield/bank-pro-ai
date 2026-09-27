@echo off
title Bank AI Dashboard
cd /d "%~dp0bank-ai-v1-ppo"

REM Enable torch.compile in self-play workers (~30-100% speedup if it works,
REM falls back silently to eager mode if it doesn't).
set BANK_AI_TORCH_COMPILE=1

echo.
echo ============================================
echo   Bank AI Dashboard (v1 + v2 training)
echo ============================================
echo.
echo Starting server. Browser will open in ~5 seconds.
echo Keep this window open while you use the dashboard.
echo Press Ctrl+C to stop.
echo.

REM Open browser after a brief delay so the server has time to start
start "" /b cmd /c "timeout /t 4 /nobreak >nul && start http://localhost:5001"

REM Try the Windows py launcher first, then fall back to python
where py >nul 2>nul
if %errorlevel% equ 0 (
    py dashboard\server.py
) else (
    python dashboard\server.py
)

echo.
echo Server stopped.
pause
