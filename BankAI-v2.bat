@echo off
REM Bank AI v2 dashboard launcher.
REM Server takes ~3-5 seconds to start (imports torch).
REM It opens the browser automatically once ready, so don't pre-open it here.
cd /d "%~dp0bank-ai-v2-mcts"
python -m dashboard.server
pause
