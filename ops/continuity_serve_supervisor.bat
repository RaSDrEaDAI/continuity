@echo off
:: Continuity Serve API supervisor (T8) - logon-tier scheduled task
:: Relaunches the serve API if it exits. Port 8195 on the ZeroTier iface.
setlocal
cd /d D:\3.AIProjs\Continuity
for /f "tokens=2 delims==" %%a in ('findstr /b "ORCHESTRATOR_HEARTBEAT_SHARED_SECRET=" D:\3.AIProjs\2.Hermes\.env') do set "CONTINUITY_SECRET=%%a"
set "CONTINUITY_SECRET=%CONTINUITY_SECRET: =%"
:loop
echo [%date% %time%] starting continuity.api.serve >> logs\continuity_serve.log
.venv\Scripts\python.exe -m continuity.api.serve >> logs\continuity_serve.log 2>&1
echo [%date% %time%] exited rc=%errorlevel% - restarting in 5s >> logs\continuity_serve.log
timeout /t 5 /nobreak >nul
goto loop
