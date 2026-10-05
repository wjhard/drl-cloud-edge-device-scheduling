@echo off
%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0demo\run_demo.ps1" %*
