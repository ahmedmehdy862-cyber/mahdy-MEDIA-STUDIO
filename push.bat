@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo === Publish Local Studio to GitHub ===
echo 1) Create an EMPTY repo on github.com (no README, no .gitignore)
echo 2) Paste its address below.
set /p REPOURL=Repo URL (https://github.com/USER/REPO.git):
git remote remove origin 2>nul
git remote add origin "%REPOURL%"
git branch -M main
git push -u origin main
echo.
echo Done. Your link is: https://github.com/ ... (your repo page)
pause
