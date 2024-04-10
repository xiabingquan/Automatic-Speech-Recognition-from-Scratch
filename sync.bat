@echo off

set default_commit_message="Auto commit by script"
echo Enter commit message (leave empty for default):
set /p commit_message=
if "%commit_message%" == "" (
  set commit_message=%default_commit_message%
)

echo 'Committing with message: %commit_message%'

git add .
git commit -m '%commit_message%'
git push origin main

echo Done.

pause
