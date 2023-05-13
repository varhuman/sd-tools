@echo off
echo Installing project dependencies...
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
echo Dependencies installed successfully!
pause