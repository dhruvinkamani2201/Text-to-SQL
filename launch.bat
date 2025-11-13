@echo off
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
python rag_text_to_sql_rag.py
pause
