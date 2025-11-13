# Text-to-SQL
This package provides a small RAG pipeline (TF-IDF retrieval  -> SQL)
and a Flask UI for interactive demos.

Files:
- rag_text_to_sql_rag.py    : Main Flask app
- survey_lung_cancer.xlsx   : Your Excel dataset (rename as necessary)
- requirements.txt          : Python deps
- launch.bat    : Simple launch scripts

How to run:
1) Create a folder and place all files inside it.
2) Create & activate a virtual env (recommended):
   python -m venv venv
   # Windows:
   venv\\Scripts\\activate

3) Install dependencies:
   pip install -r requirements.txt

4) Run the app:
   python rag_text_to_sql_rag.py

5) Open: http://127.0.0.1:5000

Demo tips:
- Use natural language queries like:
    "How many patients are there where age > 60?"
    "Average age by gender"
    "List rows where smoking_status is 'yes' and age > 50"

Security & Safety:
- Only SELECT statements are allowed.
- Row limits are enforced for safety.
- This is a demo — do NOT use in production without additional security (auth, rate limiting, query cost limits).
