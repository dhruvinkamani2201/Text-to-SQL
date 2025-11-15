"""
RAG Text->SQL Demo (Flask)
- Lightweight RAG using TF-IDF retrieval
- Optional OpenAI generation (set OPENAI_API_KEY env var)
- Fallback rule-based generator when no API key is present

How to run:
1) pip install -r requirements.txt
2) python rag_text_to_sql_rag.py
3) Open http://127.0.0.1:5000
"""

import os
import sqlite3
import re
import pandas as pd
import json
import math
import sqlparse
from flask import Flask, request, render_template_string
from pathlib import Path

# Retrieval utilities (TF-IDF)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).parent
EXCEL_NAME = "survey_lung_cancer.xlsx"
EXCEL_PATH = BASE_DIR / EXCEL_NAME
DB_PATH = BASE_DIR / "demo_dataset.db"

# Config
TOP_K = 4         # how many docs to retrieve
MAX_ROWS = 500    # safety row cap for queries

app = Flask(__name__)


###########################
# 1) Load Excel -> SQLite
###########################
def load_excel_to_sqlite(excel_path: Path, db_path: Path):
    xls = pd.ExcelFile(excel_path)
    conn = sqlite3.connect(str(db_path))
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        # normalize column names
        df.columns = [str(c).strip().replace(" ", "_").lower() for c in df.columns]
        tbl_name = sheet.strip().replace(" ", "_").lower()
        df.to_sql(tbl_name, conn, index=False, if_exists="replace")
    conn.commit()
    conn.close()

if not DB_PATH.exists():
    print("Creating SQLite DB from Excel...")
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Excel file not found: {EXCEL_PATH}")
    load_excel_to_sqlite(EXCEL_PATH, DB_PATH)
else:
    print("Using existing DB:", DB_PATH)

###########################
# 2) Build retrieval corpus
###########################
def introspect_db(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cur.fetchall()]
    schema_docs = []
    for t in tables:
        cols = [c[1] for c in conn.execute(f"PRAGMA table_info('{t}')").fetchall()]
        schema_docs.append({"title": f"table:{t}", "text": f"table {t} columns: {', '.join(cols)}"})
        # sample rows (up to 5)
        df = pd.read_sql_query(f"SELECT * FROM {t} LIMIT 5", conn)
        if not df.empty:
            schema_docs.append({"title": f"sample:{t}", "text": f"sample rows for {t}: {df.to_csv(index=False, header=True).strip()}"})
    conn.close()
    return schema_docs

# Create example NL->SQL pairs from schema heuristics for retrieval (helpful for prompting)
def build_example_pairs(db_path: Path):
    pairs = []
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cur.fetchall()]
    for t in tables:
        # simple examples
        pairs.append({"title": "example1", "text": f"NL: List all rows from {t}\nSQL: SELECT * FROM {t} LIMIT 100;"})
        pairs.append({"title": "example2", "text": f"NL: How many records in {t}\nSQL: SELECT COUNT(*) as count FROM {t};"})
        # try generating a group-by example if numeric columns present
        df = pd.read_sql_query(f"SELECT * FROM {t} LIMIT 100", conn)
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            pairs.append({"title": "example3", "text": f"NL: Average of {numeric_cols[0]} in {t}\nSQL: SELECT AVG({numeric_cols[0]}) as avg_{numeric_cols[0]} FROM {t};"})
    conn.close()
    return pairs

# Build corpus
corpus_docs = introspect_db(DB_PATH) + build_example_pairs(DB_PATH)
corpus_texts = [d["text"] for d in corpus_docs]
vectorizer = TfidfVectorizer(stop_words="english").fit(corpus_texts)
corpus_vecs = vectorizer.transform(corpus_texts)

def retrieve_context(nl_query: str, top_k: int = TOP_K):
    qv = vectorizer.transform([nl_query])
    sims = cosine_similarity(qv, corpus_vecs)[0]
    idxs = sims.argsort()[::-1][:top_k]
    retrieved = [corpus_docs[i]["text"] for i in idxs if sims[i] > 0]
    return retrieved

###########################
# 3) SQL generator
###########################
def build_prompt(nl_query: str, retrieved_context: list):
    system = (
        "You are a SQL generation assistant. Produce a single SQL SELECT statement only. "
        "Do NOT produce any INSERT/UPDATE/DELETE/CREATE/DROP statements. "
        "If a table or column is ambiguous, prefer returning a clarifying note instead of guessing. "
        "Enforce a LIMIT of at most 500 rows unless the user asked for aggregates."
    )
    ctx = "\n\n".join(retrieved_context)
    prompt = f"{system}\n\nCONTEXT:\n{ctx}\n\nNL_QUERY:\n{nl_query}\n\nSQL:"
    return prompt

# Fallback rule-based generator (improved version of earlier rule-based translator)
def nl_to_sql_fallback(nl: str, conn):
    text = nl.lower().strip()
    # find tables
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cur.fetchall()]
    col_map = {t: [c[1] for c in conn.execute(f"PRAGMA table_info('{t}')").fetchall()] for t in tables}
    # heuristic table selection
    def find_table():
        for t in tables:
            if t in text:
                return t
        for t, cols in col_map.items():
            for col in cols:
                if col in text:
                    return t
        return tables[0]
    table = find_table()
    # aggregates
    if re.search(r'\b(count|how many)\b', text):
        cond = ''
        m2 = re.search(r'where (.+)', text)
        if m2:
            cond = ' WHERE ' + _parse_condition(m2.group(1))
        return f"SELECT COUNT(*) as count FROM {table}{cond};"
    if 'average' in text or 'avg' in text:
        # choose numeric column if referenced
        for tcols in col_map.values():
            for c in tcols:
                if c in text and ('avg' in text or 'average' in text or c in text):
                    return f"SELECT AVG({c}) as avg_{c} FROM {table};"
        # fallback to first numeric
        df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 50", conn)
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                return f"SELECT AVG({c}) as avg_{c} FROM {table};"
    if 'sum' in text:
        df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 50", conn)
        for c in df.columns:
            if re.search(r'amount|total|cost|price|value', c):
                m = re.search(r'where (.+)', text)
                cond = ' WHERE ' + _parse_condition(m.group(1)) if m else ''
                return f"SELECT SUM({c}) as total_{c} FROM {table}{cond};"
    # list/select
    if any(w in text for w in ['list', 'show', 'get', 'who', 'what', 'which']):
        cols = '*'
        if 'name' in text:
            for tcols in col_map.values():
                for c in tcols:
                    if 'name' in c:
                        cols = c
                        break
        cond = ''
        m = re.search(r'where (.+)', text)
        if m:
            cond = ' WHERE ' + _parse_condition(m.group(1))
        if 'by' in text and 'group' in text:
            m2 = re.search(r'by ([a-z_]+)', text)
            if m2:
                grp = m2.group(1)
                return f"SELECT {grp}, COUNT(*) as cnt FROM {table} GROUP BY {grp};"
        if 'highest' in text or 'max' in text:
            df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 50", conn)
            for c in df.columns:
                if pd.api.types.is_numeric_dtype(df[c]):
                    return f"SELECT * FROM {table} ORDER BY {c} DESC LIMIT 1;"
        # fallback
        return f"SELECT {cols} FROM {table}{cond} LIMIT {min(200, MAX_ROWS)};"
    raise ValueError("Could not translate NL to SQL; please rephrase (e.g. 'List all where age > 60').")

def _parse_condition(cond_text: str) -> str:
    t = cond_text.strip().lower()
    # employee N
    m = re.match(r'([a-z_]+)\s*(>|<|=)\s*([\w\-]+)', t)
    if m:
        col, op, val = m.groups()
        if val.isdigit():
            return f"{col} {op} {val}"
        else:
            return f"{col} {op} '{val}'"
    m = re.match(r'([a-z_]+)\s*(?:is|equals|=)\s*([\w\-]+)', t)
    if m:
        col, val = m.groups()
        if val.isdigit():
            return f"{col} = {val}"
        else:
            return f"{col} = '{val}'"
    m = re.match(r'([a-z_]+)\s*(greater|less)\s*than\s*([\d]+)', t)
    if m:
        col, comp, val = m.groups()
        op = '>' if comp == 'greater' else '<'
        return f"{col} {op} {val}"
    # fallback simple 'col value'
    m = re.match(r'([a-z_]+)\s+([\w\-]+)', t)
    if m:
        col, val = m.groups()
        if val.isdigit():
            return f"{col} = {val}"
        else:
            return f"{col} = '{val}'"
    return cond_text


###########################
# 4) SQL Validation
###########################
BANNED_KEYWORDS = ['insert', 'update', 'delete', 'drop', 'create', 'alter', 'truncate', 'attach', 'detach']

def validate_and_sanitize_sql(sql: str) -> str:
    # remove trailing semicolons / whitespace
    sql = sql.strip().rstrip(';')
    parsed = sqlparse.parse(sql)
    if len(parsed) != 1:
        raise ValueError("Only single SQL statement allowed.")
    stmt = parsed[0]
    # check it's a SELECT
    first_token = stmt.token_first(skip_cm=True)
    if first_token is None or first_token.value.lower() != 'select':
        raise ValueError("Only SELECT statements are allowed.")
    # banned keywords
    low = sql.lower()
    for k in BANNED_KEYWORDS:
        if re.search(r'\b' + re.escape(k) + r'\b', low):
            raise ValueError(f"Disallowed keyword in SQL: {k}")
    # Ensure we have a LIMIT for safety; if absent and no aggregate, add LIMIT
    if 'limit' not in low:
        # naive check for aggregate keywords
        if not any(kw in low for kw in ('count(', 'avg(', 'sum(', 'min(', 'max(', 'group by', 'having')):
            sql = f"{sql} LIMIT {MAX_ROWS}"
    return sql

###########################
# 5) End-to-end utils
###########################
def generate_sql(nl_query: str):
    # retrieve context
    retrieved = retrieve_context(nl_query)
    # fallback: rule-based
    conn = sqlite3.connect(str(DB_PATH))
    try:
        sql = nl_to_sql_fallback(nl_query, conn)
    finally:
        conn.close()
    return sql, retrieved

###########################
# 6) Flask UI
###########################
TEMPLATE = """
<!doctype html>
<title>RAG Text->SQL Demo</title>
<h1>RAG Text → SQL Demo</h1>
<p>Dataset: <b>{{excel_name}}</b></p>
<form method=post action="/query">
  <textarea name=nl rows=4 cols=90 placeholder="Ask a question in natural language...">{{ request.form.get('nl','') }}</textarea><br>
  <label>Top-K retrieval:</label>
  <input type="number" name="topk" value="{{topk}}" min=1 max=10 />
  <input type=submit value="Run">
</form>
{% if nl %}
<hr>
<h3>Natural language</h3>
<pre>{{nl}}</pre>
<h3>Retrieved context (top {{topk}})</h3>
<pre>{{retrieved}}</pre>
<h3>Translated SQL</h3>
<pre>{{sql}}</pre>
<h3>Results (first 500 rows)</h3>
{{table|safe}}
{% endif %}
<hr>
<h4>Available tables & columns</h4>
<pre>{{schema}}</pre>
"""

def get_schema_text():
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cur.fetchall()]
    lines = []
    for t in tables:
        cols = [c[1] for c in conn.execute(f"PRAGMA table_info('{t}')").fetchall()]
        lines.append(f"{t}: {', '.join(cols)}")
    conn.close()
    return "\n".join(lines)

@app.route('/', methods=['GET'])
def index():
    return render_template_string(TEMPLATE, excel_name=EXCEL_NAME, nl=None, sql=None, table=None, retrieved=None, schema=get_schema_text(), topk=TOP_K)

@app.route('/query', methods=['POST'])
def query():
    nl = request.form.get('nl', '').strip()
    topk = int(request.form.get('topk', TOP_K))

    if not nl:
        return render_template_string(
            TEMPLATE, excel_name=EXCEL_NAME, nl=None, sql=None, 
            table=None, retrieved=None, schema=get_schema_text(), topk=topk
        )

    # 🔹 retrieve context dynamically based on topk
    retrieved = retrieve_context(nl, top_k=topk)

    try:
        sql, _ = generate_sql(nl)   # unchanged
    except Exception as e:
        return render_template_string(
            TEMPLATE, excel_name=EXCEL_NAME, nl=nl, 
            sql=f"ERROR: {e}", table=None, 
            retrieved='\n'.join(retrieved), schema=get_schema_text(), topk=topk
        )

    try:
        safe_sql = validate_and_sanitize_sql(sql)
    except Exception as e:
        return render_template_string(
            TEMPLATE, excel_name=EXCEL_NAME, nl=nl, 
            sql=f"Validation error: {e}\nGenerated SQL:\n{sql}", 
            table=None, retrieved='\n'.join(retrieved), 
            schema=get_schema_text(), topk=topk
        )

    # 🔹 Execute SQL safely
    conn = sqlite3.connect(str(DB_PATH))
    try:
        df = pd.read_sql_query(safe_sql, conn)
        table_html = df.to_html(index=False)
    except Exception as e:
        table_html = f"<pre>SQL execution error: {e}</pre>"
    finally:
        conn.close()

    return render_template_string(
        TEMPLATE, excel_name=EXCEL_NAME, nl=nl, sql=safe_sql, 
        table=table_html, retrieved='\n\n'.join(retrieved), 
        schema=get_schema_text(), topk=topk
    )

if __name__ == '__main__':
    print("Open the app at http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)
