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