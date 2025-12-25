#Entire Table is taken as default
# server.py
import os
import re
import json
import traceback
from html import escape
from dotenv import load_dotenv
from openai import OpenAI
from flask import Flask, request, jsonify, send_from_directory, g, send_file
import duckdb
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo
import time
from utils import rows_to_html_table, plot_to_base64 # your existing utils
import uuid
from pathlib import Path
import sqlite3
import datetime as dtmod
from io import BytesIO  # in-memory Excel export
from sklearn.linear_model import LinearRegression

# CONFIG / ENV
# -------------------------------------------------
load_dotenv()

DEBUG_SQL =True

TZ = ZoneInfo("Asia/Kolkata")  # user timezone

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DUCKDB_FILE = os.getenv("DUCKDB_FILE", "./data/sales.duckdb")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
PORT = int(os.getenv("PORT", "3000"))
MAX_ROWS = int(os.getenv("MAX_ROWS", "500"))
QUERY_CACHE = {}      # { sql_text: { "response": result_dict, "time": timestamp } }
CACHE_TTL = 30        # cache lifespan in seconds (you can increase to 60 if needed)

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY must be set in environment")

client = OpenAI(api_key=OPENAI_API_KEY)

# directory for Excel exports (kept, but no longer used for auto-writing per-query)
EXPORT_DIR = Path("./data/exports")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# In-memory cache for exports: token -> DataFrame or dict of DataFrames
EXPORT_CACHE = {}

# -------------------------------------------------
# SAMPLE CHAT (QUESTIONS ONLY; ALWAYS RUNS ON LATEST DATA)
# -------------------------------------------------
SAMPLE_CHAT_ID = "sample-chat"
SAMPLE_CHAT_TITLE = "Sample Questions"
SAMPLE_CHAT_CREATED_AT = 1700000000000  # stable timestamp (ms)

SAMPLE_MESSAGES = [
    {
        "role": "bot",
        "text": "Here are sample questions you can run. Click a sample question and press “Run this question” to get the latest answer.",
        "table_html": "",
        "plot_data_uri": None,
        "download_url": None,
        "time": SAMPLE_CHAT_CREATED_AT,
    },
    {"role": "user", "text": "show me the column names", "time": SAMPLE_CHAT_CREATED_AT + 1},
    {"role": "user", "text": "give me any 5 account number from august 2025", "time": SAMPLE_CHAT_CREATED_AT + 2},
    {"role": "user", "text": "give details of the account #554517388438", "time": SAMPLE_CHAT_CREATED_AT + 3},
    {
        "role": "user",
        "text": "show me the total order,installation order,installation rate,profit,disconnection rate month wise where provider is spectrum for 2024",
        "time": SAMPLE_CHAT_CREATED_AT + 4,
    },
    {
        "role": "user",
        "text": "show me the total installation order where installed order =1 based on provider for 2025",
        "time": SAMPLE_CHAT_CREATED_AT + 5,
    },
    {"role": "user", "text": "give me the account number which had different customer names limit 10", "time": SAMPLE_CHAT_CREATED_AT + 6},
    {"role": "user", "text": "show me the unique product names", "time": SAMPLE_CHAT_CREATED_AT + 7},
    {"role": "user", "text": "give top 10 product name, their profit , profit rate based on profit rate for 2024", "time": SAMPLE_CHAT_CREATED_AT + 8},
    {"role": "user", "text": "show me the cancellation count for provider spectrum where status = cancelled for august 2025", "time": SAMPLE_CHAT_CREATED_AT + 9},
    {"role": "user", "text": "show me the distinct status values", "time": SAMPLE_CHAT_CREATED_AT + 10},
    {"role": "user", "text": "show me the top 5 company names based on their profit on 2025 in a pie chart", "time": SAMPLE_CHAT_CREATED_AT + 11}
]

SAMPLE_CHAT = {
    "id": SAMPLE_CHAT_ID,
    "title": SAMPLE_CHAT_TITLE,
    "created_at": SAMPLE_CHAT_CREATED_AT,
    "messages": SAMPLE_MESSAGES,
}

PRED_CHAT_ID = "predicted-chat"
print(">>> IMPORTING SERVER FILE:", __file__)
print(">>> PRED_CHAT_ID AT IMPORT:", PRED_CHAT_ID)

PRED_CHAT_TITLE = "Predicted Questions"
PRED_CHAT_CREATED_AT = 1700000001000

PRED_MESSAGES = [
  {"role":"bot","text":"Here are prediction sample questions. Click and press “Run this question”.","table_html":"","plot_data_uri":None,"download_url":None,"time":PRED_CHAT_CREATED_AT},
  {"role":"user","text":"show me the total order,installation order,installation rate and estimated installation rate month wise for provider spectrum on 2025.","time":PRED_CHAT_CREATED_AT+1},
  {"role":"user","text":"show me the disconnection rate and estimated disconnection rate month wise for provider spectrum on 2025.","time":PRED_CHAT_CREATED_AT+2},
  {"role":"user","text":"show me the installation rate,disconnection rate and estimated installation rate, estimated disconnection rate month wise for provider spectrum on 2025.","time":PRED_CHAT_CREATED_AT+3},
  {"role":"user","text":"show me the month wise profit rate and estimated profit rate for the product TV for 2025","time":PRED_CHAT_CREATED_AT+4},
  {"role":"user","text":"show month wise revenue for the product tv and also estimate the revenue for 2025","time":PRED_CHAT_CREATED_AT+5},
  {"role":"user","text":"what is the disconnected rate and estimated disconnected rate for the provider comcast for dec 2025.","time":PRED_CHAT_CREATED_AT+6},
  {"role":"user","text":"show me the installation and estimated installation based on mainchannel wise on 2025","time":PRED_CHAT_CREATED_AT+7},
  {"role":"user","text":"show me the top 10 product name, profit and estimated profit on 2025","time":PRED_CHAT_CREATED_AT+8},
  {"role":"user","text":"show me the top 10 provider name, disconnection orders,disconnected rate and estimated disconnection rate on october 2025","time":PRED_CHAT_CREATED_AT+9}
]

PRED_CHAT = {"id":PRED_CHAT_ID,"title":PRED_CHAT_TITLE,"created_at":PRED_CHAT_CREATED_AT,"messages":PRED_MESSAGES}

# DUCKDB CONNECTION
# -------------------------------------------------
db_path = DUCKDB_FILE.strip()
if not (db_path.lower().startswith("md:") or db_path.lower().startswith("md://")):
    db_path = os.path.abspath(db_path)
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DuckDB file not found: {db_path}")

def _get_duck_con():
    """
    Per-request connection for concurrency safety.
    IMPORTANT: Must match the same config everywhere to avoid:
      "Can't open a connection to same database file with a different configuration"
    """
    local_con = duckdb.connect(db_path, read_only=True)
    local_con.execute("PRAGMA enable_progress_bar=false;")
    local_con.execute("PRAGMA threads=4;")
    return local_con

# Detect tables / schema info (one-time metadata load using SAME config: read_only=True)
with duckdb.connect(db_path, read_only=True) as _meta_con:
    available_tables = [t[0] for t in _meta_con.execute("SHOW TABLES").fetchall()]
    print("Detected tables:", available_tables)

    TABLE = "aiv_OrderItemUnit_md"
    if TABLE not in available_tables:
        for t in available_tables:
            if t.lower() == TABLE.lower():
                TABLE = t
                break

    # Basic schema info
    cols_info = _meta_con.execute(f"PRAGMA table_info('{TABLE}')").fetchall()
    COLS = [c[1] for c in cols_info] if cols_info else []

    # Sample and row count
    SAMPLE_DF = _meta_con.execute(f"SELECT * FROM {TABLE} LIMIT 10").fetchdf() if COLS else pd.DataFrame()
    try:
        ROWCOUNT = _meta_con.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
    except Exception:
        ROWCOUNT = None

print(f"Using table: {TABLE} rows={ROWCOUNT}, cols={len(COLS)}")

# HELPER: column detection (revenue/profit/date)
# -------------------------------------------------
def find_best(cols, candidates):
    cols_lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand and cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

# Candidate lists
REVENUE_CANDS = [
    'NetAfterChargeback'
]
PROFIT_CANDS = ['Profit', 'NetProfit', 'ProfitAmount', 'Margin']
DATE_CANDS = ['sale_date_parsed', 'SaleDate', 'Date', 'OrderDate']

def qident(name: str) -> str:
    """Properly quote an identifier for DuckDB (safe, avoids f-string backslash parsing)."""
    return '"' + name.replace('"', '""') + '"'

DETECTED = {
    'date': find_best(COLS, DATE_CANDS),
    'revenue': find_best(COLS, REVENUE_CANDS),
    'profit': find_best(COLS, PROFIT_CANDS),
}
PACKAGE_CANDS = ["Package", "PackageName", "OrderPackage", "PkgName"]
DETECTED.update({
    "package": find_best(COLS, PACKAGE_CANDS),
})

print("Auto-detected columns:", DETECTED)
# ---------------------------
# DATE COLUMN CANDIDATES (ADD/UPDATE THESE ABOVE WHERE DETECTED IS BUILT)
# ---------------------------
SALE_DATE_CANDS = ["SaleDate", "sale_date_parsed", "OrderDate", "Date"]
INSTALL_DATE_CANDS = ["InstallDate", "install_date_parsed"]
DISCONNECT_DATE_CANDS = ["DisconnectDate", "disconnect_date_parsed"]

# IMPORTANT: after DETECTED is initially created, extend it like this:
# (keep your existing DETECTED keys too)
try:
    DETECTED.update({
        "sale_date": find_best(COLS, SALE_DATE_CANDS),
        "install_date": find_best(COLS, INSTALL_DATE_CANDS),
        "disconnect_date": find_best(COLS, DISCONNECT_DATE_CANDS),
    })
except Exception:
    pass


def get_sale_date_col():
    return DETECTED.get("sale_date") or find_best(COLS, SALE_DATE_CANDS)

def get_install_date_col():
    return DETECTED.get("install_date") or find_best(COLS, INSTALL_DATE_CANDS)

def get_disconnect_date_col():
    return DETECTED.get("disconnect_date") or find_best(COLS, DISCONNECT_DATE_CANDS)

def get_revenue_col():
    r = DETECTED.get('revenue')
    if r and r in COLS:
        return r
    for cand in REVENUE_CANDS:
        if cand in COLS:
            return cand
    return None

def get_profit_col():
    p = DETECTED.get('profit')
    if p and p in COLS:
        return p
    for cand in PROFIT_CANDS:
        if cand in COLS:
            return cand
    return None

def get_date_col():
    d = DETECTED.get('date')
    if d and d in COLS:
        return d
    for c in DATE_CANDS:
        if c in COLS:
            return c
    return COLS[0] if COLS else None

# handy flags for customer name logic
HAS_CUST_FIRST = any(c.lower() == "customerfirstname" for c in COLS)
HAS_CUST_LAST = any(c.lower() == "customerlastname" for c in COLS)
CUST_FIRST_COL = next((c for c in COLS if c.lower() == "customerfirstname"), None)
CUST_LAST_COL = next((c for c in COLS if c.lower() == "customerlastname"), None)

def run_sql_and_fetch_df(sql: str, print_errors: bool = True):
    """Execute SQL and return a pandas DataFrame."""
    if DEBUG_SQL:
        print("DEBUG SQL:\n", sql)
    try:
        with _get_duck_con() as local_con:
            df = local_con.execute(sql).fetchdf()
        return df
    except Exception as e:
        if print_errors:
            print("SQL execution error:", e, traceback.format_exc())
        raise

def _safe_user_error(msg=None):
    return {
        "reply": msg or "Sorry — I couldn't get that data. Please try rephrasing or ask fewer filters.",
        "table_html": "",
        "plot_data_uri": None,
    }

def _log_and_mask_error(e, context=""):
    print("INTERNAL ERROR:", context, str(e), traceback.format_exc())
    return "I couldn't complete that query due to an internal error. Try rephrasing or ask a simpler question."

def is_safe_select(sql: str) -> bool:
    """Allow SELECT or WITH...SELECT only. Block DML/DDL and multi-statement."""
    if not sql:
        return False

    # strip comments
    sql_clean = re.sub(r"--.*$", "", sql, flags=re.MULTILINE)
    sql_clean = re.sub(r"/\*.*?\*/", "", sql_clean, flags=re.DOTALL).strip()
    if not sql_clean:
        return False

    # block multiple statements (very common injection vector)
    # allow trailing semicolon, but not multiple ; or ; in the middle
    parts = [p.strip() for p in sql_clean.split(";") if p.strip()]
    if len(parts) != 1:
        return False
    sql_clean = parts[0]

    # must start with SELECT or WITH
    first_word = sql_clean.split()[0].upper()
    if first_word not in ("SELECT", "WITH"):
        return False

    # must contain a SELECT somewhere (CTE must eventually select)
    if not re.search(r"\bSELECT\b", sql_clean, flags=re.IGNORECASE):
        return False

    # block dangerous keywords (word-boundary)
    forbidden = [
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "REPLACE",
        "TRUNCATE", "MERGE", "VACUUM", "ATTACH", "DETACH", "COPY", "EXPORT",
        "CALL", "EXEC", "EXECUTE","PRAGMA", "SET", "LOAD", "INSTALL"
    ]
    for w in forbidden:
        if re.search(rf"\b{w}\b", sql_clean, flags=re.IGNORECASE):
            return False

    return True

# Columns to hide by default in table output + excel download
HIDDEN_COLS_DEFAULT = {
    "CompanyKey",
    "AffiliateCompanyKey",
    "CustomerKey",
    "CampaignPhoneKey",
    "OrderItemKey",
}

def _norm_key(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', (s or '').lower())

def _split_camel(s: str) -> str:
    if not s:
        return ""
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def _hidden_cols_user_requested(question: str):
    """
    Return a set of hidden column names that the user explicitly mentioned.
    Handles: CompanyKey, company key, company_key, etc.
    """
    q = (question or "").lower()
    q_norm = _norm_key(q)  # remove spaces/punct

    hits = set()
    for c in HIDDEN_COLS_DEFAULT:
        c_norm = _norm_key(c)              # "CompanyKey" -> "companykey"
        c_spaced = _split_camel(c).lower() # "CompanyKey" -> "company key"

        if c_norm and c_norm in q_norm:
            hits.add(c)
            continue

        if c_spaced and c_spaced in q:
            hits.add(c)
            continue

    return hits

def pick_anchor_date_col(question: str, requested_metrics: set) -> str:
    # If multiple metrics are requested together, anchor on SaleDate.
    if len(requested_metrics) >= 2:
        return get_sale_date_col() or "SaleDate"

    # Single-metric:
    m = next(iter(requested_metrics)) if requested_metrics else ""
    if m == "installation_count":
        return get_install_date_col() or (get_sale_date_col() or "SaleDate")
    if m == "disconnection_count":
        return get_disconnect_date_col() or (get_sale_date_col() or "SaleDate")

    # Default:
    return get_sale_date_col() or "SaleDate"

def apply_hidden_cols_policy(df: pd.DataFrame, question: str) -> pd.DataFrame:
    """
    Default: remove hidden cols.
    Exception: if user asked for a hidden col, keep that one.
    """
    if df is None or df.empty:
        return df

    requested = _hidden_cols_user_requested(question)
    # Hide all default hidden cols EXCEPT the ones explicitly requested
    to_drop = [c for c in HIDDEN_COLS_DEFAULT if c in df.columns and c not in requested]
    if to_drop:
        return df.drop(columns=to_drop, errors="ignore")
    return df

def _extract_key_lookup(question: str):
    """
    Detect queries like:
      - 'details of the company key <uuid> limit 5'
      - 'show details for customerkey <uuid>'
      - 'orderitemkey 12345 details'
    Returns (col, value, limit) or (None, None, None)
    """
    q = (question or "").strip()
    ql = q.lower()

    # limit (default 50 for safety)
    lim = 50
    mlim = re.search(r"\blimit\s+(\d{1,5})\b", ql)
    if mlim:
        lim = max(1, min(int(mlim.group(1)), MAX_ROWS))

    # find which hidden key column user mentioned (supports: company key / companykey / company_key)
    hit_col = None
    q_norm = _norm_key(ql)
    for c in HIDDEN_COLS_DEFAULT:
        c_norm = _norm_key(c)
        c_spaced = _split_camel(c).lower()
        if (c_norm and c_norm in q_norm) or (c_spaced and c_spaced in ql):
            hit_col = c
            break

    if not hit_col:
        return (None, None, None)

    # try to extract UUID-like value
    uuid_pat = re.compile(r"\b[0-9a-fA-F]{8}\-[0-9a-fA-F]{4}\-[0-9a-fA-F]{4}\-[0-9a-fA-F]{4}\-[0-9a-fA-F]{12}\b")
    mu = uuid_pat.search(q)
    if mu:
        return (hit_col, mu.group(0), lim)

    # fallback: extract a token after the key phrase (handles numeric keys)
    # e.g. "orderitemkey 12345"
    token_pat = re.compile(rf"{re.escape(_split_camel(hit_col).lower())}\s+([A-Za-z0-9\-#]{{2,80}})", re.IGNORECASE)
    token_pat2 = re.compile(rf"{re.escape(_norm_key(hit_col))}\s*[:=]?\s*([A-Za-z0-9\-#]{{2,80}})", re.IGNORECASE)
    mt2 = token_pat2.search(q)  # use original text, not normalized
    if mt2:
        return (hit_col, mt2.group(1).strip(), lim)

    mt = token_pat.search(q)
    if mt:
        return (hit_col, mt.group(1).strip(), lim)

    return (hit_col, None, lim)

# -------------------------------------------------
# Make formulas readable (avoid LaTeX)
# -------------------------------------------------
def strip_latex_math(text: str) -> str:
    """
    Convert common LaTeX math wrappers to plain text so frontend can read it.
    Not a full LaTeX parser — just enough for \\[ \\], \\( \\), and simple \\text{}.
    (NOTE: backslashes are doubled to avoid SyntaxWarning invalid escape sequences.)
    """
    if not text:
        return text

    t = text
    t = re.sub(r"\\\[\s*", "", t)
    t = re.sub(r"\s*\\\]", "", t)
    t = re.sub(r"\\\(\s*", "", t)
    t = re.sub(r"\s*\\\)", "", t)
    t = re.sub(r"\\text\{([^}]*)\}", r"\1", t)
    t = t.replace("\\%", "%").replace("\\_", "_")
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t

def classify_intent(question: str) -> str:
    q = (question or "").strip().lower()
    if not q:
        return "empty"

    # normalize: remove punctuation so "hi!" works
    q_clean = re.sub(r'[^a-z\s]', ' ', q)
    q_clean = re.sub(r'\s+', ' ', q_clean).strip()

    # greetings / smalltalk
    greetings = {
        "hi", "hello", "hey", "yo", "hola", "sup",
        "good morning", "good afternoon", "good evening"
    }
    if q_clean in greetings:
        return "smalltalk"
    if q_clean.startswith(("hi ", "hello ", "hey ")):
        return "smalltalk"
    if "your name" in q_clean or "who are you" in q_clean:
        return "smalltalk"

    # schema / metadata
    if "what columns" in q_clean or "what fields" in q_clean or "show schema" in q_clean:
        return "meta"
    if re.search(r'\b(show|list|display|give|what are)\b.*\b(column names?|columns|fields|attributes|headings)\b', q_clean):
        return "meta"

    # everything else: treat as data question
    return "data_question"


_COL_VARIANTS = {}  # variant(lower) -> real_col
_COL_VARIANTS_NORM = {}  # normalized variant -> real_col
for _c in COLS:
    v1 = (_c or "").strip().lower()
    v2 = _split_camel(_c).lower()
    v3 = _norm_key(_c)  # no spaces
    if v1:
        _COL_VARIANTS[v1] = _c
        _COL_VARIANTS_NORM[_norm_key(v1)] = _c
    if v2:
        _COL_VARIANTS[v2] = _c
        _COL_VARIANTS_NORM[_norm_key(v2)] = _c
    if v3:
        _COL_VARIANTS_NORM[v3] = _c

_ALIAS_MAP = {
    "package": ["package", "package name", "packagename", "pkg", "pkgname", "order package", "product package"],
    "mainchannel": ["mainchannel", "main channel", "channel"],
    "marketingsource": ["marketing source", "marketingsource", "marketting source", "markettingsource", "marketing", "source"],
    "providername": ["provider", "provider name", "providername"],
    "productname": ["product", "product name", "productname"],
}
_ALIAS_MAP.update({
  "companyname": ["company", "company name", "companyname", "company_name"],
  "companykey":  ["company key", "companyid", "company id", "companykey"],
})
_ALIAS_MAP.update({
    "account": ["account", "account number", "account#", "acct", "acct number"],
    "orderid": ["orderid", "order id", "order#", "order number"],
})

def autodetect_package_col():
    # strongest: contains "package"
    for c in COLS:
        if "package" in _norm_key(c):
            return c
    # fallback: contains "bundle"
    for c in COLS:
        if "bundle" in _norm_key(c):
            return c
    return None

try:
    DETECTED["package"] = autodetect_package_col()
except Exception:
    pass

def resolve_col_name(user_phrase: str):
    """
    Resolve any user-typed column phrase to a real column in COLS.
    Works for: package, Mainchannel, Marketing source, CompanyState, etc.
    """
    if not user_phrase:
        return None
    raw = (user_phrase or "").strip().lower()
    if not raw:
        return None

    # 1) direct variant match
    if raw in _COL_VARIANTS:
        return _COL_VARIANTS[raw]

    # 2) normalized match (removes spaces/punct)
    nk = _norm_key(raw)
    if nk in _COL_VARIANTS_NORM:
        return _COL_VARIANTS_NORM[nk]

    # Package override (works even if column isn't literally named "Package")
    if _norm_key(raw) in ("package", "packagename", "pkg", "pkgname", "bundle", "bundlename"):
        pc = DETECTED.get("package")
        if pc:
            return pc

    # direct "package" auto-detect override
    if _norm_key(raw) in ("package", "packagename", "orderpackage", "bundle", "bundlename", "pkg", "pkgname"):
        pc = DETECTED.get("package")
        if pc:
            return pc
    # 3) alias map (NON-RECURSIVE to avoid infinite loops)
    for k, vals in _ALIAS_MAP.items():
        k_norm = _norm_key(k)
        vals_norm = {_norm_key(x) for x in vals}

        # match if user typed the alias key OR any alias value
        if nk == k_norm or nk in vals_norm or raw in vals:
            # First try: canonical key itself (e.g., "companyname" -> actual column)
            col = _COL_VARIANTS_NORM.get(k_norm)
            if col:
                return col

            # Second try: each alias value directly (NO recursion)
            for v in vals:
                v_raw = (v or "").strip().lower()
                v_norm = _norm_key(v_raw)

                col = _COL_VARIANTS.get(v_raw) or _COL_VARIANTS_NORM.get(v_norm)
                if col:
                    return col

    # 4) contains match (last resort)
    for c in COLS:
        if nk and nk in _norm_key(c):
            return c

    return None

# -------------------------------------------------
# PREDICTION HELPERS
# -------------------------------------------------
_PRED_WORDS = ("estimate", "estimated", "predict", "predicted", "prediction", "forecast", "projected", "projection")

_MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

def is_prediction_request(question: str) -> bool:
    q = (question or "").lower()
    return any(w in q for w in _PRED_WORDS)

def extract_year(question: str):
    m = re.search(r"\b(20\d{2})\b", (question or "").lower())
    return int(m.group(1)) if m else None

def extract_month_year(question: str):
    """
    Returns (year:int, month:int) if 'dec 2025' etc appears, else None.
    """
    q = (question or "").lower()
    m = re.search(
        r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b\D{0,15}\b(20\d{2})\b",
        q
    )
    if not m:
        return None
    mon_txt = m.group(1)
    year = int(m.group(2))
    month = _MONTHS.get(mon_txt, None)
    if not month:
        month = _MONTHS.get(mon_txt[:3], None)
    if not month:
        return None
    return (year, month)

def _month_start_end(year: int, month: int):
    start = dtmod.date(year, month, 1)
    if month == 12:
        end = dtmod.date(year + 1, 1, 1)
    else:
        end = dtmod.date(year, month + 1, 1)
    return start, end

# --- shared stopwords for entity extraction ---
_ENTITY_STOPWORDS = (
    "for|in|on|where|with|month|year|and|or|by|from|to|as|at|of|"
    "what|whats|is|are|was|were|show|give|tell|find|get|calculate|compute|"
    "name|names|"
    "profit|revenue|rate|orders|order|installation|install|disconnection|disconnect|"
    "estimate|estimated|forecast|predict|predicted|projection|projected"
)
def extract_provider(question: str):
    q = (question or "").strip()

    # ONLY treat as a provider filter if user uses an operator:
    # "provider is X" / "provider = X" / "provider: X"
    m = re.search(
        r"\bprovider\b\s*(?:is|=|:)\s*([A-Za-z0-9&._\-/ ]{2,80})",
        q,
        flags=re.IGNORECASE
    )
    if not m:
        return None

    val = (m.group(1) or "").strip()

    # cut at stopwords
    val = re.split(rf"\b(?:{_ENTITY_STOPWORDS})\b", val, flags=re.IGNORECASE)[0].strip()
    val = re.split(r"[,.?;!]", val)[0].strip()

    # block junk
    if val.lower() in ("name", "names"):
        return None

    return val if val else None
def is_bare_id_message(text: str) -> bool:
    """
    True if user message is basically just an ID like:
    19876
    1,722,757
    #8260130077300575
    """
    if not text:
        return False
    t = text.strip()
    t = t.replace(",", "")
    # allow optional leading '#'
    return bool(re.fullmatch(r"#?\d{4,20}", t))


def _find_last_details_type(history):
    """
    Returns:
      "account" if last details request was for account
      "order"   if last details request was for order
      None      if not found
    """
    if not history:
        return None

    for m in reversed(history):
        txt = (m.get("content") or m.get("text") or "").lower().strip()
        if not txt:
            continue

        # only consider previous turns that were clearly "details" intent
        if "detail" not in txt and "report" not in txt:
            continue

        # if they said account in that request -> account
        if "account" in txt or "acct" in txt:
            return "account"

        # if they said order in that request -> order
        if "order" in txt:
            return "order"

    return None

def extract_product(question: str):
    q = (question or "").strip()
    ql = q.lower()

    # If user says "product name" they are usually referring to the column, not a filter value.
    if re.search(r"\bproduct\s+name\b", ql):
        return None

    # allow "product tv" or "product is tv" or "product = tv"
    m = re.search(
        r"\bproduct\b\s*(?:is|=|:)?\s*([A-Za-z0-9&._\-/ ]{2,80})",
        q,
        flags=re.IGNORECASE
    )
    if not m:
        return None

    val = (m.group(1) or "").strip()

    # cut at stopwords
    val = re.split(rf"\b(?:{_ENTITY_STOPWORDS})\b", val, flags=re.IGNORECASE)[0].strip()
    val = re.split(r"[,.?;!]", val)[0].strip()

    # block junk values
    if val.lower() in ("name", "names", "rate", "rates"):
        return None

    return val if val else None

def _safe_sql_literal(s: str) -> str:
    return "'" + (s or "").replace("'", "''").strip() + "'"
def _wants_estimate_for_metric(q_lower: str, metric_name: str) -> bool:
    """
    True ONLY if estimate/forecast word is NEAR the FULL metric phrase.

    IMPORTANT FIX:
    - Uses strict word boundaries so "profit" does NOT match inside "profit rate"
      unless the user explicitly says "estimated profit".
    """
    if not q_lower or not metric_name:
        return False

    est_words = r"(estimate|estimated|forecast|predict|predicted|projection|projected)"

    # strict phrase match
    m = re.escape(metric_name.strip().lower())

    patterns = [
        rf"\b{est_words}\b\s+(?:\w+\s+){{0,2}}\b{m}\b",  # estimated <metric>
        rf"\b{m}\b\s+(?:\w+\s+){{0,2}}\b{est_words}\b",  # <metric> estimated
    ]
    return any(re.search(p, q_lower) for p in patterns)


def _extract_generic_filters_anycol(question: str):
    """
    Extract filters for ANY column (supports multi-word column phrases):
      - "package is DirecTV" / "package = DirecTV" / "package: DirecTV"
      - "package DirecTV"
      - "for the package DirecTV"
      - "Mainchannel Google"
      - "Marketing source xyz"
    IMPORTANT: This does not require the user to type the exact column name.
    It matches column variants like "MarketingSource" <-> "marketing source".
    """
    q = (question or "").strip()
    ql = q.lower()

    stop_words = set(["for", "the", "in", "on", "where", "with", "month", "year", "and", "or", "to", "of", "by"])
    stop_words2 = stop_words.union(set(["based", "wise", "monthwise", "yearwise"]))

    def _clean_value(val_raw: str) -> str:
        parts = (val_raw or "").strip().split()
        cleaned = []
        for p in parts:
            if p.lower() in stop_words2:
                break
            cleaned.append(p)
        val = " ".join(cleaned).strip()
        val = re.split(r"[,.?;!]", val)[0].strip()
        return val

    # Build a list of column phrases (variants) and sort longest-first so
    # "marketing source" matches before "source" etc.
    variants = sorted(list(_COL_VARIANTS.keys()), key=lambda s: len(s), reverse=True)

    used_spans = []  # avoid double-capturing overlapping ranges
    out = []

    def _span_overlaps(a, b):
        for (s, e) in used_spans:
            if not (b <= s or a >= e):
                return True
        return False
    
    # A) operator form: "<col phrase> is/=/: <value>"
    for v in variants:
        if not v:
            continue
        v_re = re.escape(v)
        if not re.search(rf"\b{v_re}\b", ql, flags=re.IGNORECASE):
            continue
        pat = re.compile(rf"\b{v_re}\b\s*(?:=|:|\bis\b)\s*([A-Za-z0-9&._\-#/ ]{{2,80}})", flags=re.IGNORECASE)
        for m in pat.finditer(q):
            a, b = m.span()
            if _span_overlaps(a, b):
                continue
            col = resolve_col_name(v)
            if not col:
                continue
            val = _clean_value(m.group(1))
            if not val:
                continue
            out.append((col, val))
            used_spans.append((a, b))

    # B) no-operator form: "for the <col phrase> <value>" OR "<col phrase> <value>"
# B) no-operator form: "<col phrase> <value>" (simple, 1-3 tokens)
    for v in variants:
        if not v:
            continue
        v_re = re.escape(v)
        if not re.search(rf"\b{v_re}\b", ql, flags=re.IGNORECASE):
            continue

        pat = re.compile(
            rf"\b{v_re}\b\s+([A-Za-z0-9&._\-#/]+(?:\s+[A-Za-z0-9&._\-#/]+){{0,2}})",
            flags=re.IGNORECASE
        )
        for m in pat.finditer(q):
            a, b = m.span()
            if _span_overlaps(a, b):
                continue
            col = resolve_col_name(v)
            if not col:
                continue
            val = _clean_value(m.group(1))
            if not val:
                continue
            out.append((col, val))
            used_spans.append((a, b))

    # C) reversed form: "DirecTV package" (keep ONLY for package-ish columns because value-first is risky)
    pkg_col = resolve_col_name("package")
    if pkg_col:
        pat = re.compile(r"\b([A-Za-z0-9&._\-#/]{2,80})\s+package\b", flags=re.IGNORECASE)
        for m in pat.finditer(q):
            a, b = m.span()
            if _span_overlaps(a, b):
                continue
            val = _clean_value(m.group(1))
            if not val:
                continue
            out.append((pkg_col, val))
            used_spans.append((a, b))
    # ---- BLOCK BAD MATCHES LIKE: "profit rate" -> (Profit, rate) ----
    bad_vals = {"rate", "rates"}
    out = [(c, v) for (c, v) in out if _norm_key(v) not in bad_vals]
    # ---------------------------------------------------------------

    # dedupe while preserving order
    try:
        out = list(dict.fromkeys(out))
    except Exception:
        pass
    return out
# -------------------------------------------------
# GROUP BY + TOP-N detection (Prediction mode)
# -------------------------------------------------
def extract_top_n(question: str):
    """
    Extracts top N from phrases like:
      - "top 10"
      - "top 20 product"
    Returns int or None
    """
    q = (question or "").lower()
    m = re.search(r"\btop\s+(\d{1,4})\b", q)
    if not m:
        return None
    n = int(m.group(1))
    n = max(1, min(n, 500))
    return n

def wants_month_wise(question: str) -> bool:
    q = (question or "").lower()
    return any(p in q for p in ["month wise", "month-wise", "monthly", "monthwise"])

def extract_groupby_phrase(question: str):
    q = (question or "").strip()

    def _clean(phrase: str) -> str:
        if not phrase:
            return None
        phrase = phrase.strip()
        # remove leading prompt fluff
        phrase = re.sub(r"^(show|give|tell|find|get|display)\s+me\s+(the\s+)?", "", phrase, flags=re.IGNORECASE).strip()
        # stop at common separators / time / limits
        phrase = re.split(r"\b(on|in|for|where|with|limit|month|year|wise)\b", phrase, flags=re.IGNORECASE)[0]
        phrase = re.split(r"\b20\d{2}\b", phrase)[0]
        return phrase.strip()

    # 1) based on ...
    m = re.search(r"\bbased\s+on\s+([A-Za-z0-9 _\-/&.]{2,120})", q, flags=re.IGNORECASE)
    if m:
        return _clean(m.group(1))

    # 2) by ...
    m = re.search(r"\bby\s+([A-Za-z0-9 _\-/&.]{2,120})", q, flags=re.IGNORECASE)
    if m:
        return _clean(m.group(1))

    # 3) <something> wise  (IMPORTANT: capture only the last phrase before 'wise')
    m = re.search(r"\b(.{2,200}?)\s+wise\b", q, flags=re.IGNORECASE)
    if m:
        left = _clean(m.group(1))
        if not left:
            return None

        # take tail tokens (prevents capturing "show me the ...")
        tokens = left.split()
        for k in (4, 3, 2, 1):
            tail = " ".join(tokens[-k:])
            if tail:
                return tail

    return None
# -------------------------------------------------
# DETAILS REPORT (OrderId / Account)
# -------------------------------------------------
DETAILS_KEYWORDS = (
    "details", "detail", "full report", "order report", "account report"
)

def is_details_request(question: str) -> bool:
    q = (question or "").lower()

    # must contain details keyword OR must look like specific order/account id lookup
    if any(k in q for k in DETAILS_KEYWORDS):
        # BUT block "any 5 account numbers" type queries
        if re.search(r"\b(any|give me)\s+\d+\s+account", q):
            return False
        if re.search(r"\b(any|give me)\s+\d+\s+order", q):
            return False
        return True
# block sample/list requests
    if re.search(r"\b(any|give me|show me)\s+\d+\s+(account|order)\b", q):
        return False

    # order id pattern
    if re.search(r"\border\s*(?:id)?\b\s*#?\s*[\d,]{4,15}\b", q):
        return True

    # account pattern (requires account/acct word)
    if re.search(r"\b(account|acct)\b\s*(?:number)?\b\s*#?\s*[a-z0-9\-]{4,30}\b", q):
        return True

    return False

def extract_account_number(question: str):
    q = (question or "").strip()

    # block list/sample requests
    if re.search(r"\b(any|give me|show me)\s+\d+\s+account", q, flags=re.IGNORECASE):
        return None

    # require digits in the captured value (accounts are numeric in your data)
    m = re.search(
        r"\b(account|acct)\b\s*(?:number)?\s*#?\s*([0-9][0-9,\- ]{3,30})",
        q,
        flags=re.IGNORECASE
    )
    if not m:
        return None

    acc = (m.group(2) or "").strip()
    acc = acc.replace(",", "").replace(" ", "")
    acc = acc.lstrip("#").strip()

    # final safety: must contain at least 4 digits
    digits_only = re.sub(r"\D", "", acc)
    if len(digits_only) < 4:
        return None

    return acc


def extract_order_id(question: str):
    q = (question or "")

    # allow commas: 1,206,911
    m = re.search(r"\border\s*id\b\s*#?\s*([\d,]{4,15})\b", q, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\borderid\b\s*#?\s*([\d,]{4,15})\b", q, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\border\b\s*#?\s*([\d,]{4,15})\b", q, flags=re.IGNORECASE)

    if not m:
        return None

    oid = (m.group(1) or "").replace(",", "").strip()
    return oid if oid.isdigit() else None

def build_details_sql(order_id: str = None, account: str = None):
    """
    Fetch ALL rows for a single order_id or account.
    Keep it deterministic; no LLM.
    """
    where = None
    if order_id:
        order_id = str(order_id).replace(",", "").strip()
        where = f"CAST({qident('OrderId')} AS BIGINT) = {int(order_id)}" if "OrderId" in COLS else None

    elif account:
        account_col = resolve_col_name("account") or ("Account" if "Account" in COLS else None)
        if account_col:
            clean_acc = re.sub(r"[^A-Za-z0-9\-]", "", account or "")
            where = f"""
            REPLACE(LOWER(TRIM(CAST({qident(account_col)} AS VARCHAR))), '#', '')
            = LOWER(TRIM({_safe_sql_literal(clean_acc)}))
            """.strip()
    if not where:
        return None

    # No LIMIT: user wants full report for that key (but safe because single key)
    sql = f"""
    SELECT *
    FROM {TABLE}
    WHERE {where}
    """.strip()
    return sql

def extract_limit(q: str, default=None, max_limit=None):
    if max_limit is None:
        max_limit = MAX_ROWS
    m = re.search(r"\blimit\s+(\d{1,4})\b", q, flags=re.IGNORECASE)
    if not m:
        return default
    n = int(m.group(1))
    return max(1, min(n, max_limit))

def resolve_group_col(question: str):
    phrase = extract_groupby_phrase(question)

    # NEW: if no "based on / by / wise", try "top N <dimension>"
    if not phrase:
        phrase = extract_groupby_from_top_phrase(question)

    if not phrase:
        return None

    col = resolve_col_name(phrase)
    if col and col in COLS:
        return col
    return None

def extract_groupby_from_top_phrase(question: str):
    """
    Detect patterns like:
      - "top 10 product name ..."
      - "top 20 company name ..."
      - "top 15 package ..."
    Returns a phrase like "product name" / "company name" etc.
    """
    q = (question or "").strip()
    ql = q.lower()

    m = re.search(r"\btop\s+\d{1,4}\s+([a-z0-9 _\-/&.]{2,60})", ql)
    if not m:
        return None

    phrase = (m.group(1) or "").strip()

    # stop phrase early at common metric words
    phrase = re.split(r"\b(profit|revenue|rate|orders?|installation|disconnection|estimated|estimate)\b", phrase)[0].strip()

    # common cleanup
    phrase = phrase.replace("wise", "").strip()
    return phrase if phrase else None

def build_prediction_monthly_sql(question: str, group_col: str = None, sparse_groups: bool = False, limit_n: int = None):
    year = extract_year(question)
    month_year = extract_month_year(question)
    # limit for grouped outputs (fallback to 50 if not provided)
    if limit_n is None:
        limit_n = extract_limit(question, default=50, max_limit=MAX_ROWS)
    else:
        limit_n = max(1, min(int(limit_n), MAX_ROWS))

    profit_col = get_profit_col() or "Profit"
    revenue_col = get_revenue_col() or "NetAfterChargeback"

    sale_date_col = get_sale_date_col() or (get_date_col() or "SaleDate")
    install_date_col = get_install_date_col() or "InstallDate"
    disconnect_date_col = get_disconnect_date_col() or "DisconnectDate"

    provider_col = resolve_col_name("providername") or "ProviderName"
    product_col  = resolve_col_name("productname")  or "ProductName"

    provider = extract_provider(question)
    # HARD BLOCK: never allow provider filter to be the word "name"
    if provider and provider.strip().lower() in ("name", "providername", "provider name"):
        provider = None
    product = extract_product(question)

    generic_filters = []
    try:
        generic_filters = _extract_generic_filters_anycol(question)
    except Exception:
        generic_filters = []

    # target range + train range
    if month_year:
        y, m = month_year
        year = month_year[0]
        target_start, target_end = _month_start_end(y, m)
        train_start = target_start - dtmod.timedelta(days=365)
        date_start = train_start
        date_end = target_end
        target_filter = ("month", y, m)
    else:
        if not year:
            year = dtmod.date.today().year
        target_start = dtmod.date(year, 1, 1)
        target_end   = dtmod.date(year + 1, 1, 1)
        train_start  = dtmod.date(year - 1, 1, 1)
        date_start   = train_start
        date_end     = dtmod.date(year + 1, 1, 1)
        target_filter = ("year", year)

    # shared dimension filters (provider/product + generic)
    where_dim = []
    if provider:
        where_dim.append(
            f"LOWER(TRIM(CAST({qident(provider_col)} AS VARCHAR))) LIKE '%' || LOWER(TRIM({_safe_sql_literal(provider)})) || '%'"
        )
    if product:
        where_dim.append(
            f"LOWER(TRIM(CAST({qident(product_col)} AS VARCHAR))) LIKE '%' || LOWER(TRIM({_safe_sql_literal(product)})) || '%'"
        )
    for col, val in generic_filters:
        if col in (provider_col, product_col):
            continue
        if col and col in COLS:
            where_dim.append(
                f"LOWER(TRIM(CAST({qident(col)} AS VARCHAR))) LIKE '%' || LOWER(TRIM({_safe_sql_literal(val)})) || '%'"
            )
    where_dim_sql = " AND ".join(where_dim) if where_dim else "1=1"

    # months MUST include training months too
    months_start = date_start
    months_end = target_end

    # Optional grouping: normalize empty to 'nan'
    group_enabled = bool(group_col and group_col in COLS)

    group_key_expr = (
        f"COALESCE(NULLIF(TRIM(CAST({qident(group_col)} AS VARCHAR)),''), 'nan')"
        if group_enabled else
        None
    )
# -------------------------------------------------
# months CTE
# - no group: months only
# - group + not sparse: months x group_key (cross join)
# - group + sparse: DO NOT build months_cte (fast-path returns later)
# -------------------------------------------------
    months_cte = ""

    if not group_enabled:
        months_cte = f"""
    months AS (
    SELECT STRFTIME('%Y-%m', d) AS month
    FROM generate_series(
        DATE '{months_start.isoformat()}',
        DATE '{(months_end - dtmod.timedelta(days=1)).isoformat()}',
        INTERVAL '1 month'
    ) t(d)
    )
    """.strip()

    elif group_enabled and (not sparse_groups):
        months_cte = f"""
    months AS (
    SELECT
        STRFTIME('%Y-%m', d) AS month,
        g.group_key
    FROM generate_series(
        DATE '{months_start.isoformat()}',
        DATE '{(months_end - dtmod.timedelta(days=1)).isoformat()}',
        INTERVAL '1 month'
    ) t(d)
    JOIN (
        SELECT DISTINCT {group_key_expr} AS group_key
        FROM {TABLE}
        WHERE {where_dim_sql}
    ) g ON 1=1
    )
    """.strip()

    else:
        # sparse group mode: do not build months_cte; fast-path SQL returns below
        months_cte = ""


    # Add group select fragments for the agg CTEs
    sale_group_select = f", {group_key_expr} AS group_key" if group_enabled else ""
    sale_group_by = ", 2" if group_enabled else ""

    inst_group_select = f", {group_key_expr} AS group_key" if group_enabled else ""
    inst_group_by = ", 2" if group_enabled else ""

    disc_group_select = f", {group_key_expr} AS group_key" if group_enabled else ""
    disc_group_by = ", 2" if group_enabled else ""

    # Join conditions
    join_sale = "s.month = m.month" + (" AND s.group_key = m.group_key" if group_enabled else "")
    join_inst = "i.month = m.month" + (" AND i.group_key = m.group_key" if group_enabled else "")
    join_disc = "d.month = m.month" + (" AND d.group_key = m.group_key" if group_enabled else "")

    select_group_key = "m.group_key," if group_enabled else ""
    order_by = "m.month" + (", m.group_key" if group_enabled else "")
    # ---------------------------------------------
    # FAST PATH: sparse grouped mode (NO months x groups CROSS JOIN)
    # IMPORTANT: Must RETURN a "month" column because downstream code filters by df_out["month"]
    # ---------------------------------------------
    if group_enabled and sparse_groups:
        sql = f"""
        WITH
        sale_agg AS (
            SELECT
            STRFTIME('%Y-%m', TRY_CAST({qident(sale_date_col)} AS DATE)) AS month,
            {group_key_expr} AS group_key,

            SUM(CASE WHEN COALESCE(ConfirmedOrder,0)=1 THEN 1 ELSE 0 END) AS total_orders,
            SUM(CASE WHEN COALESCE(InstalledOrder,0)=1 THEN 1 ELSE 0 END) AS installation_orders,
            SUM(CASE WHEN TRY_CAST({qident(disconnect_date_col)} AS TIMESTAMP) IS NOT NULL THEN 1 ELSE 0 END) AS disconnection_orders,

            SUM(COALESCE({qident(revenue_col)},0)) AS revenue,
            SUM(COALESCE({qident(profit_col)},0))  AS profit
            FROM {TABLE}
            WHERE
            TRY_CAST({qident(sale_date_col)} AS DATE) >= DATE '{date_start.isoformat()}'
            AND TRY_CAST({qident(sale_date_col)} AS DATE) <  DATE '{date_end.isoformat()}'
            AND {where_dim_sql}
            GROUP BY 1,2
        ),

        install_agg AS (
            SELECT
            STRFTIME('%Y-%m', TRY_CAST({qident(install_date_col)} AS DATE)) AS month,
            {group_key_expr} AS group_key,
            COUNT(*) AS installation_count
            FROM {TABLE}
            WHERE
            TRY_CAST({qident(install_date_col)} AS DATE) >= DATE '{date_start.isoformat()}'
            AND TRY_CAST({qident(install_date_col)} AS DATE) <  DATE '{date_end.isoformat()}'
            AND {where_dim_sql}
            GROUP BY 1,2
        ),

        disc_agg AS (
            SELECT
            STRFTIME('%Y-%m', TRY_CAST({qident(disconnect_date_col)} AS DATE)) AS month,
            {group_key_expr} AS group_key,
            COUNT(*) AS disconnection_count
            FROM {TABLE}
            WHERE
            TRY_CAST({qident(disconnect_date_col)} AS DATE) >= DATE '{date_start.isoformat()}'
            AND TRY_CAST({qident(disconnect_date_col)} AS DATE) <  DATE '{date_end.isoformat()}'
            AND {where_dim_sql}
            GROUP BY 1,2
        ),

        base AS (
            SELECT month, group_key FROM sale_agg
            UNION
            SELECT month, group_key FROM install_agg
            UNION
            SELECT month, group_key FROM disc_agg
        )

        SELECT
        b.month,
        b.group_key,

        COALESCE(s.total_orders, 0) AS total_orders,
        COALESCE(s.installation_orders, 0) AS installation_orders,
        COALESCE(
            ROUND(COALESCE(s.installation_orders,0) * 100.0 / NULLIF(COALESCE(s.total_orders,0), 0), 2),
            0
        ) AS installation_rate,

        COALESCE(s.disconnection_orders, 0) AS disconnection_orders,
        COALESCE(
            ROUND(COALESCE(s.disconnection_orders,0) * 100.0 / NULLIF(COALESCE(s.total_orders,0), 0), 2),
            0
        ) AS disconnection_rate,

        COALESCE(i.installation_count, 0) AS installation_count,
        COALESCE(d.disconnection_count, 0) AS disconnection_count,

        COALESCE(s.revenue, 0) AS revenue,
        COALESCE(s.profit, 0)  AS profit,

        COALESCE(
            ROUND(COALESCE(s.profit,0) * 100.0 / NULLIF(COALESCE(s.revenue,0), 0), 2),
            0
        ) AS profit_rate

        FROM base b
        LEFT JOIN sale_agg    s ON s.month = b.month AND s.group_key = b.group_key
        LEFT JOIN install_agg i ON i.month = b.month AND i.group_key = b.group_key
        LEFT JOIN disc_agg    d ON d.month = b.month AND d.group_key = b.group_key
        ORDER BY b.month, b.group_key
        """.strip()

        return sql, target_filter


    sql = f"""
WITH
{months_cte},

sale_agg AS (
  SELECT
    STRFTIME('%Y-%m', TRY_CAST({qident(sale_date_col)} AS DATE)) AS month
    {sale_group_select},

    -- Total orders (confirmed)
    SUM(CASE WHEN COALESCE(ConfirmedOrder,0)=1 THEN 1 ELSE 0 END) AS total_orders,

    -- Installation orders (flag on the order)
    SUM(CASE WHEN COALESCE(InstalledOrder,0)=1 THEN 1 ELSE 0 END) AS installation_orders,

    -- Disconnection orders (has disconnect date) anchored to SaleDate for the rate denominator
    SUM(CASE WHEN TRY_CAST({qident(disconnect_date_col)} AS TIMESTAMP) IS NOT NULL THEN 1 ELSE 0 END) AS disconnection_orders,

    -- Revenue/profit on SaleDate
    SUM(COALESCE({qident(revenue_col)},0)) AS revenue,
    SUM(COALESCE({qident(profit_col)},0))  AS profit

  FROM {TABLE}
  WHERE
    TRY_CAST({qident(sale_date_col)} AS DATE) >= DATE '{date_start.isoformat()}'
    AND TRY_CAST({qident(sale_date_col)} AS DATE) <  DATE '{date_end.isoformat()}'
    AND {where_dim_sql}
  GROUP BY 1{sale_group_by}
),

install_agg AS (
  SELECT
    STRFTIME('%Y-%m', TRY_CAST({qident(install_date_col)} AS DATE)) AS month
    {inst_group_select},
    COUNT(*) AS installation_count
  FROM {TABLE}
  WHERE
    TRY_CAST({qident(install_date_col)} AS DATE) >= DATE '{date_start.isoformat()}'
    AND TRY_CAST({qident(install_date_col)} AS DATE) <  DATE '{date_end.isoformat()}'
    AND {where_dim_sql}
  GROUP BY 1{inst_group_by}
),

disc_agg AS (
  SELECT
    STRFTIME('%Y-%m', TRY_CAST({qident(disconnect_date_col)} AS DATE)) AS month
    {disc_group_select},
    COUNT(*) AS disconnection_count
  FROM {TABLE}
  WHERE
    TRY_CAST({qident(disconnect_date_col)} AS DATE) >= DATE '{date_start.isoformat()}'
    AND TRY_CAST({qident(disconnect_date_col)} AS DATE) <  DATE '{date_end.isoformat()}'
    AND {where_dim_sql}
  GROUP BY 1{disc_group_by}
)

SELECT
  m.month,
  {select_group_key}

  COALESCE(s.total_orders, 0) AS total_orders,
  COALESCE(s.installation_orders, 0) AS installation_orders,

  -- Installation Rate = installation_orders / total_orders
  COALESCE(
    ROUND(COALESCE(s.installation_orders,0) * 100.0 / NULLIF(COALESCE(s.total_orders,0), 0), 2),
    0
  ) AS installation_rate,

  COALESCE(s.disconnection_orders, 0) AS disconnection_orders,

  -- Disconnection Rate = disconnection_orders / total_orders
  COALESCE(
    ROUND(COALESCE(s.disconnection_orders,0) * 100.0 / NULLIF(COALESCE(s.total_orders,0), 0), 2),
    0
  ) AS disconnection_rate,

  COALESCE(i.installation_count, 0) AS installation_count,
  COALESCE(d.disconnection_count, 0) AS disconnection_count,

  COALESCE(s.revenue, 0) AS revenue,
  COALESCE(s.profit, 0)  AS profit,

  COALESCE(
    ROUND(COALESCE(s.profit,0) * 100.0 / NULLIF(COALESCE(s.revenue,0), 0), 2),
    0
  ) AS profit_rate

FROM months m
LEFT JOIN sale_agg    s ON {join_sale}
LEFT JOIN install_agg i ON {join_inst}
LEFT JOIN disc_agg    d ON {join_disc}
ORDER BY {order_by}
""".strip()

    return sql, target_filter


def _month_to_index(ym: str) -> int:
    try:
        y, m = ym.split("-")
        return int(y) * 12 + (int(m) - 1)
    except Exception:
        return None

def add_estimated_columns(df: pd.DataFrame, cols_to_estimate, window=12, mode: str = "always"):
    """
    Adds estimated_<col> using rolling Linear Regression.
    Predicts each month using ONLY previous months.

    mode:
      - "always":   compute an estimate for every month (after at least 1-2 history points)
      - "missing":  only estimate when actual is missing (NaN)
    """
    if df is None or df.empty or "month" not in df.columns:
        return df

    df = df.copy()
    df["__t"] = df["month"].astype(str).apply(_month_to_index)
    df = df.sort_values("__t").reset_index(drop=True)

    mode = (mode or "always").strip().lower()
    if mode not in ("always", "missing"):
        mode = "always"

    for col in cols_to_estimate:
        est_col = f"estimated_{col}"
        df[est_col] = np.nan
        if col not in df.columns:
            continue

        y_all = pd.to_numeric(df[col], errors="coerce").values.astype(float)
        t_all = pd.to_numeric(df["__t"], errors="coerce").values.astype(float)

        for i in range(len(df)):
            # "missing" mode: only estimate if actual is missing
            if mode == "missing" and np.isfinite(y_all[i]):
                continue

            # always needs prior history
            if i <= 0 or not np.isfinite(t_all[i]):
                continue

            start_i = max(0, i - window)
            X_hist = t_all[start_i:i]
            y_hist = y_all[start_i:i]

            mask = np.isfinite(X_hist) & np.isfinite(y_hist)
            X_hist = X_hist[mask]
            y_hist = y_hist[mask]

            if len(X_hist) == 0:
                continue

            # 1 point: carry-forward
            if len(X_hist) == 1:
                df.loc[i, est_col] = float(y_hist[0])
                continue

            model = LinearRegression()
            model.fit(X_hist.reshape(-1, 1), y_hist)

            pred = model.predict(np.array([[t_all[i]]], dtype=float))[0]
            df.loc[i, est_col] = float(pred)

        df[est_col] = pd.to_numeric(df[est_col], errors="coerce").round(2)

    df.drop(columns=["__t"], inplace=True, errors="ignore")
    return df
def add_estimated_columns_grouped(df: pd.DataFrame, group_col_name: str, cols_to_estimate, window=12, mode: str = "always"):
    """
    Apply add_estimated_columns() per group so each group gets its own regression model.
    group_col_name should be "group_key".
    """
    if df is None or df.empty:
        return df
    if group_col_name not in df.columns:
        return add_estimated_columns(df, cols_to_estimate, window=window, mode=mode)

    out_parts = []
    for gval, part in df.groupby(group_col_name, dropna=False):
        part2 = add_estimated_columns(part, cols_to_estimate, window=window, mode=mode)
        out_parts.append(part2)

    try:
        return pd.concat(out_parts, ignore_index=True)
    except Exception:
        return df

def add_estimated_rate_columns(df: pd.DataFrame):
    """
    Only derive rate estimates from estimated numerators/denominators
    when the estimated rate column is NOT already present.

    IMPORTANT FIX:
    - Do NOT derive estimated_profit_rate from estimated_profit/estimated_revenue.
      Profit rate should be estimated directly from the profit_rate series.
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # installation_rate = installation_orders / total_orders
    if ("estimated_installation_rate" not in df.columns and
        "estimated_installation_orders" in df.columns and
        "estimated_total_orders" in df.columns):
        denom = pd.to_numeric(df["estimated_total_orders"], errors="coerce").astype(float)
        numer = pd.to_numeric(df["estimated_installation_orders"], errors="coerce").astype(float)
        df["estimated_installation_rate"] = np.where(
            denom > 0,
            (numer * 100.0 / denom),
            np.nan
        )
        df["estimated_installation_rate"] = pd.to_numeric(df["estimated_installation_rate"], errors="coerce").round(2)

    # disconnection_rate = disconnection_orders / total_orders
    if ("estimated_disconnection_rate" not in df.columns and
        "estimated_disconnection_orders" in df.columns and
        "estimated_total_orders" in df.columns):
        denom = pd.to_numeric(df["estimated_total_orders"], errors="coerce").astype(float)
        numer = pd.to_numeric(df["estimated_disconnection_orders"], errors="coerce").astype(float)
        df["estimated_disconnection_rate"] = np.where(
            denom > 0,
            (numer * 100.0 / denom),
            np.nan
        )
        df["estimated_disconnection_rate"] = pd.to_numeric(df["estimated_disconnection_rate"], errors="coerce").round(2)

    # NOTE: profit rate MUST be estimated directly from profit_rate series.
    return df

def _explain_trend(df: pd.DataFrame, est_col: str, window: int):
    vals = pd.to_numeric(df[est_col], errors="coerce").dropna().values
    if len(vals) < 2:
        return None
    delta = vals[-1] - vals[-2]
    if abs(delta) < 0.01:
        direction = "stable"
    elif delta > 0:
        direction = "increasing"
    else:
        direction = "decreasing"
    return f"- {est_col.replace('estimated_', '').replace('_', ' ').title()} estimate is {direction} based on the recent {window} months trend."

def build_prediction_reply(df_out: pd.DataFrame, cols_to_estimate, window: int):
    lines = []
    lines.append(f"Estimates are generated using rolling Linear Regression on the previous {window} months (past data only).")
    for col in cols_to_estimate:
        est_col = f"estimated_{col}"
        if est_col in df_out.columns:
            t = _explain_trend(df_out, est_col, window)
            if t:
                lines.append(t)
    return "\n".join(lines)

# LLM HELPERS
# -------------------------------------------------
def ask_model_for_sql(question: str, schema_cols: list, sample_rows: pd.DataFrame) -> str:
    try:
        schema_info = f"Columns: {', '.join(schema_cols)}"
        sample_info = sample_rows.head(10).to_csv(index=False) if not sample_rows.empty else "No sample data"

        sale_date_col = get_sale_date_col() or "SaleDate"
        install_date_col = get_install_date_col() or "InstallDate"
        disconnect_date_col = get_disconnect_date_col() or "DisconnectDate"

        # FIX: date_col must exist
        date_col = get_date_col() or sale_date_col

        rev_col = get_revenue_col()
        profit_col = get_profit_col()

        semantic_lines = []

        if CUST_FIRST_COL and CUST_LAST_COL:
            semantic_lines.append(
                f"- CustomerFirstName + CustomerLastName together form the full customer name.\n"
                f"  When the user says 'customer name' or 'customer names' or 'customer', use "
                f"TRIM({CUST_FIRST_COL}) || ' ' || TRIM({CUST_LAST_COL}) as CustomerName and "
                f"exclude rows where both are NULL or empty."
            )
            semantic_lines.append(
                f"- If the user asks whether the same account number has different customer names, "
                f"group by Account (or Account-like column) and use "
                f"HAVING COUNT(DISTINCT TRIM({CUST_FIRST_COL}) || ' ' || TRIM({CUST_LAST_COL})) > 1."
            )

        semantic_lines.append(
            "- If the user asks for 'any N X' or 'sample N X values', use SELECT DISTINCT on that "
            "column or expression, filter out NULL/empty values, and LIMIT N."
        )

        semantic_lines.append(
            "- DATE RULES:\n"
            f"  • SaleDate (anchor for most rollups and rates): {sale_date_col}\n"
            f"  • Installation COUNT (completed installs): use InstallDate: {install_date_col}\n"
            f"  • Disconnection COUNT: use DisconnectDate: {disconnect_date_col}\n"
            "  • If the user asks for multiple metrics together, anchor grouping/filtering on SaleDate.\n"
        )

        semantic_lines.append(
            "- IMPORTANT DEFINITIONS (use these exactly):\n"
            "  • Total Orders = COUNT(OrderId)\n"
            "  • Installation Orders = SUM(InstalledOrder)\n"
            "  • Installation Rate = Installation Orders / Total Orders\n"
            f"  • Disconnection Orders = SUM(CASE WHEN {disconnect_date_col} IS NOT NULL THEN 1 ELSE 0 END)\n"
            "  • Disconnection Rate = Disconnection Orders / Total Orders\n"
            "  • Profit Rate (%) = SUM(Profit)*100.0 / NULLIF(SUM(Revenue),0)\n"
            "  • If user asks 'installation and installation rate', treat 'installation' as Installation Orders (NOT InstallDate count).\n"
        )

        extra_semantics = "\n".join(semantic_lines)

        prompt = rf"""
You are an assistant that generates DuckDB SQL for a single table called {TABLE}.

Your job:
1. Decide if the user's question is about this table's data.
2. If YES → return exactly ONE DuckDB SELECT statement that answers it.
3. If NO → return exactly NO_SQL.
IMPORTANT:
- Questions asking for charts, graphs, plots, pie charts, histograms, distributions,
  rankings, top/bottom N, or comparisons are ALWAYS data questions.
- NEVER return NO_SQL for such questions.

DO NOT return explanations, markdown, or comments.
Return ONLY:
- a single SELECT ... FROM {TABLE} ... statement, OR
- the token NO_SQL.

Table name: {TABLE}
Usable columns (ONLY these): {', '.join(schema_cols)}

Business meaning hints (if these columns exist):
- Default date column: {date_col}
- SaleDate column (preferred anchor): {sale_date_col}
- InstallDate column (for installation counts): {install_date_col}
- DisconnectDate column (for disconnection counts): {disconnect_date_col}
- Revenue column (money received): {rev_col}
- Profit column (net profit): {profit_col}

Additional semantic hints:
{extra_semantics or '- (none)'}

Technical rules:
- When you work with dates, ALWAYS cast the chosen date column to DATE first:
  CAST(chosen_date_col AS DATE)
- For year-month buckets:
  STRFTIME('%Y-%m', CAST(chosen_date_col AS DATE))
- For year-only buckets:
  STRFTIME('%Y', CAST(chosen_date_col AS DATE))
- Do NOT invent aliases like date_col_expr unless you define them in the SELECT/FROM.
  If you want to reuse the expression, just repeat CAST(chosen_date_col AS DATE) wherever needed.
- For case-insensitive text comparison:
  LOWER(TRIM(CAST(col AS VARCHAR))) = LOWER(TRIM('value'))
- When returning names or values for a column, prefer to exclude NULL or empty strings:
  WHERE col IS NOT NULL AND TRIM(COALESCE(CAST(col AS VARCHAR),'')) <> ''
- When the user asks for:
  • "top N", "bottom N"
  • "based on <metric>"
  • "by <dimension>"
  • rankings or comparisons

  You MUST:
  1. GROUP BY the identifying column(s) (e.g. CompanyName, ProviderName)
  2. Aggregate metrics using SUM / COUNT / AVG as appropriate
  3. ORDER BY the aggregated metric (DESC for top, ASC for bottom)
  4. Apply LIMIT N

- IMPORTANT: If the user asks for estimate/prediction/forecast/projected values, return NO_SQL.
- If the question wants raw row details (not aggregation), include LIMIT {MAX_ROWS}, unless the user explicitly asks for "all ...".
- If the question is purely conceptual/theory (definitions, explanations) without needing actual table values, return NO_SQL.
- If the user asks for a chart, graph, plot, distribution, comparison, or "by <something>":
  return an aggregated result suitable for visualization.
  Prefer:
  • 1 numeric column → histogram
  • 2 numeric columns → scatter plot
  • 1 category + 1 numeric → pie or bar
  • 1 category + multiple numeric columns → stacked bar
  Do NOT return raw row-level data in these cases.
Schema:
{schema_info}

Tiny sample (only for understanding meaning, not for answering):
{sample_info}

User question:
{question}

Now output either:
- ONE DuckDB SELECT query (no backticks, no explanation), OR
- NO_SQL
"""

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=600,
        )

        sql = (response.choices[0].message.content or "").strip()
        sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", sql, flags=re.IGNORECASE).strip()

        if sql.upper() == "NO_SQL":
            return "NO_SQL"
        if not is_safe_select(sql):
            return "NO_SQL"
        return sql

    except Exception as e:
        print(f"Error generating SQL: {e}", traceback.format_exc())
        return "NO_SQL"
def ask_model_fix_sql(bad_sql: str, error_text: str, schema_cols: list, sample_rows: pd.DataFrame) -> str:
    """
    Attempt to fix a failing DuckDB SELECT query using the LLM.
    Returns a safe SELECT query string or NO_SQL.
    """
    try:
        schema_info = f"Columns: {', '.join(schema_cols)}"
        sample_info = sample_rows.head(10).to_csv(index=False) if not sample_rows.empty else "No sample data"

        sale_date_col = get_sale_date_col() or "SaleDate"
        install_date_col = get_install_date_col() or "InstallDate"
        disconnect_date_col = get_disconnect_date_col() or "DisconnectDate"
        date_col = get_date_col() or sale_date_col

        rev_col = get_revenue_col()
        profit_col = get_profit_col()

        prompt = rf"""
You are an assistant that fixes DuckDB SQL for a single table called {TABLE}.

You will be given:
1) A SQL query that FAILED
2) The DuckDB error message
3) The allowed schema columns
4) A tiny sample of data

Your job:
- Return exactly ONE corrected DuckDB SELECT statement that will run.
- Do NOT return explanations, markdown, or comments.
- If it cannot be fixed safely, return exactly: NO_SQL

Hard rules:
- Output must be a single SELECT statement (first keyword must be SELECT).
- Only use this table: {TABLE}
- Only use these columns: {', '.join(schema_cols)}
- Do NOT use INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/REPLACE.
- Do NOT reference other tables, files, or schemas.
- If you need dates, CAST the chosen date column AS DATE.
- If the original query had an alias/column that doesn't exist, replace it with the closest valid column from schema.
- If there is an ambiguous column name, pick the best match from schema_cols.

Helpful business hints:
- Default date column: {date_col}
- SaleDate (anchor): {sale_date_col}
- InstallDate: {install_date_col}
- DisconnectDate: {disconnect_date_col}
- Revenue column: {rev_col}
- Profit column: {profit_col}

FAILED SQL:
{bad_sql}

DUCKDB ERROR:
{error_text}

Schema:
{schema_info}

Tiny sample:
{sample_info}

Now output ONLY one corrected SELECT query, or NO_SQL.
"""

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=700,
        )

        fixed = (response.choices[0].message.content or "").strip()
        fixed = re.sub(r"^```(?:sql)?\s*|\s*```$", "", fixed, flags=re.IGNORECASE).strip()

        if fixed.upper() == "NO_SQL":
            return "NO_SQL"

        # enforce safety: must be SELECT-only
        if not is_safe_select(fixed):
            return "NO_SQL"

        return fixed

    except Exception as e:
        print("ask_model_fix_sql error:", e, traceback.format_exc())
        return "NO_SQL"

def answer_theory_question(question: str):
    """For general / non-data questions, or when no safe SQL is generated."""
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a clear, concise assistant.\n\n"
                        "IMPORTANT:\n"
                        "- Do NOT say that you cannot create charts or visualizations.\n"
                        "- If a question asks for data, charts, graphs, plots, rankings, or comparisons, answer normally using the data.\n\n"
                        "FORMATTING RULES:\n"
                        "- Do NOT use LaTeX (no \\[ \\], no \\( \\)).\n"
                        "- If you need formulas, write them in plain text, for example:\n"
                        "  Profit = Total Revenue - Total Costs\n"
                        "  Profit Rate (%) = (Profit / Total Revenue) * 100\n"
                    )

                },
                {"role": "user", "content": question},
            ],
            temperature=0.5,
            max_tokens=800,
        )
        ans = (resp.choices[0].message.content or "").strip()
        ans = strip_latex_math(ans)
        return {
            "reply": ans,
            "table_html": "",
            "plot_data_uri": None,
        }
    except Exception as e:
        _log_and_mask_error(e, "answer_theory_question")
        return _safe_user_error("I couldn't answer that right now. Please try again.")
def user_wants_summary(question: str) -> bool:
    q = (question or "").lower().strip()
    return any(k in q for k in [
        "summarize", "summary", "explain", "insights", "interpret", "what does this mean"
    ])
def _first_non_null(series):
    try:
        for x in series:
            if pd.notna(x) and str(x).strip().lower() not in ("", "nan", "nat", "none"):
                return x
    except Exception:
        pass
    return None

def build_order_or_account_report(df: pd.DataFrame, question: str) -> dict:
    """
    Returns:
      reply (markdown-ish text),
      table_html (structured mini tables),
      plot_data_uri (optional bar chart),
    """
    if df is None or df.empty:
        return {
            "reply": "No matching rows found for that Order ID / Account number.",
            "table_html": "<p><i>No data</i></p>",
            "plot_data_uri": None,
        }

    df = df.copy()

    # normalize columns used in report
    col_orderid = "OrderId" if "OrderId" in df.columns else None
    col_account = "Account" if "Account" in df.columns else resolve_col_name("account")
    col_company = "CompanyName" if "CompanyName" in df.columns else None
    col_created = "CreatedByName" if "CreatedByName" in df.columns else None
    col_salesrep = "SalesRepName" if "SalesRepName" in df.columns else None
    col_provider = "ProviderName" if "ProviderName" in df.columns else None
    cust_first = "CustomerFirstName" if "CustomerFirstName" in df.columns else None
    cust_last  = "CustomerLastName" if "CustomerLastName" in df.columns else None
    col_msource = "MarketingSource" if "MarketingSource" in df.columns else None

    sale_col = get_sale_date_col()
    inst_col = get_install_date_col()
    disc_col = get_disconnect_date_col()

    col_status = "Status" if "Status" in df.columns else None
    col_product = "ProductName" if "ProductName" in df.columns else None
    col_ordertype = "OrderType" if "OrderType" in df.columns else None
    col_package = DETECTED.get("package") if DETECTED.get("package") in df.columns else ("Package" if "Package" in df.columns else None)
    col_addon = "Addon" if "Addon" in df.columns else None

    comm_est = "EstimatedCommission" if "EstimatedCommission" in df.columns else None
    comm_recv = "ReceivedCommission" if "ReceivedCommission" in df.columns else None
    spend_ads = None

    revenue_col = get_revenue_col()
    profit_col = get_profit_col()

    # Dedup common duplicates (like repeated EERO SECURE rows)
    subset = [c for c in [col_orderid, col_account, col_product, col_ordertype, col_package, col_addon, col_status, inst_col, disc_col, comm_est, comm_recv, revenue_col, profit_col] if c]
    if subset:
        df = df.drop_duplicates(subset=subset, keep="first")

    # Header values
    order_id_val = _first_non_null(df[col_orderid]) if col_orderid else None
    account_val  = _first_non_null(df[col_account]) if col_account and col_account in df.columns else None
    company_val  = _first_non_null(df[col_company]) if col_company else None
    provider_val = _first_non_null(df[col_provider]) if col_provider else None
    created_val  = _first_non_null(df[col_created]) if col_created else None
    salesrep_val = _first_non_null(df[col_salesrep]) if col_salesrep else None
    customer_name_val = None
    if cust_first or cust_last:
        fn = _first_non_null(df[cust_first]) if cust_first else None
        ln = _first_non_null(df[cust_last]) if cust_last else None
        customer_name_val = " ".join([str(x).strip() for x in [fn, ln] if x and str(x).strip().lower() not in ("nan","none","nat")]).strip() or None

    msource_val = _first_non_null(df[col_msource]) if col_msource else None

    sale_val = _first_non_null(df[sale_col]) if sale_col and sale_col in df.columns else None
    inst_val = _first_non_null(df[inst_col]) if inst_col and inst_col in df.columns else None
    disc_val = _first_non_null(df[disc_col]) if disc_col and disc_col in df.columns else None

    # Package status table
    status_rows = []
    for _, r in df.iterrows():
        status_rows.append({
            "Product": r.get(col_product, "") if col_product else "",
            "Order Type": r.get(col_ordertype, "") if col_ordertype else "",
            "Package": r.get(col_package, "") if col_package else "",
            "Addon?": r.get(col_addon, "") if col_addon else "",
            "Status": r.get(col_status, "") if col_status else "",
            "Install Date": r.get(inst_col, "") if inst_col else "",
            "Disconnect Date": r.get(disc_col, "") if disc_col else "",
        })
    status_df = pd.DataFrame(status_rows)
    if "Status" not in status_df.columns:
        status_df["Status"] = ""
    if "Package" not in status_df.columns:
        status_df["Package"] = ""

    # installation/disconnection summary logic
    installed_mask = status_df["Status"].astype(str).str.lower().str.contains("install")
    disconnected_mask = status_df["Status"].astype(str).str.lower().str.contains("disconnect")

    installed_pkgs = status_df.loc[installed_mask, "Package"].dropna().astype(str).tolist()
    disconnected_pkgs = status_df.loc[disconnected_mask, "Package"].dropna().astype(str).tolist()
    not_installed_pkgs = status_df.loc[~installed_mask, "Package"].dropna().astype(str).tolist()

    # Financials product-wise
    fin = df.copy()
    for c in [comm_est, comm_recv, spend_ads, revenue_col, profit_col]:
        if c and c in fin.columns:
            fin[c] = pd.to_numeric(fin[c], errors="coerce").fillna(0.0)
    # ✅ ADD THIS BLOCK RIGHT HERE (before groupby)
    if col_package and col_package in fin.columns:
        fin[col_package] = fin[col_package].astype(str).replace(["nan", "None", "NaT"], "").str.strip()
        fin.loc[fin[col_package] == "", col_package] = "Unknown"
    group_cols = [c for c in [col_package] if c]
    if group_cols:
        agg = { }
        if comm_est:   agg[comm_est] = "sum"
        if comm_recv:  agg[comm_recv] = "sum"
        if spend_ads:  agg[spend_ads] = "sum"
        if revenue_col: agg[revenue_col] = "sum"
        if profit_col:  agg[profit_col] = "sum"

        fin_prod = fin.groupby(group_cols, dropna=False).agg(agg).reset_index()
        fin_prod = fin_prod.rename(columns={
            col_package: "Package",   # ✅ correct
            comm_est: "Est. Commission",
            comm_recv: "Received Commission",
            revenue_col: "Revenue",
            profit_col: "Profit"
        })

    else:
        fin_prod = pd.DataFrame()

    # totals
    total_profit = float(fin[profit_col].sum()) if profit_col else None
    total_rev = float(fin[revenue_col].sum()) if revenue_col else None
    total_est_comm = float(fin[comm_est].sum()) if comm_est else None

    # Header table HTML (simple key-value)
    header_pairs = [
        ("Order ID", order_id_val),
        ("Account #", account_val),
        ("Company", company_val),
        ("Customer Name", customer_name_val),          # ✅ added
        ("Marketing Source", msource_val), 
        ("Provider", provider_val),
        ("Created By", created_val),
        ("Sales Rep", salesrep_val),
        ("Sale Date", sale_val),
        ("Install Date", inst_val),
        ("Disconnect Date", disc_val),
    ]
    header_html = "<table class='table' style='width:100%; border-collapse:collapse;'>"
    for k, v in header_pairs:
        if v is None or str(v).strip().lower() in ("", "nan", "nat", "none"):
            v = "-"
        header_html += f"<tr><td style='padding:6px; font-weight:600; width:180px;'>{escape(str(k))}</td><td style='padding:6px;'>{escape(str(v))}</td></tr>"
    header_html += "</table>"

    # Build a neat reply text (what users read first)
    lines = []
    lines.append(f"**Order Report**")
    if order_id_val is not None:
        lines.append(f"- Order ID: **{order_id_val}**")
    if account_val is not None:
        lines.append(f"- Account: **{account_val}**")
    if company_val:
        lines.append(f"- Company: **{company_val}**")
    if provider_val:
        lines.append(f"- Provider: **{provider_val}**")
    # package summary
    if len(not_installed_pkgs) == 0 and len(status_df) > 0:
        lines.append(f"- Installation: **All packages installed** (Install date: {inst_val or '-'})")
    else:
        if installed_pkgs:
            lines.append(f"- Installed packages: {', '.join(installed_pkgs[:8])}" + (" ..." if len(installed_pkgs) > 8 else ""))
        if not_installed_pkgs:
            lines.append(f"- Not installed / ordered only: {', '.join(not_installed_pkgs[:8])}" + (" ..." if len(not_installed_pkgs) > 8 else ""))

    if disconnected_pkgs:
        lines.append(f"- Disconnected packages: {', '.join(disconnected_pkgs[:8])}" + (" ..." if len(disconnected_pkgs) > 8 else ""))
        lines.append(f"  (Disconnected date shown in table)")

    # money summary
    money_bits = []
    if total_est_comm is not None:
        money_bits.append(f"Est. Commission: {round(total_est_comm, 2)}")
    if total_rev is not None:
        money_bits.append(f"Revenue: {round(total_rev, 2)}")
    if total_profit is not None:
        money_bits.append(f"Profit: {round(total_profit, 2)}")
    if money_bits:
        lines.append("- Financials: " + " | ".join(money_bits))

    # Add-ons list
    addons_list = []
    if col_addon and col_addon in df.columns:
        addons_list = sorted({str(x) for x in df[col_addon].dropna().unique() if str(x).strip().lower() not in ("", "nan")})
    if addons_list:
        lines.append(f"- Add-ons: {', '.join(addons_list[:10])}" + (" ..." if len(addons_list) > 10 else ""))

    reply = "\u200b"  # zero-width space: looks empty, but prevents UI showing "(no reply)"

    # Make status table + financial table HTML
    status_html = "<h4 style='margin:12px 0 6px;'>Packages & Status</h4>" + rows_to_html_table(status_df.to_dict(orient="records"))
    fin_html = ""
    if not fin_prod.empty:
        fin_html = "<h4 style='margin:12px 0 6px;'>Package-wise Financials</h4>" + rows_to_html_table(fin_prod.to_dict(orient="records"))

    table_html = (
        "<h4 style='margin:0 0 6px;'>Summary</h4>"
        + header_html
        + status_html
        + fin_html
    )

    # Optional chart: product-wise profit (bar)
    plot_uri = None
    try:
        if not fin_prod.empty and "Profit" in fin_prod.columns:
            labels = fin_prod["Package"].astype(str).tolist()
            vals = pd.to_numeric(fin_prod["Profit"], errors="coerce").fillna(0).tolist()
            plot_uri = plot_to_base64(labels, vals, kind="bar", title="Profit by Package")
    except Exception:
        plot_uri = None

    return {
        "reply": reply,
        "table_html": table_html,
        "plot_data_uri": plot_uri,
    }

def handle_general_data_question(question: str, history=None):
    """
    Single generic data handler:
    - Ask LLM for SQL or NO_SQL.
    - If SQL → run it and summarize.
    - If NO_SQL → answer purely in text.
    """
    schema_cols = COLS
    qlow = (question or "").lower()
    history = history or []
    resolved_question = question
    if history:
        # take last few user turns and append (keeps it simple but effective)
        ctx = []
        for m in history[-6:]:
            t = (m.get("content") or m.get("text") or "").strip()
            if t:
                ctx.append(t)
        resolved_question = " | ".join(ctx + [resolved_question])

# -------------------------------------------------
# PREDICTION MODE (bypass LLM SQL and do stable month-wise SQL + ML)
# -------------------------------------------------
    prediction_mode = is_prediction_request(resolved_question)
    if prediction_mode:
        try:
            group_col = resolve_group_col(resolved_question)
            print("DEBUG group_col:", group_col, "phrase:", extract_groupby_phrase(resolved_question))
            top_n = extract_top_n(resolved_question)
            month_wise = wants_month_wise(resolved_question)
            sparse_groups = bool(group_col and (not month_wise))
            pred_sql, target_filter = build_prediction_monthly_sql(
                resolved_question,
                group_col=group_col,
                sparse_groups=sparse_groups,
                limit_n=extract_limit(resolved_question, default=50, max_limit=MAX_ROWS)
            )
            df = run_sql_and_fetch_df(pred_sql)

            ql = (resolved_question or "").lower()

            # normalize wording so estimate detection works consistently
            ql = ql.replace("disconnected rate", "disconnection rate")
            ql = ql.replace("disconnect rate", "disconnection rate")
            # Normalize "orders" wording -> total_orders
            ql = re.sub(r"\bestimated\s+orders?\b", "estimated total orders", ql)
            ql = re.sub(r"\bestimate\s+orders?\b", "estimate total orders", ql)
            ql = re.sub(r"\bestimated\s+order\b", "estimated total orders", ql)
            ql = re.sub(r"\bestimate\s+order\b", "estimate total orders", ql)
            # Normalize shorthand "estimated disconnection/installation" to orders
            ql = re.sub(r"\bestimat(?:e|ed)\s+disconnection\b", "estimated disconnection orders", ql)
            ql = re.sub(r"\bestimat(?:e|ed)\s+installation\b", "estimated installation orders", ql)

            # Only normalize bare "order(s)" -> total orders when it's NOT part of
            # "installation order(s)" or "disconnection order(s)"
            ql = re.sub(r"(?<!disconnection\s)(?<!installation\s)\borders\b", "total orders", ql)
            ql = re.sub(r"(?<!disconnection\s)(?<!installation\s)\border\b", "total order", ql)

            # -----------------------------
            # IMPORTANT:
            # User often says "installation" (not "installation orders").
            # For prediction-mode output selection, treat "installation" as installation_orders
            # unless they explicitly say "installation count".
            # -----------------------------
            def _wants_installation_wording(q: str) -> bool:
                return bool(re.search(r"\binstallation\b", q)) and not any(
                    p in q for p in [
                        "installation rate", "install rate",
                        "installation order", "installation orders",
                        "installed order", "installed orders",
                        "installation count", "installed count",
                    ]
                )

            # -----------------------------
            # 1) Decide what the user ACTUALLY wants estimated (VISIBLE only)
            # -----------------------------
            visible_est = []

            if _wants_estimate_for_metric(ql, "profit rate"):
                visible_est.append("profit_rate")
            if _wants_estimate_for_metric(ql, "revenue"):
                visible_est.append("revenue")
            # profit: only if user asked "estimated profit" and NOT just "profit rate"
            if _wants_estimate_for_metric(ql, "profit") and ("profit rate" not in ql):
                visible_est.append("profit")
            if _wants_estimate_for_metric(ql, "installation rate"):
                visible_est.append("installation_rate")
            if _wants_estimate_for_metric(ql, "disconnection rate"):
                visible_est.append("disconnection_rate")
            if _wants_estimate_for_metric(ql, "total orders"):
                visible_est.append("total_orders")
            if _wants_estimate_for_metric(ql, "installation orders"):
                visible_est.append("installation_orders")
            if _wants_estimate_for_metric(ql, "disconnection orders"):
                visible_est.append("disconnection_orders")

            # dedupe preserve order
            try:
                visible_est = list(dict.fromkeys(visible_est))
            except Exception:
                pass

            # -----------------------------
            # 2) Internal estimation columns (helpers) — compute them, but DON'T show unless asked
            # -----------------------------
            internal_est = list(visible_est)

            # If user asked estimated installation_rate, compute helper series internally
            if "installation_rate" in internal_est:
                for x in ["installation_orders", "total_orders"]:
                    if x not in internal_est:
                        internal_est.append(x)

            # If user asked estimated disconnection_rate, compute helper series internally
            if "disconnection_rate" in internal_est:
                for x in ["disconnection_orders", "total_orders"]:
                    if x not in internal_est:
                        internal_est.append(x)

            # dedupe preserve order
            try:
                internal_est = list(dict.fromkeys(internal_est))
            except Exception:
                pass

            window = 12
            if group_col:
                df2 = add_estimated_columns_grouped(df, "group_key", internal_est, window=window, mode="always")
            else:
                df2 = add_estimated_columns(df, internal_est, window=window, mode="always")

            # derive estimated_installation_rate / estimated_disconnection_rate if possible
            df2 = add_estimated_rate_columns(df2)

            # Filter output to target (month or year)
            df_out = df2.copy()
            if isinstance(target_filter, tuple) and target_filter[0] == "month":
                _, y, m = target_filter
                key = f"{y:04d}-{m:02d}"
                df_out = df_out[df_out["month"].astype(str) == key]
            elif isinstance(target_filter, tuple) and target_filter[0] == "year":
                _, y = target_filter
                df_out = df_out[df_out["month"].astype(str).str.startswith(f"{y:04d}-")]
            # Keep a month-wise copy for trend explanation (group-wise tables don't have month)
            df_trend_source = df_out.copy()
            # -------------------------------------------------
            # GROUP-WISE OUTPUT (collapse monthly -> one row per group_key)
            # -------------------------------------------------
            if group_col and (not month_wise):
                # df_out currently has month + group_key + metrics (+ estimated)
                # Collapse to group_key rows for the target year/month range

                # Ensure numeric
                def _num(s):
                    return pd.to_numeric(s, errors="coerce").astype(float)

                # For year mode, pick the last month available per group for estimated rates
                # (end-of-year estimate behavior)
                df_out = df_out.copy()
                df_out["__t"] = df_out["month"].astype(str).apply(_month_to_index)

                agg_rows = []

                for gval, part in df_out.groupby("group_key", dropna=False):
                    part = part.sort_values("__t")

                    row = {"group_key": gval}

                    # Actual totals
                    if "installation_orders" in part.columns:
                        row["installation_orders"] = float(_num(part["installation_orders"]).sum())
                    if "total_orders" in part.columns:
                        row["total_orders"] = float(_num(part["total_orders"]).sum())
                    if "disconnection_orders" in part.columns:
                        row["disconnection_orders"] = float(_num(part["disconnection_orders"]).sum())
                    if "revenue" in part.columns:
                        row["revenue"] = float(_num(part["revenue"]).sum())
                    if "profit" in part.columns:
                        row["profit"] = float(_num(part["profit"]).sum())

                    # Actual rates from totals
                    if "installation_orders" in row and "total_orders" in row:
                        denom = row.get("total_orders", 0.0)
                        row["installation_rate"] = round((row["installation_orders"] * 100.0 / denom), 2) if denom else 0.0

                    if "disconnection_orders" in row and "total_orders" in row:
                        denom = row.get("total_orders", 0.0)
                        row["disconnection_rate"] = round((row["disconnection_orders"] * 100.0 / denom), 2) if denom else 0.0

                    if "profit" in row and "revenue" in row:
                        denom = row.get("revenue", 0.0)
                        row["profit_rate"] = round((row["profit"] * 100.0 / denom), 2) if denom else 0.0

                    # Estimated value metrics: SUM across months in target period
                    if "estimated_revenue" in part.columns:
                        row["estimated_revenue"] = float(_num(part["estimated_revenue"]).sum())
                    if "estimated_profit" in part.columns:
                        row["estimated_profit"] = float(_num(part["estimated_profit"]).sum())
                    if "estimated_total_orders" in part.columns:
                        row["estimated_total_orders"] = float(_num(part["estimated_total_orders"]).sum())
                    if "estimated_installation_orders" in part.columns:
                        row["estimated_installation_orders"] = float(_num(part["estimated_installation_orders"]).sum())
                    if "estimated_disconnection_orders" in part.columns:
                        row["estimated_disconnection_orders"] = float(_num(part["estimated_disconnection_orders"]).sum())

                    # Estimated rates: take last month estimate available (end-of-period)
                    def _last_valid(colname):
                        if colname not in part.columns:
                            return None
                        v = _num(part[colname]).replace([np.inf, -np.inf], np.nan).dropna()
                        if len(v) == 0:
                            return None
                        # align with last rows: use last non-null in sorted time
                        vv = pd.to_numeric(part[colname], errors="coerce")
                        vv = vv.replace([np.inf, -np.inf], np.nan)
                        # pick last non-null in time order
                        for x in reversed(list(vv.values)):
                            if np.isfinite(x):
                                return float(x)
                        return None

                    if "estimated_installation_rate" in part.columns:
                        lv = _last_valid("estimated_installation_rate")
                        if lv is not None:
                            row["estimated_installation_rate"] = round(lv, 2)

                    if "estimated_disconnection_rate" in part.columns:
                        lv = _last_valid("estimated_disconnection_rate")
                        if lv is not None:
                            row["estimated_disconnection_rate"] = round(lv, 2)

                    if "estimated_profit_rate" in part.columns:
                        lv = _last_valid("estimated_profit_rate")
                        if lv is not None:
                            row["estimated_profit_rate"] = round(lv, 2)

                    agg_rows.append(row)

                df_group = pd.DataFrame(agg_rows)

                # Decide sorting for TOP N
                sort_col = None
                if "profit" in df_group.columns:
                    sort_col = "profit"
                elif "revenue" in df_group.columns:
                    sort_col = "revenue"
                elif "installation_orders" in df_group.columns:
                    sort_col = "installation_orders"
                elif "total_orders" in df_group.columns:
                    sort_col = "total_orders"
                elif "installation_rate" in df_group.columns:
                    sort_col = "installation_rate"
                elif "profit_rate" in df_group.columns:
                    sort_col = "profit_rate"

                if sort_col:
                    df_group = df_group.sort_values(sort_col, ascending=False)

                # Apply top N if asked
                if top_n:
                    df_group = df_group.head(int(top_n))
                # ✅ ALSO apply "limit N" (your current bug: limit is not applied in prediction mode)
                req_limit = extract_limit(question, default=None, max_limit=MAX_ROWS)
                if req_limit:
                    df_group = df_group.head(int(req_limit))
                # total_orders is useful for "orders" requests; hide only if user did not ask for it
                q_norm = (question or "").lower()
                asked_orders = bool(re.search(r"\b(total\s+orders?|orders?)\b", q_norm))
                asked_est_orders = bool(re.search(r"\bestimat(?:e|ed)\s+(total\s+orders?|orders?)\b", q_norm))

                if "total_orders" in df_group.columns and not (asked_orders or asked_est_orders):
                    df_group = df_group.drop(columns=["total_orders"], errors="ignore")

                df_out = df_group

            if df_out is None or df_out.empty:
                debug_hint = (
                    "No rows match those filters in the training/target window. "
                    "Try checking distinct values (e.g., 'show me distinct ProviderName') or relax filters."
                )
                return {
                    "reply": f"Estimates are generated using rolling Linear Regression on the previous {window} months (past data only).\n{debug_hint}",
                    "table_html": "<p><i>No data</i></p>",
                    "plot_data_uri": None,
                    "rows_returned": 0,
                    "truncated": False,
                    "download_url": None,
                }

            # --------- OUTPUT COLUMN SELECTION (NO EXTRA COLUMNS) ----------
            wants_total_orders = ("total order" in ql) or ("total orders" in ql) or bool(re.search(r"\btotal\s+orders?\b", ql))

            wants_install_orders = any(p in ql for p in ["installation order", "installation orders", "installed order", "installed orders"])
            wants_install_rate = any(p in ql for p in ["installation rate", "install rate"])
            # If user asked "installation, installation rate" treat "installation" as installation_orders too
            if re.search(r"\binstallation\b", ql) and ("installation rate" in ql) and not wants_install_orders:
                wants_install_orders = True

            # "installation" (alone) should behave like installation_orders
            if _wants_installation_wording(ql):
                wants_install_orders = True

            wants_disc_orders = any(p in ql for p in ["disconnection order", "disconnection orders", "disconnected order", "disconnected orders"])
            wants_disc_rate = any(p in ql for p in ["disconnection rate", "disconnect rate", "disconnected rate"])

            wants_revenue = "revenue" in ql
            wants_profit_rate = ("profit rate" in ql)

            asked_profit_explicit = bool(re.search(r"\bprofit\b\s*(?:,|and|&)", ql)) or bool(re.search(r"(?:,|and|&)\s*profit\b", ql))
            wants_profit = asked_profit_explicit or (bool(re.search(r"\bprofit\b", ql)) and not wants_profit_rate)

            # Base columns depend on output shape
            base_cols = []
            if "month" in df_out.columns:
                base_cols.append("month")
            if "group_key" in df_out.columns:
                base_cols.append("group_key")


            # total orders (only if asked OR if user explicitly requested its estimate)
            if wants_total_orders or ("total_orders" in visible_est):
                base_cols += ["total_orders"]
                if "total_orders" in visible_est and "estimated_total_orders" in df_out.columns:
                    base_cols += ["estimated_total_orders"]

            # installation orders (only if asked OR estimate requested)
            if wants_install_orders or ("installation_orders" in visible_est):
                base_cols += ["installation_orders"]
                if "installation_orders" in visible_est and "estimated_installation_orders" in df_out.columns:
                    base_cols += ["estimated_installation_orders"]

            # installation rate (only if asked OR estimate requested)
            if wants_install_rate or ("installation_rate" in visible_est):
                base_cols += ["installation_rate"]
                if ("installation_rate" in visible_est) and ("estimated_installation_rate" in df_out.columns):
                    base_cols += ["estimated_installation_rate"]

            # disconnection orders (only if asked OR estimate requested)
            if wants_disc_orders or ("disconnection_orders" in visible_est):
                base_cols += ["disconnection_orders"]
                if "disconnection_orders" in visible_est and "estimated_disconnection_orders" in df_out.columns:
                    base_cols += ["estimated_disconnection_orders"]

            # disconnection rate (only if asked OR estimate requested)
            if wants_disc_rate or ("disconnection_rate" in visible_est):
                base_cols += ["disconnection_rate"]
                if ("disconnection_rate" in visible_est) and ("estimated_disconnection_rate" in df_out.columns):
                    base_cols += ["estimated_disconnection_rate"]

            # revenue (only if asked OR estimate requested)
            if wants_revenue or ("revenue" in visible_est):
                base_cols += ["revenue"]
                if "revenue" in visible_est and "estimated_revenue" in df_out.columns:
                    base_cols += ["estimated_revenue"]

            # profit (only if asked OR estimate requested)
            if wants_profit or ("profit" in visible_est):
                base_cols += ["profit"]
                if "profit" in visible_est and "estimated_profit" in df_out.columns:
                    base_cols += ["estimated_profit"]


            # profit rate (only if asked OR estimate requested)
            if wants_profit_rate or ("profit_rate" in visible_est):
                base_cols += ["profit_rate"]
                if ("profit_rate" in visible_est) and ("estimated_profit_rate" in df_out.columns):
                    base_cols += ["estimated_profit_rate"]

            # dedupe while preserving order and keep only existing cols
            seen = set()
            final_cols = []
            for c in base_cols:
                if c in df_out.columns and c not in seen:
                    final_cols.append(c)
                    seen.add(c)

            df_out = df_out[final_cols] if final_cols else df_out
            df_out = apply_hidden_cols_policy(df_out, question)

            table_html = rows_to_html_table(df_out.to_dict(orient="records")) if not df_out.empty else "<p><i>No data</i></p>"

            trend_df = df_trend_source if (group_col and (not month_wise)) else df_out
            reply_text = build_prediction_reply(trend_df, visible_est, window=window)

            result = {
                "reply": reply_text,
                "table_html": table_html,
                "plot_data_uri": None,
                "rows_returned": int(len(df_out)),
                "truncated": False,
                "download_url": None,
            }

            if DEBUG_SQL:
                result["debug_sql"] = pred_sql
                try:
                    result["applied_filters"] = _extract_generic_filters_anycol(question)
                except Exception:
                    result["applied_filters"] = []

            return result

        except Exception as e:
            _log_and_mask_error(e, "prediction_mode")
            return _safe_user_error("Prediction failed due to an internal error. Please try again with fewer filters.")


    # --- conversation-aware context ---
    history = history or []
    if history:
        ctx_lines = []
        for m in history[-6:]:
            role = m.get("role", "").upper()
            content = (m.get("content") or m.get("text") or "").strip()
            if content:
                ctx_lines.append(f"{role}: {content}")
        context_block = "\n".join(ctx_lines)
        augmented_question = (
            "Conversation so far:\n"
            f"{context_block}\n\n"
            f"User's latest message:\nUSER: {question}\n"
        )
    else:
        augmented_question = question
    # -----------------------------------

    # --- handle ambiguous time phrases like "last year", "this month" or month without year ---
    has_year = bool(re.search(r'\b20\d{2}\b', qlow))

    month_name = re.search(
        r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
        qlow
    )

    rel_time = re.search(
        r'\b(last year|this year|next year|last month|this month|next month)\b',
        qlow
    )

    if rel_time or (month_name and not has_year):
        msg = (
            "To answer this accurately I need a specific year or month-year. "
            "For example, try: 'in 2025' or 'in October 2025'."
        )
        return {
            "reply": msg,
            "table_html": "",
            "plot_data_uri": None,
        }
    # --- END ambiguous time block ---

    try:
        with _get_duck_con() as local_con:
            sample_rows = local_con.execute(f"SELECT * FROM {TABLE} LIMIT 10").fetchdf()
    except Exception:
        sample_rows = SAMPLE_DF if not SAMPLE_DF.empty else pd.DataFrame(columns=COLS)

    # -------------------------------------------------
    # KEY LOOKUP OVERRIDE (ensures hidden key appears when user asks by key)
    # -------------------------------------------------
    key_col, key_val, key_lim = _extract_key_lookup(question)
    if key_col and key_val:
        sql_text = f"""
        SELECT *
        FROM {TABLE}
        WHERE LOWER(TRIM(CAST({qident(key_col)} AS VARCHAR))) = LOWER(TRIM({_safe_sql_literal(key_val)}))
        LIMIT {int(key_lim)}
        """.strip()

        df = run_sql_and_fetch_df(sql_text)

        total_rows = len(df)
        df_preview = df.head(MAX_ROWS) if (MAX_ROWS and total_rows > MAX_ROWS) else df
        df_preview = apply_hidden_cols_policy(df_preview, question)

        table_html = rows_to_html_table(df_preview.to_dict(orient="records")) if not df_preview.empty else "<p><i>No data</i></p>"

        result = {
            "reply": f"Here are the results for: {question}",
            "table_html": table_html,
            "plot_data_uri": None,
            "rows_returned": total_rows,
            "truncated": (MAX_ROWS and total_rows > MAX_ROWS),
            "download_url": None,
        }
        if DEBUG_SQL:
            result["debug_sql"] = sql_text
        return result
    # -------------------------------------------------
    # DETAILS REPORT OVERRIDE (OrderId / Account)
    # -------------------------------------------------
    def _find_last_order_or_account(history):
        if not history:
            return (None, None)

        # scan backward
        for m in reversed(history):
            txt = (m.get("content") or m.get("text") or "").strip()
            if not txt:
                continue
            oid = extract_order_id(txt)
            acc = extract_account_number(txt)
            if oid or acc:
                return (oid, acc)
        return (None, None)
    # ---------------------------
    if is_details_request(question) or is_bare_id_message(question):
        oid = extract_order_id(question)
        acc = extract_account_number(question)

        # ✅ If user typed ONLY a number, infer whether it's account/order from history
        if not (oid or acc) and is_bare_id_message(question):
            last_type = _find_last_details_type(history)

            bare = question.strip().replace(",", "")
            bare = bare.lstrip("#").strip()

            if last_type == "account":
                acc = bare
            elif last_type == "order":
                oid = bare
            else:
                # fallback: treat as order id (or you can choose account)
                oid = bare

        # ✅ If user said "yes give details" etc (no id), fall back to last id in history
        if not (oid or acc):
            oid2, acc2 = _find_last_order_or_account(history)
            oid = oid or oid2
            acc = acc or acc2

        if oid or acc:
            details_sql = build_details_sql(order_id=oid, account=acc)
            if details_sql:
                df_details = run_sql_and_fetch_df(details_sql)
                df_details = apply_hidden_cols_policy(df_details, question)

                report = build_order_or_account_report(df_details, question)

                result = {
                    "reply": report["reply"],
                    "table_html": report["table_html"],
                    "plot_data_uri": report.get("plot_data_uri"),
                    "rows_returned": int(len(df_details)),
                    "truncated": False,
                    "download_url": None,
                }
                if DEBUG_SQL:
                    result["debug_sql"] = details_sql
                return result

    # 1) Ask model for SQL  (conversation-aware)
    sql_text = ask_model_for_sql(augmented_question, schema_cols, sample_rows)

    # 2) If NO_SQL → theory/general mode (conversation-aware)
    if not sql_text or sql_text.strip().upper() == "NO_SQL":
        return answer_theory_question(augmented_question)
    user_limit = extract_limit(question, default=None, max_limit=MAX_ROWS)
    if user_limit and not re.search(r"\blimit\b", sql_text, flags=re.IGNORECASE):
        sql_text = sql_text.rstrip().rstrip(";") + f"\nLIMIT {int(user_limit)}"

    cache_key = sql_text.strip()
    cached = QUERY_CACHE.get(cache_key)
    if cached:
        if (time.time() - cached["time"]) < CACHE_TTL:
            return cached["response"]

    # 3) Normalize & re-check
    sql_text = sql_text.strip()
    if not is_safe_select(sql_text):
        return _safe_user_error("Generated SQL was blocked for safety. Please rephrase your question.")

    # 4) Execute with at most one auto-fix attempt
    try_count = 0
    last_error = None
    df = None

    while try_count < 2:
        try:
            df = run_sql_and_fetch_df(sql_text, print_errors=(try_count > 0))
            break
        except Exception as e:
            last_error = str(e)
            if try_count == 0:
                fixed = ask_model_fix_sql(sql_text, last_error, schema_cols, sample_rows)
                if fixed and fixed.strip().upper() != "NO_SQL":
                    sql_text = fixed.strip()
                    try_count += 1
                    continue
            _log_and_mask_error(last_error, "sql_execution")
            return _safe_user_error("I couldn't execute the query I generated. Please try asking in a simpler way.")

    if df is None:
        return _safe_user_error("No data returned for that query.")

    # 4.a) If the question talks about customer name(s), drop rows where both first & last are null/empty
    qlow = (question or "").lower()
    if re.search(r'\bcustomer\s+names?\b', qlow) and CUST_FIRST_COL and CUST_LAST_COL:
        try:
            fn = CUST_FIRST_COL
            ln = CUST_LAST_COL
            if fn in df.columns and ln in df.columns:
                fn_series = df[fn].astype(str).str.strip()
                ln_series = df[ln].astype(str).str.strip()
                mask = ~((fn_series.eq("") | fn_series.eq("nan")) &
                         (ln_series.eq("") | ln_series.eq("nan")))
                df = df[mask]
        except Exception as e:
            print("Post-filter customer names error:", e)

    total_rows = len(df)
    truncated = False
    df_preview = df

    if MAX_ROWS and total_rows > MAX_ROWS:
        truncated = True
        df_preview = df.head(MAX_ROWS)

    df_preview = apply_hidden_cols_policy(df_preview, question)
    if not df_preview.empty:
        table_html = rows_to_html_table(df_preview.to_dict(orient="records"))
    else:
        table_html = "<p><i>No data</i></p>"

    # 5) Optional plot (AUTO + USER-FORCED)
    plot_uri = None

    def _detect_forced_chart(q: str):
        q = (q or "").lower()
        if "stacked" in q:
            return "stacked_bar"
        if "pie" in q:
            return "pie"
        if "scatter" in q:
            return "scatter"
        if "hist" in q or "histogram" in q:
            return "hist"
        if "line" in q:
            return "line"
        if "bar" in q:
            return "bar"
        return None

    try:
        if not df.empty:
            forced = _detect_forced_chart(question)
            kind = forced

            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            cat_cols = [c for c in df.columns if c not in num_cols]

            # USER-FORCED CHART
            if kind == "hist" and num_cols:
                plot_uri = plot_to_base64([], df[num_cols[0]].tolist(), kind="hist", title=question)

            elif kind == "scatter" and len(num_cols) >= 2:
                plot_uri = plot_to_base64(
                    df[num_cols[0]].tolist(),
                    df[num_cols[1]].tolist(),
                    kind="scatter",
                    title=question,
                )

            elif kind == "stacked_bar" and cat_cols and len(num_cols) >= 2:
                labels = df[cat_cols[0]].astype(str).tolist()
                series = [df[c].fillna(0).tolist() for c in num_cols]
                plot_uri = plot_to_base64(labels, series, kind="stacked_bar", title=question)

            elif kind == "pie"and cat_cols and num_cols:
                labels = df[cat_cols[0]].astype(str).tolist()
                vals = df[num_cols[0]].fillna(0).tolist()
                plot_uri = plot_to_base64(labels, vals, kind="pie", title=question)

            elif kind == "line" and cat_cols and num_cols:
                plot_uri = plot_to_base64(
                    df[cat_cols[0]].astype(str).tolist(),
                    df[num_cols[0]].fillna(0).tolist(),
                    kind="line",
                    title=question,
                )

            elif kind == "bar" and cat_cols and num_cols:
                plot_uri = plot_to_base64(
                    df[cat_cols[0]].astype(str).tolist(),
                    df[num_cols[0]].fillna(0).tolist(),
                    kind="bar",
                    title=question,
                )

            # AUTO CHART (if no force)
            if plot_uri is None:
                if len(num_cols) == 1:
                    plot_uri = plot_to_base64([], df[num_cols[0]].tolist(), kind="hist", title=question)
                elif len(num_cols) == 2 and not cat_cols:
                    plot_uri = plot_to_base64(
                        df[num_cols[0]].tolist(),
                        df[num_cols[1]].tolist(),
                        kind="scatter",
                        title=question,
                    )
                elif cat_cols and len(num_cols) >= 2:
                    labels = df[cat_cols[0]].astype(str).tolist()
                    series = [df[c].fillna(0).tolist() for c in num_cols]
                    plot_uri = plot_to_base64(labels, series, kind="stacked_bar", title=question)
                elif cat_cols and len(num_cols) == 1 and len(df) <= 12:
                    plot_uri = plot_to_base64(
                        df[cat_cols[0]].astype(str).tolist(),
                        df[num_cols[0]].fillna(0).tolist(),
                        kind="pie",
                        title=question,
                    )
                elif "month" in df.columns and num_cols:
                    plot_uri = plot_to_base64(
                        df["month"].astype(str).tolist(),
                        df[num_cols[0]].fillna(0).tolist(),
                        kind="line",
                        title=question,
                    )
                elif cat_cols and num_cols:
                    plot_uri = plot_to_base64(
                        df[cat_cols[0]].astype(str).tolist(),
                        df[num_cols[0]].fillna(0).tolist(),
                        kind="bar",
                        title=question,
                    )

    except Exception as e:
        print("Plot error:", e, traceback.format_exc())
        plot_uri = None

    # 5.a) Prepare Excel export ONLY as in-memory cache; no file written to disk
    download_url = None
    try:
        file_id = uuid.uuid4().hex
        EXPORT_CACHE[file_id] = {"sid": getattr(g, "sid", ""), "df": df_preview}
        download_url = f"/download/{file_id}"

    except Exception as e:
        print("Export cache error:", e, traceback.format_exc())
        download_url = None

    # 6) Summarize ONLY if user asked
    if df_preview.empty:
        explanation = "No data found for that query."
    else:
        if user_wants_summary(question):
            # your existing LLM summarizer block goes here (keep it same)
            explanation = ""
            preview_rows = df_preview.head(50).to_dict(orient="records")
            try:
                prompt = (
                    f"User question: {question}\n"
                    f"SQL used:\n{sql_text}\n\n"
                    f"Result preview (first {min(50, len(preview_rows))} rows, JSON):\n"
                    f"{json.dumps(preview_rows, ensure_ascii=False)}\n\n"
                    f"Please provide:\n"
                    f"1. One short headline-style summary line.\n"
                    f"2. One or two sentences explaining the key points in plain language.\n\n"
                    f"IMPORTANT: Do NOT use LaTeX. If you include formulas, write them in plain text."
                )
                resp = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a concise data analyst. Avoid LaTeX; use plain text formulas."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=256,
                )
                explanation = (resp.choices[0].message.content or "").strip()
                explanation = strip_latex_math(explanation)
            except Exception:
                explanation = f"Here are the results for: {question}"
        else:
            explanation = f"Here are the results for: {question} (Ask “summarize” if you want insights.)"

    result = {
        "reply": explanation,
        "table_html": table_html,
        "plot_data_uri": plot_uri,
        "rows_returned": total_rows,
        "truncated": truncated,
        "download_url": download_url,
    }
    if DEBUG_SQL:
        result["debug_sql"] = sql_text
    try:
        QUERY_CACHE[cache_key] = {
            "response": result,
            "time": time.time()
        }
    except Exception:
        pass

    return result

# FLASK APP
# -------------------------------------------------
app = Flask(__name__, static_folder='static')

SESSION_COOKIE_NAME = "sid"
SESSION_COOKIE_TTL_DAYS = 365 * 2

def _make_sid():
    return uuid.uuid4().hex

@app.route("/debug", methods=["GET"], strict_slashes=False)
def debug():
    return jsonify({
        "server_file": __file__,
        "PRED_CHAT_ID": PRED_CHAT_ID,
        "SAMPLE_CHAT_ID": SAMPLE_CHAT_ID,
        "routes": sorted([str(r) for r in app.url_map.iter_rules()])
    })

@app.before_request
def ensure_sid_cookie():
    sid_from_cookie = request.cookies.get(SESSION_COOKIE_NAME, "").strip()
    sid_from_query = (request.args.get('sid') or "").strip()
    sid_from_payload = ""
    try:
        sid_from_payload = (request.get_json(silent=True) or {}).get('sid') or ""
    except Exception:
        sid_from_payload = ""
    sid = sid_from_payload or sid_from_query or sid_from_cookie or _make_sid()
    g.sid = sid
    g.set_sid_cookie = (sid != sid_from_cookie)

@app.after_request
def set_sid_cookie(response):
    try:
        if getattr(g, "set_sid_cookie", False):
            secure_flag = True
            try:
                if app.debug or request.host.startswith("localhost"):
                    secure_flag = False
            except Exception:
                secure_flag = True
            samesite_val = 'None' if secure_flag else 'Lax'
            response.set_cookie(
                SESSION_COOKIE_NAME,
                g.sid,
                max_age=SESSION_COOKIE_TTL_DAYS * 24 * 3600,
                httponly=True,
                samesite=samesite_val,
                secure=secure_flag,
                path="/"
            )
            try:
                response.headers['X-Debug-SID'] = g.sid
            except Exception:
                pass
    except Exception:
        pass
    return response

@app.route('/', methods=["GET"], strict_slashes=False)
def index():
    return send_from_directory('.', 'index.html')

@app.route('/chat', methods=['POST'], strict_slashes=False)
def chat():
    payload = request.json or {}
    question = (payload.get('message') or "").strip()

    history = payload.get('history') or []
    if not question:
        return jsonify({"error": "No message provided"}), 400

    qlow = question.lower()
    print(f"Q: {question}")

    # quick date helpers
    if any(phrase in qlow for phrase in [
        "what year is this",
        "what year is it now",
        "current year",
        "this year now",
        "what is today's year",
    ]):
        return jsonify({"reply": f"The current year is {dtmod.date.today().year}.", "table_html": "", "plot_data_uri": None})

    if any(phrase in qlow for phrase in [
        "what month is this",
        "current month",
        "what month are we in",
    ]):
        today = dtmod.date.today()
        return jsonify({"reply": f"We are currently in {today.strftime('%B %Y')}.", "table_html": "", "plot_data_uri": None})

    if any(phrase in qlow for phrase in [
        "what was last year",
        "last year is what",
        "which year was last year",
        "previous year",
    ]):
        return jsonify({"reply": f"Last year was {dtmod.date.today().year - 1}.", "table_html": "", "plot_data_uri": None})

    if any(phrase in qlow for phrase in [
        "what is today",
        "today's date",
        "what day is today",
        "what date is today",
    ]):
        today = dtmod.date.today()
        return jsonify({"reply": f"Today's date is {today.strftime('%B %d, %Y')}.", "table_html": "", "plot_data_uri": None})

    intent = classify_intent(question)

    # Smalltalk
    if intent == "smalltalk":
        return jsonify({
            "reply": "Hello 👋 I’m BundleAI. Ask me anything about your dataset — or click “Sample questions you can ask” to run examples.",
            "table_html": "",
            "plot_data_uri": None
        })

    # Meta / schema
    if intent == "meta":
        cols_html = "<br>".join(f"{i+1}. <strong>{escape(c)}</strong>" for i, c in enumerate(COLS))
        return jsonify({
            "reply": f"This dataset uses table '{TABLE}' with {ROWCOUNT or 'unknown'} rows and {len(COLS)} columns.",
            "table_html": cols_html,
            "plot_data_uri": None
        })

    # Row/column count quick question
# Row/column count quick question (ONLY when no filters are present)
    is_rowcount_phrase = re.search(r'\bhow many rows\b|\brow count\b|\bnumber of rows\b', qlow)

    # If the user includes filters, do NOT short-circuit; let SQL handler run
    has_filter_words = re.search(
    r'\bwhere\b|\bwith\b|\bfor\b|\bprovider\b|\bproduct\b|\bltype\b|=',
    qlow
    )


    if is_rowcount_phrase and not has_filter_words:
        try:
            rows = ROWCOUNT
            if rows is None:
                with _get_duck_con() as local_con:
                    rows = local_con.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
            cols = len(COLS)
            return jsonify({"reply": f"The dataset contains {rows:,} rows and {cols} columns.", "table_html": "", "plot_data_uri": None})
        except Exception as e:
            _log_and_mask_error(e, "count_rows_cols")
            return jsonify(_safe_user_error("Could not determine rows/cols right now.")), 500

    resp = handle_general_data_question(question, history=history)

    # Ensure no LaTeX leaks
    try:
        if isinstance(resp, dict) and "reply" in resp:
            resp["reply"] = strip_latex_math(resp.get("reply") or "")
    except Exception:
        pass

    return jsonify(resp)

# -------------------------------------------------
# DOWNLOAD ENDPOINT FOR EXCEL (in-memory only)
# -------------------------------------------------
@app.route("/download/<token>", methods=["GET"], strict_slashes=False)
def download_result(token):
    data = EXPORT_CACHE.get(token)
    if data is None:
        return "File not found or expired", 404

    # session bind
    expected_sid = (data.get("sid") or "")
    if expected_sid and expected_sid != getattr(g, "sid", ""):
        return "File not found or expired", 404

    data = data.get("df")

    output = BytesIO()

    if isinstance(data, dict):
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            for sheet_name, df in data.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        df = data
        df.to_excel(output, index=False)

    output.seek(0)
    EXPORT_CACHE.pop(token, None)

    return send_file(
        output,
        as_attachment=True,
        download_name="result.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# -------------------------------------------------
# MEMORY MANAGEMENT (SQLite + JSON fallback)
# -------------------------------------------------
MEMORY_FILE = Path('./data/conversations.json')
MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)

DB_PATH = Path('./data/chat_memory.db')
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def _ensure_chats_table_and_migration(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                title TEXT,
                created_at INTEGER
            );
        """)
        conn.commit()

        cur.execute("PRAGMA table_info(chats);")
        existing_cols = [r[1] for r in cur.fetchall()]

        if 'session_id' not in existing_cols:
            try:
                cur.execute("ALTER TABLE chats ADD COLUMN session_id TEXT;")
                conn.commit()
            except Exception:
                pass

        cur.execute("PRAGMA table_info(chats);")
        existing_cols = [r[1] for r in cur.fetchall()]

        if 'messages_json' not in existing_cols:
            try:
                cur.execute("ALTER TABLE chats ADD COLUMN messages_json TEXT;")
                conn.commit()
            except Exception:
                pass

        cur.close()
    finally:
        conn.close()

def _row_to_conv(row):
    try:
        if len(row) == 4:
            cid, title, created_at, messages_json = row
            session_id = ""
        elif len(row) == 5:
            cid, title, created_at, session_id, messages_json = row
        else:
            cid = row[0]
            title = row[1] if len(row) > 1 else None
            created_at = row[2] if len(row) > 2 else 0
            session_id = row[3] if len(row) > 3 else ''
            messages_json = row[4] if len(row) > 4 else ''
    except Exception:
        cid = row[0] if row else "unknown"
        title = ""
        created_at = 0
        session_id = ""
        messages_json = ""

    try:
        messages = json.loads(messages_json) if messages_json else []
    except Exception:
        messages = []

    return {
        "id": cid,
        "title": title,
        "created_at": created_at or 0,
        "session_id": session_id,
        "messages": messages
    }

def _load_conversations():
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            cur = conn.cursor()
            try:
                rows = cur.execute(
                    "SELECT id, title, created_at, session_id, messages_json "
                    "FROM chats ORDER BY created_at DESC"
                ).fetchall()
            except Exception:
                rows = cur.execute(
                    "SELECT id, title, created_at, messages_json "
                    "FROM chats ORDER BY created_at DESC"
                ).fetchall()

        out = {}
        for r in rows:
            conv = _row_to_conv(r)
            out[conv['id']] = {
                "id": conv['id'],
                "title": conv.get("title") or "Chat",
                "created_at": conv.get("created_at", 0),
                "messages": conv.get("messages", [])
            }
        return out
    except Exception:
        try:
            if MEMORY_FILE.exists():
                return json.load(open(MEMORY_FILE, 'r', encoding='utf8'))
        except Exception:
            pass
        return {}

def _save_conversation_single(chat_id, title, created_at, messages, session_id=""):
    _ensure_chats_table_and_migration(DB_PATH)
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(chats);")
            existing_cols = [r[1] for r in cur.fetchall()]

            if 'session_id' in existing_cols and 'messages_json' in existing_cols:
                cur.execute(
                    "INSERT OR REPLACE INTO chats (id, title, created_at, session_id, messages_json) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (chat_id, title, created_at, session_id, json.dumps(messages))
                )
            elif 'messages_json' in existing_cols:
                cur.execute(
                    "INSERT OR REPLACE INTO chats (id, title, created_at, messages_json) "
                    "VALUES (?, ?, ?, ?)",
                    (chat_id, title, created_at, json.dumps(messages))
                )
            else:
                raise RuntimeError("SQLite schema missing messages_json column")

            conn.commit()
        return True
    except Exception:
        try:
            all_conv = {}
            if MEMORY_FILE.exists():
                try:
                    all_conv = json.load(open(MEMORY_FILE, 'r', encoding='utf8'))
                except Exception:
                    all_conv = {}
            all_conv[chat_id] = {
                "id": chat_id,
                "title": title,
                "created_at": created_at,
                "session_id": session_id,
                "messages": messages
            }
            with open(MEMORY_FILE, 'w', encoding='utf8') as f:
                json.dump(all_conv, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False

@app.route('/memory/list', methods=['GET'], strict_slashes=False)
def memory_list():
    sid = (getattr(g, "sid", "") or request.args.get('sid') or "").strip()
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(chats);")
            cols = [r[1] for r in cur.fetchall()]

            if sid and 'session_id' in cols:
                rows = cur.execute(
                    "SELECT id, title, created_at FROM chats WHERE session_id = ? ORDER BY created_at DESC",
                    (sid,)
                ).fetchall()
            else:
                rows = cur.execute(
                    "SELECT id, title, created_at FROM chats ORDER BY created_at DESC"
                ).fetchall()

        chats = [{"id": r[0], "title": r[1] or "Chat", "created_at": r[2]} for r in rows]

        # ALWAYS include sample chat at top (even if user has no chats)
        chats = [c for c in chats if c.get("id") != SAMPLE_CHAT_ID]
        chats.insert(0, {"id": SAMPLE_CHAT_ID, "title": SAMPLE_CHAT_TITLE, "created_at": SAMPLE_CHAT_CREATED_AT})
        chats.insert(1, {"id": PRED_CHAT_ID, "title": PRED_CHAT_TITLE, "created_at": PRED_CHAT_CREATED_AT})
        return jsonify({"chats": chats})

    except Exception as e:
        try:
            conv = _load_conversations()
            items = []
            for k, v in conv.items():
                if sid:
                    if isinstance(v, dict) and v.get("session_id") and v.get("session_id") != sid:
                        continue
                items.append({"id": k, "title": v.get("title") or "Chat", "created_at": v.get("created_at", 0)})

            items.sort(key=lambda x: x['created_at'] or 0, reverse=True)
            items = [c for c in items if c.get("id") != SAMPLE_CHAT_ID]
            items.insert(0, {"id": SAMPLE_CHAT_ID, "title": SAMPLE_CHAT_TITLE, "created_at": SAMPLE_CHAT_CREATED_AT})
            items.insert(1, {"id": PRED_CHAT_ID, "title": PRED_CHAT_TITLE, "created_at": PRED_CHAT_CREATED_AT})

            return jsonify({"chats": items})
        except Exception:
            return jsonify({"error": f"Failed to list chats: {e}"}), 500

@app.route('/memory/load', methods=['GET'], strict_slashes=False)
def memory_load():
    cid = (request.args.get('id') or "").strip()
    sid = (getattr(g, "sid", "") or (request.args.get('sid') or "")).strip()

    if not cid:
        return jsonify({"error": "Missing chat ID"}), 400

    # Always serve sample + predicted chats from code (never DB)
    if cid == SAMPLE_CHAT_ID:
        return jsonify(SAMPLE_CHAT)

    # bulletproof: accept both spellings just in case
    if cid in (PRED_CHAT_ID, "predicted-chat", "predict-samples", "predicted-questions"):
        return jsonify(PRED_CHAT)

    # normal chats from DB
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(chats);")
            cols = [r[1] for r in cur.fetchall()]

            if sid and 'session_id' in cols:
                row = cur.execute(
                    "SELECT id, title, created_at, session_id, messages_json FROM chats WHERE id = ? AND session_id = ?",
                    (cid, sid)
                ).fetchone()
            else:
                row = cur.execute(
                    "SELECT id, title, created_at, messages_json FROM chats WHERE id = ?",
                    (cid,)
                ).fetchone()

        if not row:
            return jsonify({"error": "Chat not found"}), 404

        conv = _row_to_conv(row)

        if sid and conv.get("session_id") and conv.get("session_id") != sid:
            return jsonify({"error": "Chat not found"}), 404

        return jsonify({
            "id": conv["id"],
            "title": conv.get("title") or "Chat",
            "created_at": conv.get("created_at", 0),
            "messages": conv.get("messages", [])
        })

    except Exception as e:
        return jsonify({"error": f"Failed to load chat: {e}"}), 500

@app.route('/memory/save', methods=['POST'], strict_slashes=False)
def memory_save():
    payload = request.get_json(force=True) or {}
    sid = (getattr(g, "sid") or payload.get('sid') or "").strip()
    cid = payload.get('id') or f"chat-{int(time.time() * 1000)}"

    if cid in (SAMPLE_CHAT_ID, PRED_CHAT_ID):
        return jsonify({"ok": True, "id": cid})

    title = payload.get('title') or 'Chat'
    created_at = payload.get('created_at') or int(time.time() * 1000)
    messages = payload.get('messages') or []

    ok = _save_conversation_single(cid, title, created_at, messages, session_id=sid)
    if not ok:
        return jsonify({"error": "Failed to save chat"}), 500
    return jsonify({"ok": True, "id": cid})

@app.route("/health", methods=["GET"], strict_slashes=False)
def health():
    try:
        with _get_duck_con() as local_con:
            local_con.execute("SELECT 1")
        with sqlite3.connect(str(DB_PATH)) as conn:
            conn.execute("SELECT 1")
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "error", "details": str(e)}), 500

@app.route('/memory/delete', methods=['POST'], strict_slashes=False)
def memory_delete():
    payload = request.get_json(force=True) or {}
    cid = payload.get('id')
    sid = (getattr(g, "sid", "") or payload.get('sid') or "").strip()
    if not cid:
        return jsonify({"error": "Missing chat ID"}), 400

    # Block deleting sample + predicted chats forever
    if cid in (SAMPLE_CHAT_ID, PRED_CHAT_ID):
        return jsonify({"ok": False, "error": "This chat cannot be deleted"}), 400

    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(chats);")
            cols = [r[1] for r in cur.fetchall()]

            if sid and 'session_id' in cols:
                cur.execute("DELETE FROM chats WHERE id = ? AND session_id = ?", (cid, sid))
            else:
                cur.execute("DELETE FROM chats WHERE id = ?", (cid,))
            conn.commit()

        return jsonify({"ok": True})
    except Exception as e:
        try:
            if MEMORY_FILE.exists():
                m = json.load(open(MEMORY_FILE, 'r', encoding='utf8'))
                if cid in m:
                    if sid and m[cid].get("session_id") and m[cid].get("session_id") != sid:
                        return jsonify({"error": "Chat not found"}), 404
                    m.pop(cid, None)
                    with open(MEMORY_FILE, 'w', encoding='utf8') as f:
                        json.dump(m, f, ensure_ascii=False, indent=2)
                    return jsonify({"ok": True})
        except Exception:
            pass
        return jsonify({"error": f"Failed to delete chat: {e}"}), 500

# MAIN
if __name__ == "__main__":
    print(f"Starting server on http://localhost:{PORT} — using table {TABLE}")
    print(">>> SERVER STARTED FROM FILE:", __file__)
    print(">>> PRED_CHAT_ID IS:", PRED_CHAT_ID)

    # print routes at startup so you can confirm /memory/load is registered
    try:
        print("=== ROUTES REGISTERED ===")
        for r in sorted(app.url_map.iter_rules(), key=lambda x: str(x)):
            print(r)
        print("=========================")
    except Exception as e:
        print("Route print failed:", e)

    app.run(host="0.0.0.0", port=PORT)
