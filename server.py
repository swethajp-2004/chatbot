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
from utils import rows_to_html_table, plot_to_base64, embed_text  # your existing utils
import uuid
from pathlib import Path
import sqlite3
import datetime as dtmod
from io import BytesIO  # in-memory Excel export
from sklearn.linear_model import LinearRegression

# CONFIG / ENV
# -------------------------------------------------
load_dotenv()

DEBUG_SQL = True

TZ = ZoneInfo("Asia/Kolkata")  # user timezone

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DUCKDB_FILE = os.getenv("DUCKDB_FILE", "./data/sales.duckdb")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
PORT = int(os.getenv("PORT", "3000"))
MAX_ROWS = int(os.getenv("MAX_ROWS", "2000"))
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
    'NetAfterChargeback', 'NetAfterCb', 'GrossAfterCb',
    'Revenue', 'ProviderPaid', 'Amount', 'SaleAmount'
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

print("Auto-detected columns:", DETECTED)

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
    """Check if SQL is a safe SELECT query (no DML/DDL)."""
    sql_clean = re.sub(r"--.*$", "", sql, flags=re.MULTILINE)
    sql_clean = re.sub(r"/\*.*?\*/", "", sql_clean, flags=re.DOTALL).strip()
    if not sql_clean:
        return False
    first_word = sql_clean.split()[0].upper()
    if first_word != "SELECT":
        return False
    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "REPLACE"]
    for w in forbidden:
        if re.search(rf"\b{w}\b", sql_clean, flags=re.IGNORECASE):
            return False
    return True
def _wants_addon(question: str) -> bool:
    q = (question or "").lower()
    return ("addon" in q) or ("add-on" in q) or ("add on" in q)

def _extract_addon_value(question: str):
    """
    Detect explicit addon value in the user's question.
    Returns (addon_value:int|None, is_explicit:bool)

    Matches:
      addon = 1, addon=1, addon is 1, addon:1, addon 1
      addon = 0, addon=0, addon is 0, addon:0, addon 0
    """
    q = (question or "").lower()

    # explicit numeric addon filter (0/1)
    m = re.search(r"\badd\s*-\s*on\b\s*(?:=|:|\bis\b)?\s*(0|1)\b", q)
    if m:
        return (int(m.group(1)), True)

    m = re.search(r"\badd\s+on\b\s*(?:=|:|\bis\b)?\s*(0|1)\b", q)
    if m:
        return (int(m.group(1)), True)

    m = re.search(r"\baddon\b\s*(?:=|:|\bis\b)?\s*(0|1)\b", q)
    if m:
        return (int(m.group(1)), True)

    return (None, False)

def addon_filter_sql(question: str) -> str:
    """
    Addon rules:
    - If user explicitly says addon=1 => filter Addon = 1 (no forced Ltype)
    - If user explicitly says addon=0 => filter Addon = 0 (and default Ltype applies unless user asked Ltype)
    - If user only says "addon" with no value => DO NOT filter Addon column (it is a mode -> Ltype='A')
    """
    addon_val, is_explicit = _extract_addon_value(question)
    if not is_explicit:
        return ""

    # Use TRY_CAST for safety if Addon is stored as text
    return f"(TRY_CAST(Addon AS INTEGER) = {int(addon_val)})"

def ltype_filter_sql(question: str) -> str:
    """
    ✅ Final rules (as requested):

    Default (normal questions, no addon mentioned at all)
      -> Ltype = 'L' OR Ltype IS NULL

    If user says “addon” (but does NOT say addon=0 or addon=1)
      -> treat as a mode => Ltype = 'A'

    If user says addon = 1
      -> do NOT force any Ltype (unless user explicitly asks Ltype)
      -> Ltype can be L / A / NULL (anything)

    If user says addon = 0
      -> filter Addon = 0
      -> and behave like default for Ltype again (L or NULL), unless user explicitly asked Ltype
    """
    q = (question or "").lower()

    addon_val, is_explicit = _extract_addon_value(question)

    # If addon is explicitly specified:
    # - addon=1 => no forced Ltype
    # - addon=0 => default Ltype (L or NULL)
    if is_explicit:
        if addon_val == 1:
            return ""  # do not force any Ltype
        else:
            return "(TRIM(CAST(Ltype AS VARCHAR)) = 'L' OR Ltype IS NULL)"

    # If addon keyword is present without explicit value => Ltype must be A
    if _wants_addon(question):
        return "(TRIM(CAST(Ltype AS VARCHAR)) = 'A')"

    # Default => L + NULL
    return "(TRIM(CAST(Ltype AS VARCHAR)) = 'L' OR Ltype IS NULL)"

def policy_filter_sql(question: str, *, already_has_ltype: bool = False, already_has_addon: bool = False) -> str:
    """
    Build the combined filter to apply at query-time.

    Important behavior:
    - If user explicitly wrote Ltype in the question, we DO NOT override it (handled by caller via already_has_ltype flag).
      (In apply_ltype_policy_to_sql we treat SQL containing 'ltype' as already_has_ltype)
    - If SQL already contains addon filter, we do not inject another addon filter.
    """
    parts = []

    # Addon filter (only when addon=0/1 explicitly mentioned)
    addon_f = "" if already_has_addon else addon_filter_sql(question)
    if addon_f:
        parts.append(addon_f)

    # Ltype filter (default / addon-mode), but skip if SQL already has Ltype
    ltype_f = "" if already_has_ltype else ltype_filter_sql(question)
    if ltype_f:
        parts.append(ltype_f)

    if not parts:
        return ""

    if len(parts) == 1:
        return parts[0]
    return "(" + " AND ".join(parts) + ")"

def apply_ltype_policy_to_sql(sql: str, question: str) -> str:
    """
    Inject Ltype/AddOn policy filter into the TOP-LEVEL query for {TABLE}.
    Inserts into WHERE if present, else adds WHERE before GROUP BY/HAVING/ORDER BY/LIMIT.

    Robust to newlines/tabs because it uses regex word-boundary clause detection.

    Also fixes invalid HAVING usage:
      1) HAVING without GROUP BY -> convert first HAVING to WHERE
      2) HAVING appears before GROUP BY -> convert first HAVING to WHERE
    """
    if not sql or not isinstance(sql, str):
        return sql

    s = sql.strip().rstrip(";")
    s_low = s.lower()

    # If query doesn't reference the main table, don't touch it
    if TABLE.lower() not in s_low:
        return sql

    # If SQL already mentions these columns, don't override those parts
    already_has_ltype = re.search(r"\b(where|having)\b[\s\S]*\bltype\b", s_low, re.IGNORECASE) is not None
    already_has_addon = re.search(r"\baddon\b", s_low, flags=re.IGNORECASE) is not None

    filt = policy_filter_sql(
        question,
        already_has_ltype=already_has_ltype,
        already_has_addon=already_has_addon
    )
    if not filt:
        return sql

    # ---------- helpers ----------
    def _find_clause_span(text: str, clause: str):
        """
        Return (start, end) span of the FIRST occurrence of clause keyword as a whole word,
        or None if not found.
        """
        m = re.search(rf"\b{re.escape(clause)}\b", text, flags=re.IGNORECASE)
        return m.span() if m else None

    def _replace_first_keyword(text: str, keyword: str, replacement: str):
        return re.sub(rf"\b{re.escape(keyword)}\b", replacement, text, count=1, flags=re.IGNORECASE)

    # ---------- FIX HAVING problems BEFORE injecting ----------
    span_having = _find_clause_span(s, "HAVING")
    span_group  = _find_clause_span(s, "GROUP")
    # Note: we specifically care about "GROUP BY", but "GROUP" is good enough for ordering checks
    # because GROUP BY always starts with GROUP.

    if span_having:
        has_group_by = re.search(r"\bGROUP\s+BY\b", s, flags=re.IGNORECASE) is not None

        # 1) HAVING without GROUP BY -> invalid
        if not has_group_by:
            s = _replace_first_keyword(s, "HAVING", "WHERE")
        else:
            # 2) HAVING before GROUP BY -> invalid ordering
            span_having2 = _find_clause_span(s, "HAVING")
            span_group2  = _find_clause_span(s, "GROUP")
            if span_having2 and span_group2 and span_having2[0] < span_group2[0]:
                s = _replace_first_keyword(s, "HAVING", "WHERE")

    # refresh lowered string after potential edits
    s_low = s.lower()

    # ---------- Inject filter into WHERE or create WHERE ----------
    span_where = _find_clause_span(s, "WHERE")
    if span_where:
        # insert right after WHERE keyword
        insert_at = span_where[1]  # position just after WHERE
        head = s[:insert_at]
        tail = s[insert_at:]
        # WHERE <filt> AND (<existing conditions...>)
        return f"{head} {filt} AND ({tail.strip()})"

    # No WHERE: insert before the earliest of GROUP BY / HAVING / ORDER BY / LIMIT (if any)
    cut_spans = []
    for kw in ["GROUP", "HAVING", "ORDER", "LIMIT"]:
        sp = _find_clause_span(s, kw)
        if sp:
            cut_spans.append(sp[0])

    if cut_spans:
        cut = min(cut_spans)
        return f"{s[:cut].rstrip()} WHERE {filt} {s[cut:].lstrip()}"
    else:
        return f"{s} WHERE {filt}"

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
    mt2 = token_pat2.search(_norm_key(q))
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
def user_wants_summary(question: str) -> bool:
    q = (question or "").lower()
    return any(k in q for k in ["summarize", "summary", "explain the result", "explain results"])

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

# -------------------------------------------------
# GENERIC COLUMN NAME RESOLUTION (FIXES: package, mainchannel, marketing source, ANY COLUMN)
# -------------------------------------------------


# Build variants like:
#   "MarketingSource" -> ["marketing source", "marketingsource"]
#   "MainChannel"     -> ["main channel", "mainchannel"]
# and map them back to the real column name in COLS.
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

# Optional aliases (small set; you can add more later)
_ALIAS_MAP = {
    "package": ["package", "package name", "packagename", "pkg", "pkgname", "order package", "product package"],
    "mainchannel": ["mainchannel", "main channel", "channel"],
    "marketingsource": ["marketing source", "marketingsource", "marketing", "source"],
    "providername": ["provider", "provider name", "providername"],
    "productname": ["product", "product name", "productname"],
}

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

    # 3) alias expansion
    for k, vals in _ALIAS_MAP.items():
        if nk == _norm_key(k) or raw == k:
            for v in vals:
                col = resolve_col_name(v)  # recurse through variants
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
    "profit|revenue|rate|orders|order|installation|install|disconnection|disconnect|"
    "estimate|estimated|forecast|predict|predicted|projection|projected"
)
# Words/phrases that describe metrics, NOT column filters
_METRIC_PHRASES = (
    "profit rate",
    "installation rate",
    "disconnection rate",
    "disconnect rate",
    "installed rate",
    "install rate",
    # generic "rate" causes most false-positives (profit rate -> Profit + 'rate')
    "rate",
)


def extract_provider(question: str):
    q = (question or "").strip()
    # allow "provider spectrum" or "provider is spectrum" or "provider = spectrum"
    m = re.search(
        r"\bprovider\b\s*(?:is|=|:)?\s*([A-Za-z0-9&._\-/ ]{2,80})",
        q,
        flags=re.IGNORECASE
    )
    if not m:
        return None
    val = (m.group(1) or "").strip()

    # cut at stopwords
    val = re.split(rf"\b(?:{_ENTITY_STOPWORDS})\b", val, flags=re.IGNORECASE)[0].strip()

    # keep only first chunk before punctuation that often continues sentence
    val = re.split(r"[,.?;!]", val)[0].strip()

    return val if val else None

def extract_product(question: str):
    q = (question or "").strip()
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

    return val if val else None

def _safe_sql_literal(s: str) -> str:
    return "'" + (s or "").replace("'", "''").strip() + "'"

def _wants_estimate_for_metric(q_lower: str, metric_name: str) -> bool:
    """
    True ONLY if estimate/forecast word is NEAR the metric
    (e.g. 'estimated installation rate', 'forecast profit')
    """
    est_words = r"(estimate|estimated|forecast|predict|predicted|projection|projected)"
    m = re.escape(metric_name)

    patterns = [
        # estimated <metric>
        rf"\b{est_words}\b\s+(?:\w+\s+){{0,2}}{m}\b",
        # <metric> estimated
        rf"\b{m}\b\s+(?:\w+\s+){{0,2}}{est_words}\b",
    ]
    return any(re.search(p, q_lower) for p in patterns)

def _wants_any_estimates(q_lower: str) -> bool:
    return any(w in q_lower for w in _PRED_WORDS)

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
        if not v or v not in ql:
            continue
        # allow extra spaces in the question for multi-word phrases
        v_re = re.escape(v)
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
            val_l = (" " + val.lower().strip() + " ")
            if any(f" {mp} " in val_l for mp in _METRIC_PHRASES):
                continue
            out.append((col, val))
            used_spans.append((a, b))

    # B) no-operator form: "for the <col phrase> <value>" OR "<col phrase> <value>"
    # We only do this for variants that are at least 3 chars to reduce false hits.
    for v in variants:
        if not v or len(v) < 3 or v not in ql:
            continue
        v_re = re.escape(v)
        pat = re.compile(rf"\b(?:for\s+the\s+|for\s+)?{v_re}\b\s+([A-Za-z0-9&._\-#/ ]{{2,80}})", flags=re.IGNORECASE)
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
            val_l = (" " + val.lower().strip() + " ")
            if any(f" {mp} " in val_l for mp in _METRIC_PHRASES):
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

    # dedupe while preserving order
    try:
        out = list(dict.fromkeys(out))
    except Exception:
        pass
    return out
def build_prediction_monthly_sql(question: str):
    """
    Stable month-wise SQL for prediction mode.
    Uses your real columns:
      - SaleDate for month
      - ConfirmedOrder, InstalledOrder, DisconnectDate
      - NetAfterChargeback (revenue), Profit
    Also supports generic filters like:
      package xyz, mainchannel digital, marketing source google, companystate ca, etc.
    """
    year = extract_year(question)
    month_year = extract_month_year(question)  # (y,m) or None

    provider = extract_provider(question)
    product = extract_product(question)

    # NEW: generic filters for ANY column names (including multi-word column phrases)
    generic_filters = []
    try:
        generic_filters = _extract_generic_filters_anycol(question)
    except Exception:
        generic_filters = []

    # date range
    if month_year:
        y, m = month_year
        start_m, end_m = _month_start_end(y, m)
        train_start = start_m - dtmod.timedelta(days=365)
        date_start = train_start
        date_end = end_m
        target_filter = ("month", y, m)
    else:
        if not year:
            year = dtmod.date.today().year
        date_start = dtmod.date(year - 1, 1, 1)
        date_end = dtmod.date(year + 1, 1, 1)
        target_filter = ("year", year)

    where = []
    where.append(f"CAST(SaleDate AS DATE) >= DATE '{date_start.isoformat()}'")
    where.append(f"CAST(SaleDate AS DATE) <  DATE '{date_end.isoformat()}'")

    # ✅ Correct: In prediction mode we are building SQL ourselves, so we are NOT "already having"
    # addon/ltype in SQL. Let policy_filter_sql apply the rules from the question.
    pol = policy_filter_sql(question, already_has_ltype=False, already_has_addon=False)
    if pol:
        where.append(pol)

    if provider:
        where.append(
            f"LOWER(TRIM(CAST(ProviderName AS VARCHAR))) LIKE '%' || LOWER(TRIM({_safe_sql_literal(provider)})) || '%'"
        )

    if product:
        where.append(
            f"LOWER(TRIM(CAST(ProductName AS VARCHAR))) LIKE '%' || LOWER(TRIM({_safe_sql_literal(product)})) || '%'"
        )

    for col, val in generic_filters:
        # don't double-apply provider/product if user typed them as generic filters too
        if col and col.lower() in ("providername", "productname"):
            continue
        if not col or col not in COLS:
            continue
        where.append(
            f"LOWER(TRIM(CAST({qident(col)} AS VARCHAR))) LIKE '%' || LOWER(TRIM({_safe_sql_literal(val)})) || '%'"
        )

    where_sql = " AND ".join(where) if where else "1=1"

    sql = f"""
SELECT
  STRFTIME('%Y-%m', CAST(SaleDate AS DATE)) AS month,

  SUM(CASE WHEN COALESCE(ConfirmedOrder,0)=1 THEN 1 ELSE 0 END) AS total_orders,

  SUM(CASE WHEN COALESCE(InstalledOrder,0)=1 THEN 1 ELSE 0 END) AS installation_orders,
  ROUND(
    SUM(CASE WHEN COALESCE(InstalledOrder,0)=1 THEN 1 ELSE 0 END) * 100.0 /
    NULLIF(SUM(CASE WHEN COALESCE(ConfirmedOrder,0)=1 THEN 1 ELSE 0 END), 0),
    2
  ) AS installation_rate,

  SUM(CASE WHEN TRY_CAST(DisconnectDate AS TIMESTAMP) IS NOT NULL THEN 1 ELSE 0 END) AS disconnection_orders,
  ROUND(
    SUM(CASE WHEN TRY_CAST(DisconnectDate AS TIMESTAMP) IS NOT NULL THEN 1 ELSE 0 END) * 100.0 /
    NULLIF(SUM(CASE WHEN COALESCE(ConfirmedOrder,0)=1 THEN 1 ELSE 0 END), 0),
    2
  ) AS disconnection_rate,

  SUM(COALESCE(NetAfterChargeback,0)) AS revenue,
  SUM(COALESCE(Profit,0)) AS profit,

  ROUND(
    SUM(COALESCE(Profit,0)) * 100.0 /
    NULLIF(SUM(COALESCE(NetAfterChargeback,0)), 0),
    2
  ) AS profit_rate

FROM {TABLE}
WHERE {where_sql}
GROUP BY 1
ORDER BY 1
""".strip()

    return sql, target_filter


def _month_to_index(ym: str) -> int:
    try:
        y, m = ym.split("-")
        return int(y) * 12 + (int(m) - 1)
    except Exception:
        return None

def add_estimated_columns(df: pd.DataFrame, cols_to_estimate, window=12):
    """
    Adds estimated_<col> using rolling Linear Regression.
    Predicts each month using ONLY previous months.
    """
    if df is None or df.empty or "month" not in df.columns:
        return df

    df = df.copy()
    df["__t"] = df["month"].astype(str).apply(_month_to_index)
    df = df.sort_values("__t").reset_index(drop=True)

    for col in cols_to_estimate:
        est_col = f"estimated_{col}"
        df[est_col] = np.nan
        if col not in df.columns:
            continue

        y_all = pd.to_numeric(df[col], errors="coerce").values
        t_all = df["__t"].values

        for i in range(len(df)):
            start_i = max(0, i - window)
            X_hist = t_all[start_i:i]
            y_hist = y_all[start_i:i]

            mask = np.isfinite(X_hist) & np.isfinite(y_hist)
            X_hist = X_hist[mask]
            y_hist = y_hist[mask]

            if len(X_hist) < 3:
                continue

            model = LinearRegression()
            model.fit(X_hist.reshape(-1, 1), y_hist)

            t_pred = t_all[i]
            if not np.isfinite(t_pred):
                continue

            pred = model.predict(np.array([[t_pred]]))[0]
            df.loc[i, est_col] = float(pred)

        df[est_col] = pd.to_numeric(df[est_col], errors="coerce").round(2)

    df.drop(columns=["__t"], inplace=True, errors="ignore")
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
    """
    Ask the LLM to decide:
    - if the question is about the dataset → return one DuckDB SELECT.
    - if not → return NO_SQL.
    """
    try:
        schema_info = f"Columns: {', '.join(schema_cols)}"
        sample_info = sample_rows.head(10).to_csv(index=False) if not sample_rows.empty else "No sample data"

        date_col = get_date_col()
        rev_col = get_revenue_col()
        profit_col = get_profit_col()

        # extra semantic hints
        semantic_lines = []
        if CUST_FIRST_COL and CUST_LAST_COL:
            semantic_lines.append(
                f"- CustomerFirstName + CustomerLastName together form the full customer name.\n"
                f"  When the user says 'customer name' or 'customer names' or 'customer' , use "
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
- Date column: {date_col}
- Revenue column (money received): {rev_col}
- Profit column (net profit): {profit_col}

Additional semantic hints:
{extra_semantics or '- (none)'}

Technical rules:
- When you work with dates, ALWAYS cast the date column to DATE first:
  CAST({date_col} AS DATE)
  (Assume {date_col} is stored as text and must be cast.)
- For year-month buckets:
  STRFTIME('%Y-%m', CAST({date_col} AS DATE))
- For year-only buckets:
  STRFTIME('%Y', CAST({date_col} AS DATE))
- When filtering on dates (last month, a specific month/year, etc.), always compare using CAST({date_col} AS DATE), e.g.:
  WHERE CAST({date_col} AS DATE) >= DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '1 month'
    AND CAST({date_col} AS DATE) <  DATE_TRUNC('month', CURRENT_DATE)
  or for October 2025:
  WHERE CAST({date_col} AS DATE) >= DATE '2025-10-01'
    AND CAST({date_col} AS DATE) <  DATE '2025-11-01'
- Do NOT invent aliases like date_col_expr unless you define them in the SELECT/FROM.
  If you want to reuse the expression, just repeat CAST({date_col} AS DATE) wherever needed.
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

- IMPORTANT Addon/Ltype policy:
  1) Default (no addon mentioned at all):
     ALWAYS filter rows where (Ltype = 'L' OR Ltype IS NULL).
  2) If user mentions "addon" (addon/add-on/add on) WITHOUT specifying a value:
     treat it as addon-mode and filter rows where Ltype = 'A'.
  3) If user explicitly says addon = 1:
     filter rows where Addon = 1
     and DO NOT force any Ltype (unless the user explicitly asks Ltype).
  4) If user explicitly says addon = 0:
     filter rows where Addon = 0
     and behave like default for Ltype again (Ltype='L' OR NULL), unless the user explicitly asks Ltype.
  5) If user explicitly asks Ltype (e.g. Ltype = 'A') then respect that and do not override it.

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

def ask_model_fix_sql(bad_sql: str, error: str, schema_cols: list, sample_rows: pd.DataFrame) -> str:
    """Ask LLM to fix problematic SQL."""
    try:
        date_col = get_date_col()

        if date_col:
            date_fix_rules = f"""
- If you compare the date column {date_col} to CURRENT_DATE or use DATE_TRUNC,
  ALWAYS wrap it as TRY_CAST({date_col} AS DATE).
  Example:
    WHERE TRY_CAST({date_col} AS DATE) >= DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '1 month'
"""
        else:
            date_fix_rules = """
- Do NOT introduce date filters unless the user explicitly requires them and a real date
  column name from the schema is used.
"""

        prompt = rf"""
The following DuckDB SQL query failed with error: {error}

Broken SQL:
{bad_sql}

The table name is {TABLE}.
Usable columns: {', '.join(schema_cols)}

Fix the SQL so it works in DuckDB, while respecting these rules:
- Only a single SELECT statement.
- No INSERT/UPDATE/DELETE/DDL.
- Use only the listed columns.
- NEVER invent or use fake column names like "date_col_expr".
- Do not change the user's intent.

Additional rules for dates:
{date_fix_rules}

If you need to compare a date-like column (e.g. the main date column), you must:
- Wrap it with TRY_CAST(column AS DATE) before comparing to CURRENT_DATE or using DATE_TRUNC.

Return ONLY the fixed SQL or NO_SQL. No explanations, no markdown.
"""
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=600,
        )

        fixed_sql = (response.choices[0].message.content or "").strip()
        fixed_sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", fixed_sql, flags=re.IGNORECASE).strip()

        if fixed_sql.upper() == "NO_SQL":
            return "NO_SQL"
        if not is_safe_select(fixed_sql):
            return "NO_SQL"
        return fixed_sql

    except Exception as e:
        print(f"Error fixing SQL: {e}", traceback.format_exc())
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

def handle_general_data_question(question: str, history=None):
    """
    Single generic data handler:
    - Ask LLM for SQL or NO_SQL.
    - If SQL → run it and summarize.
    - If NO_SQL → answer purely in text.
    """
    schema_cols = COLS
    qlow = (question or "").lower()

    # -------------------------------------------------
    # PREDICTION MODE (bypass LLM SQL and do stable month-wise SQL + ML)
    # -------------------------------------------------
    prediction_mode = is_prediction_request(question)
    if prediction_mode:
        try:
            pred_sql, target_filter = build_prediction_monthly_sql(question)
            df = run_sql_and_fetch_df(pred_sql)

            ql = (question or "").lower()
            cols_to_estimate = []

            # Map metric display names to df column names
            # IMPORTANT: keep "profit rate" BEFORE "profit" to avoid wrong matching
            metric_map = {
                "total orders": "total_orders",
                "total order": "total_orders",

                "installation orders": "installation_orders",
                "installation order": "installation_orders",
                "installed orders": "installation_orders",
                "installed order": "installation_orders",

                "installation rate": "installation_rate",
                "install rate": "installation_rate",

                "disconnection orders": "disconnection_orders",
                "disconnection order": "disconnection_orders",
                "disconnected orders": "disconnection_orders",
                "disconnected order": "disconnection_orders",

                "disconnection rate": "disconnection_rate",
                "disconnect rate": "disconnection_rate",
                "disconnected rate": "disconnection_rate",

                "profit rate": "profit_rate",
                "profit": "profit",
                "revenue": "revenue",
            }

            # Only estimate metrics the user EXPLICITLY requested as estimated/forecast/predicted
            for phrase, col in metric_map.items():
                if _wants_estimate_for_metric(ql, phrase):
                    cols_to_estimate.append(col)

            # dedupe cols_to_estimate (prevents repeated trend lines)
            cols_to_estimate = list(dict.fromkeys(cols_to_estimate))

            # If user asked for estimates generally but didn't specify metric: conservative fallback
            if not cols_to_estimate and _wants_any_estimates(ql):
                if "installation rate" in ql or "install rate" in ql:
                    cols_to_estimate = ["installation_rate"]
                elif "disconnection rate" in ql or "disconnect rate" in ql or "disconnected rate" in ql:
                    cols_to_estimate = ["disconnection_rate"]
                elif "profit rate" in ql:
                    cols_to_estimate = ["profit_rate"]
                elif "profit" in ql:
                    cols_to_estimate = ["profit"]
                elif "order" in ql or "orders" in ql:
                    cols_to_estimate = ["total_orders"]
                else:
                    cols_to_estimate = ["installation_rate"]

            window = 12
            df2 = add_estimated_columns(df, cols_to_estimate, window=window)

            # Filter output to target (month or year)
            df_out = df2.copy()
            if isinstance(target_filter, tuple) and target_filter[0] == "month":
                _, y, m = target_filter
                key = f"{y:04d}-{m:02d}"
                df_out = df_out[df_out["month"].astype(str) == key]
            elif isinstance(target_filter, tuple) and target_filter[0] == "year":
                _, y = target_filter
                df_out = df_out[df_out["month"].astype(str).str.startswith(f"{y:04d}-")]

            # --------- OUTPUT COLUMN SELECTION (FIXED: no extra columns) ----------
            wants_total_orders = ("total order" in ql) or ("total orders" in ql) or bool(re.search(r"\btotal\s+orders?\b", ql))

            wants_install_orders = any(p in ql for p in ["installation order", "installation orders", "installed order", "installed orders"])
            wants_install_rate = any(p in ql for p in ["installation rate", "install rate"])

            wants_disc_orders = any(p in ql for p in ["disconnection order", "disconnection orders", "disconnected order", "disconnected orders"])
            wants_disc_rate = any(p in ql for p in ["disconnection rate", "disconnect rate", "disconnected rate"])

            wants_revenue = "revenue" in ql
            wants_profit = ("profit" in ql)  # includes profit rate too
            wants_profit_rate = ("profit rate" in ql)

            base_cols = ["month"]

            # total orders (only if asked OR if its estimate is requested)
            if wants_total_orders or ("total_orders" in cols_to_estimate):
                base_cols += ["total_orders"]
                if "total_orders" in cols_to_estimate:
                    base_cols += ["estimated_total_orders"]

            # installation orders (only if asked OR estimate requested)
            if wants_install_orders or ("installation_orders" in cols_to_estimate):
                base_cols += ["installation_orders"]
                if "installation_orders" in cols_to_estimate:
                    base_cols += ["estimated_installation_orders"]

            # installation rate (only if asked OR estimate requested)
            if wants_install_rate or ("installation_rate" in cols_to_estimate):
                base_cols += ["installation_rate"]
                if "installation_rate" in cols_to_estimate:
                    base_cols += ["estimated_installation_rate"]

            # disconnection orders (only if asked OR estimate requested)
            if wants_disc_orders or ("disconnection_orders" in cols_to_estimate):
                base_cols += ["disconnection_orders"]
                if "disconnection_orders" in cols_to_estimate:
                    base_cols += ["estimated_disconnection_orders"]

            # disconnection rate (only if asked OR estimate requested)
            if wants_disc_rate or ("disconnection_rate" in cols_to_estimate):
                base_cols += ["disconnection_rate"]
                if "disconnection_rate" in cols_to_estimate:
                    base_cols += ["estimated_disconnection_rate"]

            # revenue (only if asked OR estimate requested)
            if wants_revenue or ("revenue" in cols_to_estimate):
                base_cols += ["revenue"]
                if "revenue" in cols_to_estimate:
                    base_cols += ["estimated_revenue"]

            # profit (only if asked OR estimate requested)
            if (wants_profit and not wants_profit_rate) or ("profit" in cols_to_estimate):
                # if user only asked profit rate, don't force profit unless estimated profit requested
                base_cols += ["profit"]
                if "profit" in cols_to_estimate:
                    base_cols += ["estimated_profit"]

            # profit rate (only if asked OR estimate requested)
            if wants_profit_rate or ("profit_rate" in cols_to_estimate):
                base_cols += ["profit_rate"]
                if "profit_rate" in cols_to_estimate:
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
            reply_text = build_prediction_reply(df_out, cols_to_estimate, window=window)

            result = {
                "reply": reply_text,
                "table_html": table_html,
                "plot_data_uri": None,
                "rows_returned": int(len(df_out)),
                "truncated": False,
                "download_url": None,
            }

            # IMPORTANT: return actual SQL used in prediction mode (so you don't get fake LLM SQL)
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
            content = (m.get("content") or "").strip()
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
        # For key lookup, also respect explicit Ltype/addon constraints by applying policy_filter_sql
        # If user explicitly wrote Ltype in the question, we do not override it (so skip Ltype injection here)
        ql2 = (question or "").lower()
        key_policy = policy_filter_sql(
            question,
            already_has_ltype=("ltype" in ql2),
            already_has_addon=("addon" in ql2),
        )
        key_policy_sql = f"WHERE {key_policy}\n          AND " if key_policy else "WHERE "

        sql_text = f"""
        SELECT *
        FROM {TABLE}
        {key_policy_sql}LOWER(TRIM(CAST({qident(key_col)} AS VARCHAR))) = LOWER(TRIM({_safe_sql_literal(key_val)}))
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

    # 1) Ask model for SQL  (conversation-aware)
    sql_text = ask_model_for_sql(augmented_question, schema_cols, sample_rows)
    sql_text = apply_ltype_policy_to_sql(sql_text, question)

    # 2) If NO_SQL → theory/general mode (conversation-aware)
    if not sql_text or sql_text.strip().upper() == "NO_SQL":
        return answer_theory_question(augmented_question)

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
                    sql_text = apply_ltype_policy_to_sql(sql_text, question)
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

            # -------------------------
            # USER-FORCED CHART
            # -------------------------
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

            elif kind == "pie" and cat_cols and num_cols:
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

            # -------------------------
            # AUTO CHART (if no force)
            # -------------------------
            if plot_uri is None:
                # 1 numeric → histogram
                if len(num_cols) == 1:
                    plot_uri = plot_to_base64([], df[num_cols[0]].tolist(), kind="hist", title=question)

                # 2 numeric → scatter
                elif len(num_cols) == 2 and not cat_cols:
                    plot_uri = plot_to_base64(
                        df[num_cols[0]].tolist(),
                        df[num_cols[1]].tolist(),
                        kind="scatter",
                        title=question,
                    )

                # category + 2+ numeric → stacked bar
                elif cat_cols and len(num_cols) >= 2:
                    labels = df[cat_cols[0]].astype(str).tolist()
                    series = [df[c].fillna(0).tolist() for c in num_cols]
                    plot_uri = plot_to_base64(labels, series, kind="stacked_bar", title=question)

                # category + 1 numeric (small) → pie
                elif cat_cols and len(num_cols) == 1 and len(df) <= 12:
                    plot_uri = plot_to_base64(
                        df[cat_cols[0]].astype(str).tolist(),
                        df[num_cols[0]].fillna(0).tolist(),
                        kind="pie",
                        title=question,
                    )

                # month/time → line
                elif "month" in df.columns and num_cols:
                    plot_uri = plot_to_base64(
                        df["month"].astype(str).tolist(),
                        df[num_cols[0]].fillna(0).tolist(),
                        kind="line",
                        title=question,
                    )

                # fallback → bar
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
        EXPORT_CACHE[file_id] = df_preview
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
