# server.py — updated (keeps all original logic; adds safe column detection)
import os
import re
import json
import traceback
from html import escape
from dotenv import load_dotenv
from openai import OpenAI
from flask import Flask, request, jsonify, send_from_directory
import pyodbc  # ✅ changed from duckdb to pyodbc
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
import calendar
import time
from utils import rows_to_html_table, plot_to_base64, embed_text
import uuid
from pathlib import Path

# NEW: sqlalchemy for pooling & safe per-request connections
from sqlalchemy import create_engine, text

# NEW: sqlite3 for memory DB
import sqlite3

MEMORY_FILE = Path('./data/conversations.json')
MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
print("PWD:", os.getcwd())

# -------------------------
# Relative time & phrase parsing helpers (unchanged logic)
# -------------------------
TZ = ZoneInfo("Asia/Kolkata")  # user timezone

def _start_of_month(dt: datetime) -> date:
    return date(dt.year, dt.month, 1)

def _end_of_month(dt: datetime) -> date:
    last_day = calendar.monthrange(dt.year, dt.month)[1]
    return date(dt.year, dt.month, last_day)

def month_n_ago(today_dt: datetime, n: int):
    y = today_dt.year
    m = today_dt.month - n
    while m <= 0:
        m += 12
        y -= 1
    return y, m
def parse_relative_time_phrase(question: str):
    q = (question or "").lower()
    now = datetime.now(TZ)

    # Be strict about "current time" detection so "installation time" or "average time" won't match.
    # Match explicit clock/time-of-day requests only.
    if re.search(r"\b(what(?:'s| is)? the time|what time is it|current time|time now|tell me the time|time\?)\b", q):
        return {"type": "now", "datetime": now}
    if re.search(r"\b(today|what(?:'s| is)? the date|current date|date today|todays date|what is the date today)\b", q):
        return {"type": "day", "date": now.date().isoformat(), "datetime": now}

    # days ago / yesterday
    m = re.search(r'\b(\d+)\s+days?\s+ago\b', q)
    if m:
        n = int(m.group(1))
        target = (now - timedelta(days=n)).date()
        return {"type":"day", "date": target.isoformat()}
    if re.search(r'\byesterday\b', q):
        target = (now - timedelta(days=1)).date()
        return {"type":"day", "date": target.isoformat()}

    # months ago / last month / specific month
    m = re.search(r'\b(\d+)\s+months?\s+ago\b', q)
    if m:
        n = int(m.group(1))
        y, mth = month_n_ago(now, n)
        return {"type":"month", "month": f"{y:04d}-{mth:02d}"}
    if re.search(r'\b(previous|last)\s+month\b', q):
        y, mth = month_n_ago(now, 1)
        return {"type":"month", "month": f"{y:04d}-{mth:02d}"}
    m = re.search(r'\b(\d+)(?:st|nd|rd|th)?\s+previous\s+month\b', q)
    if m:
        n = int(m.group(1))
        y, mth = month_n_ago(now, n)
        return {"type":"month", "month": f"{y:04d}-{mth:02d}"}
    m = re.search(r'\b(\d+)(?:st|nd|rd|th)?\s+previous\b', q)
    if m:
        n = int(m.group(1))
        y, mth = month_n_ago(now, n)
        return {"type":"month", "month": f"{y:04d}-{mth:02d}"}
    if re.search(r'\b(this|current)\s+month\b', q):
        y, mth = month_n_ago(now, 0)
        return {"type":"month", "month": f"{y:04d}-{mth:02d}"}
    m = re.search(r'\blast\s+(\d+)\s+months\b', q)
    if m:
        n = int(m.group(1))
        start_y, start_m = month_n_ago(now, n)
        start = _start_of_month(datetime(start_y, start_m, 1))
        end_y, end_m = month_n_ago(now, 1)
        end = _end_of_month(datetime(end_y, end_m, 1))
        return {"type":"range", "start": start.isoformat(), "end": end.isoformat()}
    m = re.search(r'\bfor\s+([a-zA-Z]+)(?:\s+(\d{4}))?\b', q)
    if m:
        month_name = m.group(1)
        year = m.group(2)
        try:
            month_num = list(calendar.month_name).index(month_name.capitalize())
            if month_num == 0:
                month_num = list(calendar.month_abbr).index(month_name.capitalize())
            if month_num != 0:
                y = int(year) if year else now.year
                return {"type":"month", "month": f"{y:04d}-{month_num:02d}"}
        except Exception:
            pass
    return None


def get_existing_date_column():
    """
    Detect which date column actually exists in the SQL table.
    Uses _columns_from_cursor() to avoid indexing the wrong field.
    """
    date_candidates = ['sale_date_parsed', 'SaleDate', 'OrderDate', 'Date']
    existing_cols = _columns_from_cursor() or list(COLS)  # fallback to COLS if helper returns empty
    # normalize names to exact-case from COLS if possible
    existing_lower = {c.lower(): c for c in existing_cols}
    for cand in date_candidates:
        if cand.lower() in existing_lower:
            return existing_lower[cand.lower()]
    # fallback: first column that contains 'date' in name
    for c in existing_cols:
        if 'date' in c.lower():
            return c
    return existing_cols[0] if existing_cols else (DETECTED.get('date') or 'SaleDate')

def handle_relative_time(question: str, sale_date_col="sale_date_parsed"):
    parsed = parse_relative_time_phrase(question)
    if not parsed:
        return None
    if parsed.get("type") == "now":
        dt = parsed.get("datetime")
        return {"action":"now", "text": dt.strftime("%Y-%m-%d %H:%M:%S %Z")}
    if parsed.get("type") == "day":
        d = parsed.get("date")
        cond = f"CAST({sale_date_col} AS DATE) = '{d}'"
        return {"action":"filter", "filter_sql": cond, "meta": parsed}
    if parsed.get("type") == "month":
        mon = parsed.get("month")
        cond = f"FORMAT({sale_date_col}, 'yyyy-MM') = '{mon}'"
        return {"action":"filter", "filter_sql": cond, "meta": parsed}
    if parsed.get("type") == "range":
        s = parsed.get("start")
        e = parsed.get("end")
        cond = f"CAST({sale_date_col} AS DATE) BETWEEN '{s}' AND '{e}'"
        return {"action":"filter", "filter_sql": cond, "meta": parsed}
    return None

# -------------------------
# Helper: ask_model_for_sql
# -------------------------
def ask_model_for_sql(question: str, schema_cols, sample_rows: pd.DataFrame):
    """
    Uses OpenAI to convert a natural-language question into a SQL Server SELECT query.
    """
    schema_text = ", ".join(schema_cols)
    sample_text = sample_rows.head(5).to_csv(index=False) if not sample_rows.empty else "no sample rows"
    system = (
        "You are an expert SQL generator for Microsoft SQL Server. "
        "The main table is named OrderItemUnit (or dataset_clean if present). "
        "Return ONLY one SELECT query that answers the user's question. "
        "If you cannot express it as SQL, return exactly NO_SQL. "
        "Use standard SQL Server syntax — for example, use FORMAT(SaleDate, 'yyyy-MM') for months, "
        "and TOP 10 instead of LIMIT 10."
    )
    user = f"Columns: {schema_text}\nSample (CSV):\n{sample_text}\n\nQuestion: {question}\n\nReturn only SQL or NO_SQL."

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=0.0,
        max_tokens=512
    )

    return resp.choices[0].message.content.strip()

# -------------------------
# SQL safety / normalization + LLM helpers (needed by /chat)
# -------------------------
def is_safe_select(sql: str) -> bool:
    """
    Allow only SELECT/ WITH ... SELECT statements (no DDL/DML).
    Strips ``` fences and trailing semicolons, then checks for forbidden keywords.
    """
    if not sql or not isinstance(sql, str):
        return False
    s = sql.strip()
    s = re.sub(r"^```(?:sql)?\s*|\s*```$", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r";\s*$", "", s)
    # disallow multiple statements separated by semicolon
    if ";" in s:
        return False
    s_low = s.lower()
    # require SELECT or WITH
    if not (s_low.startswith("select") or s_low.startswith("with")):
        return False
    forbidden = [
        r"\binsert\b", r"\bupdate\b", r"\bdelete\b", r"\bdrop\b", r"\bcreate\b",
        r"\balter\b", r"\battach\b", r"\bmerge\b", r"\bcopy\b", r"\bvacuum\b",
        r"\bpragma\b", r"\bexport\b", r"\bimport\b", r"\bcall\b", r"\bexecute\b",
        r"\bgrant\b", r"\brevoke\b"
    ]
    for kw in forbidden:
        if re.search(kw, s_low):
            return False
    return True

def normalize_sql_columns(sql: str) -> str:
    """
    Replace bare column tokens with exact-cased column names from COLS when possible.
    This helps keep generated SQL compatible with your DB column names.
    """
    if not sql or not isinstance(sql, str):
        return sql
    col_map = {c.lower(): c for c in COLS}
    def repl(m):
        token = m.group(0)
        return col_map.get(token.lower(), token)
    # only replace identifiers (letters, underscores, digits) to avoid touching functions
    return re.sub(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", repl, sql)

def ask_model_fix_sql(bad_sql: str, error_msg: str, schema_cols, sample_rows: 'pd.DataFrame'):
    """
    Ask the LLM to fix a SQL error. Returns corrected SQL or NO_SQL.
    """
    schema_text = ", ".join(schema_cols)
    sample_text = sample_rows.head(3).to_csv(index=False) if not sample_rows.empty else "no sample rows"
    system = "You are a SQL fixer for Microsoft SQL Server. Given erroneous SQL and its error message, return a corrected single SELECT or NO_SQL."
    user = f"Schema: {schema_text}\nSample:\n{sample_text}\nSQL:\n{bad_sql}\nError:\n{error_msg}\nReturn corrected SQL or NO_SQL."
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=0.0,
            max_tokens=512
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        # if the LLM call fails, fall back to no fix
        print("ask_model_fix_sql failed:", e)
        return None

def ask_model_explain(question: str, sql: str, preview_df: 'pd.DataFrame'):
    """
    Ask the LLM to give a concise explanation of SQL results.
    """
    system = "You are a concise data analyst. Provide a one-line summary and 1-2 sentences of detail grounded in the result preview."
    preview = preview_df.head(8).to_csv(index=False) if not preview_df.empty else "no rows"
    user = f"Question: {question}\nSQL:\n{sql}\nResult preview:\n{preview}\nAnswer concisely."
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=0.0,
            max_tokens=256
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print("ask_model_explain failed:", e)
        return "Here are the results."

# (optional) nearest-neighbor helper if you later use embeddings
def topk_similar(q_emb, k=6):
    if nbrs is None:
        return []
    dists, idxs = nbrs.kneighbors([q_emb], n_neighbors=k, return_distance=True)
    idxs = idxs[0]
    out = []
    for i in idxs:
        out.append(meta[int(i)])
    return out

# -------------------------
# Small helper: safe columns_from_cursor for pyodbc
# -------------------------
def _columns_from_cursor():
    """
    Return list of column names for TABLE using pyodbc cursor.columns(). 
    Handles different pyodbc row shapes safely.
    """
    cols = []
    try:
        # use a fresh cursor for this call to avoid interfering with other cursors
        c = get_cursor()
        for r in c.columns(table=TABLE):
            # prefer attribute, otherwise fallback to index 3
            if hasattr(r, "column_name"):
                cols.append(r.column_name)
            else:
                try:
                    # many pyodbc implementations put column name at index 3
                    cols.append(r[3])
                except Exception:
                    # ultimate fallback: try all fields and pick a string that looks like a column
                    for fld in r:
                        if isinstance(fld, str) and fld and fld.lower() not in (TABLE.lower(),):
                            cols.append(fld)
                            break
        try:
            c.close()
        except Exception:
            pass
    except Exception:
        # on any error, return empty list and caller will fallback to previously fetched COLS
        return []
    return cols

def _get_existing_columns_set():
    cols = _columns_from_cursor()
    if cols:
        return set(cols)
    return set(COLS)

# -------------------------
# New helper: rewrite and run SQL (ensures date tokens converted)
# -------------------------
def _detect_date_col_once():
    """Detect date column used for filters and months; store in global DATE_COL."""
    global DATE_COL
    try:
        DATE_COL = get_existing_date_column()
    except Exception:
        # if get_existing_date_column not usable, fallback to first date-like in COLS
        DATE_COL = next((c for c in COLS if 'date' in c.lower()), (COLS[0] if COLS else 'SaleDate'))
    return DATE_COL

def rewrite_date_tokens_for_sqlserver(sql_text: str, date_col: str = None) -> str:
    """
    - Replace bare sale_date_parsed tokens with actual date column.
    - Convert DuckDB STRFTIME(sale_date_parsed, '%Y-%m') -> FORMAT(<date_col>, 'yyyy-MM')
    - Also handle STRFTIME(SaleDate, '%Y-%m') -> FORMAT(SaleDate, 'yyyy-MM') and similar.
    """
    if not sql_text or not isinstance(sql_text, str):
        return sql_text
    s = sql_text

    # choose date column
    if date_col is None:
        date_col = globals().get('DATE_COL') or _detect_date_col_once()

    # 1) replace STRFTIME(sale_date_parsed, '%Y-%m') and variants -> FORMAT(date_col, 'yyyy-MM')
    s = re.sub(
        r"STRFTIME\(\s*sale_date_parsed\s*,\s*['\"]%Y-%m['\"]\s*\)",
        f"FORMAT({date_col}, 'yyyy-MM')",
        s, flags=re.IGNORECASE
    )
    # generic STRFTIME(sale_date_parsed, '...') fallback -> FORMAT(date_col, 'yyyy-MM')
    s = re.sub(
        r"STRFTIME\(\s*sale_date_parsed\s*,\s*['\"].+?['\"]\s*\)",
        f"FORMAT({date_col}, 'yyyy-MM')",
        s, flags=re.IGNORECASE
    )

    # 2) replace STRFTIME(SaleDate, '%Y-%m') -> FORMAT(SaleDate, 'yyyy-MM')
    s = re.sub(
        r"STRFTIME\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*,\s*['\"]%Y-%m['\"]\s*\)",
        lambda m: f"FORMAT({m.group(1)}, 'yyyy-MM')",
        s, flags=re.IGNORECASE
    )

    # 3) Replace bare sale_date_parsed token (case-insensitive) with date_col
    s = re.sub(r"\bsale_date_parsed\b", date_col, s, flags=re.IGNORECASE)

    # 4) Replace CamelCase variants like SaleDate_parsed etc.
    s = re.sub(r"\bsale_date_parsed\b", date_col, s, flags=re.IGNORECASE)

    # 5) If code erroneously used STRFTIME(..., '%Y-%m') with other tokens, leave as-is or convert above
    return s

def run_sql_and_fetch_df(sql: str, con_obj=None):
    """
    Centralized helper to rewrite sql, print it, and execute with pandas.
    Returns DataFrame or raises the same exceptions pandas would.
    """
    if con_obj is None:
        con_obj = con
    sql_rewritten = rewrite_date_tokens_for_sqlserver(sql)
    # debug print so you can copy exact SQL that was executed
    print("Executing SQL →\n", sql_rewritten)
    # use pandas to fetch (consistent with previous code)
    # Note: pandas accepts SQLAlchemy engine or raw DBAPI connection. We keep using the engine (con) when possible.
    return pd.read_sql(sql_rewritten, con_obj)

def run_sql_no_return(sql: str, con_obj=None):
    """
    Execute a SQL that doesn't return a dataframe (e.g., DDL) via cursor after rewrite.
    """
    if con_obj is None:
        con_obj = con
    sql_rewritten = rewrite_date_tokens_for_sqlserver(sql)
    print("Executing SQL (no-return) →\n", sql_rewritten)
    # Use a raw cursor for non-returning calls
    cur = get_cursor()
    try:
        res = cur.execute(sql_rewritten)
        # commit if possible (if con_obj is a DBAPI connection)
        try:
            # if we got a DBAPI connection from engine.raw_connection(), commit on that
            raw_conn = cur.connection
            raw_conn.commit()
        except Exception:
            pass
        return res
    finally:
        try:
            cur.close()
        except Exception:
            pass

# -------------------------
# Load environment, DB, OpenAI client
# -------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
PORT = int(os.getenv("PORT", "3000"))
MAX_ROWS = int(os.getenv("MAX_ROWS", "5000"))

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY must be set in environment")

client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------
# Connect to SQL Server (replaces DuckDB)
# -------------------------
print("Connecting to SQL Server...")

# Build SQLAlchemy engine URL (keeps your credentials and driver — adjust if necessary)
# Note: credentials are used inline here (same as your previous code). If you later prefer env vars, swap them.
username = "devswetha"
password = "xwnpZgdX"
host = "cableportalstage.westus2.cloudapp.azure.com"
database = "bundle_root"
# driver string must be URL-encoded for spaces; sqlalchemy handles it via + for spaces
conn_url = (
    f"mssql+pyodbc://{username}:{password}@{host}/{database}"
    "?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes&MARS_Connection=Yes"
)

try:
    # create engine with pooling to avoid "connection is busy" contention
    engine = create_engine(
        conn_url,
        pool_size=5,
        max_overflow=5,
        pool_pre_ping=True,
        connect_args={"timeout": 15}
    )
    # Expose `con` as the SQLAlchemy engine so pandas.read_sql(..., con) works as before
    con = engine
    print("✅ SQLAlchemy engine created (connection pooling enabled).")
except Exception as e:
    # Fallback: attempt a direct pyodbc connection (keeps compatibility)
    # but still set `con` to that connection (less ideal for concurrency)
    try:
        fallback_conn_str = (
            "DRIVER={ODBC Driver 18 for SQL Server};"
            f"SERVER={host};"
            f"DATABASE={database};"
            f"UID={username};"
            f"PWD={password};"
            "TrustServerCertificate=yes;MARS_Connection=Yes;"
        )
        py_conn = pyodbc.connect(fallback_conn_str)
        con = py_conn
        engine = None
        print("⚠️ SQLAlchemy engine creation failed — using raw pyodbc connection as fallback.", e)
    except Exception as e2:
        raise RuntimeError(f"❌ Failed to connect to SQL Server: {e} ; fallback also failed: {e2}")

# Helper functions to get raw DBAPI connections/cursors when code expects cursor()
def get_dbapi_conn():
    """
    Return a raw DBAPI connection (pyodbc connection) from engine.raw_connection()
    If engine is not available, return the existing pyodbc connection `con`.
    Caller MUST close the returned connection when done.
    """
    if 'engine' in globals() and engine is not None:
        raw_conn = engine.raw_connection()
        return raw_conn
    # fallback: `con` might itself be a DBAPI pyodbc connection
    try:
        return con
    except Exception:
        raise RuntimeError("No DBAPI connection available")

def get_cursor():
    """
    Return a DBAPI cursor. Caller should close cursor (and connection if created here) when done.
    We return a cursor object; if it came from a raw connection created here, caller should close the connection via cursor.connection.close()
    """
    raw_conn = get_dbapi_conn()
    # if raw_conn is a SQLAlchemy Engine (shouldn't be), raise — but we avoid that above
    cur = raw_conn.cursor()
    return cur

TABLE = "OrderItemUnit"  # use your clean table/view or actual table name

# fetch column names dynamically (use cursor.description fallback)
cursor = None
try:
    # use a raw cursor to fetch a single row's description (works with DBAPI cursor)
    cur = get_cursor()
    cur.execute(f"SELECT TOP 1 * FROM {TABLE}")
    # some cursors populate .description; handle robustly
    try:
        COLS = [column[0] for column in cur.description]
    except Exception:
        # fallback to pandas if description missing
        try:
            SAMPLE_DF = pd.read_sql(f"SELECT TOP 10 * FROM {TABLE}", con)
            COLS = list(SAMPLE_DF.columns)
        except Exception:
            COLS = []
    try:
        cur.close()
    except Exception:
        pass
except Exception:
    # fallback: try a pandas read
    try:
        SAMPLE_DF = pd.read_sql(f"SELECT TOP 10 * FROM {TABLE}", con)
        COLS = list(SAMPLE_DF.columns)
    except Exception:
        COLS = []

# try to get a sample df for LLM prompts
try:
    SAMPLE_DF = pd.read_sql(f"SELECT TOP 10 * FROM {TABLE}", con)
except Exception:
    SAMPLE_DF = pd.DataFrame()

try:
    # use raw cursor for count to avoid pandas overhead here
    cur = get_cursor()
    cur.execute(f"SELECT COUNT(*) FROM {TABLE}")
    try:
        total_count = cur.fetchone()[0]
    except Exception:
        total_count = None
    try:
        cur.close()
    except Exception:
        pass
except Exception:
    total_count = None

# debug prints to verify columns (remove if noisy)
print(f"Using table: {TABLE} rows={total_count}, cols={len(COLS)}")
print("Columns detected (COLS variable):", COLS)
# safe call to _columns_from_cursor() will use get_cursor internally
print("Columns from cursor.columns():", _columns_from_cursor())

# detect date column once
DATE_COL = _detect_date_col_once()
print("Using date column for filters:", DATE_COL)

def find_best(cols, candidates):
    cols_lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand and cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

REVENUE_CANDS = ['Revenue','ProviderPaid','NetAfterCb','GrossAfterCb','Amount','SaleAmount']
PROFIT_CANDS = ['Profit','NetProfit','ProfitAmount','Margin']
DATE_CANDS = ['sale_date_parsed','SaleDate','Date','OrderDate']
CHANNEL_CANDS = ['MainChannel','SalesChannel','Channel','ChannelName']

DETECTED = {
    'date': find_best(COLS, DATE_CANDS),
    'revenue': find_best(COLS, REVENUE_CANDS),
    'profit': find_best(COLS, PROFIT_CANDS),
    'channel': find_best(COLS, CHANNEL_CANDS)
}
print("Auto-detected columns:", DETECTED)

# -------------------------
# Flask app
# -------------------------
app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    payload = request.json or {}
    question = (payload.get('message') or "").strip()
    qlow = question.lower()

    # --- Column list handler ---
    if re.search(r'\b(show|list|display|give|what are)\b.*\b(columns|fields|attributes|headings)\b', qlow):
        try:
            cols = COLS
            cols_html = "<br>".join(f"{i+1}. <strong>{c}</strong>" for i, c in enumerate(cols))
            return jsonify({
                "reply": f"The dataset contains {len(cols)} columns.",
                "table_html": cols_html,
                "columns": cols
            })
        except Exception as e:
            return jsonify({"reply": f"Error fetching columns: {e}"}), 500

    # Handle relative time queries
    rel = handle_relative_time(question)
        # --- right after: rel = handle_relative_time(question) --

    # Ensure the relative-time filter uses a real date column in your SQL Server table.
    if rel and isinstance(rel, dict) and rel.get("action") == "filter":
        # detect a date column that actually exists (falling back to what's auto-detected)
        try:
            date_col = get_existing_date_column()  # helper already present in your file
        except Exception:
            date_col = DETECTED.get('date') or 'SaleDate'

        # replace the placeholder sale_date_parsed (if present) with the real column name
        rel_filter_sql = rel.get('filter_sql', '')
        # common mismatches: sale_date_parsed or SaleDate_parsed etc. replace any "sale_date_parsed" token
        rel_filter_sql = re.sub(r"\bsale_date_parsed\b", date_col, rel_filter_sql, flags=re.IGNORECASE)

        # update rel so the rest of your code uses the fixed filter expression
        rel['filter_sql'] = rel_filter_sql

    if rel:
        if rel["action"] == "now":
            return jsonify({"reply": f"Current date & time (Asia/Kolkata): {rel['text']}"})

        if rel["action"] == "filter":
            ql = question.lower()
            rev_col = DETECTED.get('revenue') or 'NetAfterCb'
            profit_col = DETECTED.get('profit') or 'Profit'
            date_col = get_existing_date_column()

            # revenue query
            if any(tok in ql for tok in ("revenue", "sales", "amount", "total")):
                date_col = get_existing_date_column()
                # only include rows where revenue-like column is present
                rev_check = f"({rev_col} IS NOT NULL AND {rev_col} <> '')" if rev_col else "1=1"
                sql = f"""
                    SELECT FORMAT({date_col}, 'yyyy-MM') AS month,
                        SUM(COALESCE(CAST({rev_col} AS FLOAT), 0)) AS revenue,
                        COUNT(*) AS orders
                    FROM {TABLE}
                    WHERE {rel['filter_sql']} AND {rev_check}
                    GROUP BY FORMAT({date_col}, 'yyyy-MM')
                    ORDER BY month;
                """
                try:
                    df = run_sql_and_fetch_df(sql, con)
                    # drop null/empty group keys (first column) for display/explanation
                    if not df.empty:
                        first_col = df.columns[0]
                        df = df[df[first_col].notna() & (df[first_col].astype(str).str.strip() != "")]
                    table_html = rows_to_html_table(df.to_dict(orient='records')) if not df.empty else "<p><i>No data</i></p>"
                    plot_uri = None
                    if not df.empty:
                        plot_uri = plot_to_base64(df['month'].astype(str).tolist(), df['revenue'].fillna(0).tolist(), kind='line', title=question)
                    explanation = ask_model_explain(question, sql, df.head(8)) if not df.empty else "No data available."
                    return jsonify({"reply": explanation, "sql": sql, "table_html": table_html, "plot_data_uri": plot_uri})
                except Exception as e:
                    return jsonify({"reply": f"Error executing SQL: {e}", "sql": sql}), 500
            # installation / disconnection rates
            if any(tok in ql for tok in ("install", "installation", "disconnection", "disconnect", "installation rate", "disconnection rate")):
                date_col = get_existing_date_column()
                sql = f"""
                    SELECT FORMAT({date_col}, 'yyyy-MM') AS month,
                        COUNT(*) AS total_orders,
                        SUM(CASE WHEN (install_date_parsed IS NOT NULL OR InstallDate IS NOT NULL OR IsInstalled = 1) THEN 1 ELSE 0 END) AS installations,
                        SUM(CASE WHEN (DisconnectDate IS NOT NULL OR LOWER(COALESCE(Status,'')) LIKE '%disconnect%') THEN 1 ELSE 0 END) AS disconnections
                    FROM {TABLE}
                    WHERE {rel['filter_sql']}
                    GROUP BY FORMAT({date_col}, 'yyyy-MM')
                    ORDER BY month;
                """
                try:
                    df = run_sql_and_fetch_df(sql, con)
                    if not df.empty:
                        first_col = df.columns[0]
                        df = df[df[first_col].notna() & (df[first_col].astype(str).str.strip() != "")]
                    table_html = rows_to_html_table(df.to_dict(orient='records')) if not df.empty else "<p><i>No data</i></p>"
                    plot_uri = None
                    if not df.empty:
                        plot_uri = plot_to_base64(df['month'].astype(str).tolist(), df['installations'].fillna(0).tolist(), kind='bar', title=question)
                    explanation = ask_model_explain(question, sql, df.head(8)) if not df.empty else f"No install/disconnect info for {rel['meta']}"
                    return jsonify({"reply": explanation, "sql": sql, "table_html": table_html, "plot_data_uri": plot_uri, "rows_returned": len(df)})
                except Exception as e:
                    return jsonify({"reply": f"Error executing filtered install/disconnect query: {e}", "sql": sql}), 500


            # profit query
            if any(tok in ql for tok in ("profit", "profits", "net profit", "total profit")):
                date_col = get_existing_date_column()
                profit_check = f"({profit_col} IS NOT NULL AND {profit_col} <> '')" if profit_col else "1=1"
                sql = f"""
                    SELECT FORMAT({date_col}, 'yyyy-MM') AS month,
                        SUM(COALESCE(CAST({profit_col} AS FLOAT), 0)) AS profit,
                        COUNT(*) AS orders
                    FROM {TABLE}
                    WHERE {rel['filter_sql']} AND {profit_check}
                    GROUP BY FORMAT({date_col}, 'yyyy-MM')
                    ORDER BY month;
                """
                try:
                    df = run_sql_and_fetch_df(sql, con)
                    if not df.empty:
                        first_col = df.columns[0]
                        df = df[df[first_col].notna() & (df[first_col].astype(str).str.strip() != "")]
                    table_html = rows_to_html_table(df.to_dict(orient='records')) if not df.empty else "<p><i>No data</i></p>"
                    plot_uri = None
                    if not df.empty:
                        plot_uri = plot_to_base64(df['month'].astype(str).tolist(), df['profit'].fillna(0).tolist(), kind='line', title=question)
                    return jsonify({"reply": f"Profit details for {rel['meta']}", "sql": sql, "table_html": table_html, "plot_data_uri": plot_uri})
                except Exception as e:
                    return jsonify({"reply": f"Error executing SQL: {e}", "sql": sql}), 500


            # order count
            if any(tok in ql for tok in ("order", "orders", "count")):
                date_col = get_existing_date_column()
                sql = f"""
                    SELECT FORMAT({date_col}, 'yyyy-MM') AS month,
                        COUNT(*) AS orders
                    FROM {TABLE}
                    WHERE {rel['filter_sql']}
                    GROUP BY FORMAT({date_col}, 'yyyy-MM')
                    ORDER BY month;
                """
                try:
                    df = run_sql_and_fetch_df(sql, con)
                    if not df.empty:
                        first_col = df.columns[0]
                        df = df[df[first_col].notna() & (df[first_col].astype(str).str.strip() != "")]
                    table_html = rows_to_html_table(df.to_dict(orient='records')) if not df.empty else "<p><i>No data</i></p>"
                    explanation = ask_model_explain(question, sql, df.head(8)) if not df.empty else "No data available."
                    return jsonify({"reply": explanation, "sql": sql, "table_html": table_html})
                except Exception as e:
                    return jsonify({"reply": f"Error executing SQL: {e}", "sql": sql}), 500


    if not question:
        return jsonify({"error": "No message provided"}), 400

    # metadata
    if re.search(r'\bhow many rows\b|\bnumber of rows\b', qlow):
        try:
            cur = get_cursor()
            cur.execute(f"SELECT COUNT(*) FROM {TABLE}")
            rows = cur.fetchone()[0]
            try:
                cur.close()
            except Exception:
                pass
            return jsonify({"reply": f"The dataset contains {rows:,} rows."})
        except Exception as e:
            return jsonify({"reply": f"Could not count rows: {e}"}), 500

    # generate SQL using LLM
    schema_cols = COLS
    try:
        sample_rows = run_sql_and_fetch_df(f"SELECT TOP 50 * FROM {TABLE}", con)
    except Exception:
        sample_rows = SAMPLE_DF if not SAMPLE_DF.empty else pd.DataFrame(columns=COLS)

    sql_text = ask_model_for_sql(question, schema_cols, sample_rows)
    if sql_text.strip().upper() == "NO_SQL":
        sample_csv = sample_rows.head(20).to_csv(index=False) if not sample_rows.empty else "no sample rows"
        prompt = f"Columns: {', '.join(COLS)}\nSample:\n{sample_csv}\nQuestion: {question}"
        resp = client.chat.completions.create(model=CHAT_MODEL, messages=[{"role": "system", "content": "You are a data analyst."}, {"role": "user", "content": prompt}], temperature=0.2)
        return jsonify({"reply": resp.choices[0].message.content})

    sql_text = re.sub(r"^```(?:sql)?\s*|\s*```$", "", sql_text, flags=re.IGNORECASE).strip()
    sql_text = re.sub(r"\bdataset\b", TABLE, sql_text, flags=re.IGNORECASE)
    sql_text = normalize_sql_columns(sql_text)

    if not is_safe_select(sql_text):
        return jsonify({"reply": "Generated SQL blocked for safety."}), 400

    try:
        df = run_sql_and_fetch_df(sql_text, con)
    except Exception as e:
        # try one LLM-based auto-fix attempt
        try:
            fixed = ask_model_fix_sql(sql_text, str(e), schema_cols, sample_rows)
            if fixed and fixed.strip().upper() != "NO_SQL":
                fixed = re.sub(r"^```(?:sql)?\s*|\s*```$", "", fixed, flags=re.IGNORECASE).strip()
                fixed = re.sub(r"\bdataset\b", TABLE, fixed, flags=re.IGNORECASE)
                fixed = normalize_sql_columns(fixed)
                if not is_safe_select(fixed):
                    return jsonify({"reply":"Auto-fixed SQL blocked for safety."}), 400
                try:
                    df = run_sql_and_fetch_df(fixed, con)
                    sql_text = fixed
                except Exception as e2:
                    return jsonify({"reply": f"SQL execution error after auto-fix: {e2}", "sql": fixed}), 500
            else:
                return jsonify({"reply": f"SQL execution error: {e}", "sql": sql_text}), 500
        except Exception:
            return jsonify({"reply": f"SQL execution error: {e}", "sql": sql_text}), 500
        # --- START: drop NULL/empty group labels so UI doesn't show "null" as top bucket ---
    # Defensive: ensure df was created by the block above
    if 'df' not in locals():
        return jsonify({"reply": "No data returned from query.", "sql": sql_text}), 500

    # If DataFrame returned, drop rows where the first column (group label) is NULL or empty.
    # Prefer non-null groups. If ALL are null/empty, attempt a fallback SQL that explicitly excludes nulls.
    if df is not None and not df.empty:
        first_col = df.columns[0]
        try:
            # mask of meaningful group labels
            non_null_mask = df[first_col].notna() & (df[first_col].astype(str).str.strip() != "")
            if non_null_mask.any():
                # There are valid group labels — remove the null/empty rows
                df = df[non_null_mask]
            else:
                # All group labels are NULL/empty — attempt to re-run SQL excluding nulls.
                # Build a safety exclusion expression for the group column.
                group_col = first_col
                excl = f"{group_col} IS NOT NULL AND LTRIM(RTRIM(COALESCE({group_col},''))) <> ''"

                new_sql = None
                try:
                    # If original SQL already contains a WHERE, inject the exclusion at the start of the WHERE.
                    if re.search(r"\bWHERE\b", sql_text, flags=re.IGNORECASE):
                        # inject excl after the first WHERE
                        new_sql = re.sub(r"(\bWHERE\b\s*)", r"\1" + excl + " AND ", sql_text, flags=re.IGNORECASE, count=1)
                    else:
                        # No WHERE — try inserting before GROUP BY or ORDER BY if present, otherwise append WHERE
                        if re.search(r"\bGROUP\s+BY\b", sql_text, flags=re.IGNORECASE):
                            new_sql = re.sub(r"(\bGROUP\s+BY\b)", "WHERE " + excl + " \\1", sql_text, flags=re.IGNORECASE, count=1)
                        elif re.search(r"\bORDER\s+BY\b", sql_text, flags=re.IGNORECASE):
                            new_sql = re.sub(r"(\bORDER\s+BY\b)", "WHERE " + excl + " \\1", sql_text, flags=re.IGNORECASE, count=1)
                        else:
                            new_sql = sql_text + " WHERE " + excl

                    # Try executing fallback query (if we produced one)
                    if new_sql:
                        try:
                            df2 = run_sql_and_fetch_df(new_sql, con)
                            # if fallback returned rows, use them (and update sql_text to reflect real executed SQL)
                            if df2 is not None and not df2.empty:
                                df = df2
                                sql_text = new_sql
                        except Exception:
                            # fallback failed — keep original df
                            pass
                except Exception:
                    # any unexpected error during fallback — ignore and keep original df
                    pass
        except Exception:
            # If something goes wrong (type coercion etc.), don't modify df
            pass
    # --- END: drop NULL/empty group labels ---

    df_preview = df.head(MAX_ROWS)
    table_html = rows_to_html_table(df_preview.to_dict(orient='records')) if not df_preview.empty else "<p><i>No data</i></p>"

    plot_uri = None
    try:
        if not df.empty:
            if 'month' in df.columns:
                ycol = next((c for c in df.columns if c != 'month' and pd.api.types.is_numeric_dtype(df[c])), None)
                if ycol:
                    plot_uri = plot_to_base64(df['month'].astype(str).tolist(), df[ycol].fillna(0).tolist(), kind='line', title=question)
            elif df.shape[1] >= 2 and pd.api.types.is_numeric_dtype(df.iloc[:,1]):
                labels = df.iloc[:,0].astype(str).tolist()
                vals = df.iloc[:,1].astype(float).fillna(0).tolist()
                plot_uri = plot_to_base64(labels, vals, kind='bar', title=question)
    except Exception as e:
        print("Plot error:", e, traceback.format_exc())

    explanation = "No data found."
    if not df_preview.empty:
        preview_rows = df_preview.head(50).to_dict(orient='records')
        try:
            prompt = f"Question: {question}\nSQL: {sql_text}\nPreview:\n{json.dumps(preview_rows)}"
            resp = client.chat.completions.create(model=CHAT_MODEL, messages=[{"role": "system", "content": "You are a concise analyst."}, {"role": "user", "content": prompt}], temperature=0.0)
            explanation = resp.choices[0].message.content.strip()
        except Exception:
            explanation = f"Here are the results for: {question}"

    return jsonify({
        "reply": explanation,
        "sql": sql_text,
        "table_html": table_html,
        "plot_data_uri": plot_uri,
        "rows_returned": len(df),
        "truncated": len(df) > MAX_ROWS
    })

# -------------------------
# Memory DB (SQLite) implementation (CONSOLIDATED/CORRECTED)
# -------------------------
# We'll keep the same function names (_load_conversations, _save_conversations) that your endpoints call,
# but back them with a single SQLite database to avoid JSON file race conditions and scale better.
# -------------------------
# Memory DB (SQLite) implementation (CONSOLIDATED/CORRECTED + safe migration)
# -------------------------
# We'll keep the same function names (_load_conversations, _save_conversations) that your endpoints call,
# but back them with a single SQLite database to avoid JSON file race conditions and scale better.

DB_PATH = Path('./data/chat_memory.db')
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def _ensure_chats_table_and_migration(db_path: Path):
    """
    Ensure the chats table exists with the canonical schema:
      id TEXT PRIMARY KEY,
      title TEXT,
      created_at INTEGER,
      messages_json TEXT
    If the table exists but messages_json is missing, add it.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        # create table if missing with minimal columns (without messages_json)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                title TEXT,
                created_at INTEGER
                -- messages_json column may be added by migration below
            );
        """)
        conn.commit()

        # fetch existing columns
        cur.execute("PRAGMA table_info(chats);")
        existing_cols = [r[1] for r in cur.fetchall()]  # r[1] is column name

        if 'messages_json' not in existing_cols:
            try:
                # Add messages_json column
                cur.execute("ALTER TABLE chats ADD COLUMN messages_json TEXT;")
                conn.commit()
                print("Migration: added 'messages_json' column to chats table.")
            except Exception as e:
                # On some SQLite setups ALTER TABLE may fail; log and continue (we'll fallback to JSON file)
                print("⚠️ Failed to add messages_json column during migration:", e)

        # close cursor
        try:
            cur.close()
        except Exception:
            pass
    finally:
        conn.close()

# Run migration once at startup
_ensure_chats_table_and_migration(DB_PATH)

def _row_to_conv(row):
    """Convert sqlite row (id,title,created_at,messages_json) -> dict as before."""
    cid, title, created_at, messages_json = row
    try:
        messages = json.loads(messages_json) if messages_json else []
    except Exception:
        messages = []
    return {
        "id": cid,
        "title": title,
        "created_at": created_at or 0,
        "messages": messages
    }

def _load_conversations():
    """Return all saved chats as a dict."""
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            rows = conn.execute("SELECT id, title, created_at, messages_json FROM chats ORDER BY created_at DESC").fetchall()
        out = {}
        for r in rows:
            out[r[0]] = {
                "id": r[0],
                "title": r[1],
                "created_at": r[2],
                "messages": json.loads(r[3]) if r[3] else []
            }
        return out
    except Exception as e:
        print("⚠️ Failed to load chats from DB:", e)
        # fallback: if JSON file exists, read it (keeps compatibility)
        try:
            if MEMORY_FILE.exists():
                return json.load(open(MEMORY_FILE, 'r', encoding='utf8'))
        except Exception as ex:
            print("⚠️ Fallback JSON load failed:", ex)
        return {}

def _save_conversation_single(chat_id, title, created_at, messages):
    """Save or update a single chat record (tries SQLite first, then fallback JSON)."""
    # ensure migration ran before we attempt to save (defensive)
    _ensure_chats_table_and_migration(DB_PATH)

    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO chats (id, title, created_at, messages_json) VALUES (?, ?, ?, ?)",
                (chat_id, title, created_at, json.dumps(messages))
            )
            conn.commit()
        return True
    except Exception as e:
        print(f"⚠️ Failed to save chat {chat_id} to SQLite DB:", e)
        # fallback: persist to JSON file as a best-effort fallback
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
                "messages": messages
            }
            with open(MEMORY_FILE, 'w', encoding='utf8') as f:
                json.dump(all_conv, f, ensure_ascii=False, indent=2)
            print(f"Saved chat {chat_id} to fallback JSON file.")
            return True
        except Exception as ex:
            print("Fallback JSON save also failed:", ex)
            return False

# -------------------------
# Flask Memory Endpoints (use the consolidated DB functions)
# -------------------------

@app.route('/memory/list', methods=['GET'])
def memory_list():
    """List saved chat sessions (for sidebar)"""
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            rows = conn.execute(
                "SELECT id, title, created_at FROM chats ORDER BY created_at DESC"
            ).fetchall()
        chats = [
            {"id": r[0], "title": r[1] or "Chat", "created_at": r[2]}
            for r in rows
        ]
        return jsonify({"chats": chats})
    except Exception as e:
        # fallback to JSON if SQLite fails
        try:
            conv = _load_conversations()
            items = []
            for k, v in conv.items():
                items.append({"id": k, "title": v.get("title") or "Chat", "created_at": v.get("created_at", 0)})
            items.sort(key=lambda x: x['created_at'] or 0, reverse=True)
            return jsonify({"chats": items})
        except Exception:
            return jsonify({"error": f"Failed to list chats: {e}"}), 500

@app.route('/memory/load', methods=['GET'])
def memory_load():
    """Load a specific chat by ID"""
    cid = (request.args.get('id') or "").strip()
    if not cid:
        return jsonify({"error": "Missing chat ID"}), 400
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            row = conn.execute(
                "SELECT id, title, created_at, messages_json FROM chats WHERE id = ?",
                (cid,)
            ).fetchone()
        if not row:
            # fallback to JSON
            conv = _load_conversations()
            conv_row = conv.get(cid)
            if conv_row:
                return jsonify(conv_row)
            return jsonify({"error": "Chat not found"}), 404
        return jsonify({
            "id": row[0],
            "title": row[1],
            "created_at": row[2],
            "messages": json.loads(row[3]) if row[3] else []
        })
    except Exception as e:
        return jsonify({"error": f"Failed to load chat: {e}"}), 500

@app.route('/memory/save', methods=['POST'])
def memory_save():
    """Save or update a conversation"""
    payload = request.get_json(force=True) or {}
    cid = payload.get('id') or f"chat-{int(time.time() * 1000)}"
    title = payload.get('title') or 'Chat'
    created_at = payload.get('created_at') or int(time.time() * 1000)
    messages = payload.get('messages') or []
    ok = _save_conversation_single(cid, title, created_at, messages)
    if not ok:
        return jsonify({"error": "Failed to save chat"}), 500
    return jsonify({"ok": True, "id": cid})

@app.route('/memory/delete', methods=['POST'])
def memory_delete():
    """Delete a chat by ID"""
    payload = request.get_json(force=True) or {}
    cid = payload.get('id')
    if not cid:
        return jsonify({"error": "Missing chat ID"}), 400
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            conn.execute("DELETE FROM chats WHERE id = ?", (cid,))
            conn.commit()
        return jsonify({"ok": True})
    except Exception as e:
        # try fallback deletion from JSON
        try:
            if MEMORY_FILE.exists():
                m = json.load(open(MEMORY_FILE, 'r', encoding='utf8'))
                if cid in m:
                    m.pop(cid, None)
                    with open(MEMORY_FILE, 'w', encoding='utf8') as f:
                        json.dump(m, f, ensure_ascii=False, indent=2)
                    return jsonify({"ok": True})
        except Exception:
            pass
        return jsonify({"error": f"Failed to delete chat: {e}"}), 500

# Run the Flask app
# -------------------------
if __name__ == "__main__":
    print(f"🚀 Starting server on http://localhost:{PORT}")
    print(f"Connected to SQL Server table: {TABLE}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
