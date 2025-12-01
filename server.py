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
from datetime import date
from zoneinfo import ZoneInfo
import time
from utils import rows_to_html_table, plot_to_base64, embed_text  # your existing utils
import uuid
from pathlib import Path
import sqlite3
import datetime as dtmod 
from datetime import date as _date
import uuid as _uuid
import traceback as _traceback
from html import escape
from datetime import datetime as _dt
from io import BytesIO  # <-- added for in-memory Excel export

# CONFIG / ENV
# -------------------------------------------------
load_dotenv()

DEBUG_SQL = os.getenv("DEBUG_SQL", "false").lower() in ("1", "true", "yes")

TZ = ZoneInfo("Asia/Kolkata")  # user timezone

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DUCKDB_FILE = os.getenv("DUCKDB_FILE", "./data/sales.duckdb")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
PORT = int(os.getenv("PORT", "3000"))
MAX_ROWS = int(os.getenv("MAX_ROWS", "5000"))

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY must be set in environment")

client = OpenAI(api_key=OPENAI_API_KEY)

# directory for Excel exports (kept, but no longer used for auto-writing per-query)
EXPORT_DIR = Path("./data/exports")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# In-memory cache for exports: token -> DataFrame or dict of DataFrames
EXPORT_CACHE = {}

# DUCKDB CONNECTION
# -------------------------------------------------
db_path = DUCKDB_FILE.strip()
if not (db_path.lower().startswith("md:") or db_path.lower().startswith("md://")):
    db_path = os.path.abspath(db_path)
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DuckDB file not found: {db_path}")

con = duckdb.connect(db_path)

# Detect tables
available_tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
print("Detected tables:", available_tables)

TABLE = "aiv_OrderItemUnit_md"
if TABLE not in available_tables:
    for t in available_tables:
        if t.lower() == TABLE.lower():
            TABLE = t
            break
# Basic schema info
cols_info = con.execute(f"PRAGMA table_info('{TABLE}')").fetchall()
COLS = [c[1] for c in cols_info] if cols_info else []

# Sample and row count
SAMPLE_DF = con.execute(f"SELECT * FROM {TABLE} LIMIT 10").fetchdf() if COLS else pd.DataFrame()
try:
    ROWCOUNT = con.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
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

# GENERIC HELPERS
# -------------------------------------------------
def run_sql_and_fetch_df(sql: str, print_errors: bool = True):
    """Execute SQL and return a pandas DataFrame."""
    if DEBUG_SQL:
        print("DEBUG SQL:\n", sql)
    try:
        df = con.execute(sql).fetchdf()
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

# INTENT CLASSIFICATION (cheap)
# -------------------------------------------------
def classify_intent(question: str) -> str:
    q = (question or "").strip().lower()
    if not q:
        return "empty"

    # greetings
    if q in {"hi", "hello", "hey", "yo", "hola", "sup"} or q.startswith(("hi ", "hello ", "hey ")):
        return "smalltalk"
    if "your name" in q or "who are you" in q:
        return "smalltalk"

    # schema / metadata
    if "what columns" in q or "what fields" in q or "show schema" in q:
        return "meta"
    if re.search(r'\b(show|list|display|give|what are)\b.*\b(column names?|columns|fields|attributes|headings)\b', q):
        return "meta"

    # everything else: treat as data question
    return "data_question"

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

        prompt = f"""
You are an assistant that generates DuckDB SQL for a single table called {TABLE}.

Your job:
1. Decide if the user's question is about this table's data.
2. If YES → return exactly ONE DuckDB SELECT statement that answers it.
3. If NO → return exactly NO_SQL.

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
- When the user asks for results "based on", "by", "where", "sorted by" or "whose <metric> ..."
  (e.g. "based on profit", "by profit", "estimated commission = 250", "orders > 150"):
  ALWAYS include both:
  • the identifying column(s) (e.g. ProviderName, MainChannel, Package), AND
  • the metric column(s) you are using (Profit, Revenue, EstimatedCommission, COUNT(*) AS orders, etc.)
  in the SELECT list.

  Examples:
  - "give me any 20 provider names based on profits" ⇒
    SELECT ProviderName,
           SUM(CAST(Profit AS DOUBLE)) AS total_profit
    FROM {TABLE}
    GROUP BY ProviderName
    ORDER BY total_profit DESC
    LIMIT 20;

  - "give me the provider name whose estimated commission is 250" ⇒
    SELECT ProviderName,
           EstimatedCommission
    FROM {TABLE}
    WHERE EstimatedCommission = 250;

  - "give me the mainchannel whose order > 150" ⇒
    SELECT MainChannel,
           COUNT(*) AS orders
    FROM {TABLE}
    GROUP BY MainChannel
    HAVING COUNT(*) > 150;

- If the question wants raw row details (not aggregation), include LIMIT {MAX_ROWS}, unless the user explicitly asks for "all ...".
- If the question is purely conceptual/theory (definitions, explanations) without needing actual table values, return NO_SQL.

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

        # Strip code fences if model added them
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

        prompt = f"""
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
                {"role": "system", "content": "You are a clear, concise assistant. Explain things in simple terms."},
                {"role": "user", "content": question},
            ],
            temperature=0.5,
            max_tokens=800,
        )
        ans = (resp.choices[0].message.content or "").strip()
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

    # --- conversation-aware context ---
    history = history or []
    if history:
        ctx_lines = []
        # keep last 6 messages to stay small
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

    # If user used relative time ("last year", "this month") OR mentioned a month with no year, ask for clarification
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
        sample_rows = con.execute(f"SELECT * FROM {TABLE} LIMIT 50").fetchdf()
    except Exception:
        sample_rows = SAMPLE_DF if not SAMPLE_DF.empty else pd.DataFrame(columns=COLS)

    # 1) Ask model for SQL  (conversation-aware)
    sql_text = ask_model_for_sql(augmented_question, schema_cols, sample_rows)

    # 2) If NO_SQL → theory/general mode (conversation-aware)
    if not sql_text or sql_text.strip().upper() == "NO_SQL":
        return answer_theory_question(augmented_question)

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
            # first attempt: don't spam error logs
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

    if not df_preview.empty:
        table_html = rows_to_html_table(df_preview.to_dict(orient="records"))
    else:
        table_html = "<p><i>No data</i></p>"

    # 5) Optional plot
    plot_uri = None
    try:
        if not df.empty:
            if "month" in df.columns:
                nums = [c for c in df.columns if c != "month" and pd.api.types.is_numeric_dtype(df[c])]
                if nums:
                    ycol = nums[0]
                    plot_uri = plot_to_base64(
                        df["month"].astype(str).tolist(),
                        df[ycol].fillna(0).tolist(),
                        kind="line",
                        title=question,
                    )
            else:
                if df.shape[1] >= 2 and pd.api.types.is_numeric_dtype(df.iloc[:, 1]):
                    labels = df.iloc[:, 0].astype(str).tolist()
                    vals = df.iloc[:, 1].astype(float).fillna(0).tolist()
                    plot_uri = plot_to_base64(labels, vals, kind="bar", title=question)
    except Exception as e:
        print("Plot error:", e, traceback.format_exc())
        plot_uri = None

    # 5.a) Prepare Excel export ONLY as in-memory cache; no file written to disk
    download_url = None
    try:
        file_id = uuid.uuid4().hex
        EXPORT_CACHE[file_id] = df_preview  # store DataFrame in memory
        download_url = f"/download/{file_id}"
    except Exception as e:
        print("Export cache error:", e, traceback.format_exc())
        download_url = None

    # 6) Let LLM summarize results (we keep showing the latest question here)
    explanation = ""
    if not df_preview.empty:
        preview_rows = df_preview.head(50).to_dict(orient="records")
        try:
            prompt = (
                f"User question: {question}\n"
                f"SQL used:\n{sql_text}\n\n"
                f"Result preview (first {min(50, len(preview_rows))} rows, JSON):\n"
                f"{json.dumps(preview_rows, ensure_ascii=False)}\n\n"
                f"Please provide:\n"
                f"1. One short headline-style summary line.\n"
                f"2. One or two sentences explaining the key points in plain language."
            )
            resp = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a concise data analyst."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=256,
            )
            explanation = (resp.choices[0].message.content or "").strip()
        except Exception:
            explanation = f"Here are the results for: {question}"
    else:
        explanation = "No data found for that query."

    result = {
        "reply": explanation,
        "table_html": table_html,
        "plot_data_uri": plot_uri,
        "rows_returned": total_rows,
        "truncated": truncated,
        "download_url": download_url,   # frontend can show 'Download Excel'
    }
    if DEBUG_SQL:
        result["debug_sql"] = sql_text

    return result

# FLASK APP
# -------------------------------------------------
app = Flask(__name__, static_folder='static')

SESSION_COOKIE_NAME = "sid"
SESSION_COOKIE_TTL_DAYS = 365 * 2

def _make_sid():
    return uuid.uuid4().hex

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

# --- Provider endpoints: unified generator (updated to include 'installed' in Total) ---
from datetime import datetime as _dt

def _make_provider_route(route: str, provider_filter_sql: str, route_name: str):
    """
    Register a GET endpoint at `route` that returns the monthly table for the provider.
    provider_filter_sql should be a SQL boolean expression using table columns
    e.g. "LOWER(TRIM(ProviderName)) LIKE '%spectrum%'"
    route_name must be unique (used as Flask endpoint name).
    """

    def handler():
        try:
            # optional ?year=YYYY, defaults to current year in TZ
            try:
                req_year = int(request.args.get("year")) if request.args.get("year") else None
            except Exception:
                req_year = None
            if not req_year:
                req_year = _dt.now(TZ).year

            sale_date = qident('SaleDate')
            status_col = qident('Status')

            # SQL implementing the corrected status groups (installed included in Total)
            sql = f"""
            WITH flagged AS (
              SELECT
                STRFTIME('%Y', TRY_CAST({sale_date} AS DATE)) AS SaleYear,
                STRFTIME('%m', TRY_CAST({sale_date} AS DATE)) AS SaleMonth,
                LOWER(TRIM(COALESCE({status_col}, ''))) AS status_norm
              FROM {qident(TABLE)}
              WHERE TRY_CAST({sale_date} AS DATE) IS NOT NULL
                AND ({provider_filter_sql})
                AND STRFTIME('%Y', TRY_CAST({sale_date} AS DATE)) = '{req_year}'
            ),
            marks AS (
              SELECT
                CAST(SaleYear AS INTEGER) AS SaleYear,
                CAST(SaleMonth AS INTEGER) AS SaleMonth,
                -- CORRECTED: Total now includes both 'installed' AND 'installation'
                CASE WHEN status_norm IN (
                  'installed',
                  'callinorder','onlineorder','pending activation',
                  'pending cancel','spectrumpending','cancelled','disconnected'
                ) THEN 1 ELSE 0 END AS is_total,
                -- Install statuses (unchanged)
                CASE WHEN status_norm IN ('Installed','Pending Activation','Disconnected') THEN 1 ELSE 0 END AS is_install,
                -- Cancel statuses (unchanged)
                CASE WHEN status_norm ='cancelled' THEN 1 ELSE 0 END AS is_cancel,
                CASE WHEN status_norm = 'disconnected' THEN 1 ELSE 0 END AS is_disconnect
              FROM flagged
            ),
            monthly AS (
              SELECT
                SaleYear,
                SaleMonth,
                SUM(is_total) AS Total,
                SUM(is_install) AS Install,
                SUM(is_cancel) AS Cancel,
                SUM(is_disconnect) AS Disconnect
              FROM marks
              GROUP BY SaleYear, SaleMonth
            ),
            with_pending AS (
              SELECT
                SaleYear,
                SaleMonth,
                Total,
                Install,
                Cancel,
                Disconnect,
                (Total - Install - Cancel) AS Pending,
                CASE WHEN Total = 0 THEN 0.0 ELSE 100.0 * Install / Total END AS InstallPct,
                CASE WHEN Install = 0 THEN 0.0 ELSE 100.0 * Disconnect / Install END AS DisconnectPct,
                LAG(Total - Install - Cancel) OVER (ORDER BY SaleYear DESC, SaleMonth DESC) AS PrevPending
              FROM monthly
            )
            SELECT
              SaleYear,
              SaleMonth,
              Total,
              Pending,
              CASE WHEN PrevPending IS NULL THEN CAST(Pending AS VARCHAR) || ' --> (CL:0)'
                   ELSE CAST(Pending AS VARCHAR) || ' --> (CL:' || (Pending - PrevPending) || ')' END AS PendingWithCL,
              Install,
              Cancel,
              ROUND(InstallPct, 2) AS InstallPct,
              Disconnect,
              ROUND(DisconnectPct, 2) AS DisconnectPct
            FROM with_pending
            ORDER BY SaleYear DESC, SaleMonth DESC
            ;
            """

            df = run_sql_and_fetch_df(sql)
            rows = df.to_dict(orient="records")
            table_html = rows_to_html_table(rows)

            # export (in-memory only)
            import uuid
            file_id = uuid.uuid4().hex
            EXPORT_CACHE[file_id] = df
            download_url = f"/download/{file_id}"

            return jsonify({
                "reply": f"{route_name} monthly table ({req_year}) generated.",
                "table_html": table_html,
                "plot_data_uri": None,
                "download_url": download_url
            })

        except Exception as e:
            print(f"{route_name} endpoint error:", e, traceback.format_exc())
            return jsonify({
                "reply": f"Failed to compute {route_name} table.",
                "table_html": f"<pre>Error: {escape(str(e))}</pre>",
                "plot_data_uri": None
            }), 500

    endpoint_name = f"provider_{route_name.replace(' ','_').lower()}"
    if endpoint_name in app.view_functions:
        app.view_functions.pop(endpoint_name, None)
    app.add_url_rule(route, endpoint=endpoint_name, view_func=handler, methods=['GET'])


# Register providers (same registration as before)
_make_provider_route(
    route="/spectrum",
    provider_filter_sql="LOWER(TRIM(ProviderName)) LIKE '%spectrum%'",
    route_name="Spectrum"
)

_make_provider_route(
    route="/att",
    provider_filter_sql="(LOWER(TRIM(ProviderName)) LIKE '%at&t%' OR LOWER(TRIM(ProviderName)) LIKE '%att%')",
    route_name="AT&T"
)

_make_provider_route(
    route="/comcast",
    provider_filter_sql="LOWER(TRIM(ProviderName)) LIKE '%comcast%'",
    route_name="Comcast"
)

_make_provider_route(
    route="/directv",
    provider_filter_sql="LOWER(TRIM(ProviderName)) LIKE '%directv%'",
    route_name="DIRECTV"
)
# --- end updated provider endpoints ---


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    payload = request.json or {}
    question = (payload.get('message') or "").strip()
    history = payload.get('history') or []
    if not question:
        return jsonify({"error": "No message provided"}), 400

    qlow = question.lower()
    print(f"Q: {question}")
    if any(phrase in qlow for phrase in [
        "what year is this",
        "what year is it now",
        "current year",
        "this year now",
        "what is today's year",
    ]):
        return jsonify({
            "reply": f"The current year is {dtmod.date.today().year}.",
            "table_html": "",
            "plot_data_uri": None
        })

    if any(phrase in qlow for phrase in [
        "what month is this",
        "current month",
        "what month are we in",
    ]):
        today = dtmod.date.today()
        return jsonify({
            "reply": f"We are currently in {today.strftime('%B %Y')}.",
            "table_html": "",
            "plot_data_uri": None
        })
    if any(phrase in qlow for phrase in [
        "what was last year",
        "last year is what",
        "which year was last year",
        "previous year",
    ]):
        today = dtmod.date.today()
        return jsonify({
            "reply": f"Last year was {dtmod.date.today().year - 1}.",
            "table_html": "",
            "plot_data_uri": None
        })
    if any(phrase in qlow for phrase in [
        "what is today",
        "today's date",
        "what day is today",
        "what date is today",
    ]):
        today = dtmod.date.today()
        return jsonify({
            "reply": f"Today's date is {today.strftime('%B %d, %Y')}.",
            "table_html": "",
            "plot_data_uri": None
        })

    intent = classify_intent(question)

    # Smalltalk
    if intent == "smalltalk":
        return jsonify({
            "reply": "Hi! I’m your data assistant. Ask me anything about this dataset or general concepts.",
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
    if re.search(r'\bhow many rows\b|\brow count\b|\bnumber of rows\b', qlow):
        try:
            rows = ROWCOUNT if ROWCOUNT is not None else con.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
            cols = len(COLS)
            return jsonify({
                "reply": f"The dataset contains {rows:,} rows and {cols} columns.",
                "table_html": "",
                "plot_data_uri": None
            })
        except Exception as e:
            _log_and_mask_error(e, "count_rows_cols")
            return jsonify(_safe_user_error("Could not determine rows/cols right now.")), 500

    resp = handle_general_data_question(question, history=history)
    return jsonify(resp)

# -------------------------------------------------
# DOWNLOAD ENDPOINT FOR EXCEL (in-memory only)
# -------------------------------------------------
@app.route("/download/<token>")
def download_result(token):
    # Look up the data for this token in the in-memory cache.
    data = EXPORT_CACHE.get(token)
    if data is None:
        return "File not found or expired", 404

    output = BytesIO()

    # Data can be a single DataFrame or a dict of sheet_name -> DataFrame
    if isinstance(data, dict):
        # multiple sheets (used by /summary)
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            for sheet_name, df in data.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        # single sheet
        df = data
        df.to_excel(output, index=False)

    output.seek(0)

    # Optional: remove from cache after first download so memory doesn't grow forever
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
    except Exception as e:
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
    except Exception as e:
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


@app.route("/summary", methods=["GET"])
def summary():
    """
    Returns dataset-level summary for the CURRENT YEAR (TZ = Asia/Kolkata).
    JSON includes:
      - year
      - total_companies (distinct providers)
      - total_orders (count rows)
      - total_revenue (sum NetAfterChargeback)
      - total_profit  (sum Profit)
      - product counts: tv_count, mobile_count, homephone_count
      - top_providers: small table (provider, orders, revenue, profit) top 10 by orders
    Also returns table_html for top_providers and download_url (xlsx).
    """
    try:
        # allow optional ?year=YYYY override
        try:
            req_year = int(request.args.get("year")) if request.args.get("year") else None
        except Exception:
            req_year = None
        if not req_year:
            req_year = _dt.now(TZ).year

        # quoted identifiers
        sale_date = qident("SaleDate")
        provider_col = qident("ProviderName")
        product_col = qident("ProductName")
        revenue_col = qident("NetAfterChargeback")
        profit_col = qident("Profit")
        status_col = qident("Status")

        # product keyword checks (case-insensitive)
        # tv: look for 'tv', 'television', 'set-top', 'stb'
        # mobile: 'mobile','cell','phone','smartphone'
        # homephone/landline: 'homephone','home phone','landline'
        sql = f"""
        WITH year_rows AS (
          SELECT
            *
          FROM {qident(TABLE)}
          WHERE TRY_CAST({sale_date} AS DATE) IS NOT NULL
            AND STRFTIME('%Y', TRY_CAST({sale_date} AS DATE)) = '{req_year}'
        ),
        metrics AS (
          SELECT
            COUNT(*) AS total_orders,
            COUNT(DISTINCT LOWER(TRIM({provider_col}))) AS total_companies,
            SUM(TRY_CAST(COALESCE({revenue_col}, '0') AS DOUBLE)) AS total_revenue,
            SUM(TRY_CAST(COALESCE({profit_col}, '0') AS DOUBLE)) AS total_profit,
            SUM(CASE WHEN LOWER(TRIM(COALESCE({product_col},''))) LIKE '%tv%' 
                      OR LOWER(TRIM(COALESCE({product_col},''))) LIKE '%television%'
                      OR LOWER(TRIM(COALESCE({product_col},''))) LIKE '%set-top%'
                      OR LOWER(TRIM(COALESCE({product_col},''))) LIKE '%stb%'
                     THEN 1 ELSE 0 END) AS tv_count,
            SUM(CASE WHEN LOWER(TRIM(COALESCE({product_col},''))) LIKE '%mobile%'
                      OR LOWER(TRIM(COALESCE({product_col},''))) LIKE '%cell%'
                      OR LOWER(TRIM(COALESCE({product_col},''))) LIKE '%smartphone%'
                      OR LOWER(TRIM(COALESCE({product_col},''))) LIKE '%phone%' -- catches 'mobile phone'
                     THEN 1 ELSE 0 END) AS mobile_count,
            SUM(CASE WHEN LOWER(TRIM(COALESCE({product_col},''))) LIKE '%homephone%'
                      OR LOWER(TRIM(COALESCE({product_col},''))) LIKE '%home phone%'
                      OR LOWER(TRIM(COALESCE({product_col},''))) LIKE '%landline%'
                     THEN 1 ELSE 0 END) AS homephone_count
          FROM year_rows
        ),
        top_providers AS (
          SELECT
            {provider_col} AS provider,
            COUNT(*) AS orders,
            SUM(TRY_CAST(COALESCE({revenue_col}, '0') AS DOUBLE)) AS revenue,
            SUM(TRY_CAST(COALESCE({profit_col}, '0') AS DOUBLE)) AS profit
          FROM year_rows
          GROUP BY provider
          ORDER BY orders DESC
          LIMIT 10
        )
        SELECT
          m.total_orders,
          m.total_companies,
          ROUND(m.total_revenue, 2) AS total_revenue,
          ROUND(m.total_profit, 2) AS total_profit,
          m.tv_count,
          m.mobile_count,
          m.homephone_count,
          tp.provider,
          tp.orders,
          ROUND(tp.revenue, 2) AS provider_revenue,
          ROUND(tp.profit, 2) AS provider_profit
        FROM metrics m
        LEFT JOIN top_providers tp ON 1=1
        ;
        """

        # run query
        df = run_sql_and_fetch_df(sql)

        # df will have repeated metrics rows for each top provider row; extract metrics from first row
        if df is None or df.empty:
            return jsonify({
                "reply": f"No data for year {req_year}",
                "table_html": "<p><i>No data</i></p>",
                "plot_data_uri": None
            }), 200

        # metrics
        first = df.iloc[0]
        result_metrics = {
            "year": req_year,
            "total_orders": int(first["total_orders"]) if not pd.isna(first["total_orders"]) else 0,
            "total_companies": int(first["total_companies"]) if not pd.isna(first["total_companies"]) else 0,
            "total_revenue": float(first["total_revenue"]) if not pd.isna(first["total_revenue"]) else 0.0,
            "total_profit": float(first["total_profit"]) if not pd.isna(first["total_profit"]) else 0.0,
            "tv_count": int(first["tv_count"]) if not pd.isna(first["tv_count"]) else 0,
            "mobile_count": int(first["mobile_count"]) if not pd.isna(first["mobile_count"]) else 0,
            "homephone_count": int(first["homephone_count"]) if not pd.isna(first["homephone_count"]) else 0,
        }

        # build top providers table (take columns provider, orders, provider_revenue, provider_profit)
        top_cols = ["provider", "orders", "provider_revenue", "provider_profit"]
        top_df = df[top_cols].dropna(subset=["provider"]).drop_duplicates(subset=["provider"]).reset_index(drop=True)
        table_html = rows_to_html_table(top_df.to_dict(orient="records"))

        # prepare in-memory export (multi-sheet: metrics + top_providers)
        import uuid
        file_id = uuid.uuid4().hex
        EXPORT_CACHE[file_id] = {
            "metrics": pd.DataFrame([result_metrics]),
            "top_providers": top_df
        }
        download_url = f"/download/{file_id}"

        response = {
            "reply": f"Dataset summary for {req_year}",
            "year": req_year,
            "metrics": result_metrics,
            "top_providers": top_df.to_dict(orient="records"),
            "table_html": table_html,
            "download_url": download_url,
            "plot_data_uri": None
        }
        return jsonify(response)

    except Exception as e:
        print("Summary endpoint error:", e, traceback.format_exc())
        return jsonify({
            "reply": "Failed to compute summary.",
            "table_html": f"<pre>Error: {escape(str(e))}</pre>",
            "plot_data_uri": None
        }), 500

@app.route('/memory/list', methods=['GET'])
def memory_list():
    sid = (getattr(g, "sid", "") or request.args.get('sid') or "").strip()
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(chats);")
            cols = [r[1] for r in cur.fetchall()]
            if sid and 'session_id' in cols:
                rows = cur.execute(
                    "SELECT id, title, created_at FROM chats WHERE session_id = ? "
                    "ORDER BY created_at DESC",
                    (sid,)
                ).fetchall()
            else:
                rows = cur.execute(
                    "SELECT id, title, created_at FROM chats ORDER BY created_at DESC"
                ).fetchall()
        chats = [{"id": r[0], "title": r[1] or "Chat", "created_at": r[2]} for r in rows]
        return jsonify({"chats": chats})
    except Exception as e:
        try:
            conv = _load_conversations()
            items = []
            for k, v in conv.items():
                if sid:
                    if isinstance(v, dict) and v.get("session_id") and v.get("session_id") != sid:
                        continue
                items.append({
                    "id": k,
                    "title": v.get("title") or "Chat",
                    "created_at": v.get("created_at", 0)
                })
            items.sort(key=lambda x: x['created_at'] or 0, reverse=True)
            return jsonify({"chats": items})
        except Exception:
            return jsonify({"error": f"Failed to list chats: {e}"}), 500

@app.route('/memory/load', methods=['GET'])
def memory_load():
    cid = (request.args.get('id') or "").strip()
    sid = (getattr(g, "sid", "") or (request.args.get('sid') or "")).strip()
    if not cid:
        return jsonify({"error": "Missing chat ID"}), 400
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(chats);")
            cols = [r[1] for r in cur.fetchall()]
            if sid and 'session_id' in cols:
                row = cur.execute(
                    "SELECT id, title, created_at, session_id, messages_json "
                    "FROM chats WHERE id = ? AND session_id = ?",
                    (cid, sid)
                ).fetchone()
            else:
                row = cur.execute(
                    "SELECT id, title, created_at, messages_json FROM chats WHERE id = ?",
                    (cid,)
                ).fetchone()
        if not row:
            conv = _load_conversations()
            conv_row = conv.get(cid)
            if conv_row:
                if sid and conv_row.get("session_id") and conv_row.get("session_id") != sid:
                    return jsonify({"error": "Chat not found"}), 404
                return jsonify(conv_row)
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

@app.route('/memory/save', methods=['POST'])
def memory_save():
    payload = request.get_json(force=True) or {}
    sid = (getattr(g, "sid") or payload.get('sid') or "").strip()
    cid = payload.get('id') or f"chat-{int(time.time() * 1000)}"
    title = payload.get('title') or 'Chat'
    created_at = payload.get('created_at') or int(time.time() * 1000)
    messages = payload.get('messages') or []
    ok = _save_conversation_single(cid, title, created_at, messages, session_id=sid)
    if not ok:
        return jsonify({"error": "Failed to save chat"}), 500
    return jsonify({"ok": True, "id": cid})

@app.route("/health", methods=["GET"])
def health():
    try:
        # Check DuckDB
        con.execute("SELECT 1")
        
        # Check SQLite
        with sqlite3.connect(str(DB_PATH)) as conn:
            conn.execute("SELECT 1")
        
        # Check export folder exists
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)

        return jsonify({"status": "ok"}), 200

    except Exception as e:
        return jsonify({"status": "error", "details": str(e)}), 500

@app.route('/memory/delete', methods=['POST'])
def memory_delete():
    payload = request.get_json(force=True) or {}
    cid = payload.get('id')
    sid = (getattr(g, "sid", "") or payload.get('sid') or "").strip()
    if not cid:
        return jsonify({"error": "Missing chat ID"}), 400
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
    app.run(host="0.0.0.0", port=PORT)
