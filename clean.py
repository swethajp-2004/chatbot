# clean_resume.py
import os
import pyodbc
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
import glob
import math

# ----- Configure these for your environment -----
DSN = "bundle_root"            # your ODBC DSN name
DB_USER = "devswetha"         # your DB user
DB_PASS = "xwnpZgdX" # replace
DB_NAME = "bundle_root"       # database name
TABLE = "dbo.OrderItemUnit"    # the table to read (adjust if needed)

OUT_CSV = "cleaned_dataset.csv"
CHUNK_DIR = "chunks"          # folder to store chunk files
CHUNKSIZE = 10000             # same chunk size you used before

os.makedirs(CHUNK_DIR, exist_ok=True)

def parse_excel_or_string_date(x):
    if pd.isna(x):
        return None
    if isinstance(x, datetime):
        return x.date().isoformat()
    try:
        if hasattr(x, "date"):
            return x.date().isoformat()
    except Exception:
        pass
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return None
    try:
        if isinstance(x, (int, float, Decimal)):
            days = float(x)
            base = datetime(1899, 12, 30)
            dt = base + timedelta(days=days)
            return dt.date().isoformat()
        days = float(s)
        base = datetime(1899, 12, 30)
        dt = base + timedelta(days=days)
        return dt.date().isoformat()
    except Exception:
        pass
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except Exception:
            pass
    try:
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
        if pd.isna(dt):
            return None
        return dt.date().isoformat()
    except Exception:
        return None

def connect():
    conn_str = f"DSN={DSN};UID={DB_USER};PWD={DB_PASS};DATABASE={DB_NAME};TrustServerCertificate=Yes;"
    print("Connecting to SQL Server via ODBC DSN:", DSN)
    conn = pyodbc.connect(conn_str, autocommit=True)
    print("Connected ✅")
    return conn

def existing_chunk_indexes():
    files = glob.glob(os.path.join(CHUNK_DIR, "chunk_*.csv"))
    idxs = []
    for f in files:
        bn = os.path.basename(f)
        try:
            num = int(bn.split("_")[1].split(".")[0])
            idxs.append(num)
        except Exception:
            pass
    return set(idxs)

def combine_chunks_to_output(out_csv=OUT_CSV):
    chunk_files = sorted(glob.glob(os.path.join(CHUNK_DIR, "chunk_*.csv")))
    if not chunk_files:
        print("No chunk files found to combine.")
        return False
    print(f"Combining {len(chunk_files)} chunk files into {out_csv} ...")
    # Remove existing OUT_CSV if present (we'll create fresh)
    if os.path.exists(out_csv):
        try:
            os.remove(out_csv)
        except Exception as e:
            print("Could not remove existing output file:", e)
            return False
    first = True
    for f in chunk_files:
        print("Appending", f)
        # Use pandas to_csv append safely
        df = pd.read_csv(f, dtype=str)  # load as string to avoid unexpected dtype conversion
        if first:
            df.to_csv(out_csv, index=False, mode="w", encoding="utf-8")
            first = False
        else:
            df.to_csv(out_csv, index=False, header=False, mode="a", encoding="utf-8")
    print("Combined into", out_csv)
    # optional: remove chunk files
    for f in chunk_files:
        try:
            os.remove(f)
        except Exception:
            print("Warning: could not remove chunk file", f)
    print("Temporary chunk files removed.")
    return True

def main():
    conn = connect()
    cur = conn.cursor()
    # count total rows
    try:
        cur.execute(f"SELECT COUNT(*) FROM {TABLE}")
        total = cur.fetchone()[0]
        print("Total rows in", TABLE, ":", total)
    except Exception as e:
        print("Could not determine total rows:", e)
        total = None

    sql = f"SELECT * FROM {TABLE}"
    # We will use pandas read_sql_query with chunksize
    reader = pd.read_sql_query(sql, conn, chunksize=CHUNKSIZE)

    existing = existing_chunk_indexes()
    print("Already existing chunk indexes:", sorted(existing) if existing else "None")

    written = sum(1 for _ in existing) * CHUNKSIZE  # not exact for last chunk but okay for progress message
    chunk_no = 0
    for i, chunk in enumerate(reader, start=1):
        chunk_no = i
        if i in existing:
            print(f"Skipping chunk #{i} (chunk file already present).")
            continue
        print(f"Processing chunk #{i} with {len(chunk)} rows...")
        # parse date cols
        if "SaleDate" in chunk.columns:
            chunk["sale_date_parsed"] = chunk["SaleDate"].apply(parse_excel_or_string_date)
        else:
            chunk["sale_date_parsed"] = None
        if "InstallDate" in chunk.columns:
            chunk["install_date_parsed"] = chunk["InstallDate"].apply(parse_excel_or_string_date)
        else:
            chunk["install_date_parsed"] = None

        chunk_file = os.path.join(CHUNK_DIR, f"chunk_{i:05d}.csv")
        try:
            chunk.to_csv(chunk_file, index=False, encoding="utf-8")
            written += len(chunk)
            print(f"Saved chunk file {chunk_file}  — written rows so far (approx): {written:,}")
        except PermissionError as e:
            print("PermissionError while writing chunk file:", e)
            print("Make sure the folder is writable and no process is locking it.")
            conn.close()
            return
        except Exception as e:
            print("Unexpected error while saving chunk file:", e)
            conn.close()
            return

    conn.close()
    print("All chunks processed (or already present). Now combining...")
    ok = combine_chunks_to_output(OUT_CSV)
    if ok:
        print("Done. Final cleaned dataset is at:", OUT_CSV)
    else:
        print("Combine failed — check permissions and that chunk files exist.")

if __name__ == "__main__":
    main()
