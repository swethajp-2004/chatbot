# test_sqlserver.py
import pyodbc
import pandas as pd
import sys

# ---- EDIT ONLY THESE VALUES ----
SERVER = "cableportalstage.westus2.cloudapp.azure.com"        # e.g. cableportalstage.westus2.cloudapp.azure.com
DATABASE = "bundle_root"    # e.g. bundle_root
USERNAME = "devswetha"
PASSWORD = "xwnpZgdX"
# --------------------------------

DRIVER = "ODBC Driver 18 for SQL Server"  # keep this exactly if you installed Driver 18
# If you installed Driver 17, change to "ODBC Driver 17 for SQL Server"

# Build connection string. Using Encrypt=yes and TrustServerCertificate=yes for testing.
conn_str = (
    f"DRIVER={{{DRIVER}}};"
    f"SERVER={SERVER};"
    f"DATABASE={DATABASE};"
    f"UID={USERNAME};"
    f"PWD={PASSWORD};"
    f"Encrypt=yes;"
    f"TrustServerCertificate=yes;"
)

print("Connecting with connection string (password hidden)...")
try:
    conn = pyodbc.connect(conn_str, timeout=10)
    print("Connected ✅")
except Exception as e:
    print("Connection failed ❌")
    print(e)
    sys.exit(1)

# Simple test query: show top 5 rows from the table used by chatbot
# Replace the table name below if different
TEST_TABLE = "dbo.OrderItemUnit"

try:
    sql = f"SELECT TOP 5 * FROM {TEST_TABLE};"
    df = pd.read_sql_query(sql, conn)
    print("\n--- Query result (first 5 rows) ---")
    print(df.head().to_string(index=False))
    print("\nColumns in returned table:")
    print(list(df.columns))
except Exception as e:
    print("Query failed:")
    print(e)
finally:
    conn.close()
