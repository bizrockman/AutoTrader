import sqlite3

conn = sqlite3.connect("autotrader.db")

tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print(f"Tables: {[r[0] for r in tables]}")

count = conn.execute("SELECT COUNT(*) FROM strategies").fetchone()[0]
print(f"Strategies: {count}")

# Check if there are any trade records
for t in tables:
    name = t[0]
    c = conn.execute(f"SELECT COUNT(*) FROM [{name}]").fetchone()[0]
    print(f"  {name}: {c} rows")

conn.close()
