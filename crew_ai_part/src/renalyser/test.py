import psycopg2

try:
    pg_conn = os.getenv("psql")
    print("Connection successful!")
    pg_conn.close()
except Exception as e:
    print(f"Connection failed: {e}")
