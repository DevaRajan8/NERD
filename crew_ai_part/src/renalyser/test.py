import psycopg2

try:
    pg_conn = psycopg2.connect("postgresql://postgres:devarajan#8@localhost:5432/demo")
    print("Connection successful!")
    pg_conn.close()
except Exception as e:
    print(f"Connection failed: {e}")
