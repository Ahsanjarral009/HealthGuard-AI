import pyodbc

def get_connection():
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=Development\\SQLEXPRESS;"
        "DATABASE=HealthGuard_AI;"
        "UID=sa;"
        "PWD=ahsan;"
        "TrustServerCertificate=yes;"
    )
    return conn
def fetch_data(query):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

