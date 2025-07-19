import sqlite3

db_path = "shared_data.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS tickets (
        project_id TEXT,
        last_updated TIMESTAMP,
        tickets TEXT,
        PRIMARY KEY (project_id)
    )
''')
cursor.execute('''
    CREATE TABLE IF NOT EXISTS updates (
        project_id TEXT,
        last_updated TIMESTAMP,
        has_changes BOOLEAN,
        PRIMARY KEY (project_id)
    )
''')
conn.commit()
conn.close()
print("SQLite database initialized at shared_data.db")