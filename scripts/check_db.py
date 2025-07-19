import sqlite3
import json

conn = sqlite3.connect("shared_data.db")
cursor = conn.cursor()
cursor.execute("SELECT project_id, last_updated, tickets FROM tickets WHERE project_id = 'PROJ123'")
result = cursor.fetchone()
conn.close()

if result:
    project_id, last_updated, tickets_json = result
    tickets = json.loads(tickets_json)
    print(f"Project: {project_id}")
    print(f"Last Updated: {last_updated}")
    print(f"Number of Tickets: {len(tickets)}")
    for ticket in tickets:
        print(f"- {ticket['key']}: Status = {ticket['fields']['status']['name']}")
else:
    print("No data found for PROJ123")