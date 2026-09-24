import sqlite3
import json
from encoder import create_embedding

DATABASE_NAME = "database.db"


def connect():
    return sqlite3.connect(DATABASE_NAME)


def create_tables():
    conn = connect()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            image_path TEXT,
            embedding TEXT NOT NULL,
            FOREIGN KEY(person_id) REFERENCES people(id)
        )
    """)

    conn.commit()
    conn.close()


def add_person(name):
    conn = connect()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT OR IGNORE INTO people(name) VALUES(?)",
        (name,)
    )

    conn.commit()
    conn.close()


def add_embedding(name, image_path):
    conn = connect()
    cursor = conn.cursor()

    # Make sure the person exists
    cursor.execute(
        "INSERT OR IGNORE INTO people(name) VALUES(?)",
        (name,)
    )

    # Find person's id
    cursor.execute(
        "SELECT id FROM people WHERE name=?",
        (name,)
    )

    person_id = cursor.fetchone()[0]

    # Create embedding
    embedding = create_embedding(image_path)

    # Convert list -> JSON
    embedding_json = json.dumps(embedding)

    # Store it
    cursor.execute("""
        INSERT INTO embeddings(person_id, image_path, embedding)
        VALUES (?, ?, ?)
    """, (person_id, str(image_path), embedding_json))

    conn.commit()
    conn.close()

def get_known_people():
    conn = connect()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT name
        FROM people
        ORDER BY name
    """)

    people = [row[0] for row in cursor.fetchall()]

    conn.close()

    return people


def load_database():
    conn = connect()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT people.name, embeddings.embedding
        FROM embeddings
        JOIN people
        ON people.id = embeddings.person_id
    """)

    database = []

    for name, embedding_json in cursor.fetchall():
        database.append({
            "name": name,
            "embedding": json.loads(embedding_json)
        })

    conn.close()

    return database
create_tables()