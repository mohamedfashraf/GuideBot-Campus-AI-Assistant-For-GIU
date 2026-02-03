import os
import sqlite3
from datetime import datetime


def ensure_dirs(base_dir):
    data_dir = os.path.join(base_dir, "data")
    pdf_dir = os.path.join(data_dir, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)
    return data_dir, pdf_dir


def init_db(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_type TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            source TEXT,
            updated_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts
        USING fts5(title, content, content='documents', content_rowid='id')
        """
    )

    cur.execute(
        """
        CREATE TRIGGER IF NOT EXISTS documents_ai
        AFTER INSERT ON documents BEGIN
          INSERT INTO documents_fts(rowid, title, content)
          VALUES (new.id, new.title, new.content);
        END
        """
    )

    cur.execute(
        """
        CREATE TRIGGER IF NOT EXISTS documents_ad
        AFTER DELETE ON documents BEGIN
          INSERT INTO documents_fts(documents_fts, rowid, title, content)
          VALUES('delete', old.id, old.title, old.content);
        END
        """
    )

    cur.execute(
        """
        CREATE TRIGGER IF NOT EXISTS documents_au
        AFTER UPDATE ON documents BEGIN
          INSERT INTO documents_fts(documents_fts, rowid, title, content)
          VALUES('delete', old.id, old.title, old.content);
          INSERT INTO documents_fts(rowid, title, content)
          VALUES (new.id, new.title, new.content);
        END
        """
    )

    conn.commit()
    conn.close()


def seed_default_data(db_path):
    now = datetime.utcnow().isoformat()
    docs = [
        (
            "admissions",
            "Admissions Office",
            "Admissions handles applications, registration, and enrollment. Visit the Admissions office in Building M.",
            "campus",
            now,
        ),
        (
            "financial",
            "Financial Office",
            "Financial Office assists with tuition payment, billing, and scholarships.",
            "campus",
            now,
        ),
        (
            "student_affairs",
            "Student Affairs",
            "Student Affairs helps with course enrollment, add/drop, and schedules.",
            "campus",
            now,
        ),
        (
            "faq",
            "General Help",
            "You can ask about rooms, doctor availability, admissions, financial services, or navigation.",
            "campus",
            now,
        ),
    ]

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM documents")
    if cur.fetchone()[0] == 0:
        cur.executemany(
            """
            INSERT INTO documents (doc_type, title, content, source, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            docs,
        )
        conn.commit()
    conn.close()


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir, _ = ensure_dirs(base_dir)
    db_path = os.path.join(data_dir, "guidebot.db")
    init_db(db_path)
    seed_default_data(db_path)
    print(f"Initialized RAG DB at: {db_path}")


if __name__ == "__main__":
    main()
