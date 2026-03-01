import sqlite3
from pathlib import Path
try:
    from . import db_manager as manager
except ImportError:
    import db_manager as manager

DB = str(Path(__file__).resolve().with_name("atenta"))
SCHEMA = Path(__file__).parent / "build2.sql"


def _table_exists(cur, table_name):
    cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?;",
        (table_name,),
    )
    return cur.fetchone() is not None


def db_init(build_clean):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    needs_schema = build_clean or not (
        _table_exists(cur, "template") and _table_exists(cur, "session")
    )
    if needs_schema:
        schema = SCHEMA.read_text(encoding="utf-8")
        cur.executescript(schema)

    print("DB init")

    path_templates = Path(__file__).resolve().parent / "../Model/Templates"
    json_files = list(path_templates.glob("*.json"))

    conn.commit()
    conn.close()
    for x in json_files:
        manager.add_template(x)


if __name__ == "__main__":
    db_init(True)
