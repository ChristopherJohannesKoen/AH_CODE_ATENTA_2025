import sqlite3
from pathlib import Path
import db_manager as manager

DB = "atenta"
SCHEMA = Path(__file__).parent / "build2.sql"


def db_init(build_clean):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    if build_clean:
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
