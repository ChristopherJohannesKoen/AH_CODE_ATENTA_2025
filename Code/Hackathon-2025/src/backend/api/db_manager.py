import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

DB = str(Path(__file__).resolve().with_name("atenta"))


def _run_sql(
    statement: str,
    params: Iterable[Any] = (),
    *,
    fetch_one: bool = False,
    fetch_all: bool = False,
):
    with sqlite3.connect(DB) as conn:
        cur = conn.cursor()
        cur.execute(statement, tuple(params))
        if fetch_all:
            return cur.fetchall()
        if fetch_one:
            return cur.fetchone()
        return cur.lastrowid


def query(state, params: Iterable[Any] = ()):
    return _run_sql(state, params)


def query_fetch(state, params: Iterable[Any] = ()):
    return _run_sql(state, params, fetch_all=True)


def query_fetch_one(state, params: Iterable[Any] = ()):
    row = _run_sql(state, params, fetch_one=True)
    return row[0] if row else None


def add_template(filename):
    template_content = filename.read_text(encoding="utf-8")
    query(
        "INSERT OR IGNORE INTO template(name, data) VALUES(?, ?);",
        (filename.stem, template_content),
    )


def get_template(name):
    template_row = query_fetch_one("SELECT data FROM template WHERE name = ?;", (name,))
    if template_row is None:
        return {}
    return json.loads(template_row)


def get_template_names():
    return query_fetch("SELECT name FROM template;")


def save_template_json(name, json_object):
    query(
        "INSERT INTO template(name, data) VALUES(?, ?)"
        " ON CONFLICT(name) DO UPDATE SET data = excluded.data;",
        (name, json.dumps(json_object)),
    )


def save_session(patient_number, data):
    return query(
        "INSERT INTO session(patient_number, data) VALUES(?, ?);",
        (patient_number, json.dumps(data)),
    )


def overwrite_data(session_id, data):
    query(
        "UPDATE session SET data = ? WHERE id = ?;",
        (json.dumps(data), int(session_id)),
    )


def get_session_meta(patient_number):
    return query_fetch(
        "SELECT id, date_time FROM session WHERE patient_number = ?;",
        (patient_number,),
    )


def get_data_set(patient_number):
    return query_fetch(
        "SELECT date_time, data FROM session WHERE patient_number = ?"
        " ORDER BY date_time DESC;",
        (patient_number,),
    )


def get_session_data(session_id):
    return query_fetch_one("SELECT data FROM session WHERE id = ?;", (int(session_id),))


def generate_txt(structured_data):
    output = [""]
    for key, value in structured_data.items():
        output.append(f"{key}: {value if value else 'Not specified'}")
    return "\n".join(output)
