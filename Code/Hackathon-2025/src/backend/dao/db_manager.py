import db_init as util
import sqlite3
import json

DB = util.DB


def query(state):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute(state)

    conn.commit()
    conn.close()


def query_fetch(state):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute(state)
    tables = cur.fetchall()

    conn.commit()
    conn.close()
    return tables


def query_fetch_one(state):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute(state)
    row = cur.fetchone()
    conn.commit()
    conn.close()
    return row[0]


def add_template(filename):
    temp = filename.read_text(encoding="utf-8")
    query(
        "INSERT INTO template(name, data) values('"
        + filename.stem
        + "', '"
        + temp
        + "');"
    )


def get_template(name):
    lis = query_fetch_one("SELECT data FROM template WHERE name='" + name + "';")
    print(lis)
    return json.loads(lis)


def get_template_names():
    return query_fetch("SELECT name FROM template")


def save_template_json(name, json_object):
    query(
        "INSERT INTO template(name, data) values('"
        + name
        + "', '"
        + json.dumps(json_object)
        + "');"
    )


def save_session(patient_number, data):
    query(
        "INSERT INTO session(patient_number, data) values('"
        + patient_number
        + "', '"
        + json.dumps(data)
        + "');"
    )
    return


def overwrite_data(id, data):
    query("UPDATE session SET data = '" + json.dumps(data) + "' WHERE id = " + id + ";")


def get_session_meta(patient_number):
    return query_fetch(
        "SELECT id, date_time FROM session WHERE patient_number='"
        + patient_number
        + "';"
    )


def generate_txt(structured_data):
    output = [""]
    for key, value in structured_data.items():
        output.append(f"{key}: {value if value else 'Not specified'}")
    return "\n".join(output)
