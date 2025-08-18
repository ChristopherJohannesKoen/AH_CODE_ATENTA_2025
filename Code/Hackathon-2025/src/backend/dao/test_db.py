import sqlite3

import db_manager as manager

names = manager.query_fetch("select * from session")
print(names)
