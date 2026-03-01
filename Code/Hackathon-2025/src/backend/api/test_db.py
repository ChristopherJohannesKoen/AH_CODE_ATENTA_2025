try:
    from . import db_init
    from . import db_manager as manager
except ImportError:
    import db_init
    import db_manager as manager

if __name__ == "__main__":
    db_init.db_init(False)
    sessions = manager.query_fetch("SELECT * FROM session;")
    print(sessions)
