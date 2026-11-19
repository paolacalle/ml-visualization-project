"""Open the configured PostgreSQL connection and execute SQL text or query files.

Environment keys: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD. The wrapper
loads local dotenv settings and requires TLS. Callers must close their instance.
The direct-script entrypoint runs schema creation, not a connection-only check.
"""

import psycopg
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import os

class Database:
    """Own a psycopg connection shared by query helpers and the Parquet importer."""
    
    def __init__(self, *, env_file=None):
        """Connect using environment variables and an optional dotenv file.

        Without ``env_file``, look for .env from the working directory upward.
        Supply an explicit path when running notebooks outside the checkout.
        Existing environment variables take precedence over dotenv values.
        """
        self.env_file = env_file
        self.connection = self.connect()
        
    def connect(self):
        """Load dotenv settings and return a PostgreSQL connection with TLS required.

        Missing required environment keys raise ``KeyError``; connection errors
        propagate from psycopg. Passwords are passed as arguments, not URI-encoded.
        """
        load_dotenv(
            dotenv_path=self.env_file if self.env_file is not None else find_dotenv(usecwd=True),
            interpolate=False,
        )
        connection = psycopg.connect(
            host=os.environ["DB_HOST"],
            port=os.environ["DB_PORT"],
            dbname=os.environ["DB_NAME"],
            user=os.environ["DB_USER"],
            password=os.environ["DB_PASSWORD"],
            sslmode="require",
        )
        return connection
    
    def execute(self, query, params=None):
        """Execute SQL text or a filename relative to ``database/queries``.

        A stripped string ending in ``.sql`` selects a UTF-8 query file; other
        strings are SQL text. ``params`` are bound separately by psycopg. Each
        call uses a transaction/savepoint and returns fetched tuples for a result
        set, or None for statements without results. No string interpolation of
        parameter values is needed. Query-file paths are based on this module,
        independent of the caller's working directory. SQL resources are included
        in both editable and regular package installations.
        """
        if query.strip().lower().endswith(".sql"):
            sql_path = Path(__file__).resolve().parent / "queries" / query.strip()
            sql = sql_path.read_text(encoding="utf-8")
        else:
            sql = query

        with self.connection.transaction():
            with self.connection.cursor() as cursor:
                cursor.execute(sql, params)

                if cursor.description is not None:
                    return cursor.fetchall()

    def close(self):
        """Close the owned connection and clear it; repeated calls do nothing."""
        if self.connection is not None:
            self.connection.close()
            self.connection = None
            
if __name__ == "__main__":
    db = Database()

    try:
        print(db.execute("create_schema.sql"))
        
        # print(db.execute("SELECT 1;"))
        
        # print(db.execute(
        #     "SELECT * FROM cmv.posts WHERE post_id = %s;",
        #     ("t3_example",),
        # ))
        
    finally:
        db.close()
    
    
    
    
