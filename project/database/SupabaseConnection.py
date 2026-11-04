import psycopg
from dotenv import load_dotenv
from pathlib import Path
import os

class Database:
    """Safely create a DB connection."""
    
    def __init__(self):
        self.connection = self.connect()
        
    def connect(self):
        load_dotenv()
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
    
    
    
    
