import psycopg
from urllib.parse import quote
from dotenv import load_dotenv
from pathlib import Path
import os

class Database:
    """Safely create a DB connection."""
    
    def __init__(self):
        self.connection = self.connect()
        
    def clean_pass(self, password):
        return quote(password)

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
    
    def execute(self, sql_filename):
        sql_path = (
            Path(__file__).resolve().parent
                / "queries"
                / sql_filename
        )
        
        sql = Path(sql_path).read_text(encoding="utf-8")

        with self.connection.transaction():
            with self.connection.cursor() as cursor:
                cursor.execute(sql)
                
                # has something to return
                if cursor.description is not None:
                    return cursor.fetchall()

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
            
if __name__ == "__main__":
    db = Database()

    try:
        print(db.execute("test.sql"))
    finally:
        db.close()
    
    
    
    
