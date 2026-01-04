import psycopg2
import pandas as pd

from sqlalchemy import create_engine, text
import pandas as pd

class PostgresReader:
    def __init__(self, dsn: str):
        # Convert DSN to SQLAlchemy URL format
        self.dsn = dsn
        self.engine = create_engine(
            dsn.replace(" ", "+"),  # handles spaces safely
            pool_size=10,
            max_overflow=5
        )

    def fetch(self, query: str) -> pd.DataFrame:
        with self.engine.connect() as conn:
            return pd.read_sql(text(query), conn)
