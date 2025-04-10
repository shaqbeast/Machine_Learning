import psycopg2
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

class PostgresDB:
    def __init__(self, dbname=None, user=None, password=None, host=None, port=None):
        """Initialize database connection using environment variables as fallbacks"""
        load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "database.env"))

        self.dbname = dbname or os.getenv("DB_NAME")
        user = user or os.getenv("DB_USER")
        password = password or os.getenv("DB_PASSWORD")
        host = host or os.getenv("DB_HOST", "localhost")
        port = port or os.getenv("DB_PORT", "5432")

        if not self.dbname:
            raise ValueError("Database name must be provided either as a parameter or in .env as DB_NAME")

        self.conn = None
        self.cur = None
        self._connect(user, password, host, port)

    def _connect(self, user, password, host, port):
        try:
            self.conn = psycopg2.connect(
                dbname=self.dbname,
                user=user,
                password=password,
                host=host,
                port=port,
            )
            self.cur = self.conn.cursor()
            print(f"Connected to database: {self.dbname} at {host}:{port}")
        except Exception as e:
            print(f"Database connection failed: {e}")
            raise

    def _ensure_cursor(self):
        if self.conn.closed or self.cur.closed:
            print("Reconnecting due to closed cursor...")
            self._connect(
                os.getenv("DB_USER"),
                os.getenv("DB_PASSWORD"),
                os.getenv("DB_HOST"),
                os.getenv("DB_PORT", "5432"),
            )

    def execute_query(self, query, values=None, fetch=False):
        """Execute a SQL query (with optional values)"""
        self._ensure_cursor()
        try:
            self.cur.execute(query, values if values else ())
            self.conn.commit()
            if fetch:
                return self.cur.fetchall()
        except Exception as e:
            self.conn.rollback()
            print(f"Error executing query: {e}")

    def insert_bulk(self, table, data, batch_size=256):
        """Insert data into a table in chunks using Pandas DataFrame, list of tuples, or dict"""
        def to_python_types(row):
            return tuple(
                (v.item() if isinstance(v, (np.generic, np.bool_)) else v)
                for v in row
            )

        if isinstance(data, pd.DataFrame):
            records = data.to_records(index=False)
            all_values = [to_python_types(row) for row in records]
            columns = ", ".join(data.columns)
        elif isinstance(data, list) and all(isinstance(i, tuple) for i in data):
            all_values = [to_python_types(row) for row in data]
            columns = None
        else:
            raise ValueError("Unsupported data type")

        placeholders = ", ".join(["%s"] * len(all_values[0]))
        query = f'INSERT INTO "{table}" ({columns}) VALUES ({placeholders})' if columns else f'INSERT INTO "{table}" VALUES ({placeholders})'

        total = len(all_values)
        for i in range(0, total, batch_size):
            self._ensure_cursor()
            chunk = all_values[i:i+batch_size]
            try:
                self.cur.executemany(query, chunk)
                self.conn.commit()
                percent = int(((i + len(chunk)) / total) * 100)
                print(f"✅ Staged {i + len(chunk)} of {total} rows — {percent}% complete")
            except Exception as e:
                self.conn.rollback()
                print(f"❌ Error inserting batch starting at row {i}: {e}")
                break

    def create_table(self, table_name, columns):
        """Create a table in the database based on a column dictionary"""
        columns_query = ", ".join([f"{col} {dtype}" for col, dtype in columns.items()])
        query = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({columns_query})'

        try:
            self.cur.execute(query)
            self.conn.commit()
            print(f"✅ Table '{table_name}' created successfully.")
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Error creating table: {e}")

    def stage(self, table_name, csv_path):
        """
        Create table from CSV schema and insert data in chunks
        """
        try:
            df = pd.read_csv(csv_path)

            def infer_sql_type(dtype):
                if pd.api.types.is_integer_dtype(dtype):
                    return "INT"
                elif pd.api.types.is_float_dtype(dtype):
                    return "FLOAT"
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    return "TIMESTAMP"
                else:
                    return "TEXT"

            columns = {
                col: infer_sql_type(dtype)
                for col, dtype in df.dtypes.items()
            }

            self.create_table(table_name, columns)
            self.insert_bulk(table_name, df, batch_size=256)

            print(f"🎉 Finished staging CSV '{csv_path}' into table '{table_name}'")
        except Exception as e:
            print(f"❌ Failed to stage CSV '{csv_path}' into table '{table_name}': {e}")
    
    
    def fetch_dataframe(self, query, table_name=None):
        """
        Executes a SQL query and returns the result as a Pandas DataFrame.
        Optionally includes the table name for display/logging.
        """
        self._ensure_cursor()
        try:
            self.cur.execute(query)
            colnames = [desc[0] for desc in self.cur.description]
            rows = self.cur.fetchall()
            df = pd.DataFrame(rows, columns=colnames)
            if table_name:
                print(f"📥 Pulled {len(df)} rows from '{table_name}'")
            return df
        except Exception as e:
            print(f"❌ Failed to fetch data{f' from {table_name}' if table_name else ''}: {e}")
            return pd.DataFrame()

    def close(self):
        """Close the database connection"""
        self.cur.close()
        self.conn.close()
        print(f"🔒 Connection to {self.dbname} closed")
