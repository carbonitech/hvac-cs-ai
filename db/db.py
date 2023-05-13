"""Interface for the database with methods designed to handle embeddings, and CRUD on entities that surround the reference files"""

import ast
import psycopg2 as pg
import pandas as pd

class db:

    def __init__(self, connection: str):
        self.connection = connection

    def __enter__(self):
        self.conn = pg.connect(self.connection)
        return self
    
    def __exit__(self, e_type, e_val, e_tb):
        if e_tb:
            return False
        self.close()
        return True

    def get_embeddings(self) -> pd.DataFrame:
        sql = "SELECT text, embedding FROM embeddings;"
        with self.conn:
            with self.conn.cursor() as curr:
                curr.execute(sql) 
                result = pd.DataFrame(curr.fetchall())
                result.columns = ['text', 'embedding']
                result['embedding'] = result['embedding'].apply(ast.literal_eval)
                return result

    def post_embeddings(self, values: list[tuple|dict]) -> bool:
        success = False
        if isinstance(values[0], dict):
            sql = "INSERT INTO embeddings (text, embedding, file_id, category) VALUES (%(text)s, %(embedding)s,%(file_id)s, %(category)s)"
        elif isinstance(values[0], tuple):
            sql = "INSERT INTO embeddings (text, embedding, file_id, category) VALUES (%s, %s, %s, %s);"

        with self.conn:
            with self.conn.cursor() as curr:
                if len(values) == 1:
                    insert = curr.execute
                elif len(values) > 1:
                    insert = curr.executemany
                else:
                    success = False
                insert(sql,values)
                success = True
        return success

    def add_entity(self, entity_name: str):
        with self.conn:
            with self.conn.cursor() as curr:
                sql = "INSERT INTO entities (name) VALUES (%s) RETURNING id;"
                curr.execute(sql, (entity_name,))
                result = curr.fetchone()[0]
        return result

    def get_entities(self):
        with self.conn:
            with self.conn.cursor() as curr:
                sql = "SELECT * FROM entities;"
                curr.execute(sql)
                result = curr.fetchall()
        if result:
            entities = pd.DataFrame(result, columns=["id","name"])
        else:
            entities = pd.DataFrame()
        return entities 

    def add_file(self, filename: str, entity: int):
        with self.conn:
            with self.conn.cursor() as curr:
                sql = "INSERT INTO files (name, entity_id) VALUES (%s,%s) RETURNING id;"
                curr.execute(sql, (filename,entity))
                return curr.fetchone()[0]

    def close(self):
        self.conn.close()
