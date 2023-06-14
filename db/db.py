"""Interface for the database with methods designed to handle embeddings, and CRUD on entities that surround the reference files"""

import psycopg2 as pg
import pandas as pd
from datetime import datetime

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

    def get_embeddings(self, file_id: int|list[int] = None) -> pd.DataFrame:
        sql = "SELECT text, embedding, file_id FROM embeddings;"
        with self.conn:
            with self.conn.cursor() as curr:
                if file_id:
                    match file_id:
                        case [*ids]:
                            sql = sql.replace(';', ' WHERE file_id IN %s;')
                            curr.execute(sql, (tuple(ids),)) 
                        case _:
                            sql = sql.replace(';', ' WHERE file_id = %s;')
                            curr.execute(sql, (file_id,)) 
                else:
                    curr.execute(sql) 
                result = pd.DataFrame(curr.fetchall())
                result.columns = ['text', 'embedding', 'file_id']
                result['embedding'] = result['embedding'].apply(tuple)
                return result

    def post_embeddings(self, values: list[tuple|dict]) -> bool:
        success = False
        if isinstance(values[0], dict):
            sql = "INSERT INTO embeddings (text, embedding, file_id) VALUES (%(text)s, %(embedding)s,%(file_id)s)"
        elif isinstance(values[0], tuple):
            sql = "INSERT INTO embeddings (text, embedding, file_id) VALUES (%s, %s, %s)"
        # avoid breaking when duplicated text gets kicked back due to the uniqueness constraint in the database
        sql += " ON CONFLICT DO NOTHING;"
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

    def get_entities(self, entity_id: int|None=None, entity_name: str|None=None):
        with self.conn:
            with self.conn.cursor() as curr:
                sql = "SELECT * FROM entities;"
                if entity_id:
                    sql = sql.replace(';', ' WHERE id = %s;')
                    param = (entity_id,)
                elif entity_name:
                    sql = sql.replace(';', ' WHERE name = %s;')
                    param = (entity_name,)
                else:
                    param = None
                curr.execute(sql,param)
                result = curr.fetchall()
        if result:
            entities = pd.DataFrame(result, columns=["id","name"])
        else:
            entities = pd.DataFrame()
        return entities 

    def add_file(self, filename: str, entity: int, category: str, embedding: list[float]):
        _now = datetime.utcnow()
        with self.conn:
            with self.conn.cursor() as curr:
                sql = """INSERT INTO files (name, entity_id, category, uploaded_at, embedding)
                        VALUES (%s,%s,%s,%s,%s)
                        RETURNING id;"""
                params = (filename, entity, category, _now, embedding)
                curr.execute(sql,params)
                return curr.fetchone()[0]
    
    def get_files(self, file_id: int=0) -> pd.DataFrame:
        """Getting only file metadata, not the text-chunk embeddings"""
        with self.conn:
            with self.conn.cursor() as curr:
                sql = "SELECT id, name, entity_id, category, uploaded_at, embedding FROM files;"
                param = None
                if file_id:
                    sql = sql.replace(';', " WHERE id = %s;")
                    param = (file_id,)
                curr.execute(sql,param)
                result = pd.DataFrame(curr.fetchall(), columns=['id','name','entity_id','category','uploaded_at','embedding']) 
        return result

    def del_file(self, file_id: int):
        with self.conn:
            with self.conn.cursor() as curr:
                sql_del_embeddings = "DELETE FROM embeddings WHERE file_id = %s;"
                sql_del_file = "DELETE FROM files WHERE id = %s;"
                curr.execute(sql_del_embeddings, (file_id,))
                curr.execute(sql_del_file, (file_id,))

    def del_entity(self, entity_id: int):
        with self.conn:
            with self.conn.cursor() as curr:
                sub_query = "SELECT files.id FROM files JOIN entities ON entities.id = files.entity_id WHERE entities.id = %s"
                sql = f'DELETE FROM embeddings WHERE file_id IN ({sub_query});'
                curr.execute(sql, (entity_id,))
                sql_del_files = f"DELETE FROM files WHERE id IN ({sub_query})"
                curr.execute(sql_del_files, (entity_id,))
                sql_del_entity = "DELETE FROM entities WHERE id = %s;"
                curr.execute(sql_del_entity, (entity_id,))


    def close(self):
        self.conn.close()
