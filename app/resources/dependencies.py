import os
from db.db import db
from ai.ai import AI

EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
GPT_MODEL = os.getenv('GPT_MODEL')

def get_ai():
    database = db(connection=os.getenv('DATABASE_URL'))
    yield AI(
        embedding_model_name=EMBEDDING_MODEL,
        gpt_model_name=GPT_MODEL,
        database=database
    )

def get_db():
    database = db(connection=os.getenv('DATABASE_URL'))
    yield database