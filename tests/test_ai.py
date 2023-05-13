from ai.ai import AI
from random import random, randint
from app.main import EMBEDDING_MODEL, GPT_MODEL
import pandas as pd
from app.file_handler import File
from db.db import db
from os import getenv

class TestAI(AI):

    def _create_embedding(self, text):
        """override to fake call to OpenAI
        The structure of the embedding matches tne expected response from OpenAI,
        but the embedding values are randomly generated
        """

        embedding = {
            'object': 'list',
            'data': [
                {
                    'object': 'embedding',
                    'embedding': [
                        (random()*(10**-(randint(2,3))))*((-1)**randint(1,2))
                        for i in range(1536)
                    ],
                    'index': 0
                }
            ],
            'model': EMBEDDING_MODEL,
            'usage': {
                'prompt_tokens': 99999,
                'total_tokens': 99999
            }
        }
        return embedding


def test_generate_embeddings_table():
    database = db(connection=getenv('DATABASE_URL'))
    ai = TestAI(EMBEDDING_MODEL,GPT_MODEL,database)
    file_path = '/home/carboni/projects/hvac-cs-ai/5e133f6d27f35743210648.pdf'
    entity = 'ADP'
    category = 'Warranty'
    file = File(entity=entity, category=category, file_path=file_path)
    embeddings_table: pd.DataFrame = ai.generate_embeddings_table(file=file)
    assert len(embeddings_table) == file.num_pages*3
