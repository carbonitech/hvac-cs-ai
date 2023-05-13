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
    file_path = '/home/carboni/projects/hvac-cs-ai/5e133f6d27f35743210648.pdf'
    entity = 'ADP'
    category = 'Warranty'
    file = File(entity=entity, category=category, file_path=file_path)
    ai = TestAI(EMBEDDING_MODEL,GPT_MODEL,database)
    embeddings_table: pd.DataFrame = ai.generate_embeddings_table(file=file)
    # check that the data has been chunked in processing
    assert len(embeddings_table) == file.num_pages*3
    # check that there are no NaN values
    assert not embeddings_table.isnull().values.any()


def test__register_file_with_the_database():
    database = db(connection=getenv('DATABASE_URL'))
    file_path = '/home/carboni/projects/hvac-cs-ai/5e133f6d27f35743210648.pdf'
    entity = 'ADP'
    category = 'Warranty'
    file = File(entity=entity, category=category, file_path=file_path)
    ai = TestAI(EMBEDDING_MODEL,GPT_MODEL,database)
    file_id = ai._register_file_with_the_database(file=file)
    # check that an id has been returned
    assert file_id
    assert isinstance(file_id, int)
    # check that this file id has been recorded under the expected file name 
    with database as session:
        file_record = session.get_file(file_id=file_id)
    assert file_id == file_record['id']
    assert file.file_name() == file_record['name']
    
    # delete the data created
    with database as session:
        session.del_file(file_id=file_id)

def test_save_embeddings():
    database = db(connection=getenv('DATABASE_URL'))
    file_path = '/home/carboni/projects/hvac-cs-ai/5e133f6d27f35743210648.pdf'
    entity = 'ADP'
    category = 'Warranty'
    file = File(entity=entity, category=category, file_path=file_path)
    ai = TestAI(EMBEDDING_MODEL,GPT_MODEL,database)
    embeddings_table: pd.DataFrame = ai.generate_embeddings_table(file=file)
    save_successful = ai.save_embeddings(embeddings_table)
    # check for the successful save
    assert save_successful

    file_id = int(embeddings_table["file_id"].iat[0])
    with database as session:
        records = session.get_embeddings(file_id=file_id)
    
    # embeddings_table had its embedding column converted to str in the save_embeddings method 
    records['embedding'] = records['embedding'].astype(str)
    # both dfs need the embeddings column as a str (hashable type) for this merge to work
    merged = embeddings_table[['text','embedding']].merge(records, on=['text','embedding'], indicator=True)
    # care only that the df's contain all of the same values, nothing else
    assert (merged['_merge'] == 'both').all()

    with database as session:
        session.del_file(file_id=file_id)