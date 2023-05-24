from ai.ai import AI
from random import random, randint
from app.resources.dependencies import EMBEDDING_MODEL, GPT_MODEL
import pandas as pd
from app.file_handler import File
from db.db import db
from os import getenv

class TestAI(AI):

    __test__ = False

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

    def _complete_chat(self, messages: list[dict[str, str]]):
        """override to fake call to OpenAI
        The structure of the response matches tne expected response from OpenAI
        """
        response = {
            'id': 'testid',
            'object': 'testobj',
            'created': 999999,
            'model': GPT_MODEL,
            'usage': {'prompt_tokens': 99, 'completion_tokens': 99, 'total_tokens': 99*2},
            'choices': [
                {
                    'message': {
                        'role': 'tester',
                        'content': 'this is a test'
                    }, 
                    'finish_reason': 'test complete',
                    'index': 0
                }
            ]
        }
        return response


def test_generate_embeddings_table():
    # first test with a file path
    database = db(connection=getenv('DATABASE_URL'))
    file_path = '/home/carboni/projects/hvac-cs-ai/5e133f6d27f35743210648.pdf'
    entity = 'ADP'
    category = 'Warranty TEST'
    file = File(entity=entity, category=category, file_path=file_path)

    file_ids = []

    ai = TestAI(EMBEDDING_MODEL,GPT_MODEL,database)
    embeddings_table: pd.DataFrame = ai.generate_embeddings_table(file=file)
    # check that the data has been chunked in processing
    assert len(embeddings_table) == file.num_pages*3
    # check that there are no NaN values
    assert not embeddings_table.isnull().values.any()
    file_ids.append(int(embeddings_table.at[0,"file_id"]))

    # now test with file bytes
    with open(file_path, 'rb') as handler:
        file_data = handler.read()
    file = File(entity=entity, category=category, file_data=file_data)
    embeddings_table: pd.DataFrame = ai.generate_embeddings_table(file=file)
    # check that the data has been chunked in processing
    assert len(embeddings_table) == file.num_pages*3
    # check that there are no NaN values
    assert not embeddings_table.isnull().values.any()
    file_ids.append(int(embeddings_table.at[0,"file_id"]))

    # delete the data created
    for file_id in file_ids:
        delete_data(database=database, file_id=file_id)

def test__register_file_with_the_database():
    database = db(connection=getenv('DATABASE_URL'))
    file_path = '/home/carboni/projects/hvac-cs-ai/5e133f6d27f35743210648.pdf'
    entity = 'ADP'
    category = 'Warranty TEST'
    file = File(entity=entity, category=category, file_path=file_path)
    ai = TestAI(EMBEDDING_MODEL,GPT_MODEL,database)
    file_id = ai._register_file_with_the_database(file=file)
    # check that an id has been returned
    assert file_id and isinstance(file_id, int)
    # check that this file id has been recorded under the expected file name 
    with database as session:
        file_record = session.get_files(file_id=file_id).loc[0]
    assert file_id == file_record['id']
    assert file.file_name() == file_record['name']
    
    delete_data(database=database, file_id=file_id)

def test_save_embeddings():
    database = db(connection=getenv('DATABASE_URL'))
    file_path = '/home/carboni/projects/hvac-cs-ai/5e133f6d27f35743210648.pdf'
    entity = 'ADP'
    category = 'Warranty TEST'
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
    merged = embeddings_table[['text','embedding']].merge(records[['text','embedding']], on=['text','embedding'], indicator=True)
    # care only that the df's contain all of the same values, nothing else
    assert (merged['_merge'] == 'both').all()

    delete_data(database=database, file_id=file_id)
    
def test_num_tokens():
    database = db(connection=getenv('DATABASE_URL'))
    test_str = "This is a test string of words that don't really mean anything but represents a non-zero number of tokens."
    test_str_list = [test_str[:len(test_str)], test_str[len(test_str):]]
    ai = TestAI(EMBEDDING_MODEL,GPT_MODEL,database)
    result = ai.num_tokens(text=test_str)
    assert isinstance(result, int) and result > 0
    result = ai.num_tokens(text=test_str_list)
    assert isinstance(result, int) and result > 0

def test_build_query_message():
    ## setup ##
    database = db(connection=getenv('DATABASE_URL'))
    file_path = '/home/carboni/projects/hvac-cs-ai/5e133f6d27f35743210648.pdf'
    entity = 'ADP'
    category = 'Warranty TEST'
    file = File(entity=entity, category=category, file_path=file_path)
    ai = TestAI(EMBEDDING_MODEL,GPT_MODEL,database)
    embeddings_table: pd.DataFrame = ai.generate_embeddings_table(file=file)
    ai.save_embeddings(embeddings_table)

    query = 'What does a test question look like?'
    full_msg = ai.build_query_message(query=query)
    assert len(full_msg) != 0 and full_msg.__contains__('Document segment:')
    assert full_msg.__contains__('Question:')
    delete_data(database=database, file_id=int(embeddings_table.at[0,"file_id"]))

def delete_data(database: db, file_id: int):
    with database as session:
        session.del_file(file_id=file_id)