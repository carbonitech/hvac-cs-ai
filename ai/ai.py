"""Interface for the AI"""
from dotenv import load_dotenv; load_dotenv()        # expose the OPEN_API_KEY before the openai module attempts to pull it in __init__
import openai
import pandas as pd
from db.db import db
from app.file_handler import File
from scipy import spatial  # for calculating vector similarities for search

class AI:

    def __init__(self, embedding_model_name: str, gpt_model_name: str, database: db) -> None:
        self.embedding_model_name = embedding_model_name
        self.gpt_model_name = gpt_model_name
        self.db = database

    def _create_embedding(self, text):
        """encapsulate the embeddings creation call -- enables mocking for tests"""
        return openai.Embedding.create(
            model=self.embedding_model_name,
            input=text
        )

    def generate_embeddings_table(self, file: File) -> pd.DataFrame:
        """Takes a file and creates an embeddings table"""
        embeddings = []
        segments = file.read_and_chunk(3)
        for segment in segments:
            response = self._create_embedding(segment)
            for i, resp_data_obj in enumerate(response["data"]):
                assert i == resp_data_obj["index"]  # double check embeddings are in same order as input
            batch_embeddings = [embedding_object["embedding"] for embedding_object in response["data"]]
            embeddings.extend(batch_embeddings)
        result = pd.DataFrame({'text': segments, 'embedding': embeddings})
        result['file_id'] = self._register_file_with_the_database(file=file)
        return result
    
    def _register_file_with_the_database(self, file: File) -> int:
        """Checks against the database for an existing entity BY NAME. If it doesn't exist, create it
            Either way, add the file to the database under an entity"""
        entity = file.entity
        category = file.category
        with self.db as session:
            entities = session.get_entities(entity_name=entity)
            if not entities.empty:
                id_val = entities.loc[entities["name"] == entity,"id"].item()
                file_id = session.add_file(file.file_name(), id_val, category)
            else:
                entity_id = session.add_entity(entity)
                file_id = session.add_file(file.file_name(), entity=entity_id, category=category)
        return file_id

    
    def save_embeddings(self, df: pd.DataFrame) -> bool:
        # convert embeddings to strings so they aren't converted to python sets by Postgres
        # NOTE this change affects changes the state of the df inputted
        df["embedding"] = df["embedding"].apply(str)
        records: list[dict] = df.to_dict(orient="records") 
        with self.db as session:
            success = session.post_embeddings(records)
        return success

    def ranked_strings_by_relatedness(
            query: str,
            relatedness_fn = lambda x, y: 1 - spatial.distance.cosine(x,y),
            top_n: int = 100
    ) -> tuple[list[str], list[float]]:
        """Returns a list of strings and relatedness, sorted from most related to least"""