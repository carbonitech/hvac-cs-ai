"""Interface for the AI

Borrowing from the OpenAI Cookbook Question/Answering example using embeddings

https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb

Full procedure

1. Prepare search data (once)
    A. Collect: We'll download a few hundred Wikipedia articles about the 2022 Olympics
    B.Chunk: Documents are split into short, mostly self-contained sections to be embedded
    C.Embed: Each section is embedded with the OpenAI API
    D.Store: Embeddings are saved (for large datasets, use a vector database)
2. Search (once per query)
    A.Given a user question, generate an embedding for the query from the OpenAI API
    B.Using the embeddings, rank the text sections by relevance to the query
3. Ask (once per query)
    A.Insert the question and the most relevant sections into a message to GPT
    B.Return GPT's answer
    There's no perfect recipe for splitting text into sections.

Regarding the tradeoffs of document splitting strategy gernerally
    -   Longer sections may be better for questions that require more context
    -   Longer sections may be worse for retrieval, as they may have more topics muddled together
    -   Shorter sections are better for reducing costs (which are proportional to the number of tokens)
    -   Shorter sections allow more sections to be retrieved, which may help with recall
    -   Overlapping sections may help prevent answers from being cut by section boundaries
"""
from dotenv import load_dotenv; load_dotenv()        # expose the OPEN_API_KEY before the openai module attempts to pull it in __init__
import openai
import pandas as pd
from db.db import db
from app.file_handler import File
from scipy import spatial  # for calculating vector similarities for search
import tiktoken  # for counting tokens

class AI:

    def __init__(
            self,
            embedding_model_name: str,
            gpt_model_name: str,
            database: db,
            token_limit: int=4096-500
        ) -> None:
        self.embedding_model_name = embedding_model_name
        self.gpt_model_name = gpt_model_name
        self.db = database
        self.token_limit = token_limit

    def _create_embedding(self, text):
        """
            Call the OpenAI embeddings engine to get eembedding values for a string
            Encapsulating this allows for testing with a mock substitute
            (OpenAI charges per token)
        """
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
            self,
            query: str,
            doc_embeddings: pd.DataFrame,
            relatedness_fn = lambda x, y: 1 - spatial.distance.cosine(x,y),
            top_n: int = 5
    ) -> pd.DataFrame:
   #) -> tuple[list[str], list[float]]:
        """Returns a dataframe of the top_n document embeddings in the database, scored and ranked in descending order
            The dataframe will have columns upon return -> text, embedding, file_id, score"""
        query_embedding_resp = self._create_embedding(query)
        query_embedding = query_embedding_resp['data'][0]['embedding']
        doc_embeddings['score'] = doc_embeddings['embedding'].apply(lambda emb: relatedness_fn(query_embedding, emb))
        doc_embeddings = doc_embeddings.sort_values(by='score', ascending=False)
        return doc_embeddings.head(n=top_n)

    def num_tokens(self, text: str|list[str]) -> int:
        encoding = tiktoken.encoding_for_model(model_name=self.gpt_model_name)
        if isinstance(text, str):
            tokens = encoding.encode(text=text)
            num_tokens = len(tokens)
        elif isinstance(text, list):
            tokens = encoding.encode_batch(text=text)
            num_tokens = 0
            for batch in tokens:
                num_tokens += len(batch)
        return num_tokens
    
    def build_query_message(self, query: str) -> str:
        """Build a message for GPT with an introduction, relevant source texts, and the question/query to answer"""
        with self.db as session:
            all_file_embeddings = session.get_embeddings()
        top_related = self.ranked_strings_by_relatedness(query=query, doc_embeddings=all_file_embeddings)
        introduction = 'Use the document segments below provided by vendors that explain topics such as warranty policies, product specifications, and installation instructions to answer the subsequent question. If the answer cannot be found in these document segments, write \"Sorry, I could not find an answer.\"\n'
        question = f"\n\nQuestion: {query}"
        full_message = introduction
        for row in top_related.itertuples():
            next_segment = f'\n\nDocument segment:\n"""\n{row.text}'
            if self.num_tokens(full_message + next_segment + question) > self.token_limit:
                break
            else:
                full_message += next_segment
        return full_message + question
    
    def _complete_chat(self, messages: list[dict[str,str]]):
        """
        Response format:
        {
            'id': str,
            'object': str,
            'created': int,
            'model': str,
            'usage': {'prompt_tokens': int, 'completion_tokens': int, 'total_tokens': int}
            'choices': [
                {
                    'message': {
                        'role': str,
                        'content': str
                    }, 
                    'finish_reason': str,
                    'index': int
                }
            ]
        }
        """
        return openai.ChatCompletion.create(
            model=self.gpt_model_name,
            messages=messages,
            temperature=0
        )
    def ask_model(self, query: str, print_msg: bool=False) -> str:
        """Answers a question (query) using GPT and a database of relevant text segments from uploaded vendor documents"""
        message = self.build_query_message(query=query)
        messages = [
            {"role": "system", "content": "You answer questions about vendor documents in the HVAC industry, which are typically handled by customer service agents"},
            {"role": "user", "content": message}
        ]
        if print_msg:
            print(message)
        response = self._complete_chat(messages=messages)
        return response['choices'][0]['message']['content']