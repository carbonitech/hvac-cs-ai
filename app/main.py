"""
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

# imports - openai cookbook
import ast  # for converting embeddings saved as strings back to arrays
import openai  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search

# imports - other
from dotenv import load_dotenv  # for making environment variable OPENAI_API_KEY accessible
from PyPDF2 import PdfReader    # for loading PDf file data
import numpy as np   # for splitting page text into equal size sub-strings
import re
load_dotenv()

# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

def convert_to_tokens(text: str|list[str], model: str) -> list:
    encoding = tiktoken.encoding_for_model(model_name=model)
    if isinstance(text, str):
        return encoding.encode(text=text)
    elif isinstance(text, list):
        return encoding.encode_batch(text=text)

def read_file(file):
    """
    Reads in an entire file and splits the text into 3 equal sections per page 
    Returns: a list of text sections
    """
    reader = PdfReader(file)
    text_by_page = [page.extract_text() for page in reader.pages]
    text_by_page_split = []

    for page in text_by_page:
        # make sure special unicode characters are removed
        page = np.array(page.encode('ascii','ignore').decode().split()) 
        split_string = [" ".join(section) for section in np.array_split(page,3)]
        text_by_page_split.extend(list(split_string))

    return text_by_page_split

def create_embeddings(text: list, embeddings_model: str) -> pd.DataFrame:
    """Create embeddings for each text segment and return as a DataFrame
        The embeddings model returns a data object of this format
        {
            "object": "list",
            "data": [
                {
                "object": "embedding",
                "embedding": [
                    0.0023064255,
                    -0.009327292,
                    .... (1536 floats total for ada-002)
                    -0.0028842222,
                ],
                "index": 0
                }
            ],
            "model": "text-embedding-ada-002",
            "usage": {
                "prompt_tokens": 8,
                "total_tokens": 8
            }
        }
"""
    embeddings = []
    for segment in text:
        response = openai.Embedding.create(model=embeddings_model, input=segment)
        for i, resp_data_obj in enumerate(response["data"]):
            assert i == resp_data_obj["index"]  # double check embeddings are in same order as input
        batch_embeddings = [embedding_object["embedding"] for embedding_object in response["data"]]
        embeddings.extend(batch_embeddings)
    return pd.DataFrame({'text': text, 'embedding': embeddings})

def save_results(df: pd.DataFrame):
    save_path = './file_embeddings.csv'
    df.to_csv(save_path, index=False)