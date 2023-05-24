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
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search

# project module imports
from fastapi import FastAPI
from app.resources.files import files
app = FastAPI()

def convert_to_tokens(text: str|list[str], model: str) -> list:
    encoding = tiktoken.encoding_for_model(model_name=model)
    if isinstance(text, str):
        return encoding.encode(text=text)
    elif isinstance(text, list):
        return encoding.encode_batch(text=text)

app.include_router(files)