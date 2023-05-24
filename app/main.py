from fastapi import FastAPI
from app.resources.files import files
from app.resources.chat import chat

app = FastAPI()
app.include_router(files)
app.include_router(chat)