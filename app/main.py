from fastapi import FastAPI
from app.resources.files import files

app = FastAPI()
app.include_router(files)