from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
from app.resources.files import files
from app.resources.chat import chat

app = FastAPI()
app.include_router(files)
app.include_router(chat)

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/test')
async def add_file(
        entity: str,
        category: str,
        name: str,
        file: bytes = File()
    ):

    print(file[:10])
    return {entity, category, name}