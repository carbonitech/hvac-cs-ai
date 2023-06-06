from fastapi import FastAPI
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

@app.get('/test')
def test_route(param1: str):
    return {'data': param1}