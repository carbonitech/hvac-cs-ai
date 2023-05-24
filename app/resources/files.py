from app.resources.dependencies import get_ai, get_db
from ai.ai import AI
from db.db import db
from app.file_handler import File as FileHandler
from fastapi import APIRouter, Depends
from fastapi import File as FileFastAPI
from pydantic import BaseModel

class FilesRecords(BaseModel):
    id: int
    name: str
    entity_id: int
    category: str

class Data(BaseModel):
    type: str
    records: list[FilesRecords]

class FilesResponse(BaseModel):
    data: Data

class Embedding(BaseModel):
    text: str
    embedding: list[float]

class FileFullRecord(BaseModel):
    type: str
    file_id: int
    file_name: str
    entity_id: int
    entity_name: str
    category: str
    embeddings: list[Embedding]

class OneFileRespose(BaseModel):
    data: FileFullRecord

files = APIRouter(prefix='/files')


@files.post('')
async def add_file(
        entity: str,
        category: str,
        name: str,
        file: bytes = FileFastAPI(),
        ai: AI=Depends(get_ai)
    ):
   file = FileHandler(entity=entity, category=category, name=name, file_data=file)
   embeddings_table = ai.generate_embeddings_table(file=file) 
   ai.save_embeddings(embeddings_table)
   return {"detail": "File Saved"}

@files.get('')
async def get_files(db: db=Depends(get_db)) -> FilesResponse:
    result = {
        'data':{
            'type': 'files',
            'records': []
        }
    }
    with db as session:
        result['data']['records'] = session.get_files().to_dict(orient='records')
    return result

@files.get('/{file_id}')
async def get_file(file_id: int, db: db=Depends(get_db)) -> OneFileRespose:
    with db as session:
        filedf = session.get_files(file_id=file_id).loc[0]
        entdf = session.get_entities(entity_id=int(filedf["entity_id"])).loc[0]
        embeddingsdf = session.get_embeddings(file_id=file_id)
    result = {
        'data':{
            'type': 'file',
            'file_id': file_id,
            'file_name': filedf["name"],
            'entity_id': entdf["id"],
            'entity_name': entdf["name"],
            'category': filedf["category"],
            'embeddings': embeddingsdf[['text','embedding']].to_dict(orient='records')
        }
    }
    return result