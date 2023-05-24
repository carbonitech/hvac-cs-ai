from app.resources.dependencies import get_ai
from ai.ai import AI
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

chat = APIRouter(prefix='/chat')

class Query(BaseModel):
    query: str

class AIResponse(BaseModel):
    query: str
    response: str

@chat.post('')
async def send_query(query: Query, ai: AI=Depends(get_ai)) -> AIResponse:
    query_txt = query.query
    gpt_response = ai.ask_model(query=query_txt)
    return AIResponse(query=query.query, response=gpt_response)
