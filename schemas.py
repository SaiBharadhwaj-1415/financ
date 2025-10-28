from pydantic import BaseModel

class QueryRequest(BaseModel):
    mode: str
    query: str

class QueryResponse(BaseModel):
    title: str
    body: str
