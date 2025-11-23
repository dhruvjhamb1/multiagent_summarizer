from pydantic import BaseModel
from typing import List

class DocumentRequest(BaseModel):
    content: str

class AnalysisResponse(BaseModel):
    summary: str
    entities: List[str]
    sentiment: str