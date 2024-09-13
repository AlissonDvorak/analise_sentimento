from typing import List
from pydantic import BaseModel
class TextRequest(BaseModel):
    texts: List[str]