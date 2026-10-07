from typing import List, Literal
from pydantic import BaseModel


class QuestionItem(BaseModel):
    question: str
    difficulty: Literal["easy", "medium", "hard"]


class QuestionExtractionResponse(BaseModel):
    questions: List[QuestionItem]
