from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field

class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous")
    # superfluous means unnecessary information

class AnswerQuestion(BaseModel):
    """ Answer the question. """

    answer: str = Field(description="~250 words detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection to the initial answer.")
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )

class ReviseAnswer(BaseModel):
    """ Revise your original answer to your question. """

    references: List[str] = Field(
        description="Citations motivating your updated answer."
    )