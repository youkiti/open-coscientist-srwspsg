import uuid
from typing import List

from pydantic import BaseModel, Field


class ParsedHypothesis(BaseModel):
    """Structured output for parsed hypothesis."""

    uid: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the hypothesis",
    )
    hypothesis: str = Field(description="The main hypothesis statement")
    predictions: List[str] = Field(
        description="A list of predictions that could be tested to disprove the hypothesis"
    )
    assumptions: List[str] = Field(
        description="A list of assumptions that are implicit or explicit in the hypothesis"
    )


class HypothesisWithID(BaseModel):
    """A hypothesis with an ID."""

    id: int
    content: str
    review: str
