from pydantic import BaseModel, Field


class LiteratureReview(BaseModel):
    """A review of the literature."""

    articles_with_reasoning: str = Field(description="A review of the literature.")


class ParsedHypothesis(BaseModel):
    """Structured output for parsed hypothesis."""

    hypothesis: str = Field(description="The main hypothesis statement")
    reasoning: str = Field(
        description="The reasoning and justification for the hypothesis"
    )
    assumptions: str = Field(description="The assumptions and falsifiable predictions")


class HypothesisWithID(BaseModel):
    """A hypothesis with an ID."""

    id: int
    content: str
    review: str
