from typing import List

from pydantic import BaseModel, Field


class ResearchPlanConfig(BaseModel):
    """Configuration for a research plan."""

    preferences: str = Field(
        description="A single sentence or two at most describing the focus for a high-quality research plan."
    )
    attributes: List[str] = Field(
        description="A list of up to 5 important attributes that the research plan should possess."
    )
    constraints: List[str] = Field(
        description="A list of up to 5 constraints that the research plan must satisfy."
    )


class Hypothesis(BaseModel):
    """"""

    content: str = Field(description="The actual hypothesis in a few short sentences.")
    review: List[str] = Field(
        description=(
            "Highly detailed mechanistic explanation, reasoning and elaboration on the hypothesis. "
            "Makes the argument for why the hypothesis is likely to be true. Citing "
            "relevant literature and previous studies where appropriate."
        )
    )


class LiteratureReview(BaseModel):
    """A review of the literature."""

    articles_with_reasoning: str = Field(description="A review of the literature.")


class GeneratedHypothesis(BaseModel):
    """Data model for the structured output of hypothesis generation."""

    reasoning: str
    hypothesis: str


class HypothesisWithID(BaseModel):
    """A hypothesis with an ID."""

    id: int
    content: str
    review: str
