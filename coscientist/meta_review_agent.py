"""
Meta review agent
-----------------
- Formulates a research overview with memory
- Feedback from this agent is appended to the prompts of the
others in subsequent rounds.

More details:
- Takes in the tournament state with all debates and ELO ratings,
summarizes common patterns in the reviews and debates to synthesize
the meta-review feedback.
- Feedback helps to steer the Reflection agent so that it accounts
for common reasoning failures.
- Writes top hypotheses into a research overview that highlights
areas to follow up with real and specific experiments. This
gets fed to the Generation agent in later rounds. Format of the
overview can match the style of a review paper or a grant proposal
(like an NIH Specific Aims Page).
- Creates a list of research contacts from authors of papers
that are relevant to the research plan.
"""

from typing import List

from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

from coscientist.custom_types import LiteratureReview, ResearchPlanConfig
from coscientist.ranking_agent import EloTournament

REVIEW_GENERATION_PROMPT = """
You are an expert in scientific research and meta-analysis.
Synthesize a comprehensive meta-review of provided reviews
pertaining to the following research goal:

Goal: {goal}

Preferences:
{preferences}

Additional instructions:
{instructions}

Provided reviews for meta-analysis:
{reviews}

Instructions:
* Generate a structured meta-analysis report of the provided reviews.
* Focus on identifying recurring critique points and common issues raised by reviewers.
* The generated meta-analysis should provide actionable insights for researchers
developing future proposals.
* Refrain from evaluating individual proposals or reviews;
focus on producing a synthesized meta-analysis.

Response:
"""


def get_meta_review_prompt(
    goal: str,
    preferences: str,
    reviews: List[str],
    additional_instructions: str = "",
) -> str:
    """
    Compile a review of the literature germane to the given topic.
    """
    prompt = PromptTemplate.from_template(REVIEW_GENERATION_PROMPT)
    return prompt.format(
        goal=goal,
        preferences=preferences,
        reviews="\nReview:\n".join(reviews),
        instructions=additional_instructions,
    )


def review_current_state(
    llm: BaseChatModel,
    goal: str,
    research_plan_config: ResearchPlanConfig,
    tournament: EloTournament,
    additional_instructions: str = "",
) -> LiteratureReview:
    """
    Summarize and meta-review the current state of the tournament.

    Parameters
    ----------
    llm: BaseChatModel
        The LLM to use for the review
    goal: str
        The goal of the research
    research_plan_config: ResearchPlanConfig
        The research plan configuration
    tournament: EloTournament
        The tournament state
    additional_instructions: str
        Additional instructions for the meta-review

    Returns
    -------
    LiteratureReview
        The meta-review of the current state of the tournament
    """
    reviews = []
    for match_result in tournament.match_history.values():
        id1 = match_result.id1
        id2 = match_result.id2
        hypo1 = tournament.hypotheses[id1].content
        hypo2 = tournament.hypotheses[id2].content
        reviews.append(
            f"Hypothesis 1: {hypo1}\nHypothesis 2: {hypo2}\nDebate: {match_result.debate}"
        )

    prompt = get_meta_review_prompt(
        goal=goal,
        preferences=research_plan_config.preferences,
        reviews=reviews,
        additional_instructions=additional_instructions,
    )
    return LiteratureReview(articles_with_reasoning=llm.invoke(prompt).content)
