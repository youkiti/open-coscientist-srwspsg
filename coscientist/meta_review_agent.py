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
TODO: Adjust the meta-review prompt to include the tournament state with
ELO ratings. Probably need to add the hypotheses with IDs to the debates
so that the LLM knows which ones are being discussed. Right now it just says
Hypothesis 1 vs Hypothesis 2. Should also specify something like which round
a debate is from?
"""

from typing import List, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, StateGraph

from coscientist.common import load_prompt


class MetaReviewTournamentState(TypedDict):
    """
    State for the `meta_review_tournament` prompt agent.
    """

    goal: str
    debates: List[str]
    result: str


def _meta_review_node(
    state: MetaReviewTournamentState,
    llm: BaseChatModel,
) -> MetaReviewTournamentState:
    """
    Meta-review node that synthesizes debates into a comprehensive meta-analysis.
    """
    # Convert list of debates to formatted string
    reviews_text = "\n\n".join(
        [f"Review {i+1}:\n{debate}" for i, debate in enumerate(state["debates"])]
    )

    prompt = load_prompt(
        "meta_review_tournament",
        goal=state["goal"],
        reviews=reviews_text,
    )
    response_content = llm.invoke(prompt).content
    return {**state, "result": response_content}


def build_meta_review_agent(llm: BaseChatModel) -> StateGraph:
    """
    Builds and configures a LangGraph for meta-review analysis.

    Parameters
    ----------
    llm : BaseChatModel
        The language model to use for meta-review generation.

    Returns
    -------
    StateGraph
        A compiled LangGraph for the meta-review agent.
    """
    graph = StateGraph(MetaReviewTournamentState)

    graph.add_node(
        "meta_review",
        lambda state: _meta_review_node(state, llm),
    )

    graph.add_edge("meta_review", END)
    graph.set_entry_point("meta_review")
    return graph.compile()
