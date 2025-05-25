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

from typing import List, TypedDict

from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, StateGraph

from coscientist.common import load_prompt
from coscientist.custom_types import (
    HypothesisWithID,
    LiteratureReview,
    ResearchPlanConfig,
)
from coscientist.ranking_agent import EloTournament


def get_meta_review_prompt(
    goal: str,
    preferences: str,
    reviews: List[str],
    additional_instructions: str = "",
) -> str:
    """
    Compile a review of the literature germane to the given topic.
    """
    return load_prompt(
        "meta_review",
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


class MetaReviewState(TypedDict):
    """
    Represents the state of the meta-review process.

    Parameters
    ----------
    goal: str
        The research goal
    research_plan_config: ResearchPlanConfig
        Configuration with preferences, attributes, and constraints
    tournament: EloTournament
        Current tournament state with debates and ratings
    individual_reviews: List[str]
        Collection of individual hypothesis reviews
    pattern_analysis: str
        Analysis of recurring patterns in reviews
    agent_optimization_suggestions: str
        Suggestions for improving agent performance
    research_overview: str
        Comprehensive research overview for scientists
    """

    goal: str
    research_plan_config: ResearchPlanConfig
    tournament: EloTournament
    individual_reviews: List[str]
    pattern_analysis: str
    agent_optimization_suggestions: str
    research_overview: str


def pattern_identification_node(
    state: MetaReviewState, llm: BaseChatModel
) -> MetaReviewState:
    """
    Identifies recurring patterns in reviews and debate outcomes.

    Parameters
    ----------
    state: MetaReviewState
        Current meta-review state
    llm: BaseChatModel
        Language model for pattern analysis

    Returns
    -------
    MetaReviewState
        Updated state with pattern analysis
    """
    # Collect all reviews and debates
    all_reviews = state["individual_reviews"].copy()

    # Add tournament debates
    for match_result in state["tournament"].match_history.values():
        all_reviews.append(f"Debate: {match_result.debate}")

    if not all_reviews:
        return {
            **state,
            "pattern_analysis": "No reviews available for pattern analysis.",
        }

    prompt = load_prompt(
        "pattern_identification",
        goal=state["goal"],
        reviews=chr(10).join(all_reviews[:10]),  # Limit to first 10 for context
    )
    response = llm.invoke(prompt)

    return {**state, "pattern_analysis": response.content}


def agent_optimization_node(
    state: MetaReviewState, llm: BaseChatModel
) -> MetaReviewState:
    """
    Generates suggestions for optimizing agent performance based on patterns.

    Parameters
    ----------
    state: MetaReviewState
        Current meta-review state
    llm: BaseChatModel
        Language model for optimization suggestions

    Returns
    -------
    MetaReviewState
        Updated state with optimization suggestions
    """
    prompt = load_prompt(
        "agent_optimization",
        goal=state["goal"],
        pattern_analysis=state["pattern_analysis"],
    )
    response = llm.invoke(prompt)

    return {**state, "agent_optimization_suggestions": response.content}


def research_overview_node(
    state: MetaReviewState, llm: BaseChatModel
) -> MetaReviewState:
    """
    Synthesizes findings into a comprehensive research overview.

    Parameters
    ----------
    state: MetaReviewState
        Current meta-review state
    llm: BaseChatModel
        Language model for overview generation

    Returns
    -------
    MetaReviewState
        Updated state with research overview
    """
    # Get top hypotheses from tournament
    top_hypotheses = []
    if state["tournament"].hypotheses:
        sorted_hypotheses = state["tournament"].get_sorted_hypotheses()
        top_hypotheses = [
            state["tournament"].hypotheses[h_id]
            for h_id, _ in sorted_hypotheses[:5]  # Top 5
        ]

    top_hypotheses_text = (
        "\n\n".join(
            [
                f"Hypothesis {hyp.id}: {hyp.content}\nReview: {hyp.review[:500]}..."
                for hyp in top_hypotheses
            ]
        )
        if top_hypotheses
        else "No hypotheses available."
    )

    overview_prompt = f"""
    You are creating a comprehensive research overview for a human scientist.

    Research Goal: {state["goal"]}
    
    Research Preferences: {state["research_plan_config"].preferences}
    
    Top-Ranked Hypotheses:
    {top_hypotheses_text}
    
    Pattern Analysis: {state["pattern_analysis"]}
    
    Create a structured research overview that includes:
    1. Executive Summary
    2. Key Hypotheses and Their Strengths
    3. Recommended Next Steps for Experimental Validation
    4. Identified Knowledge Gaps
    5. Suggested Collaborations or Resources
    6. Risk Assessment and Mitigation Strategies

    Format this as a professional research report suitable for grant applications or research planning.
    """

    response = llm.invoke(overview_prompt)

    return {**state, "research_overview": response.content}


def build_meta_review_agent(llm: BaseChatModel):
    """
    Builds and configures a LangGraph for the meta-review agent process.

    The graph synthesizes research insights through:
    1. Pattern identification - finding recurring themes in reviews
    2. Agent optimization - suggesting improvements for other agents
    3. Research overview - creating comprehensive summaries for scientists

    Parameters
    ----------
    llm: BaseChatModel
        The language model to use for meta-review analysis

    Returns
    -------
    StateGraph
        A compiled LangGraph for the meta-review agent
    """
    graph = StateGraph(MetaReviewState)

    # Add nodes
    graph.add_node(
        "pattern_identification", lambda state: pattern_identification_node(state, llm)
    )
    graph.add_node(
        "agent_optimization", lambda state: agent_optimization_node(state, llm)
    )
    graph.add_node(
        "research_overview_node", lambda state: research_overview_node(state, llm)
    )

    # Define transitions
    graph.add_edge("pattern_identification", "agent_optimization")
    graph.add_edge("agent_optimization", "research_overview_node")
    graph.add_edge("research_overview_node", END)

    # Set entry point
    graph.set_entry_point("pattern_identification")

    return graph.compile()
