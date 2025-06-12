"""
Evolution agent
--------------
- Inspiration from other ideas
- Simplification
- Research extension

More details:
- Looks for weaknesses in a hypothesis, makes search queries to
evaluate them and suggests improvements to fill in reasoning gaps.
- Tries to fix invalid assumptions identified by the Reflection agent
and to improve practicality and feasibility for testing.
- Creates new hypotheses using multiple top-ranked ones as inspiration or
by combining them in new ways.
- Intentionally tries to generate out-of-the-box ideas that are
divergent from existing ones.
- Never replaces an existing hypothesis, but always adds a new one
that should in principle be better.
"""

from dataclasses import dataclass
from typing import List, TypedDict, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, StateGraph

from coscientist.common import load_prompt


class EvolveFromFeedbackState(TypedDict):
    """
    State for the `evolve_from_feedback` prompt agent.
    """

    goal: str
    hypothesis: str
    feedback: str
    evolution_response: str
    result: str


class OutOfTheBoxState(TypedDict):
    """
    State for the `out_of_the_box` prompt agent.
    """

    goal: str
    top_hypotheses: List[str]
    evolution_response: str
    result: str


@dataclass
class EvolutionConfig:
    """Configuration for evolution agents."""

    llm: BaseChatModel


def build_evolution_agent(
    mode: str,
    config: EvolutionConfig,
) -> StateGraph:
    """
    Unified builder function for evolution agents that supports both evolve_from_feedback and out_of_the_box modes.

    Parameters
    ----------
    mode : str
        The mode of operation, either "evolve_from_feedback" or "out_of_the_box".
    config : EvolutionConfig
        Configuration object containing the LLM to use for both evolution and standardization.

    Returns
    -------
    StateGraph
        A compiled LangGraph for the evolution agent.

    Raises
    ------
    ValueError
        If mode is invalid.
    """
    if mode == "evolve_from_feedback":
        return _build_evolve_from_feedback_agent(config.llm)
    elif mode == "out_of_the_box":
        return _build_out_of_the_box_agent(config.llm)
    else:
        raise ValueError(
            "mode must be either 'evolve_from_feedback' or 'out_of_the_box'"
        )


def _evolve_from_feedback_node(
    state: EvolveFromFeedbackState,
    llm: BaseChatModel,
) -> EvolveFromFeedbackState:
    """
    Evolution node for evolving a hypothesis based on feedback.
    """
    prompt = load_prompt(
        "evolve_from_feedback",
        goal=state["goal"],
        hypothesis=state["hypothesis"],
        feedback=state["feedback"],
    )
    response_content = llm.invoke(prompt).content
    return {**state, "evolution_response": response_content}


def _out_of_the_box_node(
    state: OutOfTheBoxState,
    llm: BaseChatModel,
) -> OutOfTheBoxState:
    """
    Evolution node for generating out-of-the-box ideas from top hypotheses.
    """
    # Convert list of hypotheses to formatted string
    hypotheses_text = "\n".join([f"- {hyp}" for hyp in state["top_hypotheses"]])

    prompt = load_prompt(
        "out_of_the_box",
        goal=state["goal"],
        hypotheses=hypotheses_text,
    )
    response_content = llm.invoke(prompt).content
    return {**state, "evolution_response": response_content}


def _standardization_node(
    state: Union[EvolveFromFeedbackState, OutOfTheBoxState],
    llm: BaseChatModel,
) -> Union[EvolveFromFeedbackState, OutOfTheBoxState]:
    """
    Standardization node that formats the evolution response using the standardize_hypothesis prompt.
    """
    prompt = load_prompt(
        "standardize_hypothesis",
        transcript=state["evolution_response"],
    )
    response_content = llm.invoke(prompt).content
    return {**state, "result": response_content}


def _build_evolve_from_feedback_agent(llm: BaseChatModel) -> StateGraph:
    """
    Builds and configures a LangGraph for evolving hypotheses from feedback.

    Parameters
    ----------
    llm : BaseChatModel
        The language model to use for both evolution and standardization.

    Returns
    -------
    StateGraph
        A compiled LangGraph for the evolve-from-feedback agent.
    """
    graph = StateGraph(EvolveFromFeedbackState)

    graph.add_node(
        "evolution",
        lambda state: _evolve_from_feedback_node(state, llm),
    )
    graph.add_node(
        "standardization",
        lambda state: _standardization_node(state, llm),
    )

    graph.add_edge("evolution", "standardization")
    graph.add_edge("standardization", END)

    graph.set_entry_point("evolution")
    return graph.compile()


def _build_out_of_the_box_agent(llm: BaseChatModel) -> StateGraph:
    """
    Builds and configures a LangGraph for generating out-of-the-box ideas.

    Parameters
    ----------
    llm : BaseChatModel
        The language model to use for both evolution and standardization.

    Returns
    -------
    StateGraph
        A compiled LangGraph for the out-of-the-box agent.
    """
    graph = StateGraph(OutOfTheBoxState)

    graph.add_node(
        "evolution",
        lambda state: _out_of_the_box_node(state, llm),
    )
    graph.add_node(
        "standardization",
        lambda state: _standardization_node(state, llm),
    )

    graph.add_edge("evolution", "standardization")
    graph.add_edge("standardization", END)

    graph.set_entry_point("evolution")
    return graph.compile()
