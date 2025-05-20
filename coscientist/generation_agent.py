"""
Generation agent
---------------
- Literature exploration
- Simulated scientific debates

More details:
- Searches the web for papers, reads the articles, and summarizes prior
work. Using these summaries, it generates hypotheses.
- Runs scientific debates for self-critique and self-play. These
Socratic dialogues go into refining the hypotheses.
- Each hypothesis comes with a set of testable assumptions and sub-assumptions
This looks something like multihop reasoning with MCTS.
- New hypotheses can be generated in later steps conditioned on the past hypotheses
and feedback from the meta-review agent.

"""

import json
import os
from typing import List, Tuple, TypedDict

from jinja2 import Environment, FileSystemLoader, select_autoescape
from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, StateGraph

from coscientist.custom_types import (
    GeneratedHypothesis,
    LiteratureReview,
    ResearchPlanConfig,
)
from coscientist.reasoning_types import ReasoningType

_env = Environment(
    loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "prompts")),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)


class IndependentState(TypedDict):
    goal: str
    literature_review: str
    hypothesis: str


class CollaborativeState(TypedDict):
    """
    Represents the state of the debate.

    Parameters
    ----------
    goal: str
        The main objective or topic of the debate.
    literature_review: str
        A summary of relevant background information or literature.
    transcript: List[Tuple[str, str]]
        List of (agent_name, message) tuples representing the conversation history.
    turn: int
        The current turn number in the debate.
    next_agent: str
        The name of the agent who will speak next.
    finished: bool
        Flag indicating whether the debate has concluded.
    """

    goal: str
    literature_review: str
    transcript: List[Tuple[str, str]]  # List of (agent_name, message) tuples
    turn: int
    next_agent: str
    finished: bool


def independent_generation_node(
    state: IndependentState,
    field: str,
    reasoning_type: ReasoningType,
    llm: BaseChatModel,
) -> IndependentState:
    """
    Represents the action of a single generation agent using the independent_generation.md template.
    The output is expected to be markdown with sections: Evidence, Hypothesis, Reasoning, Assumptions Table.
    """
    template = _env.get_template("independent_generation.md")
    prompt = template.render(
        goal=state["goal"],
        field=field,
        literature_review=state["literature_review"],
        reasoning_type=reasoning_type.value,
    )
    response_content = llm.invoke(prompt).content
    return {**state, "hypothesis": response_content}


def debater_node(
    state: CollaborativeState,
    field: str,
    reasoning_type: ReasoningType,
    llm: BaseChatModel,
) -> CollaborativeState:
    """
    Represents the action of a debater agent in the debate.

    This function generates a new message based on the current debate transcript and updates
    the debate state with the new message.

    Parameters
    ----------
    state : CollaborativeState
        The current state of the debate.
    field : str
        The field or domain of expertise of the debater.
    reasoning_type : ReasoningType
        The type of reasoning the debater should employ.
    llm : BaseChatModel
        The language model to use for generating responses.

    Returns
    -------
    CollaborativeState
        The updated debate state with the new message added to the transcript.
    """
    current_transcript_str = "\n".join(
        [f"{name}: {msg}" for name, msg in state["transcript"]]
    )
    template = _env.get_template("collaborative_generation.md")

    prompt = template.render(
        goal=state["goal"],
        field=field,
        literature_review=state["literature_review"],
        reasoning_type=reasoning_type.value,
        transcript=current_transcript_str,
    )
    response_content = llm.invoke(prompt).content

    new_transcript = state["transcript"] + [(field, response_content)]

    return {**state, "transcript": new_transcript}


def moderator_node(
    state: CollaborativeState, agent_names: List[str], max_turns: int = 10
) -> CollaborativeState:
    """
    Controls the debate flow and manages turn order between agents.

    This function determines the next agent to speak, checks for termination conditions,
    and updates the debate state accordingly.

    Parameters
    ----------
    state : CollaborativeState
        The current state of the debate.
    agent_names : List[str]
        List of names of all participating agents.
    max_turns : int, optional
        Maximum number of turns allowed in the debate, by default 10.

    Returns
    -------
    CollaborativeState
        The updated debate state with new turn information and next agent assignment.
    """
    last_message = state["transcript"][-1][1] if state["transcript"] else ""

    if "HYPOTHESIS" in last_message or state["turn"] >= max_turns:
        return {**state, "finished": True, "next_agent": ""}

    # Simple round-robin for now
    current_agent_index = agent_names.index(state["next_agent"])
    next_agent_index = (current_agent_index + 1) % len(agent_names)

    return {
        **state,
        "finished": False,
        "next_agent": agent_names[next_agent_index],
        "turn": state["turn"] + 1,
    }


def build_independent_generation_agent(
    field: str, reasoning_type: ReasoningType, llm: BaseChatModel
):
    """
    Builds and configures a LangGraph for a single-agent generation process using the independent_generation.md template.
    The agent's output is markdown with sections: Evidence, Hypothesis, Reasoning, Assumptions Table.

    Parameters
    ----------
    agent_name : str
        Name of the agent.
    field : str
        Field or domain of expertise.
    reasoning_type : ReasoningType
        Reasoning type for the agent.
    llm : BaseChatModel
        The language model to use.

    Returns
    -------
    StateGraph
        A compiled LangGraph for the generation agent.
    """
    graph = StateGraph(IndependentState)
    graph.add_node(
        "generator",
        lambda state: independent_generation_node(state, field, reasoning_type, llm),
    )
    graph.add_edge("generator", END)

    graph.set_entry_point("generator")
    return graph.compile()


def build_collaborative_generation_agent(
    agent_names: List[str],
    agent_fields: dict[str, str],
    agent_reasoning_types: dict[str, ReasoningType],
    llms: dict[str, BaseChatModel],
    max_turns: int = 10,
):
    """
    Builds and configures the LangGraph for the multi-agent debate system.

    This function creates a directed graph where nodes represent agents and the moderator,
    and edges define the flow of conversation between them.

    Parameters
    ----------
    agent_names : List[str]
        List of names of all participating agents.
    agent_fields : dict[str, str]
        Dictionary mapping agent names to their fields of expertise.
    agent_reasoning_types : dict[str, ReasoningType]
        Dictionary mapping agent names to their reasoning types.
    llms : dict[str, BaseChatModel]
        Dictionary mapping agent names to their respective language models.
    max_turns : int, optional
        Maximum number of turns allowed in the debate, by default 10.

    Returns
    -------
    StateGraph
        A compiled LangGraph representing the debate system.

    Raises
    ------
    ValueError
        If the agent_names list is empty.
    """
    graph = StateGraph(CollaborativeState)

    # Add debater nodes
    for name in agent_names:
        # Ensure the lambda correctly captures the 'name' for each node
        graph.add_node(
            name,
            lambda state, agent_name=name: debater_node(
                state,
                agent_fields[agent_name],
                agent_reasoning_types[agent_name],
                llms[agent_name],
            ),
        )

    # Add moderator node
    graph.add_node(
        "moderator", lambda state: moderator_node(state, agent_names, max_turns)
    )

    # Define transitions: After each debater, go to moderator
    for name in agent_names:
        graph.add_edge(name, "moderator")

    # Conditional edges from moderator
    def route_after_moderator(state: CollaborativeState):
        if state["finished"]:
            return END
        return state["next_agent"]

    graph.add_conditional_edges(
        "moderator",
        route_after_moderator,
        # Provide a mapping from the output of route_after_moderator to node names
        {name: name for name in agent_names + [END]},
    )

    # Set entry point
    if not agent_names:
        raise ValueError("Agent names list cannot be empty.")

    graph.set_entry_point(agent_names[0])

    return graph.compile()
