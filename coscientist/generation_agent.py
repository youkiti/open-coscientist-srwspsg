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

TODO: Web search generation agent. This agent uses web search summaries to generate hypotheses.
Unclear if these summaries should be compiled once and shared with other generation agents, or
if it's a completely separate generation mode. My impression is that we should do deep research
on the topic once at the beginning and the generation web search is just asking a handful of
additional questions before formalizing a hypothesis.
TODO: Add fields in the prompts for meta-review agent feedback and prior hypotheses.
TODO: Consider treating the collaborative generation as a chatbot with Memory using LangGraph.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, TypedDict, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from coscientist.common import load_prompt
from coscientist.reasoning_types import ReasoningType


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
    hypothesis: str
        The standardized hypothesis output after the debate concludes.
    """

    goal: str
    literature_review: str
    transcript: List[Tuple[str, str]]  # List of (agent_name, message) tuples
    turn: int
    next_agent: str
    finished: bool
    hypothesis: str


class ParsedHypothesis(BaseModel):
    """Structured output for parsed hypothesis."""

    hypothesis: str = Field(description="The main hypothesis statement")
    reasoning: str = Field(
        description="The reasoning and justification for the hypothesis"
    )
    assumptions: str = Field(description="The assumptions and falsifiable predictions")


@dataclass
class IndependentConfig:
    """Configuration for independent generation mode."""

    field: str
    reasoning_type: ReasoningType
    llm: BaseChatModel


@dataclass
class CollaborativeConfig:
    """Configuration for collaborative generation mode."""

    agent_names: List[str]
    agent_fields: Dict[str, str]
    agent_reasoning_types: Dict[str, ReasoningType]
    llms: Dict[str, BaseChatModel]
    standardization_llm: BaseChatModel
    max_turns: int = 10


def build_generation_agent(
    mode: str,
    config: Union[IndependentConfig, CollaborativeConfig],
) -> StateGraph:
    """
    Unified builder function for generation agents that supports both independent and collaborative modes.

    Parameters
    ----------
    mode : str
        The mode of operation, either "independent" or "collaborative".
    config : Union[IndependentConfig, CollaborativeConfig]
        Configuration object containing all necessary parameters for the selected mode.

    Returns
    -------
    StateGraph
        A compiled LangGraph for the generation agent.

    Raises
    ------
    ValueError
        If mode is invalid or required parameters are missing for the selected mode.
    """
    if mode == "independent":
        if not isinstance(config, IndependentConfig):
            raise ValueError("config must be an IndependentConfig instance")
        return _build_independent_generation_agent(
            config.field, config.reasoning_type, config.llm
        )
    elif mode == "collaborative":
        if not isinstance(config, CollaborativeConfig):
            raise ValueError("config must be a CollaborativeConfig instance")
        return _build_collaborative_generation_agent(
            config.agent_names,
            config.agent_fields,
            config.agent_reasoning_types,
            config.llms,
            config.standardization_llm,
            config.max_turns,
        )
    else:
        raise ValueError("mode must be either 'independent' or 'collaborative'")


def _independent_generation_node(
    state: IndependentState,
    field: str,
    reasoning_type: ReasoningType,
    llm: BaseChatModel,
) -> IndependentState:
    """
    Represents the action of a single generation agent using the independent_generation.md template.
    The output is expected to be markdown with sections: Evidence, Hypothesis, Reasoning, Assumptions Table.
    """
    prompt = load_prompt(
        "independent_generation",
        goal=state["goal"],
        field=field,
        literature_review=state["literature_review"],
        reasoning_type=reasoning_type.value,
    )
    response_content = llm.invoke(prompt).content
    return {**state, "hypothesis": response_content}


def _debater_node(
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
    prompt = load_prompt(
        "collaborative_generation",
        goal=state["goal"],
        field=field,
        literature_review=state["literature_review"],
        reasoning_type=reasoning_type.value,
        transcript=current_transcript_str,
    )
    response_content = llm.invoke(prompt).content
    new_transcript = state["transcript"] + [(field, response_content)]

    return {**state, "transcript": new_transcript}


def _moderator_node(
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

    if "FINAL HYPOTHESIS:" in last_message or state["turn"] >= max_turns:
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


def _standardize_hypothesis_node(
    state: CollaborativeState, llm: BaseChatModel
) -> CollaborativeState:
    """
    Standardizes the hypothesis from the debate transcript using the standardize_hypothesis.md prompt.

    This function processes the complete debate transcript to extract and standardize
    the final hypothesis that was agreed upon by the experts.

    Parameters
    ----------
    state : CollaborativeState
        The current state of the debate (should have finished=True).
    llm : BaseChatModel
        The language model to use for standardization.

    Returns
    -------
    CollaborativeState
        The updated debate state with the standardized hypothesis.
    """
    # Convert transcript to string format
    transcript_str = "\n".join([f"{name}: {msg}" for name, msg in state["transcript"]])

    # Load and invoke the standardize_hypothesis prompt
    prompt = load_prompt("standardize_hypothesis", transcript=transcript_str)
    response_content = llm.invoke(prompt).content

    return {**state, "hypothesis": response_content}


def _build_independent_generation_agent(
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
        lambda state: _independent_generation_node(state, field, reasoning_type, llm),
    )
    graph.add_edge("generator", END)

    graph.set_entry_point("generator")
    return graph.compile()


def _build_collaborative_generation_agent(
    agent_names: List[str],
    agent_fields: dict[str, str],
    agent_reasoning_types: dict[str, ReasoningType],
    llms: dict[str, BaseChatModel],
    standardization_llm: BaseChatModel,
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
    standardization_llm : BaseChatModel
        The language model to use for standardization.
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
            lambda state, agent_name=name: _debater_node(
                state,
                agent_fields[agent_name],
                agent_reasoning_types[agent_name],
                llms[agent_name],
            ),
        )

    # Add moderator node
    graph.add_node(
        "moderator", lambda state: _moderator_node(state, agent_names, max_turns)
    )

    # Add standardization node
    graph.add_node(
        "standardizer",
        lambda state: _standardize_hypothesis_node(state, standardization_llm),
    )

    # Define transitions: After each debater, go to moderator
    for name in agent_names:
        graph.add_edge(name, "moderator")

    # Add edge from standardizer to END
    graph.add_edge("standardizer", END)

    # Conditional edges from moderator
    def route_after_moderator(state: CollaborativeState):
        if state["finished"]:
            return "standardizer"
        return state["next_agent"]

    graph.add_conditional_edges(
        "moderator",
        route_after_moderator,
        # Provide a mapping from the output of route_after_moderator to node names
        {name: name for name in agent_names} | {"standardizer": "standardizer"},
    )

    # Set entry point
    if not agent_names:
        raise ValueError("Agent names list cannot be empty.")

    graph.set_entry_point(agent_names[0])

    return graph.compile()


def parse_generation_output(text: str, llm: BaseChatModel) -> ParsedHypothesis:
    """
    Parse free-form generation agent output into structured fields.

    Parameters
    ----------
    text : str
        The free-form text output from the generation agent
    llm : BaseChatModel
        The language model to use for parsing

    Returns
    -------
    ParsedHypothesis
        Structured output with hypothesis, reasoning, and assumptions fields
    """
    structured_llm = llm.with_structured_output(ParsedHypothesis)

    prompt = f"""Parse the following scientific text into structured components:

    {text}

    Extract:
    1. The main hypothesis statement
    2. The reasoning and justification 
    3. The assumptions and falsifiable predictions

    If any section is missing, provide an empty string for that field."""

    return structured_llm.invoke(prompt)


def parse_hypothesis_markdown(markdown_text: str) -> ParsedHypothesis:
    """
    Parse markdown text with # headings to extract Hypothesis, Reasoning, and Assumptions sections.

    Parameters
    ----------
    markdown_text : str
        Markdown text containing sections with # headings for Hypothesis, Reasoning, and Assumptions

    Returns
    -------
    ParsedHypothesis
        Structured output with hypothesis, reasoning, and assumptions fields extracted from markdown
    """
    # Split the text by # to get sections
    sections = markdown_text.split("#")

    # Initialize fields
    hypothesis = ""
    reasoning = ""
    assumptions = ""

    # Process each section
    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Split section into title and content
        lines = section.split("\n", 1)
        if len(lines) < 2:
            continue

        title = lines[0].strip().lower()
        content = lines[1].strip()

        # Match section titles (case-insensitive)
        if "hypothesis" in title:
            hypothesis = content
        elif "reasoning" in title:
            reasoning = content
        elif "assumption" in title:  # Matches both "Assumption" and "Assumptions"
            assumptions = content

    return ParsedHypothesis(
        hypothesis=hypothesis, reasoning=reasoning, assumptions=assumptions
    )
