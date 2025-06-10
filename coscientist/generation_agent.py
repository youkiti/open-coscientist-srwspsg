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
from typing import Dict, List, TypedDict, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from coscientist import multiturn
from coscientist.common import load_prompt
from coscientist.reasoning_types import ReasoningType


class IndependentState(TypedDict):
    goal: str
    literature_review: str
    result: str


class CollaborativeState(IndependentState, multiturn.MultiTurnState):
    pass


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
        # Use the simplified multi-turn system
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
    return {**state, "result": response_content}


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
    agent_fields: Dict[str, str],
    agent_reasoning_types: Dict[str, ReasoningType],
    llms: Dict[str, BaseChatModel],
    standardization_llm: BaseChatModel,
    max_turns: int = 10,
) -> StateGraph:
    """Build collaborative generation agent."""

    # Create agent node functions
    agent_node_fns = {}
    for agent_name in agent_names:
        agent_node_fns[agent_name] = multiturn.create_agent_node_fn(
            agent_name=agent_name,
            llm=llms[agent_name],
            prompt_name="collaborative_generation",
            prompt_keys_from_state=["goal", "literature_review"],
            # kwargs for the prompt
            field=agent_fields[agent_name],
            reasoning_type=agent_reasoning_types[agent_name].value,
        )

    # Create moderator and post-processor
    moderator_fn = multiturn.create_moderator_node_fn(
        agent_names, lambda msg: "FINAL HYPOTHESIS:" in msg, max_turns
    )
    post_processor_fn = multiturn.create_post_processor_node_fn(
        standardization_llm, "standardize_hypothesis"
    )

    return multiturn.build_multi_turn_agent(
        CollaborativeState, agent_node_fns, moderator_fn, post_processor_fn
    )


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
