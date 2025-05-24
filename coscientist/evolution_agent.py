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

import json
from typing import List

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, StateGraph

from coscientist.common import load_prompt
from coscientist.custom_types import (
    GeneratedHypothesis,
    HypothesisWithID,
    ResearchPlanConfig,
)

FEASIBILITY_PROMPT = """
You are an expert in scientific research and technological feasibility analysis.
Your task is to refine the provided conceptual idea, enhancing its practical implementability
by leveraging contemporary technological capabilities. Ensure the revised concept retains
its novelty, logical coherence, and specific articulation.

Goal: {goal}

Guidelines:
1. Begin with an introductory overview of the relevant scientific domain.
2. Provide a concise synopsis of recent pertinent research findings and related investigations,
highlighting successful methodologies and established precedents.
3. Articulate a reasoned argument for how current technological advancements can facilitate
the realization of the proposed concept.
4. CORE CONTRIBUTION: Develop a detailed, innovative, and technologically viable alternative
to achieve the objective, emphasizing simplicity and practicality.

Evaluation Criteria:
{preferences}

Original Conceptualization:
{hypothesis}
"""

OUT_OF_THE_BOX_PROMPT = """
You are an expert researcher tasked with generating a novel, singular hypothesis
inspired by analogous elements from provided concepts.

Goal: {goal}

Instructions:
1. Provide a concise introduction to the relevant scientific domain.
2. Summarize recent findings and pertinent research, highlighting successful approaches.
3. Identify promising avenues for exploration that may yield innovative hypotheses.
4. CORE HYPOTHESIS: Develop a detailed, original, and specific single hypothesis
for achieving the stated goal, leveraging analogous principles from the provided
ideas. This should not be a mere aggregation of existing methods or entities. Think out-of-the-box.

Criteria for a robust hypothesis:
{preferences}

Inspiration may be drawn from the following concepts (utilize analogy and inspiration,
not direct replication):
{hypotheses}
"""


class EvolutionState(TypedDict):
    """
    Represents the state of the evolution process.

    Parameters
    ----------
    goal: str
        The research goal
    research_plan_config: ResearchPlanConfig
        Configuration with preferences, attributes, and constraints
    top_hypotheses: List[HypothesisWithID]
        Top-ranked hypotheses to evolve
    evolved_hypotheses: List[HypothesisWithID]
        Newly evolved hypotheses
    """

    goal: str
    research_plan_config: ResearchPlanConfig
    top_hypotheses: List[HypothesisWithID]
    evolved_hypotheses: List[HypothesisWithID]


def feasibility_refinement_node(
    state: EvolutionState, llm: BaseChatModel
) -> EvolutionState:
    """
    Refines hypotheses for feasibility and practicality.

    Parameters
    ----------
    state: EvolutionState
        Current evolution state
    llm: BaseChatModel
        Language model for refinement

    Returns
    -------
    EvolutionState
        Updated state with refined hypotheses
    """
    evolved = []

    for hypothesis in state["top_hypotheses"]:
        prompt = load_prompt(
            "feasibility",
            goal=state["goal"],
            preferences=state["research_plan_config"].preferences,
            hypothesis=hypothesis.content,
        )

        suffix = """
        Return your output strictly in the following JSON format:
        {
            "reasoning": "<full detailed reasoning including analytical steps, literature synthesis, and logical progression>",
            "hypothesis": "<fully refined hypothesis, stated in detail and tailored for domain experts>"
        }

        {
            "reasoning": "
        """

        response_json_str = llm.invoke(prompt + suffix).content.replace("\n", " ")
        response_json_str = response_json_str.removeprefix("```json").removesuffix(
            "```"
        )

        try:
            data = json.loads(response_json_str)
            refined_hypothesis = GeneratedHypothesis(**data)

            # Create new hypothesis with ID
            evolved_hyp = HypothesisWithID(
                id=len(state["evolved_hypotheses"])
                + len(evolved)
                + 1000,  # Ensure unique ID
                content=refined_hypothesis.hypothesis,
                review=refined_hypothesis.reasoning,
            )
            evolved.append(evolved_hyp)

        except json.JSONDecodeError:
            # Skip malformed responses
            continue

    return {**state, "evolved_hypotheses": state["evolved_hypotheses"] + evolved}


def inspiration_generation_node(
    state: EvolutionState, llm: BaseChatModel
) -> EvolutionState:
    """
    Generates new hypotheses inspired by top-ranked ones.

    Parameters
    ----------
    state: EvolutionState
        Current evolution state
    llm: BaseChatModel
        Language model for inspiration

    Returns
    -------
    EvolutionState
        Updated state with inspired hypotheses
    """
    if len(state["top_hypotheses"]) < 2:
        return state

    # Combine top hypotheses for inspiration
    combined_hypotheses = "\n\n".join(
        [
            f"Hypothesis {i + 1}: {hyp.content}"
            for i, hyp in enumerate(state["top_hypotheses"][:3])  # Use top 3
        ]
    )

    prompt = load_prompt(
        "out_of_the_box",
        goal=state["goal"],
        preferences=state["research_plan_config"].preferences,
        hypotheses=combined_hypotheses,
    )

    suffix = """
    Return your output strictly in the following JSON format:
    {
        "reasoning": "<full detailed reasoning including analytical steps, literature synthesis, and logical progression>",
        "hypothesis": "<fully original hypothesis inspired by analogous principles, stated in detail>"
    }

    {
        "reasoning": "
    """

    response_json_str = llm.invoke(prompt + suffix).content.replace("\n", " ")
    response_json_str = response_json_str.removeprefix("```json").removesuffix("```")

    try:
        data = json.loads(response_json_str)
        inspired_hypothesis = GeneratedHypothesis(**data)

        # Create new hypothesis with ID
        inspired_hyp = HypothesisWithID(
            id=len(state["evolved_hypotheses"]) + 2000,  # Ensure unique ID
            content=inspired_hypothesis.hypothesis,
            review=inspired_hypothesis.reasoning,
        )

        return {
            **state,
            "evolved_hypotheses": state["evolved_hypotheses"] + [inspired_hyp],
        }

    except json.JSONDecodeError:
        return state


def build_evolution_agent(llm: BaseChatModel):
    """
    Builds and configures a LangGraph for the evolution agent process.

    The graph evolves hypotheses through:
    1. Feasibility refinement - improving practicality and implementability
    2. Inspiration generation - creating new hypotheses from top-ranked ones

    Parameters
    ----------
    llm: BaseChatModel
        The language model to use for evolution

    Returns
    -------
    StateGraph
        A compiled LangGraph for the evolution agent
    """
    graph = StateGraph(EvolutionState)

    # Add nodes
    graph.add_node(
        "feasibility_refinement", lambda state: feasibility_refinement_node(state, llm)
    )
    graph.add_node(
        "inspiration_generation", lambda state: inspiration_generation_node(state, llm)
    )

    # Define transitions
    graph.add_edge("feasibility_refinement", "inspiration_generation")
    graph.add_edge("inspiration_generation", END)

    # Set entry point
    graph.set_entry_point("feasibility_refinement")

    return graph.compile()


def evolve_hypothesis(
    llm: BaseChatModel,
    goal: str,
    research_plan_config: ResearchPlanConfig,
    hypothesis: GeneratedHypothesis,
) -> GeneratedHypothesis:
    """
    Legacy function for single hypothesis refinement.

    Parameters
    ----------
    llm: BaseChatModel
        The language model to use for hypothesis evolution
    goal: str
        The research goal
    research_plan_config: ResearchPlanConfig
        The research plan configuration (for preferences)
    hypothesis: GeneratedHypothesis
        The hypothesis to refine

    Returns
    -------
    GeneratedHypothesis
        The improved hypothesis and reasoning
    """
    prompt_template = PromptTemplate(
        input_variables=["goal", "preferences", "hypothesis"],
        template=FEASIBILITY_PROMPT,
    )
    prompt = prompt_template.format(
        goal=goal,
        preferences=research_plan_config.preferences,
        hypothesis=hypothesis.hypothesis,
    )

    suffix = """
    Return your output strictly in the following JSON format:
    {
        "reasoning": "<full detailed reasoning including analytical steps, literature synthesis, and logical progression>",
        "hypothesis": "<fully refined hypothesis, stated in detail and tailored for domain experts>"
    }

    {
        "reasoning": "
    """

    response_json_str = llm.invoke(prompt + suffix).content.replace("\n", " ")
    response_json_str = response_json_str.removeprefix("```json").removesuffix("```")
    try:
        data = json.loads(response_json_str)
        return GeneratedHypothesis(**data)
    except json.JSONDecodeError as e:
        # Return original if parsing fails
        return hypothesis
