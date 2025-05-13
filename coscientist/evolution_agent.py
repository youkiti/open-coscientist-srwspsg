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
from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from coscientist.types import ResearchPlanConfig, GeneratedHypothesis

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


def evolve_hypothesis(
    llm: BaseChatModel,
    goal: str,
    research_plan_config: ResearchPlanConfig,
    hypothesis: GeneratedHypothesis,
) -> GeneratedHypothesis:
    """
    Refine a hypothesis for feasibility and practicality using the FEASIBILITY_PROMPT and an LLM.

    Parameters
    ----------
    llm: BaseChatModel
        The language model to use for hypothesis evolution
    goal: str
        The research goal
    research_plan_config: ResearchPlanConfig
        The research plan configuration (for preferences)
    hypothesis: str
        The hypothesis to refine (as a string)

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
        # print(f"Error decoding JSON from LLM: {e}")
        # print(f"LLM Output was: {response_json_str}")
        # raise
        return response_json_str
