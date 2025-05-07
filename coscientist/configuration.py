"""
Configuration
-------------
- Takes a user prompt and create a configuration for the research plan to be
executed by the Supervisor.
"""

from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

from coscientist.types import ResearchPlanConfig


_PROMPT = """
You are an expert in scientific research planning. Given a goal from
a scientist, parse it into a simple JSON configuration specifying
the preferences, attributes, and constraints for an effective research plan.

- Preferences: A single sentence or two at most describing the focus for a high-quality research plan.
- Attributes: A list of up to 5 important attributes that the research plan should possess.
- Constraints: A list of up to 5 constraints that the research plan must satisfy.

For example, given the following goal:

Scientist research goal:
Develop a novel hypothesis for the key factor or process which causes ALS related to phosphorylation of a Nuclear Pore
Complex (NPC) nucleoporin. Explain mechanism of action in detail. Include also a feasible experiment to test the
hypothesis.

A valid parsed research plan configuration in JSON is:

{
  "preferences": "Focus on providing a novel hypothesis, with detailed explanation of the mechanism of action.",
  "attributes": ["Novelty", "Feasibility"],
  "constraints": ["Should be correct", "Should be novel"]
}
"""


def get_configuration_prompt(goal: str) -> str:
    """
    Create a research plan configuration from a goal using LangChain.

    Parameters
    ----------
    goal: str
        The research goal to parse into a configuration

    Returns
    -------
    str
        The configuration parsing prompt
    """
    prompt_to_format = """Now, parse the following goal into a
    research plan configuration:

    Scientist research goal
    {goal}

    Parsed research plan configuration
    """
    # Create the prompt template
    prompt_template = PromptTemplate(
        input_variables=["goal"], template=prompt_to_format
    )

    return "\n".join([_PROMPT, prompt_template.format(goal=goal)])


def goal_to_configuration(llm: BaseChatModel, goal: str) -> ResearchPlanConfig:
    """
    Creates the prompt and uses the LLM to parse the goal into a configuration.

    Parameters
    ----------
    llm: BaseChatModel
        The LLM to use for parsing the goal
    goal: str
        The goal to parse into a configuration

    Returns
    -------
    ResearchPlanConfig
        The parsed configuration
    """
    formatted_prompt = get_configuration_prompt(goal)
    structured_llm = llm.with_structured_output(ResearchPlanConfig)
    return structured_llm.invoke(formatted_prompt)
