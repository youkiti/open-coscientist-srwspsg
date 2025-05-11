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
from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

from coscientist.types import LiteratureReview, ResearchPlanConfig, GeneratedHypothesis

GENERATION_PROMPT = """
You are an expert tasked with formulating a novel and robust hypothesis to address
the following objective.

Describe the proposed hypothesis in detail, including specific entities, mechanisms,
and anticipated outcomes. The hypothesis must clearly link proposed causes and effects
step-by-step, and make explicit and actionable predictions along the way.
Be as specific as possible, eliminate vagueness and hand-waving ruthlessly. 

This description is intended for an audience of domain experts.

You have conducted a thorough review of relevant literature and developed a logical framework
for addressing the objective. The articles consulted, along with your analytical reasoning,
are provided below.

Goal: {goal}

Criteria for a strong hypothesis:
{preferences}

Existing hypothesis (if applicable):
{source_hypothesis}

{instructions}

Literature review and analytical rationale (chronologically ordered, beginning
with the most recent analysis):

{articles_with_reasoning}
"""

DEBATE_PROMPT = """
You are an expert participating in a collaborative discourse concerning the generation
of a {idea_attributes} hypothesis. You will engage in a simulated discussion with other experts.
The overarching objective of this discourse is to collaboratively develop a novel
and robust {idea_attributes} hypothesis.

Goal: {goal}

Criteria for a high-quality hypothesis:
{preferences}

Instructions:
{instructions}

Review Overview:
{reviews_overview}

Procedure:

Initial contribution (if initiating the discussion):
Propose three distinct {idea_attributes} hypotheses.

Subsequent contributions (continuing the discussion):
* Pose clarifying questions if ambiguities or uncertainties arise.
* Critically evaluate the hypotheses proposed thus far, addressing the following aspects:
- Adherence to {idea_attributes} criteria.
- Utility and practicality.
- Level of detail and specificity.
* Identify any weaknesses or potential limitations.
* Propose concrete improvements and refinements to address identified weaknesses.
* Conclude your response with a refined iteration of the hypothesis.

General guidelines:
* Exhibit boldness and creativity in your contributions.
* Maintain a helpful and collaborative approach.
* Prioritize the generation of a high-quality {idea_attributes} hypothesis.

Termination condition:
When sufficient discussion has transpired (typically 3-5 conversational turns,
with a maximum of 10 turns) and all relevant questions and points have been
thoroughly addressed and clarified, conclude the process by writing "HYPOTHESIS"
(in all capital letters) followed by a concise and self-contained exposition of the finalized idea.

#BEGIN TRANSCRIPT#
{transcript}
#END TRANSCRIPT#

Your Turn:
"""


def get_generation_prompt(
    goal: str,
    literature_review: LiteratureReview,
    research_plan_config: ResearchPlanConfig,
    source_hypothesis: str = "",
    additional_instructions: str = "",
) -> str:
    """
    Generate a prompt for hypothesis generation based on literature review and research plan.

    Parameters
    ----------
    goal: str
        The research goal to address
    literature_review: LiteratureReview
        The literature review containing articles and reasoning
    research_plan_config: ResearchPlanConfig
        The research plan configuration
    source_hypothesis: str
        The source hypothesis to build upon
    additional_instructions: str
        Additional instructions to include in the prompt

    Returns
    -------
    str
        The formatted prompt for hypothesis generation
    """
    prompt_template = PromptTemplate(
        input_variables=[
            "goal",
            "preferences",
            "source_hypothesis",
            "instructions",
            "articles_with_reasoning",
        ],
        template=GENERATION_PROMPT,
    )
    suffix = """
    Return your output strictly in the following JSON format:
    {
        "reasoning": "<full detailed reasoning including analytical steps, literature synthesis, and logical progression>",
        "hypothesis": "<fully elaborated hypothesis, stated in detail and tailored for domain experts>"
    }

    {
        "reasoning": "
    """

    return (
        prompt_template.format(
            goal=goal,
            preferences=research_plan_config.preferences,
            source_hypothesis=source_hypothesis,
            instructions=additional_instructions,
            articles_with_reasoning=literature_review.articles_with_reasoning,
        )
        + suffix
    )


def generate_hypothesis(
    llm: BaseChatModel,
    goal: str,
    literature_review: LiteratureReview,
    research_plan_config: ResearchPlanConfig,
    source_hypothesis: str = "",
    additional_instructions: str = "",
) -> GeneratedHypothesis:
    """
    Generate a hypothesis using the literature review and research plan.

    Parameters
    ----------
    llm: BaseChatModel
        The language model to use for hypothesis generation
    goal: str
        The research goal to address
    literature_review: LiteratureReview
        The literature review containing articles and reasoning
    research_plan_config: ResearchPlanConfig
        The research plan configuration
    source_hypothesis: str
        The source hypothesis to build upon
    additional_instructions: str
        Additional instructions to include in the prompt

    Returns
    -------
    str
        The generated hypothesis
    """
    prompt = get_generation_prompt(
        goal=goal,
        literature_review=literature_review,
        research_plan_config=research_plan_config,
        source_hypothesis=source_hypothesis,
        additional_instructions=additional_instructions,
    )
    response_json_str = llm.invoke(prompt).content.replace("\n", " ")
    response_json_str = response_json_str.removeprefix("```json").removesuffix("```")
    try:
        data = json.loads(response_json_str)
        return GeneratedHypothesis(**data)
    except json.JSONDecodeError as e:
        # print(f"Error decoding JSON from LLM: {e}")
        # print(f"LLM Output was: {response_json_str}")
        # raise
        return response_json_str
