"""
Reflection agent
---------------
- Full review with web search
- Simulation review
- Tournament review
- Deep verification

More details:
- Does an initial review to assess the correctness, quality,
and novelty of the hypothesis. Does not use web search -- meant
to be a quick filter of bad ideas.
- Fully reviews a hypothesis with web search
- Deep verification decomposes a hypothesis into constituent
assumptions and sub-assumptions, and checks them for correctness.
Flawed assumptions don't necessarily invalidate an idea, but they
are flagged as areas for refinement/evolution.
- Observation review checks to see if there is unexplained observational
data that would be explained by the hypothesis.
- Simulation does a step-by-step rollout of a proposed mechanism of
action or experiment.
- Tournament review uses the output from the Ranking agent to find
recurring issues and opportunities for improvement.
"""

from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

INITIAL_FILTER_PROMPT = """
You are an expert in scientific hypothesis evaluation. Your task is to analyze a hypothesis
and determine if it is correct, novel, and high-quality. 

Instructions:

1. Correctness: Assess if the hypothesis is consistent with your extensive knowledge of the field.
Your primary concern is plausibility the hypothesis itself may be speculative and unproven.
2. Novelty: Assess if the hypothesis is a meaningfully new idea.
3. Quality: A high-quality hypothesis is well-motivated, clear, concise, and scientifically sound.
It must also be testable and grounded in valid assumptions.

Provide your reasoning for each of the three criteria, then conclude with an overall final evaluation
of "pass" or "fail": Final evaluation: <pass or fail>. To pass, the hypothesis must receive a pass
rating for each of the three criteria.

Hypothesis:
{hypothesis}

Response:
"""

DEEP_VERIFICATION_PROMPT = """
"""

SIMULATION_PROMPT = """
"""

TOURNAMENT_REVIEW_PROMPT = """
"""

OBSERVATION_PROMPT = """
You are an expert in scientific hypothesis evaluation. Your task is to analyze the
relationship between a provided hypothesis and observations from a scientific article.
Specifically, determine if the hypothesis provides a novel causal explanation
for the observations, or if they contradict it.

Instructions:

1. Observation extraction: list relevant observations from the article.
2. Causal analysis (individual): for each observation:
a. State if its cause is already established.
b. Assess if the hypothesis could be a causal factor (hypothesis => observation).
c. Start with: "would we see this observation if the hypothesis was true:".
d. Explain if it’s a novel explanation. If not, or if a better explanation exists,
state: "not a missing piece."
3. Causal analysis (summary): determine if the hypothesis offers a novel explanation
for a subset of observations. Include reasoning. Start with: "would we see some of
the observations if the hypothesis was true:".
4. Disproof analysis: determine if any observations contradict the hypothesis.
Start with: "does some observations disprove the hypothesis:".
5. Conclusion: state: "hypothesis: <already explained, other explanations more likely,
missing piece, neutral, or disproved>".

Scoring:
* Already explained: hypothesis consistent, but causes are known. No novel explanation.
* Other explanations more likely: hypothesis *could* explain, but better explanations exist.
* Missing piece: hypothesis offers a novel, plausible explanation.
* Neutral: hypothesis neither explains nor is contradicted.
* Disproved: observations contradict the hypothesis.

Important: if observations are expected regardless of the hypothesis, and don’t disprove it,
it’s neutral.

Article:
{article}

Hypothesis:
{hypothesis}

Response {provide reasoning. end with: "hypothesis: <already explained, other explanations
more likely, missing piece, neutral, or disproved>".)
"""


def get_initial_filter_prompt(hypothesis: str) -> str:
    """
    Create a prompt for the initial hypothesis filter.

    Parameters
    ----------
    hypothesis: str
        The hypothesis to filter

    Returns
    -------
    str
        The prompt for the initial hypothesis filter
    """
    # Create the prompt template
    prompt_template = PromptTemplate(
        input_variables=["hypothesis"],
        template=INITIAL_FILTER_PROMPT,
    )

    return prompt_template.format(hypothesis=hypothesis)


def filter_hypothesis(llm: BaseChatModel, hypothesis: str) -> bool:
    """
    Filter a hypothesis using the initial filter prompt.

    Parameters
    ----------
    llm: BaseChatModel
        The LLM to use for filtering the hypothesis
    hypothesis: str
        The hypothesis to filter

    Returns
    -------
    bool
        True if the hypothesis is passed, False otherwise
    """
    formatted_prompt = get_initial_filter_prompt(hypothesis)
    response = llm.invoke(formatted_prompt)
    return "pass" in response.content.split("Final evaluation:")[-1].lower()
