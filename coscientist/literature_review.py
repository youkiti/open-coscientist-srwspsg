"""
System for agentic literature review that's used by other agents.
"""

from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

from coscientist.types import LiteratureReview

_PROMPT = """
You are an expect in biomedical research and especially skilled at writing 
comprehensive and accurate literature reviews. You focus on high-quality scientific
publications from reputable journals, and always provide citations. Importantly, you
do not take specific position in your research. Instead, you pursue a fair and balanced
perspective that draws no conclusions itself but provides other experts with enough
information to draw their own inferences.

Now you are tasked with compiling a review of the literature germane
to the following topic:

{topic}

Before writing the actual review, create a detailed plan of topics to cover. Then
systematically write the review, one section at a time.
"""


def get_review_prompt(topic: str) -> str:
    """
    Compile a review of the literature germane to the given topic.
    """
    prompt = PromptTemplate.from_template(_PROMPT)
    return prompt.invoke({"topic": topic})


def review_literature(llm: BaseChatModel, topic: str) -> LiteratureReview:
    """
    Compile a review of the literature germane to the given topic.

    Parameters
    ----------
    llm: BaseChatModel
        The LLM to use for the review
    topic: str
        The topic to review the literature for


    """
    prompt = get_review_prompt(topic)
    return LiteratureReview(articles_with_reasoning=llm.invoke(prompt).content)
