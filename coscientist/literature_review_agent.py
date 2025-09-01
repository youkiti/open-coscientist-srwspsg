"""
System for agentic literature review that's used by other agents.

Implementation uses LangGraph to:
1. Decompose research goals into modular topics
2. Dispatch each topic to GPTResearcher workers in parallel
3. Synthesize topic reports into executive summary
"""

import asyncio
import logging
import os
import re
from typing import TypedDict
from datetime import datetime

from gpt_researcher import GPTResearcher
from gpt_researcher.utils.enum import Tone
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, StateGraph

from coscientist.common import load_prompt

# Configure loggers
lit_review_logger = logging.getLogger('coscientist.literature_review')
gpt_researcher_logger = logging.getLogger('coscientist.gpt_researcher')

# Timeouts and retry controls to prevent infinite waits
RESEARCH_TIMEOUT_SECONDS = int(os.environ.get("COSCI_RESEARCH_TIMEOUT_SECONDS", "420"))
WRITE_TIMEOUT_SECONDS = int(os.environ.get("COSCI_WRITE_TIMEOUT_SECONDS", "240"))
RESEARCH_MAX_RETRIES = int(os.environ.get("COSCI_RESEARCH_MAX_RETRIES", "0"))


class LiteratureReviewState(TypedDict):
    """State for the literature review agent."""

    goal: str
    max_subtopics: int
    subtopics: list[str]
    subtopic_reports: list[str]
    meta_review: str


def parse_topic_decomposition(markdown_text: str) -> list[str]:
    """
    Parse the topic decomposition markdown into strings.

    Parameters
    ----------
    markdown_text : str
        The markdown output from topic_decomposition prompt

    Returns
    -------
    list[str]
        Parsed subtopics strings
    """
    # Split by subtopic headers (### Subtopic N)
    sections = re.split(r"### Subtopic \d+", markdown_text)
    return [section.strip() for section in sections[1:]]


def _topic_decomposition_node(
    state: LiteratureReviewState,
    llm: BaseChatModel,
) -> LiteratureReviewState:
    """
    Node that decomposes the research goal into focused subtopics.
    """
    lit_review_logger.info("Starting topic decomposition")
    lit_review_logger.debug(f"Research goal: {state['goal'][:200]}...")
    lit_review_logger.debug(f"Max subtopics: {state['max_subtopics']}")
    
    prompt = load_prompt(
        "topic_decomposition",
        goal=state["goal"],
        max_subtopics=state["max_subtopics"],
        subtopics=state.get("subtopics", ""),
        meta_review=state.get("meta_review", ""),
    )
    
    lit_review_logger.info(f"Invoking LLM for topic decomposition (model: {type(llm).__name__})")
    start_time = datetime.now()
    response_content = llm.invoke(prompt).content
    elapsed = (datetime.now() - start_time).total_seconds()
    lit_review_logger.debug(f"LLM response received in {elapsed:.1f}s")

    # Parse the topics from the markdown response
    subtopics = parse_topic_decomposition(response_content)

    if not subtopics:
        lit_review_logger.error("Failed to parse any topics from decomposition response")
        raise ValueError("Failed to parse any topics from decomposition response")

    if state.get("subtopics", False):
        subtopics = state["subtopics"] + subtopics
    
    lit_review_logger.info(f"Topic decomposition completed: {len(subtopics)} subtopics identified")
    for i, topic in enumerate(subtopics, 1):
        lit_review_logger.debug(f"  Subtopic {i}: {topic[:100]}...")

    return {"subtopics": subtopics}


async def _write_subtopic_report(subtopic: str, main_goal: str) -> str:
    """
    Conduct research for a single subtopic using GPTResearcher.

    Parameters
    ----------
    subtopic : str
        The subtopic to research
    main_goal : str
        The main research goal for context

    Returns
    -------
    str
        The research report
    """
    gpt_researcher_logger.info(f"Starting research for subtopic: {subtopic[:100]}...")
    gpt_researcher_logger.debug(f"Parent goal: {main_goal[:100]}...")
    
    # Create a focused query combining the research focus and key terms
    researcher = GPTResearcher(
        query=subtopic,
        report_type="subtopic_report",
        report_format="markdown",
        parent_query=main_goal,
        verbose=False,
        tone=Tone.Objective,
        config_path=os.path.join(os.path.dirname(__file__), "researcher_config.json"),
    )
    
    gpt_researcher_logger.debug("GPTResearcher instance created")

    # Conduct research and generate report
    attempts = RESEARCH_MAX_RETRIES + 1
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            gpt_researcher_logger.info(
                f"Conducting research phase for: {subtopic[:50]}... (attempt {attempt}/{attempts})"
            )
            start_time = datetime.now()
            await asyncio.wait_for(
                researcher.conduct_research(), timeout=RESEARCH_TIMEOUT_SECONDS
            )

            research_elapsed = (datetime.now() - start_time).total_seconds()
            gpt_researcher_logger.info(
                f"Research phase completed in {research_elapsed:.1f}s"
            )

            gpt_researcher_logger.info("Writing research report...")
            report_start = datetime.now()
            report = await asyncio.wait_for(
                researcher.write_report(), timeout=WRITE_TIMEOUT_SECONDS
            )

            report_elapsed = (datetime.now() - report_start).total_seconds()
            total_elapsed = (datetime.now() - start_time).total_seconds()
            gpt_researcher_logger.info(
                f"Report written in {report_elapsed:.1f}s (total: {total_elapsed:.1f}s)"
            )
            gpt_researcher_logger.debug(
                f"Report length: {len(report)} characters"
            )
            return report
        except asyncio.TimeoutError as e:
            last_error = e
            gpt_researcher_logger.error(
                f"Timeout during research/report for subtopic '{subtopic[:50]}...' "
                f"on attempt {attempt}/{attempts}",
                exc_info=True,
            )
        except Exception as e:
            last_error = e
            gpt_researcher_logger.error(
                f"Error researching subtopic '{subtopic[:50]}...' on attempt {attempt}/{attempts}: {str(e)}",
                exc_info=True,
            )

    # If all attempts failed or timed out, return a placeholder so the system can continue
    error_msg = (
        f"Timed out or failed to research subtopic after {attempts} attempt(s). "
        f"Subtopic: {subtopic[:100]}... Error: {str(last_error) if last_error else 'unknown'}"
    )
    gpt_researcher_logger.warning(error_msg)
    return f"[Research unavailable due to timeout/error]\n\n{subtopic}\n\n{error_msg}"


async def _parallel_research_node(
    state: LiteratureReviewState,
) -> LiteratureReviewState:
    """
    Node that conducts parallel research for all subtopics using GPTResearcher.
    """
    subtopics = state["subtopics"]
    main_goal = state["goal"]
    
    lit_review_logger.info(f"Starting parallel research for {len(subtopics)} subtopics")
    for i, topic in enumerate(subtopics, 1):
        lit_review_logger.debug(f"  Task {i}: {topic[:100]}...")

    # Create research tasks for all subtopics
    research_tasks = [_write_subtopic_report(topic, main_goal) for topic in subtopics]

    # Execute all research tasks in parallel
    try:
        lit_review_logger.info("Executing research tasks in parallel...")
        start_time = datetime.now()
        
        subtopic_reports = await asyncio.gather(*research_tasks, return_exceptions=True)
        # Convert any exceptions to placeholder strings to avoid aborting the whole phase
        clean_reports: list[str] = []
        for idx, r in enumerate(subtopic_reports):
            if isinstance(r, Exception):
                err_text = (
                    f"[Research task failed]\n\n{subtopics[idx]}\n\n{str(r)}"
                )
                clean_reports.append(err_text)
            else:
                clean_reports.append(r)
        subtopic_reports = clean_reports
        
        elapsed = (datetime.now() - start_time).total_seconds()
        lit_review_logger.info(f"All research tasks completed in {elapsed:.1f}s")
        
        # Log statistics about the reports
        total_length = sum(len(report) for report in subtopic_reports)
        avg_length = total_length / len(subtopic_reports) if subtopic_reports else 0
        lit_review_logger.info(f"Generated {len(subtopic_reports)} reports, total: {total_length} chars, avg: {avg_length:.0f} chars")
        
    except Exception as e:
        lit_review_logger.error(f"Failed to conduct parallel research: {str(e)}", exc_info=True)
        raise RuntimeError(f"Failed to conduct research for subtopics: {str(e)}")

    if state.get("subtopic_reports", False):
        subtopic_reports = state["subtopic_reports"] + subtopic_reports

    lit_review_logger.info(f"Parallel research completed successfully with {len(subtopic_reports)} reports")
    return {"subtopic_reports": subtopic_reports}


def build_literature_review_agent(llm: BaseChatModel) -> StateGraph:
    """
    Builds and configures a LangGraph for literature review.

    Parameters
    ----------
    llm : BaseChatModel
        The language model to use for topic decomposition and executive summary.

    Returns
    -------
    StateGraph
        A compiled LangGraph for the literature review agent.
    """
    lit_review_logger.debug(f"Building literature review agent with LLM: {type(llm).__name__}")
    graph = StateGraph(LiteratureReviewState)

    # Add nodes
    graph.add_node(
        "topic_decomposition",
        lambda state: _topic_decomposition_node(state, llm),
    )

    graph.add_node(
        "parallel_research",
        _parallel_research_node,
    )

    graph.add_edge("topic_decomposition", "parallel_research")
    graph.add_edge("parallel_research", END)

    graph.set_entry_point("topic_decomposition")

    return graph.compile()
