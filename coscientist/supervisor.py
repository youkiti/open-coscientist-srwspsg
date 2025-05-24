"""
Supervisor agent
---------------
- Manages a task queue, and assigns tasks to Agents.
- Assesses current progress and decides when to halt.
- Periodically computes and writes to the context memory, a
suite of statistics, including number of hypotheses generated and
those requiring review, and the progress of the tournament.
- It also summarizes the effectiveness of different Agents (e.g,
are new ideas from the Generation agent better than refined ideas
from the Evolution agent?), and gives them more work if they are
performing well.
"""

import json
from typing import Any, Dict, List

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, StateGraph

from coscientist.common import load_prompt
from coscientist.custom_types import HypothesisWithID, ResearchPlanConfig


class SupervisorState(TypedDict):
    """
    Represents the state of the supervisor and research process.

    Parameters
    ----------
    goal: str
        The main research objective
    research_plan_config: ResearchPlanConfig
        Configuration with preferences, attributes, and constraints
    hypotheses: List[HypothesisWithID]
        All hypotheses generated so far
    task_queue: List[Dict[str, Any]]
        Queue of tasks to be executed by specialized agents
    iteration: int
        Current iteration number
    agent_performance: Dict[str, Any]
        Performance metrics for each agent type
    should_continue: bool
        Whether the research process should continue
    research_summary: str
        Current summary of research progress
    """

    goal: str
    research_plan_config: ResearchPlanConfig
    hypotheses: List[HypothesisWithID]
    task_queue: List[Dict[str, Any]]
    iteration: int
    agent_performance: Dict[str, Any]
    should_continue: bool
    research_summary: str


def planning_node(state: SupervisorState, llm: BaseChatModel) -> SupervisorState:
    """
    Creates the initial research plan and populates the task queue.

    Parameters
    ----------
    state: SupervisorState
        Current supervisor state
    llm: BaseChatModel
        Language model for planning

    Returns
    -------
    SupervisorState
        Updated state with initial tasks queued
    """

    planning_prompt = f"""
    You are a scientific research supervisor planning a multi-agent research process.

    Research Goal: {state["goal"]}
    
    Research Plan Configuration:
    - Preferences: {state["research_plan_config"].preferences}
    - Attributes: {state["research_plan_config"].attributes}
    - Constraints: {state["research_plan_config"].constraints}

    Create an initial research plan with specific tasks for the following agents:
    1. Generation Agent - Generate initial hypotheses
    2. Reflection Agent - Review hypotheses  
    3. Ranking Agent - Run tournaments to rank hypotheses
    4. Evolution Agent - Refine top hypotheses
    5. Meta-review Agent - Synthesize findings

    Return a JSON list of initial tasks in this format:
    [
        {{
            "agent_type": "generation",
            "task_type": "independent_generation", 
            "priority": 1,
            "parameters": {{"field": "biology", "reasoning_type": "deductive"}}
        }},
        ...
    ]
    """

    response = llm.invoke(planning_prompt)
    try:
        tasks = json.loads(response.content)
        if not isinstance(tasks, list):
            tasks = []
    except (json.JSONDecodeError, AttributeError):
        # Fallback to default tasks
        tasks = [
            {
                "agent_type": "generation",
                "task_type": "independent_generation",
                "priority": 1,
                "parameters": {"field": "biology", "reasoning_type": "deductive"},
            }
        ]

    return {
        **state,
        "task_queue": tasks,
        "iteration": 1,
        "agent_performance": {},
        "should_continue": True,
        "research_summary": "Research planning initiated.",
    }


def progress_assessment_node(
    state: SupervisorState, llm: BaseChatModel
) -> SupervisorState:
    """
    Assesses current research progress and decides whether to continue.

    Parameters
    ----------
    state: SupervisorState
        Current supervisor state
    llm: BaseChatModel
        Language model for assessment

    Returns
    -------
    SupervisorState
        Updated state with progress assessment
    """

    num_hypotheses = len(state["hypotheses"])

    assessment_prompt = f"""
    You are assessing the progress of a scientific research process.

    Research Goal: {state["goal"]}
    Current Iteration: {state["iteration"]}
    Number of Hypotheses Generated: {num_hypotheses}
    
    Research Summary: {state["research_summary"]}
    
    Agent Performance: {state["agent_performance"]}

    Evaluate whether the research should continue based on:
    1. Quality and quantity of hypotheses generated
    2. Progress toward the research goal
    3. Diminishing returns from further iteration
    4. Resource efficiency

    Respond with either "CONTINUE" or "STOP" followed by a brief explanation.
    If continuing, suggest next priority tasks.
    """

    response = llm.invoke(assessment_prompt)
    response_text = response.content.upper()

    should_continue = "CONTINUE" in response_text

    # Generate next tasks if continuing
    next_tasks = []
    if should_continue and state["iteration"] < 10:  # Max 10 iterations
        if num_hypotheses < 5:
            # Need more hypotheses
            next_tasks.append(
                {
                    "agent_type": "generation",
                    "task_type": "independent_generation",
                    "priority": 1,
                    "parameters": {"field": "biology", "reasoning_type": "inductive"},
                }
            )
        elif num_hypotheses >= 5:
            # Ready for ranking and evolution
            next_tasks.extend(
                [
                    {
                        "agent_type": "ranking",
                        "task_type": "tournament",
                        "priority": 2,
                        "parameters": {},
                    },
                    {
                        "agent_type": "evolution",
                        "task_type": "refine_top_hypotheses",
                        "priority": 3,
                        "parameters": {},
                    },
                ]
            )

    return {
        **state,
        "should_continue": should_continue,
        "task_queue": state["task_queue"] + next_tasks,
        "research_summary": response.content,
    }


def task_dispatch_node(state: SupervisorState, llm: BaseChatModel) -> SupervisorState:
    """
    Dispatches the next highest priority task from the queue.

    Parameters
    ----------
    state: SupervisorState
        Current supervisor state
    llm: BaseChatModel
        Language model (not used in this node)

    Returns
    -------
    SupervisorState
        Updated state with task dispatched
    """

    if not state["task_queue"]:
        return {
            **state,
            "should_continue": False,
            "research_summary": "No more tasks in queue. Research complete.",
        }

    # Sort tasks by priority (lower number = higher priority)
    sorted_tasks = sorted(state["task_queue"], key=lambda x: x.get("priority", 999))

    # Take the highest priority task
    next_task = sorted_tasks[0]
    remaining_tasks = sorted_tasks[1:]

    # Update agent performance tracking
    agent_type = next_task["agent_type"]
    if agent_type not in state["agent_performance"]:
        state["agent_performance"][agent_type] = {
            "tasks_completed": 0,
            "hypotheses_generated": 0,
        }

    state["agent_performance"][agent_type]["tasks_completed"] += 1

    return {
        **state,
        "task_queue": remaining_tasks,
        "iteration": state["iteration"] + 1,
        "research_summary": f"Dispatched {next_task['task_type']} task to {agent_type} agent.",
    }


def build_supervisor_agent(llm: BaseChatModel):
    """
    Builds and configures a LangGraph for the supervisor agent process.

    The graph orchestrates the overall research process by:
    1. Creating initial research plans
    2. Assessing progress and deciding whether to continue
    3. Dispatching tasks to specialized agents

    Parameters
    ----------
    llm: BaseChatModel
        The language model to use for supervisor decisions

    Returns
    -------
    StateGraph
        A compiled LangGraph for the supervisor agent
    """
    graph = StateGraph(SupervisorState)

    # Add nodes
    graph.add_node("planning", lambda state: planning_node(state, llm))
    graph.add_node(
        "progress_assessment", lambda state: progress_assessment_node(state, llm)
    )
    graph.add_node("task_dispatch", lambda state: task_dispatch_node(state, llm))

    # Define transitions
    def route_after_planning(state: SupervisorState):
        return "progress_assessment"

    def route_after_assessment(state: SupervisorState):
        if state["should_continue"] and state["task_queue"]:
            return "task_dispatch"
        return END

    def route_after_dispatch(state: SupervisorState):
        return "progress_assessment"

    # Add conditional edges
    graph.add_edge("planning", "progress_assessment")

    graph.add_conditional_edges(
        "progress_assessment",
        route_after_assessment,
        {"task_dispatch": "task_dispatch", END: END},
    )

    graph.add_edge("task_dispatch", "progress_assessment")

    # Set entry point
    graph.set_entry_point("planning")

    return graph.compile()
