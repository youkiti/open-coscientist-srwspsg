"""
The async task framework with a Supervisor that manages
a task queue, and assigns tasks to Agents.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel

from coscientist.custom_types import HypothesisWithID, ResearchPlanConfig
from coscientist.evolution_agent import build_evolution_agent
from coscientist.generation_agent import (
    build_collaborative_generation_agent,
    build_independent_generation_agent,
)
from coscientist.meta_review_agent import build_meta_review_agent
from coscientist.proximity_agent import ProximityGraph, build_proximity_agent
from coscientist.ranking_agent import EloTournament
from coscientist.reasoning_types import ReasoningType
from coscientist.reflection_agent import build_reflection_agent
from coscientist.supervisor import build_supervisor_agent


class TaskStatus(Enum):
    """Status of a task in the queue."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentType(Enum):
    """Types of specialized agents."""

    GENERATION = "generation"
    REFLECTION = "reflection"
    RANKING = "ranking"
    EVOLUTION = "evolution"
    PROXIMITY = "proximity"
    META_REVIEW = "meta_review"
    SUPERVISOR = "supervisor"


@dataclass
class Task:
    """Represents a task to be executed by an agent."""

    id: str
    agent_type: AgentType
    task_type: str
    parameters: Dict[str, Any]
    priority: int = 5
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class FrameworkState(TypedDict):
    """
    Represents the global state of the research framework.

    Parameters
    ----------
    goal: str
        The main research objective
    research_plan_config: ResearchPlanConfig
        Configuration with preferences, attributes, and constraints
    hypotheses: List[HypothesisWithID]
        All hypotheses generated across all agents
    task_queue: List[Task]
        Queue of tasks to be executed
    active_tasks: List[Task]
        Currently running tasks
    completed_tasks: List[Task]
        Completed tasks with results
    tournament: EloTournament
        Tournament for ranking hypotheses
    proximity_graph: ProximityGraph
        Graph of hypothesis similarities
    agent_performance: Dict[str, Any]
        Performance metrics for each agent
    iteration: int
        Current iteration number
    should_continue: bool
        Whether the research process should continue
    final_results: Dict[str, Any]
        Final research results and recommendations
    """

    goal: str
    research_plan_config: ResearchPlanConfig
    hypotheses: List[HypothesisWithID]
    task_queue: List[Task]
    active_tasks: List[Task]
    completed_tasks: List[Task]
    tournament: EloTournament
    proximity_graph: ProximityGraph
    agent_performance: Dict[str, Any]
    iteration: int
    should_continue: bool
    final_results: Dict[str, Any]


class CoScientistFramework:
    """
    Main framework for coordinating the AI co-scientist multi-agent system.
    """

    def __init__(self, llm: BaseChatModel, max_concurrent_tasks: int = 3):
        self.llm = llm
        self.max_concurrent_tasks = max_concurrent_tasks

        # Build all agent graphs
        # TODO: Add the plan configuration agent
        # TODO: Add the literature review agent
        self.agents = {
            AgentType.GENERATION: {
                "independent": build_independent_generation_agent(
                    "biology", ReasoningType.DEDUCTIVE, llm
                ),
                "collaborative": build_collaborative_generation_agent(
                    ["biologist", "chemist"],
                    {"biologist": "biology", "chemist": "chemistry"},
                    {
                        "biologist": ReasoningType.DEDUCTIVE,
                        "chemist": ReasoningType.INDUCTIVE,
                    },
                    {"biologist": llm, "chemist": llm},
                ),
            },
            AgentType.REFLECTION: build_reflection_agent(llm),
            AgentType.EVOLUTION: build_evolution_agent(llm),
            AgentType.PROXIMITY: build_proximity_agent(llm),
            AgentType.META_REVIEW: build_meta_review_agent(llm),
            AgentType.SUPERVISOR: build_supervisor_agent(llm),
        }

        self._task_counter = 0

    def _generate_task_id(self) -> str:
        """Generate a unique task ID."""
        self._task_counter += 1
        return f"task_{self._task_counter:06d}"

    async def execute_task(self, task: Task, state: FrameworkState) -> Task:
        """
        Execute a single task using the appropriate agent.

        Parameters
        ----------
        task: Task
            The task to execute
        state: FrameworkState
            Current framework state

        Returns
        -------
        Task
            The task with updated status and result
        """
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()

        try:
            # Route task to appropriate agent
            if task.agent_type == AgentType.GENERATION:
                result = await self._execute_generation_task(task, state)
            elif task.agent_type == AgentType.REFLECTION:
                result = await self._execute_reflection_task(task, state)
            elif task.agent_type == AgentType.RANKING:
                result = await self._execute_ranking_task(task, state)
            elif task.agent_type == AgentType.EVOLUTION:
                result = await self._execute_evolution_task(task, state)
            elif task.agent_type == AgentType.PROXIMITY:
                result = await self._execute_proximity_task(task, state)
            elif task.agent_type == AgentType.META_REVIEW:
                result = await self._execute_meta_review_task(task, state)
            else:
                raise ValueError(f"Unknown agent type: {task.agent_type}")

            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()

        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()

        return task

    async def _execute_generation_task(self, task: Task, state: FrameworkState) -> Any:
        """Execute a generation agent task."""
        if task.task_type == "independent_generation":
            # TODO: Move all local imports to the top of the file
            from coscientist.generation_agent import IndependentState

            agent_state = IndependentState(
                goal=state["goal"],
                literature_review="",  # Could be populated from previous tasks
                hypothesis="",
            )

            agent = self.agents[AgentType.GENERATION]["independent"]
            result = agent.invoke(agent_state)

            # Extract hypothesis and add to state
            if "hypothesis" in result:
                new_hypothesis = HypothesisWithID(
                    id=len(state["hypotheses"]) + 1,
                    content=result["hypothesis"],
                    review="",
                )
                state["hypotheses"].append(new_hypothesis)

            return result

        else:
            raise ValueError(f"Unknown generation task type: {task.task_type}")

    async def _execute_reflection_task(self, task: Task, state: FrameworkState) -> Any:
        """Execute a reflection agent task."""
        if not state["hypotheses"]:
            return {"error": "No hypotheses to reflect on"}

        # Take the most recent hypothesis for reflection
        hypothesis = state["hypotheses"][-1]

        from coscientist.reflection_agent import ReflectionState

        agent_state = ReflectionState(
            hypothesis=hypothesis.content,
            initial_filter_assessment="",
            passed_initial_filter=False,
            verification_result="",
        )

        agent = self.agents[AgentType.REFLECTION]
        result = agent.invoke(agent_state)

        # Update hypothesis with review
        hypothesis.review = result.get("verification_result", "")

        return result

    async def _execute_ranking_task(self, task: Task, state: FrameworkState) -> Any:
        """Execute a ranking agent task."""
        if len(state["hypotheses"]) < 2:
            return {"error": "Need at least 2 hypotheses for ranking"}

        # Add hypotheses to tournament if not already added
        for hypothesis in state["hypotheses"]:
            try:
                state["tournament"].add_hypothesis(hypothesis)
            except ValueError:
                # Hypothesis already exists
                pass

        # Run tournament
        state["tournament"].run_tournament()

        return {
            "tournament_complete": True,
            "rankings": state["tournament"].get_sorted_hypotheses(),
        }

    async def _execute_evolution_task(self, task: Task, state: FrameworkState) -> Any:
        """Execute an evolution agent task."""
        # Get top hypotheses from tournament
        if not state["hypotheses"]:
            return {"error": "No hypotheses to evolve"}

        # TODO: Pick random hypotheses instead of top ones
        # Take top 3 hypotheses
        sorted_hypotheses = state["tournament"].get_sorted_hypotheses()
        top_hypotheses = [
            state["tournament"].hypotheses[h_id] for h_id, _ in sorted_hypotheses[:3]
        ]

        from coscientist.evolution_agent import EvolutionState

        agent_state = EvolutionState(
            goal=state["goal"],
            research_plan_config=state["research_plan_config"],
            top_hypotheses=top_hypotheses,
            evolved_hypotheses=[],
        )

        agent = self.agents[AgentType.EVOLUTION]
        result = agent.invoke(agent_state)

        # Add evolved hypotheses to main state
        state["hypotheses"].extend(result["evolved_hypotheses"])

        return result

    async def _execute_proximity_task(self, task: Task, state: FrameworkState) -> Any:
        """Execute a proximity agent task."""
        from coscientist.proximity_agent import ProximityState

        agent_state = ProximityState(
            hypotheses=state["hypotheses"],
            proximity_graph=state["proximity_graph"],
            clusters=[],
        )

        agent = self.agents[AgentType.PROXIMITY]
        result = agent.invoke(agent_state)

        # Update proximity graph in state
        state["proximity_graph"] = result["proximity_graph"]

        return result

    async def _execute_meta_review_task(self, task: Task, state: FrameworkState) -> Any:
        """Execute a meta-review agent task."""
        from coscientist.meta_review_agent import MetaReviewState

        # Collect individual reviews
        individual_reviews = [hyp.review for hyp in state["hypotheses"] if hyp.review]

        agent_state = MetaReviewState(
            goal=state["goal"],
            research_plan_config=state["research_plan_config"],
            tournament=state["tournament"],
            individual_reviews=individual_reviews,
            pattern_analysis="",
            agent_optimization_suggestions="",
            research_overview="",
        )

        agent = self.agents[AgentType.META_REVIEW]
        result = agent.invoke(agent_state)

        # Store results for final output
        state["final_results"]["research_overview"] = result["research_overview"]
        state["final_results"]["optimization_suggestions"] = result[
            "agent_optimization_suggestions"
        ]

        return result

    async def run_framework(
        self,
        goal: str,
        research_plan_config: ResearchPlanConfig,
        max_iterations: int = 10,
    ) -> Dict[str, Any]:
        """
        Run the complete AI co-scientist framework.

        Parameters
        ----------
        goal: str
            Research goal
        research_plan_config: ResearchPlanConfig
            Research configuration
        max_iterations: int
            Maximum number of iterations

        Returns
        -------
        Dict[str, Any]
            Final research results
        """
        # Initialize state
        tournament = EloTournament(
            llm=self.llm,
            goal=goal,
            preferences=research_plan_config.preferences,
            notes="",
            idea_attributes=research_plan_config.attributes,
        )

        state = FrameworkState(
            goal=goal,
            research_plan_config=research_plan_config,
            hypotheses=[],
            task_queue=[],
            active_tasks=[],
            completed_tasks=[],
            tournament=tournament,
            proximity_graph=ProximityGraph(),
            agent_performance={},
            iteration=0,
            should_continue=True,
            final_results={},
        )

        # Create initial tasks
        initial_tasks = [
            Task(
                id=self._generate_task_id(),
                agent_type=AgentType.GENERATION,
                task_type="independent_generation",
                parameters={"field": "biology", "reasoning_type": "deductive"},
                priority=1,
            ),
            Task(
                id=self._generate_task_id(),
                agent_type=AgentType.GENERATION,
                task_type="independent_generation",
                parameters={"field": "chemistry", "reasoning_type": "inductive"},
                priority=1,
            ),
        ]

        state["task_queue"] = initial_tasks

        # Main execution loop
        for iteration in range(max_iterations):
            state["iteration"] = iteration

            if not state["should_continue"] or not state["task_queue"]:
                break

            # Execute tasks concurrently
            await self._execute_task_batch(state)

            # Add follow-up tasks based on current state
            self._schedule_follow_up_tasks(state)

        # TODO: The meta-review agent shouldn't be executed only once at the end. The
        # results of meta-review will be fed back to generation, reflection, and evolution agents.
        # Final meta-review
        final_meta_review_task = Task(
            id=self._generate_task_id(),
            agent_type=AgentType.META_REVIEW,
            task_type="final_review",
            parameters={},
            priority=1,
        )

        state["task_queue"] = [final_meta_review_task]
        await self._execute_task_batch(state)

        return state["final_results"]

    async def _execute_task_batch(self, state: FrameworkState):
        """Execute a batch of tasks concurrently."""
        # Sort tasks by priority
        state["task_queue"].sort(key=lambda t: t.priority)

        # Take up to max_concurrent_tasks
        tasks_to_run = state["task_queue"][: self.max_concurrent_tasks]
        state["task_queue"] = state["task_queue"][self.max_concurrent_tasks :]
        state["active_tasks"].extend(tasks_to_run)

        # Execute tasks concurrently
        if tasks_to_run:
            _ = await asyncio.gather(
                *[self.execute_task(task, state) for task in tasks_to_run],
                return_exceptions=True,
            )

            # Move completed tasks
            for task in tasks_to_run:
                state["active_tasks"].remove(task)
                state["completed_tasks"].append(task)

    def _schedule_follow_up_tasks(self, state: FrameworkState):
        """Schedule follow-up tasks based on current state."""
        # TODO: We should have the supervisor agent decide on next tasks and priorities.
        num_hypotheses = len(state["hypotheses"])

        # Schedule reflection for new hypotheses
        unreviewed_hypotheses = [h for h in state["hypotheses"] if not h.review]
        for _ in unreviewed_hypotheses[:2]:  # Review up to 2 at a time
            task = Task(
                id=self._generate_task_id(),
                agent_type=AgentType.REFLECTION,
                task_type="review_hypothesis",
                parameters={},
                priority=2,
            )
            state["task_queue"].append(task)

        # Schedule ranking when we have enough hypotheses
        if num_hypotheses >= 5 and state["iteration"] % 3 == 0:
            task = Task(
                id=self._generate_task_id(),
                agent_type=AgentType.RANKING,
                task_type="tournament",
                parameters={},
                priority=3,
            )
            state["task_queue"].append(task)

        # Schedule evolution after ranking
        if num_hypotheses >= 3 and state["iteration"] % 4 == 0:
            task = Task(
                id=self._generate_task_id(),
                agent_type=AgentType.EVOLUTION,
                task_type="evolve_top_hypotheses",
                parameters={},
                priority=4,
            )
            state["task_queue"].append(task)

        # Schedule proximity analysis periodically
        if num_hypotheses >= 4 and state["iteration"] % 5 == 0:
            task = Task(
                id=self._generate_task_id(),
                agent_type=AgentType.PROXIMITY,
                task_type="analyze_similarity",
                parameters={},
                priority=5,
            )
            state["task_queue"].append(task)

        # Stop condition
        if state["iteration"] >= 8 or num_hypotheses >= 20:
            state["should_continue"] = False
