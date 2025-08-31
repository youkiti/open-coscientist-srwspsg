"""
The overall framework that takes a CoscientistStateManager from global_state.py,
setups the agents, and organizes the multi-agent system. The framework will be controlled
by a supervisor agent.
"""

import logging
import math
import random

import numpy as np
from langchain_anthropic import ChatAnthropic
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datetime import datetime

from coscientist.openai_client import create_openai_responses_client
from coscientist.progress_tracker import ProgressTracker, ProgressPhase, ProgressStatus

from coscientist.evolution_agent import build_evolution_agent
from coscientist.final_report_agent import build_final_report_agent
from coscientist.generation_agent import (
    CollaborativeConfig,
    IndependentConfig,
    build_generation_agent,
)
from coscientist.global_state import CoscientistStateManager
from coscientist.literature_review_agent import build_literature_review_agent
from coscientist.meta_review_agent import build_meta_review_agent
from coscientist.reasoning_types import ReasoningType
from coscientist.reflection_agent import build_deep_verification_agent
from coscientist.supervisor_agent import build_supervisor_agent

# Configure logger for framework
framework_logger = logging.getLogger('coscientist.framework')

# Generally reasoning models are better suited for the scientific reasoning
# tasks entailed by the Coscientist system.
_SMARTER_LLM_POOL = {
    "o3": ChatOpenAI(model="o3", max_tokens=50_000, max_retries=3),
    "gpt-5": create_openai_responses_client(
        model="gpt-5", 
        max_tokens=50_000, 
        max_retries=3,
        reasoning_effort="high"
    ),
    "gemini-2.5-pro": ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=1.0,
        max_retries=3,
        max_tokens=50_000,
    ),
    "claude-opus-4-1-20250805": ChatAnthropic(
        model="claude-opus-4-1-20250805", max_tokens=32_000, max_retries=3
    ),
    "claude-sonnet-4-20250514": ChatAnthropic(
        model="claude-sonnet-4-20250514", max_tokens=32_000, max_retries=3
    ),
}
_CHEAPER_LLM_POOL = {
    "o4-mini": ChatOpenAI(model="o4-mini", max_tokens=50_000, max_retries=3),
    "gpt-5-mini": create_openai_responses_client(
        model="gpt-5",
        max_tokens=25_000,
        max_retries=3,
        reasoning_effort="medium"
    ),
    "gemini-2.5-flash": ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=1.0,
        max_retries=3,
        max_tokens=50_000,
    ),
    # Anthropic doesn't have a good cheaper model
    "claude-sonnet-4-20250514": ChatAnthropic(
        model="claude-sonnet-4-20250514", max_tokens=32_000, max_retries=3
    ),
}


class CoscientistConfig:
    """
    Configuration for the Coscientist system.

    Note that the config for GPTResearcher which is used throughout the system
    is defined in `researcher_config.json`.

    Attributes
    ----------
    literature_review_agent_llm : BaseChatModel
        The language model for the literature review. This LLM decides on the research
        subtopics for GPTResearcher.
    generation_agent_llms : dict[str, BaseChatModel]
        The language models for the generation agents
    reflection_agent_llms : dict[str, BaseChatModel]
        The language models for the reflection agents
    evolution_agent_llms : dict[str, BaseChatModel]
        The language models for the evolution agents
    meta_review_agent_llm : BaseChatModel
        The language model for the meta-review. Gemini works best because of the long
        context window that isn't severely rate limited like other providers.
    proximity_agent_embedding_model : Embeddings
        The embedding model for the proximity agent
    specialist_fields : list[str]
        The fields of expertise for generation agents. This list should be expanded
        by the configuration agent.
    debug_mode : bool
        Enable debug mode with verbose logging and phase checkpoints.
    pause_after_literature_review : bool
        Pause execution after literature review phase for debugging.
    save_on_error : bool
        Automatically save checkpoint when errors occur.

    """

    def __init__(
        self,
        literature_review_agent_llm: BaseChatModel = _SMARTER_LLM_POOL[
            "claude-opus-4-1-20250805"
        ],
        generation_agent_llms: dict[str, BaseChatModel] = _SMARTER_LLM_POOL,
        reflection_agent_llms: dict[str, BaseChatModel] = _SMARTER_LLM_POOL,
        evolution_agent_llms: dict[str, BaseChatModel] = _SMARTER_LLM_POOL,
        meta_review_agent_llm: BaseChatModel = _CHEAPER_LLM_POOL["gemini-2.5-flash"],
        supervisor_agent_llm: BaseChatModel = _SMARTER_LLM_POOL[
            "claude-opus-4-1-20250805"
        ],
        final_report_agent_llm: BaseChatModel = _SMARTER_LLM_POOL[
            "claude-opus-4-1-20250805"
        ],
        proximity_agent_embedding_model: Embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", dimensions=256
        ),
        specialist_fields: list[str] | None = None,
        debug_mode: bool = False,
        pause_after_literature_review: bool = False,
        save_on_error: bool = True,
    ):
        # TODO: Add functionality for overriding GPTResearcher config.
        self.literature_review_agent_llm = literature_review_agent_llm
        self.generation_agent_llms = generation_agent_llms
        self.reflection_agent_llms = reflection_agent_llms
        self.evolution_agent_llms = evolution_agent_llms
        self.meta_review_agent_llm = meta_review_agent_llm
        self.supervisor_agent_llm = supervisor_agent_llm
        self.proximity_agent_embedding_model = proximity_agent_embedding_model
        self.final_report_agent_llm = final_report_agent_llm
        if specialist_fields is None:
            self.specialist_fields = ["biology"]
        else:
            self.specialist_fields = specialist_fields
        self.debug_mode = debug_mode
        self.pause_after_literature_review = pause_after_literature_review
        self.save_on_error = save_on_error


class CoscientistFramework:
    """
    The framework that takes a CoscientistStateManager from global_state.py,
    setups the agents, and organizes the multi-agent system. The framework will be controlled
    by a supervisor agent.
    """

    def __init__(
        self, config: CoscientistConfig, state_manager: CoscientistStateManager
    ):
        self.config = config
        self.state_manager = state_manager
        
        # Initialize progress tracker
        self.progress_tracker = ProgressTracker(goal=state_manager.goal)
        self.progress_tracker.start_phase(
            ProgressPhase.INITIALIZING,
            "Framework initialized, ready to begin research"
        )
        
        framework_logger.info(f"Initialized CoscientistFramework for goal: {state_manager.goal}")
        framework_logger.debug(f"Output directory: {state_manager.output_dir}")
        framework_logger.debug(f"Configuration: LLMs loaded for all agents")

    def list_generation_llm_names(self) -> list[str]:
        """
        List the names of the generation agents.
        """
        return list(self.config.generation_agent_llms.keys())

    def list_generation_modes(self) -> list[str]:
        """
        List the names of the generation modes.
        """
        return ["independent", "collaborative"]

    def list_reflection_llm_names(self) -> list[str]:
        """
        List the names of the reflection agents.
        """
        return list(self.config.reflection_agent_llms.keys())

    def list_evolution_llm_names(self) -> list[str]:
        """
        List the names of the evolution agents.
        """
        return list(self.config.evolution_agent_llms.keys())

    def list_evolution_modes(self) -> list[str]:
        """
        List the names of the evolution modes.
        """
        return ["evolve_from_feedback", "out_of_the_box"]

    def list_specialist_fields(self) -> list[str]:
        """
        List the names of the specialist fields.
        """
        return self.config.specialist_fields

    def list_reasoning_types(self) -> list[str]:
        """
        List the names of the reasoning types.
        """
        return list(ReasoningType.__members__.keys())

    def get_semantic_communities(
        self, resolution: float = 1.0, min_weight: float = 0.85
    ) -> list[set[str]]:
        """
        Get the semantic communities of the hypotheses.
        """
        self.state_manager.proximity_graph.update_edges()
        return self.state_manager.proximity_graph.get_semantic_communities(
            resolution=resolution, min_weight=min_weight
        )

    def process_reflection_queue(self) -> None:
        """
        Process all hypotheses in the reflection queue through deep verification.

        This method pops hypotheses from the reflection queue until it's empty,
        runs them through deep verification, and adds the reviewed hypotheses
        to the state manager.
        """
        while not self.state_manager.reflection_queue_is_empty:
            # This pops from the reflection queue until it's empty
            initial_reflection_state = self.state_manager.next_reflection_state()
            llm_name = random.choice(self.list_reflection_llm_names())
            reflection_agent = build_deep_verification_agent(
                llm=self.config.reflection_agent_llms[llm_name],
                review_llm=self.config.meta_review_agent_llm,
                parallel=False,
                checkpointer=None,
            )
            final_reflection_state = reflection_agent.invoke(initial_reflection_state)
            if final_reflection_state["passed_initial_filter"]:
                self.state_manager.add_reviewed_hypothesis(
                    final_reflection_state["reviewed_hypothesis"]
                )
                self.state_manager.advance_reviewed_hypothesis()

    def _generate_new_hypothesis(self) -> None:
        """
        Run the hypothesis generation for a given mode and config.
        """
        # TODO: The mode and roles should be selected by the supervisor agent.
        # Randomly pick a mode, a reasoning type, and a specialist field.
        mode = random.choice(self.list_generation_modes())
        if mode == "independent":
            llm_name = random.choice(self.list_generation_llm_names())
            reasoning_type = random.choice(self.list_reasoning_types())
            specialist_field = random.choice(self.list_specialist_fields())
            config = IndependentConfig(
                llm=self.config.generation_agent_llms[llm_name],
                reasoning_type=getattr(ReasoningType, reasoning_type),
                field=specialist_field,
            )
            first_agent_name = None
        elif mode == "collaborative":
            llm_names = np.random.choice(self.list_generation_llm_names(), 2).tolist()
            specialist_fields = np.random.choice(
                self.list_specialist_fields(), 2
            ).tolist()
            reasoning_types = np.random.choice(self.list_reasoning_types(), 2).tolist()

            agent_names = [
                f"{llm_name}_{field}"
                for llm_name, field in zip(llm_names, specialist_fields)
            ]
            config = CollaborativeConfig(
                agent_names=agent_names,
                agent_fields=dict(zip(agent_names, specialist_fields)),
                agent_reasoning_types={
                    name: getattr(ReasoningType, reasoning_type)
                    for name, reasoning_type in zip(agent_names, reasoning_types)
                },
                llms={
                    name: self.config.generation_agent_llms[llm_name]
                    for name, llm_name in zip(agent_names, llm_names)
                },
                max_turns=10,
            )
            first_agent_name = agent_names[0]

        # TODO: Make this async
        generation_agent = build_generation_agent(mode, config)
        initial_generation_state = self.state_manager.next_generation_state(
            mode, first_agent_name
        )
        final_generation_state = generation_agent.invoke(initial_generation_state)
        self.state_manager.add_generated_hypothesis(
            final_generation_state["hypothesis"]
        )

    async def start(self, n_hypotheses: int = 8) -> None:
        """
        Starts the Coscientist system with a fixed number of initial
        hypotheses.
        """
        framework_logger.info(f"Starting Coscientist system with {n_hypotheses} initial hypotheses")
        assert n_hypotheses >= 2, "Must generate at least two hypotheses to start"
        if self.state_manager.is_started:
            framework_logger.warning("Attempted to start already-started system")
            raise ValueError(
                "Coscientist system has already been started. "
                f"Use one of {self.available_actions()} instead!"
            )

        try:
            # Perform the initial literature review.
            if not self.state_manager.has_literature_review:
                framework_logger.info("=== PHASE: LITERATURE REVIEW ===")
                self.progress_tracker.start_phase(
                    ProgressPhase.LITERATURE_REVIEW,
                    f"Starting comprehensive literature review for: {self.state_manager.goal}"
                )
                
                framework_logger.info(f"Building literature review agent with LLM: {type(self.config.literature_review_agent_llm).__name__}")
                literature_review_agent = build_literature_review_agent(
                    self.config.literature_review_agent_llm
                )
                initial_lit_review_state = self.state_manager.next_literature_review_state(
                    # TODO: Make this configurable
                    max_subtopics=5
                )
                framework_logger.debug(f"Initial literature review state prepared with max_subtopics=5")
                
                self.progress_tracker.update_phase_progress(
                    "Decomposing research goal into focused subtopics"
                )
                framework_logger.info("Starting topic decomposition and research...")
                
                start_time = datetime.now()
                final_lit_review_state = await literature_review_agent.ainvoke(
                    initial_lit_review_state
                )
                elapsed = (datetime.now() - start_time).total_seconds()
                
                self.state_manager.update_literature_review(final_lit_review_state)
                
                subtopics = final_lit_review_state.get('subtopics', [])
                framework_logger.info(f"Literature review completed in {elapsed:.1f}s with {len(subtopics)} subtopics")
                for i, topic in enumerate(subtopics, 1):
                    framework_logger.debug(f"  Subtopic {i}: {topic[:100]}...")
                
                self.progress_tracker.complete_phase(
                    f"Literature review completed with {len(subtopics)} subtopics"
                )
                
                # Save checkpoint if debug mode enabled
                if self.config.debug_mode:
                    framework_logger.info("Debug mode: Saving checkpoint after literature review")
                    checkpoint_path = self.state_manager._state.save()
                    framework_logger.info(f"Checkpoint saved to: {checkpoint_path}")
                
                # Pause if requested for debugging
                if self.config.pause_after_literature_review:
                    framework_logger.warning("Pausing after literature review as requested")
                    framework_logger.warning("Set config.pause_after_literature_review=False to continue")
                    return

            # TODO: Make this async
            hypotheses_to_generate = max(0, n_hypotheses - self.state_manager.total_hypotheses)
            framework_logger.info(f"Proceeding to generate {hypotheses_to_generate} hypotheses")
            _ = await self.generate_new_hypotheses(
                n_hypotheses=hypotheses_to_generate
            )

            # Run the EloTournament
            # The top k for the bracket should the nearest power of
            # 2 less than the number of hypotheses and no more than 16.
            k_bracket = min(16, 2 ** math.floor(math.log2(n_hypotheses)))
            framework_logger.info(f"Running initial tournament with k_bracket={k_bracket}")
            # TODO: Figure out the right LLM for this job; should it be different from meta-review?
            # Feels like it should be fixed for the sake of consistency though
            _ = await self.run_tournament(k_bracket=k_bracket)
            _ = await self.run_meta_review(k_bracket=k_bracket)
            
        except Exception as e:
            framework_logger.error(f"Critical error during startup phase: {str(e)}", exc_info=True)
            
            # Save checkpoint on error if configured
            if self.config.save_on_error:
                try:
                    framework_logger.info("Attempting to save error checkpoint...")
                    checkpoint_path = self.state_manager._state.save()
                    framework_logger.info(f"Error checkpoint saved to: {checkpoint_path}")
                except Exception as save_error:
                    framework_logger.error(f"Failed to save error checkpoint: {save_error}")
            
            self.progress_tracker.report_error(
                f"Error during startup phase: {str(e)}",
                error_info=str(e)
            )
            raise

    async def generate_new_hypotheses(self, n_hypotheses: int = 2) -> None:
        """
        Generate new hypotheses.
        """
        if n_hypotheses > 0:
            framework_logger.info(f"=== PHASE: HYPOTHESIS GENERATION ===")
            framework_logger.info(f"Generating {n_hypotheses} new hypotheses")
            self.progress_tracker.start_phase(
                ProgressPhase.HYPOTHESIS_GENERATION,
                f"Generating {n_hypotheses} new hypotheses using multi-agent collaboration"
            )
            
            for i in range(n_hypotheses):
                framework_logger.info(f"Generating hypothesis {i + 1} of {n_hypotheses}")
                start_time = datetime.now()
                progress_pct = ((i + 1) / n_hypotheses) * 50  # Generation is 50% of this phase
                self.progress_tracker.update_phase_progress(
                    f"Generated hypothesis {i + 1} of {n_hypotheses}",
                    progress_percentage=progress_pct
                )
                self._generate_new_hypothesis()
                self.state_manager.advance_hypothesis(kind="generated")
                elapsed = (datetime.now() - start_time).total_seconds()
                framework_logger.debug(f"Hypothesis {i + 1} generated in {elapsed:.1f}s")

            # Now run through the review queue and perform deep verification
            framework_logger.info("=== PHASE: REFLECTION/VERIFICATION ===")
            self.progress_tracker.update_phase_progress(
                "Starting deep verification and reflection process",
                progress_percentage=50
            )
            framework_logger.info("Processing reflection queue for generated hypotheses")
            self.process_reflection_queue()
            self.state_manager.update_proximity_graph_edges()
            
            framework_logger.info(f"Successfully generated and verified {n_hypotheses} hypotheses")
            self.progress_tracker.complete_phase(
                f"Generated and verified {n_hypotheses} hypotheses successfully"
            )

    async def evolve_hypotheses(self, n_hypotheses: int = 4) -> None:
        """
        Takes the top (n_hypotheses // 2) hypotheses and evolves them. Also
        randomly selects (n_hypotheses // 2) hypotheses to evolve.
        """
        assert n_hypotheses >= 2, "Must evolve at least two hypotheses"
        assert self.state_manager.is_started, "Coscientist system must be started first"
        
        self.progress_tracker.start_phase(
            ProgressPhase.EVOLUTION,
            f"Evolving {n_hypotheses} hypotheses based on tournament feedback"
        )
        
        evolution_candidate_uids = (
            self.state_manager.get_tournament_hypotheses_for_evolution()
        )
        if len(evolution_candidate_uids) < n_hypotheses:
            logging.warning(
                f"Only {len(evolution_candidate_uids)} hypotheses are qualified for evolution. "
                f"Evolving {len(evolution_candidate_uids)} hypotheses."
            )
            n_hypotheses = len(evolution_candidate_uids)

        # The first uids are the top ranked hypotheses
        top_ranked_uids = evolution_candidate_uids[: (n_hypotheses // 2)]
        # The rest are randomly selected
        random_uids = np.random.choice(
            evolution_candidate_uids[(n_hypotheses // 2) :],
            size=n_hypotheses // 2,
            replace=False,
        ).tolist()

        # Evolve the top ranked and random hypotheses based on feedback
        total_evolutions = len(top_ranked_uids + random_uids) + 1  # +1 for out-of-the-box
        current_evolution = 0
        
        for uid in top_ranked_uids + random_uids:
            progress_pct = (current_evolution / total_evolutions) * 70  # Evolution is 70% of this phase
            self.progress_tracker.update_phase_progress(
                f"Evolving hypothesis {current_evolution + 1} of {total_evolutions} from feedback",
                progress_percentage=progress_pct
            )
            
            initial_evolution_state = self.state_manager.next_evolution_state(
                mode="evolve_from_feedback", uid_to_evolve=uid
            )
            llm_name = random.choice(self.list_evolution_llm_names())
            evolution_agent = build_evolution_agent(
                mode="evolve_from_feedback",
                llm=self.config.evolution_agent_llms[llm_name],
            )
            final_evolution_state = evolution_agent.invoke(initial_evolution_state)
            self.state_manager.add_evolved_hypothesis(
                final_evolution_state["evolved_hypothesis"]
            )
            self.state_manager.advance_hypothesis(kind="evolved")
            current_evolution += 1

        # Run one round instance of evolving the top ranked hypotheses
        # into something new
        self.progress_tracker.update_phase_progress(
            "Creating novel hypothesis through out-of-the-box evolution",
            progress_percentage=70
        )
        
        out_of_box_initial_state = self.state_manager.next_evolution_state(
            mode="out_of_the_box",
            top_k=n_hypotheses // 2,
        )
        llm_name = random.choice(self.list_evolution_llm_names())
        evolution_agent = build_evolution_agent(
            mode="out_of_the_box", llm=self.config.evolution_agent_llms[llm_name]
        )
        out_of_box_state = evolution_agent.invoke(out_of_box_initial_state)
        self.state_manager.add_evolved_hypothesis(
            out_of_box_state["evolved_hypothesis"]
        )

        # Move the evolved hypotheses to the reflection queue
        self.progress_tracker.update_phase_progress(
            "Processing evolved hypotheses through reflection queue",
            progress_percentage=85
        )
        self.state_manager.advance_hypothesis(kind="evolved")

        # TODO: Do we have to worry about reflecting on hypotheses that are
        # already in the reflection queue but weren't advanced yet?
        # Do we always want to run reflection immediately after a hypothesis
        # is generated?
        self.process_reflection_queue()

        # Move the reviewed hypothesis to the EloTournament.
        self.state_manager.update_proximity_graph_edges()
        
        self.progress_tracker.complete_phase(
            f"Evolution completed - generated {total_evolutions} evolved hypotheses"
        )

    async def expand_literature_review(self) -> None:
        """
        Expands the literature review by adding more subtopics.
        """
        initial_lit_review_state = self.state_manager.next_literature_review_state(
            # TODO: Make this configurable
            max_subtopics=5
        )
        literature_review_agent = build_literature_review_agent(
            self.config.literature_review_agent_llm
        )
        final_lit_review_state = await literature_review_agent.ainvoke(
            initial_lit_review_state
        )
        self.state_manager.update_literature_review(final_lit_review_state)

    async def run_tournament(self, k_bracket: int = 8) -> None:
        k_bracket = min(
            k_bracket,
            2 ** math.floor(math.log2(self.state_manager.num_tournament_hypotheses)),
        )
        
        self.progress_tracker.start_phase(
            ProgressPhase.TOURNAMENT,
            f"Running ELO tournament with top {k_bracket} hypotheses"
        )
        
        self.progress_tracker.update_phase_progress(
            f"Starting head-to-head competitions for {self.state_manager.num_tournament_hypotheses} hypotheses"
        )
        
        self.state_manager.run_tournament(
            llm=self.config.meta_review_agent_llm, k_bracket=k_bracket
        )
        
        self.progress_tracker.complete_phase(
            f"Tournament completed with {k_bracket} hypotheses ranked by ELO rating"
        )

    async def run_meta_review(self, k_bracket: int = 8) -> None:
        self.progress_tracker.start_phase(
            ProgressPhase.META_REVIEW,
            f"Analyzing and synthesizing insights from top {k_bracket} hypotheses"
        )
        
        initial_meta_review_state = self.state_manager.next_meta_review_state(
            top_k=k_bracket
        )
        
        self.progress_tracker.update_phase_progress(
            "Performing comprehensive meta-analysis of research findings"
        )
        
        meta_review_agent = build_meta_review_agent(self.config.meta_review_agent_llm)
        final_meta_review_state = meta_review_agent.invoke(initial_meta_review_state)
        self.state_manager.update_meta_review(final_meta_review_state)
        
        self.progress_tracker.complete_phase(
            f"Meta-review completed with comprehensive analysis of {k_bracket} top hypotheses"
        )

    async def finish(self) -> None:
        self.progress_tracker.start_phase(
            ProgressPhase.FINAL_REPORT,
            "Generating comprehensive final research report"
        )
        
        initial_final_report_state = self.state_manager.next_final_report_state(top_k=3)
        
        self.progress_tracker.update_phase_progress(
            "Compiling research findings and generating final report"
        )
        
        final_report_agent = build_final_report_agent(
            self.config.final_report_agent_llm
        )
        final_report_state = final_report_agent.invoke(initial_final_report_state)
        self.state_manager.update_final_report(final_report_state)
        
        self.progress_tracker.complete_phase(
            "Final research report generated successfully"
        )
        
        # Mark the entire research process as completed
        self.progress_tracker.complete_research()

    @classmethod
    def available_actions(self) -> list[str]:
        """
        List the available actions for the Coscientist system.
        """
        return [
            "generate_new_hypotheses",
            "evolve_hypotheses",
            "expand_literature_review",
            "run_tournament",
            "run_meta_review",
            "finish",
        ]

    async def run(self) -> tuple[str, str]:
        """
        Runs the coscientist system until it is finished.
        """
        framework_logger.info("="*60)
        framework_logger.info("Starting Coscientist Research Session")
        framework_logger.info(f"Goal: {self.state_manager.goal}")
        framework_logger.info("="*60)
        
        try:
            # Start off with 4 hypotheses
            if not self.state_manager.is_started:
                framework_logger.info("System not started, initializing with 4 hypotheses")
                _ = await self.start(n_hypotheses=4)

            supervisor_agent = build_supervisor_agent(self.config.supervisor_agent_llm)
            framework_logger.info(f"Supervisor agent initialized with LLM: {type(self.config.supervisor_agent_llm).__name__}")

            current_action = None
            iteration = 0
            while not self.state_manager.is_finished:
                iteration += 1
                framework_logger.info(f"\n--- Supervisor Decision Iteration {iteration} ---")
                
                initial_supervisor_state = self.state_manager.next_supervisor_state()
                framework_logger.debug(f"Available actions: {self.available_actions()}")
                
                final_supervisor_state = supervisor_agent.invoke(initial_supervisor_state)
                current_action = final_supervisor_state["action"]
                reasoning = final_supervisor_state.get("decision_reasoning", "")
                
                framework_logger.info(f"Supervisor decision: {current_action}")
                framework_logger.debug(f"Reasoning: {reasoning}")
                
                assert (
                    current_action in self.available_actions()
                ), f"Invalid action: {current_action}. Available actions: {self.available_actions()}"
                self.state_manager.update_supervisor_decision(final_supervisor_state)
                self.state_manager.add_action(current_action)
                
                # Update progress before executing action
                self.progress_tracker.update_phase_progress(
                    f"Supervisor decided to: {current_action}",
                    details={"action": current_action, "reasoning": reasoning}
                )
                
                framework_logger.info(f"Executing action: {current_action}")
                action_start = datetime.now()
                _ = await getattr(self, current_action)()
                action_elapsed = (datetime.now() - action_start).total_seconds()
                framework_logger.info(f"Action '{current_action}' completed in {action_elapsed:.1f}s")

            framework_logger.info("="*60)
            framework_logger.info("Research Session Completed Successfully")
            framework_logger.info(f"Total iterations: {iteration}")
            framework_logger.info("="*60)
            
            return self.state_manager.final_report, self.state_manager.meta_reviews[-1]
            
        except Exception as e:
            framework_logger.error(f"Critical error in research process: {str(e)}", exc_info=True)
            self.progress_tracker.report_error(
                f"Critical error in research process: {str(e)}",
                error_info=str(e)
            )
            raise
