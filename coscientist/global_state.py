import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import List, Literal, Optional, Union

from langchain_core.language_models import BaseChatModel

from coscientist.custom_types import ParsedHypothesis, ReviewedHypothesis
from coscientist.evolution_agent import EvolveFromFeedbackState, OutOfTheBoxState
from coscientist.generation_agent import CollaborativeState, IndependentState
from coscientist.literature_review_agent import LiteratureReviewState
from coscientist.meta_review_agent import MetaReviewTournamentState
from coscientist.proximity_agent import ProximityGraph
from coscientist.ranking_agent import EloTournament
from coscientist.reflection_agent import ReflectionState

# Global configuration for output directory
_OUTPUT_DIR = os.environ.get("COSCIENTIST_DIR", os.path.expanduser("~/.coscientist"))


def _maybe_save(n: int = 1):
    """
    Decorator to auto-save state after every n method calls.

    Each decorated method maintains its own independent counter.

    Parameters
    ----------
    n : int
        Save frequency - save after every n calls (default: 1, set to 0 to disable)
    """

    def decorator(func):
        # Initialize counter for this specific method
        func._save_counter = 0
        func._save_frequency = n

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Execute the original method
            result = func(self, *args, **kwargs)

            # Handle auto-save logic if enabled
            if func._save_frequency > 0:
                func._save_counter += 1
                if func._save_counter >= func._save_frequency:
                    func._save_counter = 0  # Reset counter
                    self._state.save()

            return result

        return wrapper

    return decorator


@dataclass
class CoscientistState:
    """
    Global state for the Coscientist multi-agent system.

    This class manages the outputs from all six main agents and provides
    persistence capabilities for the entire system state.
    """

    goal: str
    literature_review: Optional[LiteratureReviewState]
    generated_hypotheses: List[Union[IndependentState, CollaborativeState]]
    reviewed_hypotheses: List[ReviewedHypothesis]
    tournament: Optional[EloTournament]
    evolved_hypotheses: List[Union[EvolveFromFeedbackState, OutOfTheBoxState]]
    meta_review: Optional[MetaReviewTournamentState]
    proximity_graph: Optional[ProximityGraph]
    reflection_queue: List[ParsedHypothesis]
    _iteration: int = 0  # Hidden parameter for tracking saves

    def __init__(self, goal: str):
        self.goal = goal
        self.literature_review = None
        self.generated_hypotheses = []
        self.reviewed_hypotheses = []
        self.tournament = None
        self.evolved_hypotheses = []
        self.meta_review = None
        self.proximity_graph = None
        self.reflection_queue = []
        self._iteration = 0

    # Persistence methods
    def save(self) -> str:
        """
        Save the current state to a pickle file.

        Returns
        -------
        str
            Path to the saved checkpoint file
        """
        directory = _OUTPUT_DIR

        # Create directory if it doesn't exist
        Path(directory).mkdir(parents=True, exist_ok=True)

        # Generate filename with datetime and iteration
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"coscientist_state_{timestamp}_iter_{self._iteration:04d}.pkl"
        filepath = os.path.join(directory, filename)

        # Save state to pickle file
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

        # Increment iteration counter
        self._iteration += 1

        return filepath

    @classmethod
    def load(cls, filepath: str) -> "CoscientistState":
        """
        Load a CoscientistState from a pickle file.

        Parameters
        ----------
        filepath : str
            Path to the pickle file to load

        Returns
        -------
        CoscientistState
            The loaded state object
        """
        with open(filepath, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def list_checkpoints(directory: Optional[str] = None) -> List[str]:
        """
        List all available checkpoint files in the directory.

        Parameters
        ----------
        directory : Optional[str]
            Directory to search for checkpoints. Defaults to ~/.coscientist

        Returns
        -------
        List[str]
            List of checkpoint filepaths, sorted by modification time (newest first)
        """
        if directory is None:
            directory = _OUTPUT_DIR

        if not os.path.exists(directory):
            return []

        # Find all pickle files matching our naming pattern
        checkpoint_files = []
        for filename in os.listdir(directory):
            if filename.startswith("coscientist_state_") and filename.endswith(".pkl"):
                filepath = os.path.join(directory, filename)
                checkpoint_files.append(filepath)

        # Sort by modification time (newest first)
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)

        return checkpoint_files

    @classmethod
    def load_latest(
        cls, directory: Optional[str] = None
    ) -> Optional["CoscientistState"]:
        """
        Load the most recent checkpoint from the directory.

        Parameters
        ----------
        directory : Optional[str]
            Directory to search for checkpoints. Defaults to ~/.coscientist

        Returns
        -------
        Optional[CoscientistState]
            The loaded state object, or None if no checkpoints found
        """
        checkpoints = cls.list_checkpoints(directory)
        if not checkpoints:
            return None

        return cls.load(checkpoints[0])  # Load the newest checkpoint


class CoscientistStateManager:
    """
    Manager class for coordinating operations on CoscientistState.

    This class provides higher-level operations for managing the flow
    of hypotheses through the multi-agent pipeline.
    """

    def __init__(self, state: CoscientistState):
        """
        Initialize the manager with a CoscientistState.

        Parameters
        ----------
        state : CoscientistState
            The state object to manage
        """
        # This state should not be accessed directly. All the methods
        # that update the state have logic to save checkpoints.
        self._state = state
        self._setup()

    @property
    def is_started(self) -> bool:
        """
        Check if the Coscientist system has started.
        """
        # Completing the first meta-review signals that
        # the system has finished one full iteration.
        return self._state.meta_review is not None

    @property
    def has_literature_review(self) -> bool:
        """
        Check if the Coscientist system has completed the literature review.
        """
        return self._state.literature_review is not None

    def get_tournament_hypotheses_for_evolution(self) -> List[str]:
        """
        Get the ranked tournament hypotheses that are qualified for evolution.
        """
        # This gets us all the hypotheses in the tournament ordered
        # by rank. Some of them may not have competed yet though.
        ranked_order = self._state.tournament.get_sorted_hypotheses()
        # The records dictionary tells us which hypotheses have competed
        # already and are therefore qualified for evolution.
        have_competed = self._state.tournament.get_win_loss_records().keys()

        return [h_id for h_id, _ in ranked_order if h_id in have_competed]

    def get_hypothesis_by_uid(
        self,
        uid: str,
        location: Literal[
            "tournament", "generated", "reviewed", "evolved", "reflection_queue"
        ],
    ) -> ParsedHypothesis | ReviewedHypothesis:
        """
        Get a hypothesis by its UID. Must specify the location of the hypothesis.
        This prevents us from accidentally using a hypothesis that isn't at the right
        stage of development for a task.
        """
        if location == "tournament":
            return self._state.tournament.hypotheses[uid]
        elif location == "generated":
            for hypothesis in self._state.generated_hypotheses:
                if hypothesis.uid == uid:
                    return hypothesis
            raise ValueError(
                f"Hypothesis with UID '{uid}' not found in generated hypotheses"
            )
        elif location == "reviewed":
            for hypothesis in self._state.reviewed_hypotheses:
                if hypothesis.uid == uid:
                    return hypothesis
            raise ValueError(
                f"Hypothesis with UID '{uid}' not found in reviewed hypotheses"
            )
        elif location == "evolved":
            for hypothesis in self._state.evolved_hypotheses:
                if hypothesis.uid == uid:
                    return hypothesis
            raise ValueError(
                f"Hypothesis with UID '{uid}' not found in evolved hypotheses"
            )
        elif location == "reflection_queue":
            for hypothesis in self._state.reflection_queue:
                if hypothesis.uid == uid:
                    return hypothesis
            raise ValueError(
                f"Hypothesis with UID '{uid}' not found in reflection queue"
            )
        else:
            raise ValueError(
                f"Invalid location '{location}'. Must be 'tournament', 'generated', "
                "'reviewed', 'evolved', or 'reflection_queue'"
            )

    @property
    def total_hypotheses(self) -> int:
        """
        Get the total number of hypotheses in the system.
        """
        return (
            len(self._state.generated_hypotheses)
            + len(self._state.reviewed_hypotheses)
            + len(self._state.evolved_hypotheses)
            + len(self._state.tournament.hypotheses)
        )

    @property
    def reflection_queue_is_empty(self) -> bool:
        """
        Check if the reflection queue is empty.
        """
        return len(self._state.reflection_queue) == 0

    @_maybe_save(n=1)
    def add_generated_hypothesis(
        self, hypothesis: Union[IndependentState, CollaborativeState]
    ) -> None:
        """
        Add a generated hypothesis to the collection.

        Parameters
        ----------
        hypothesis : Union[IndependentState, CollaborativeState]
            The hypothesis state to add
        """
        self._state.generated_hypotheses.append(hypothesis)

    @_maybe_save(n=1)
    def add_reviewed_hypothesis(self, reviewed_hypothesis: ReviewedHypothesis) -> None:
        """
        Add a reviewed hypothesis result to the collection.

        Parameters
        ----------
        reviewed_hypothesis : ReviewedHypothesis
            The reviewed hypothesis state to add
        """
        self._state.reviewed_hypotheses.append(reviewed_hypothesis)

    @_maybe_save(n=1)
    def add_evolved_hypothesis(
        self, evolved_hypothesis: Union[EvolveFromFeedbackState, OutOfTheBoxState]
    ) -> None:
        """
        Add an evolved hypothesis to the collection.

        Parameters
        ----------
        evolved_hypothesis : Union[EvolveFromFeedbackState, OutOfTheBoxState]
            The evolved hypothesis state to add
        """
        self._state.evolved_hypotheses.append(evolved_hypothesis)

    @_maybe_save(n=1)
    def update_literature_review(
        self, literature_review: LiteratureReviewState
    ) -> None:
        """
        Update the literature review state.

        Parameters
        ----------
        literature_review : LiteratureReviewState
            The new literature review state
        """
        self._state.literature_review = literature_review

    @_maybe_save(n=1)
    def update_meta_review(self, meta_review: MetaReviewTournamentState) -> None:
        """
        Update the meta-review state.

        Parameters
        ----------
        meta_review : MetaReviewTournamentState
            The new meta-review state
        """
        self._state.meta_review = meta_review

    @_maybe_save(n=3)
    def advance_hypothesis(self, kind: Literal["generated", "evolved"]) -> None:
        """
        Move a hypothesis from generation/evolution to the reflection queue.

        This method pops the first hypothesis from the specified list,
        adds it to the reflection queue, and updates the proximity graph.

        Parameters
        ----------
        kind : Literal["generated", "evolved"]
            The type of hypothesis to advance - either "generated" or "evolved"

        Raises
        ------
        IndexError
            If the specified list is empty
        ValueError
            If kind is not "generated" or "evolved"
        """
        if kind == "generated":
            if not self._state.generated_hypotheses:
                raise IndexError("No generated hypotheses available to advance")

            parsed_hypothesis = self._state.generated_hypotheses.pop(0)

        elif kind == "evolved":
            if not self._state.evolved_hypotheses:
                raise IndexError("No evolved hypotheses available to advance")

            parsed_hypothesis = self._state.evolved_hypotheses.pop(0)

        else:
            raise ValueError(f"Invalid kind '{kind}'. Must be 'generated' or 'evolved'")

        # Add to reflection queue
        self._state.reflection_queue.append(parsed_hypothesis)

        # Add to proximity graph if it exists
        assert (
            self._state.proximity_graph is not None
        ), "Proximity graph is not initialized"
        self._state.proximity_graph.add_hypothesis(parsed_hypothesis)

    @_maybe_save(n=1)
    def advance_reviewed_hypothesis(self) -> None:
        """
        Move a hypothesis from reviewed_hypotheses to the EloTournament.

        This method pops the first reviewed hypothesis from the reviewed_hypotheses list
        and adds it to the EloTournament by calling the add_hypothesis method.

        Raises
        ------
        IndexError
            If the reviewed_hypotheses list is empty
        """
        if not self._state.reviewed_hypotheses:
            raise IndexError("No reviewed hypotheses available to advance")

        reviewed_hypothesis_state = self._state.reviewed_hypotheses.pop(0)
        if not reviewed_hypothesis_state["passed_initial_filter"]:
            return

        # Add to tournament
        assert self._state.tournament is not None, "Tournament is not initialized"
        self._state.tournament.add_hypothesis(reviewed_hypothesis_state)

    def advance_all_hypotheses(
        self, kind: Literal["generated", "evolved", "reviewed"]
    ) -> None:
        """
        Advance all hypotheses of a given kind to the reflection queue.

        Parameters
        ----------
        kind : Literal["generated", "evolved", "reviewed"]
            The type of hypothesis to advance - either "generated", "evolved", or "reviewed"
        """
        if kind == "generated":
            while self._state.generated_hypotheses:
                self.advance_hypothesis(kind="generated")
        elif kind == "evolved":
            while self._state.evolved_hypotheses:
                self.advance_hypothesis(kind="evolved")
        elif kind == "reviewed":
            while self._state.reviewed_hypotheses:
                self.advance_reviewed_hypothesis()
        else:
            raise ValueError(
                f"Invalid kind '{kind}'. Must be 'generated', 'evolved', or 'reviewed'"
            )

    @_maybe_save(n=1)
    def run_tournament(self, llm: BaseChatModel, k_bracket: int = 16) -> None:
        """
        Run the tournament.
        """
        assert self._state.tournament is not None, "Tournament is not initialized"
        self._state.tournament.run_tournament(llm=llm, k_bracket=k_bracket)

    def _setup(self) -> None:
        """
        Initialize EloTournament and ProximityGraph if they are None.

        This method ensures that the required objects are created
        and available for use by other components.
        """
        if self._state.tournament is None:
            self._state.tournament = EloTournament(self._state.goal)

        if self._state.proximity_graph is None:
            self._state.proximity_graph = ProximityGraph()

    def next_literature_review_state(
        self, max_subtopics: int = 5
    ) -> LiteratureReviewState:
        """
        Create an initial state for the literature review agent.

        Parameters
        ----------
        max_subtopics : int
            Maximum number of subtopics to decompose the research goal into

        Returns
        -------
        LiteratureReviewState
            Initial state with goal and max_subtopics set, empty lists for subtopics and reports
        """
        if self._state.literature_review is not None:
            subtopics = self._state.literature_review.get("subtopics", [])
            subtopic_reports = self._state.literature_review.get("subtopic_reports", [])
        else:
            subtopics = []
            subtopic_reports = []

        return LiteratureReviewState(
            goal=self._state.goal,
            max_subtopics=max_subtopics,
            subtopics=subtopics,
            subtopic_reports=subtopic_reports,
        )

    def next_generation_state(
        self,
        mode: Literal["independent", "collaborative"],
        first_agent_name: str | None = None,
    ) -> Union[IndependentState, CollaborativeState]:
        """
        Create an initial state for the generation agent.

        Parameters
        ----------
        mode : Literal["independent", "collaborative"]
            The type of generation state to create
        first_agent_name : str | None
            The name of the first agent in the collaborative mode. If None, the
            mode must be "independent".

        Returns
        -------
        Union[IndependentState, CollaborativeState]
            Initial state with goal and literature_review set, meta_review included if available
        """
        # Get literature review content
        if self._state.literature_review is None:
            raise ValueError(
                "Literature review must be completed before generation can begin"
            )

        # Join subtopic reports into a single literature review string
        literature_review_content = "\n\n".join(
            self._state.literature_review["subtopic_reports"]
        )

        # Create base state with required fields
        base_state = {
            "goal": self._state.goal,
            "literature_review": literature_review_content,
        }

        # Add meta_review if available
        if self._state.meta_review is not None:
            base_state["meta_review"] = self._state.meta_review["result"]

        if mode == "independent":
            return IndependentState(**base_state)
        elif mode == "collaborative":
            # Add MultiTurnState fields for collaborative mode
            collaborative_state = CollaborativeState(
                **base_state,
                transcript=[],
                turn=0,
                next_agent=first_agent_name,
                finished=False,
            )
            return collaborative_state
        else:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be 'independent' or 'collaborative'"
            )

    def next_reflection_state(self) -> ReflectionState:
        """
        Create an initial state for the reflection agent.

        Pops a hypothesis from the reflection queue and creates a ReflectionState
        with that hypothesis and default initial values for all other fields.

        Returns
        -------
        ReflectionState
            Initial state with hypothesis_to_review set and all other fields initialized

        Raises
        ------
        IndexError
            If the reflection queue is empty
        """
        if not self._state.reflection_queue:
            raise IndexError(
                "No hypotheses available in reflection queue. Please advance a hypothesis first."
            )

        # Pop the first hypothesis from the queue
        hypothesis_to_review = self._state.reflection_queue.pop(0)

        return ReflectionState(hypothesis_to_review=hypothesis_to_review)

    def next_evolution_state(
        self,
        mode: Literal["evolve_from_feedback", "out_of_the_box"],
        uid_to_evolve: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Union[EvolveFromFeedbackState, OutOfTheBoxState]:
        """
        Create an initial state for the evolution agent.

        Parameters
        ----------
        mode : Literal["evolve_from_feedback", "out_of_the_box"]
            The type of evolution state to create
        uid_to_evolve : Optional[str]
            The UID of the hypothesis to evolve (required for "evolve_from_feedback" mode)
        top_k : Optional[int]
            Number of top hypotheses to include (required for "out_of_the_box" mode)

        Returns
        -------
        Union[EvolveFromFeedbackState, OutOfTheBoxState]
            Initial state with goal and required hypothesis data

        Raises
        ------
        ValueError
            If tournament is not initialized or required parameters are missing
        KeyError
            If uid_to_evolve is not found in tournament hypotheses
        """
        if self._state.tournament is None:
            raise ValueError(
                "Tournament must be initialized before evolution can begin"
            )

        if mode == "evolve_from_feedback":
            if uid_to_evolve is None:
                raise ValueError(
                    "uid_to_evolve is required for 'evolve_from_feedback' mode"
                )

            if uid_to_evolve not in self._state.tournament.hypotheses:
                raise KeyError(
                    f"Hypothesis with UID '{uid_to_evolve}' not found in tournament"
                )

            parent_hypothesis = self._state.tournament.hypotheses[uid_to_evolve]

            # Create base state
            base_state = {
                "goal": self._state.goal,
                "parent_hypothesis": parent_hypothesis,
            }

            # Add meta_review if available
            if self._state.meta_review is not None:
                base_state["meta_review"] = self._state.meta_review["result"]
            else:
                base_state["meta_review"] = "Not Available"

            return EvolveFromFeedbackState(**base_state)

        elif mode == "out_of_the_box":
            if top_k is None:
                raise ValueError("top_k is required for 'out_of_the_box' mode")

            # Get sorted hypotheses and select top k
            sorted_hypotheses = self._state.tournament.get_sorted_hypotheses()
            if len(sorted_hypotheses) < top_k:
                raise ValueError(
                    f"Not enough hypotheses in tournament. Requested {top_k}, but only {len(sorted_hypotheses)} available"
                )

            # Get the top k hypotheses
            elo_ratings = []
            top_hypotheses = []
            for uid, rating in sorted_hypotheses[:top_k]:
                top_hypotheses.append(self._state.tournament.hypotheses[uid])
                elo_ratings.append(rating)

            return OutOfTheBoxState(
                goal=self._state.goal,
                top_hypotheses=top_hypotheses,
                elo_ratings=elo_ratings,
            )

        else:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be 'evolve_from_feedback' or 'out_of_the_box'"
            )

    def next_meta_review_state(self, top_k: int) -> MetaReviewTournamentState:
        """
        Create an initial state for the meta-review agent.

        Parameters
        ----------
        top_k : int
            Number of top hypotheses to include in the meta-review analysis

        Returns
        -------
        MetaReviewTournamentState
            Initial state with goal, tournament, and top_k set

        Raises
        ------
        ValueError
            If tournament is not initialized
        """
        if self._state.tournament is None:
            raise ValueError(
                "Tournament must be initialized before meta-review can begin"
            )

        return MetaReviewTournamentState(
            goal=self._state.goal,
            tournament=self._state.tournament,
            top_k=top_k,
        )
