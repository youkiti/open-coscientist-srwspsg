"""
Defines a global state for the Coscientist system. This global state will have persistence by
writing to pickle files after every run of the meta-review agent. It will need
to manage outputs from the six main agents. This will facilitate passing inputs and outputs
throughout the multi-agent system.

Nodes:
* Generation
- A list of IndependentState or CollaborativeState typed dictionaries.
* Reflection
- A list of ReflectionState typed dictionaries.
* Ranking
- The latest EloTournament object.
* Evolution
- A list of EvolveFromFeedbackState or OutOfTheBoxState typed dictionaries.
* Meta-review
- The latest MetaReviewState
* Proximity
- The latest ProximityGraph object.
"""

import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Union

from coscientist.custom_types import ParsedHypothesis, ReviewedHypothesis
from coscientist.evolution_agent import EvolveFromFeedbackState, OutOfTheBoxState
from coscientist.generation_agent import CollaborativeState, IndependentState
from coscientist.literature_review_agent import LiteratureReviewState
from coscientist.meta_review_agent import MetaReviewTournamentState
from coscientist.proximity_agent import ProximityGraph
from coscientist.ranking_agent import EloTournament
from coscientist.reflection_agent import ReflectionState


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
        self._iteration = 0

    # Methods for appending to lists
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
        self.generated_hypotheses.append(hypothesis)

    def add_reviewed_hypothesis(self, reviewed_hypothesis: ReviewedHypothesis) -> None:
        """
        Add a reviewed hypothesis result to the collection.

        Parameters
        ----------
        reviewed_hypothesis : ReviewedHypothesis
            The reviewed hypothesis state to add
        """
        self.reviewed_hypotheses.append(reviewed_hypothesis)

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
        self.evolved_hypotheses.append(evolved_hypothesis)

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
        self.literature_review = literature_review

    # Methods for updating single objects
    def update_tournament(self, tournament: EloTournament) -> None:
        """
        Update the tournament state.

        Parameters
        ----------
        tournament : EloTournament
            The new tournament state
        """
        self.tournament = tournament

    def update_meta_review(self, meta_review: MetaReviewTournamentState) -> None:
        """
        Update the meta-review state.

        Parameters
        ----------
        meta_review : MetaReviewTournamentState
            The new meta-review state
        """
        self.meta_review = meta_review

    def update_proximity_graph(self, proximity_graph: ProximityGraph) -> None:
        """
        Update the proximity graph.

        Parameters
        ----------
        proximity_graph : ProximityGraph
            The new proximity graph
        """
        self.proximity_graph = proximity_graph

    # Persistence methods
    def save(self, directory: Optional[str] = None) -> str:
        """
        Save the current state to a pickle file.

        Parameters
        ----------
        directory : Optional[str]
            Directory to save the checkpoint. Defaults to ~/.coscientist

        Returns
        -------
        str
            Path to the saved checkpoint file
        """
        if directory is None:
            directory = os.path.expanduser("~/.coscientist")

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
            directory = os.path.expanduser("~/.coscientist")

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
        self.state = state
        self.reflection_queue: List[ParsedHypothesis] = []
        self._setup()

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
            if not self.state.generated_hypotheses:
                raise IndexError("No generated hypotheses available to advance")

            hypothesis_state = self.state.generated_hypotheses.pop(0)
            # Extract ParsedHypothesis from IndependentState or CollaborativeState
            parsed_hypothesis = hypothesis_state["hypothesis"]

        elif kind == "evolved":
            if not self.state.evolved_hypotheses:
                raise IndexError("No evolved hypotheses available to advance")

            hypothesis_state = self.state.evolved_hypotheses.pop(0)
            # Extract ParsedHypothesis from EvolveFromFeedbackState or OutOfTheBoxState
            parsed_hypothesis = hypothesis_state["evolved_hypothesis"]

        else:
            raise ValueError(f"Invalid kind '{kind}'. Must be 'generated' or 'evolved'")

        # Add to reflection queue
        self.reflection_queue.append(parsed_hypothesis)

        # Add to proximity graph if it exists
        assert (
            self.state.proximity_graph is not None
        ), "Proximity graph is not initialized"
        self.state.proximity_graph.add_hypothesis(parsed_hypothesis)

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
        if not self.state.reviewed_hypotheses:
            raise IndexError("No reviewed hypotheses available to advance")

        reviewed_hypothesis_state = self.state.reviewed_hypotheses.pop(0)

        # Add to tournament
        assert self.state.tournament is not None, "Tournament is not initialized"
        self.state.tournament.add_hypothesis(reviewed_hypothesis_state)

    def _setup(self) -> None:
        """
        Initialize EloTournament and ProximityGraph if they are None.

        This method ensures that the required objects are created
        and available for use by other components.
        """
        if self.state.tournament is None:
            self.state.tournament = EloTournament()

        if self.state.proximity_graph is None:
            self.state.proximity_graph = ProximityGraph()

    def make_literature_review_initial_state(
        self, max_subtopics: int
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
        return LiteratureReviewState(
            goal=self.state.goal,
            max_subtopics=max_subtopics,
            subtopics=[],
            subtopic_reports=[],
        )

    def make_generation_initial_state(
        self, mode: Literal["independent", "collaborative"]
    ) -> Union[IndependentState, CollaborativeState]:
        """
        Create an initial state for the generation agent.

        Parameters
        ----------
        mode : Literal["independent", "collaborative"]
            The type of generation state to create

        Returns
        -------
        Union[IndependentState, CollaborativeState]
            Initial state with goal and literature_review set, meta_review included if available
        """
        # Get literature review content
        if self.state.literature_review is None:
            raise ValueError(
                "Literature review must be completed before generation can begin"
            )

        # Join subtopic reports into a single literature review string
        literature_review_content = "\n\n".join(
            self.state.literature_review["subtopic_reports"]
        )

        # Create base state with required fields
        base_state = {
            "goal": self.state.goal,
            "literature_review": literature_review_content,
        }

        # Add meta_review if available
        if self.state.meta_review is not None:
            base_state["meta_review"] = self.state.meta_review["result"]

        if mode == "independent":
            return IndependentState(**base_state)
        elif mode == "collaborative":
            # Add MultiTurnState fields for collaborative mode
            collaborative_state = CollaborativeState(
                **base_state, transcript=[], turn=0, next_agent="", finished=False
            )
            return collaborative_state
        else:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be 'independent' or 'collaborative'"
            )

    def make_reflection_initial_state(self) -> ReflectionState:
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
        if not self.reflection_queue:
            raise IndexError(
                "No hypotheses available in reflection queue. Please advance a hypothesis first."
            )

        # Pop the first hypothesis from the queue
        hypothesis_to_review = self.reflection_queue.pop(0)

        return ReflectionState(hypothesis_to_review=hypothesis_to_review)

    def make_evolution_initial_state(
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
        if self.state.tournament is None:
            raise ValueError(
                "Tournament must be initialized before evolution can begin"
            )

        if mode == "evolve_from_feedback":
            if uid_to_evolve is None:
                raise ValueError(
                    "uid_to_evolve is required for 'evolve_from_feedback' mode"
                )

            if uid_to_evolve not in self.state.tournament.hypotheses:
                raise KeyError(
                    f"Hypothesis with UID '{uid_to_evolve}' not found in tournament"
                )

            parent_hypothesis = self.state.tournament.hypotheses[uid_to_evolve]

            # Create base state
            base_state = {
                "goal": self.state.goal,
                "parent_hypothesis": parent_hypothesis,
            }

            # Add meta_review if available
            if self.state.meta_review is not None:
                base_state["meta_review"] = self.state.meta_review["result"]
            else:
                base_state["meta_review"] = "Not Available"

            return EvolveFromFeedbackState(**base_state)

        elif mode == "out_of_the_box":
            if top_k is None:
                raise ValueError("top_k is required for 'out_of_the_box' mode")

            # Get sorted hypotheses and select top k
            sorted_hypotheses = self.state.tournament.get_sorted_hypotheses()
            if len(sorted_hypotheses) < top_k:
                raise ValueError(
                    f"Not enough hypotheses in tournament. Requested {top_k}, but only {len(sorted_hypotheses)} available"
                )

            # Get the top k hypotheses
            top_hypothesis_uids = [uid for uid, rating in sorted_hypotheses[:top_k]]
            top_hypotheses = [
                self.state.tournament.hypotheses[uid] for uid in top_hypothesis_uids
            ]

            return OutOfTheBoxState(goal=self.state.goal, top_hypotheses=top_hypotheses)

        else:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be 'evolve_from_feedback' or 'out_of_the_box'"
            )

    def make_meta_review_initial_state(self, top_k: int) -> MetaReviewTournamentState:
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
        if self.state.tournament is None:
            raise ValueError(
                "Tournament must be initialized before meta-review can begin"
            )

        return MetaReviewTournamentState(
            goal=self.state.goal,
            tournament=self.state.tournament,
            top_k=top_k,
            result="",
        )
