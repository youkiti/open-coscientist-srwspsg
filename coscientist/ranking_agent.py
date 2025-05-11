"""
Ranking agent
-------------
- Runs tournaments and assigns ELO ratings to hypotheses

More details:
- Newly added hypotheses are added to the tournament with
an ELO rating of 1200.
- Top and bottom ranked hypotheses are evaluated differently.
Two top-ranked hypotheses are paired against each other and
there is a multi-turn scientific debate. Lower ranked hypotheses
are evaluated with a single turn debate. Final output is the number
of the winning hypothesis.
- Based on the Proximity agents graph, similar hypotheses are ranked
against each other. New and top-ranked hypotheses are prioritized.
"""

from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
import math
from tqdm import tqdm
import re  # Add re for parsing
import itertools  # Add itertools for combinations
from typing import Dict, List, Tuple, Any, Set, Optional  # Add Optional

from coscientist.types import HypothesisWithID

# Constants
DEFAULT_ELO = 1200
K_FACTOR = 32


TOURNAMENT_PROMPT = """
You are an expert evaluator tasked with comparing two hypotheses.

Evaluate the two provided hypotheses (hypothesis 1 and hypothesis 2) and determine which one
is superior based on the specified {idea_attributes}.
Provide a concise rationale for your selection, concluding with the phrase "better idea: <1 or 2>".

Goal: {goal}

Evaluation criteria:
{preferences}

Considerations:
{notes}
Each hypothesis includes an independent review. These reviews may contain numerical scores.
Disregard these scores in your comparative analysis, as they may not be directly comparable across reviews.

Hypothesis 1:
{hypothesis_1}

Hypothesis 2:
{hypothesis_2}

Review of hypothesis 1:
{review_1}

Review of hypothesis 2:
{review_2}

Reasoning and conclusion (end with "better hypothesis: <1 or 2>"):
"""

SIMULATED_DEBATE_PROMPT = """
You are an expert in comparative analysis, simulating a panel of domain experts
engaged in a structured discussion to evaluate two competing hypotheses.
The objective is to rigorously determine which hypothesis is superior based on
a predefined set of attributes and criteria.
The experts possess no pre-existing biases toward either hypothesis and are solely
focused on identifying the optimal choice, given that only one can be implemented.

Goal: {goal}
Criteria for hypothesis superiority:
{preferences}

Hypothesis 1:
{hypothesis_1}

Hypothesis 2:
{hypothesis_2}

Initial review of hypothesis 1:
{review_1}

Initial review of hypothesis 2:
{review_2}

Debate procedure:

The discussion will unfold in a series of turns, typically ranging from 3 to 5, with a maximum of 10.

Turn 1: begin with a concise summary of both hypotheses and their respective initial reviews.

Subsequent turns:

* Pose clarifying questions to address any ambiguities or uncertainties.
* Critically evaluate each hypothesis in relation to the stated Goal and Criteria.
This evaluation should consider aspects such as:
- Potential for correctness/validity.
- Utility and practical applicability.
- Sufficiency of detail and specificity.
- Novelty and originality.
- Desirability for implementation.
* Identify and articulate any weaknesses, limitations, or potential flaws in either hypothesis.

Additional notes:
{notes}

Termination and judgment:

Once the discussion has reached a point of sufficient depth (typically 3-5 turns, up to 10 turns)
and all relevant questions and concerns have been thoroughly addressed, provide a conclusive judgment.
This judgment should succinctly state the rationale for the selection.
Then, indicate the superior hypothesis by writing the phrase "better idea: ",
followed by "1" (for hypothesis 1) or "2" (for hypothesis 2).
"""


def calculate_expected_score(rating1: float, rating2: float) -> tuple[float, float]:
    """Calculates the expected scores for two players based on their ELO ratings."""
    expected1 = 1 / (1 + 10 ** ((rating2 - rating1) / 400))
    expected2 = 1 / (1 + 10 ** ((rating1 - rating2) / 400))
    return expected1, expected2


def update_elo(rating1: float, rating2: float, winner: int) -> tuple[float, float]:
    """
    Updates the ELO ratings of two players based on the match outcome.

    Parameters
    ----------
    rating1 : float
        ELO rating of hypothesis 1.
    rating2 : float
        ELO rating of hypothesis 2.
    winner : int
        1 if hypothesis 1 won, 2 if hypothesis 2 won.

    Returns
    -------
    tuple of float
        A tuple containing the updated ELO ratings (new_rating1, new_rating2).
    """
    expected1, expected2 = calculate_expected_score(rating1, rating2)

    if winner == 1:
        score1, score2 = 1, 0
    elif winner == 2:
        score1, score2 = 0, 1
    else:
        raise ValueError("Winner must be 1 or 2")  # Assuming no draws for now

    new_rating1 = rating1 + K_FACTOR * (score1 - expected1)
    new_rating2 = rating2 + K_FACTOR * (score2 - expected2)

    return new_rating1, new_rating2


class EloTournament:
    """Manages a two-stage ELO ranking tournament for hypotheses."""

    def __init__(
        self,
        llm: BaseChatModel,
        goal: str,
        preferences: str,
        notes: str,
        idea_attributes: List[str] = [],
    ):
        self.llm = llm
        self.goal = goal
        self.preferences = preferences
        self.notes = notes
        self.idea_attributes = ", ".join(idea_attributes)
        self.hypotheses: Dict[str, HypothesisWithID] = {}  # id -> Hypothesis object
        self.ratings: Dict[str, float] = {}  # id -> ELO rating
        self.match_history: Dict[Tuple[int, int], Tuple[int, int]] = {}

    def add_hypothesis(
        self, hypothesis: HypothesisWithID, initial_rating: float = DEFAULT_ELO
    ):
        """Adds a new hypothesis to the tournament."""
        if hypothesis.id not in self.hypotheses:
            self.hypotheses[hypothesis.id] = hypothesis
            self.ratings[hypothesis.id] = initial_rating
        else:
            raise ValueError(f"Hypothesis {hypothesis.id} already exists.")

    def get_sorted_hypotheses(self) -> List[Tuple[str, float]]:
        """Returns hypotheses sorted by ELO rating (descending)."""
        return sorted(self.ratings.items(), key=lambda item: item[1], reverse=True)

    def _determine_winner(
        self,
        hypo1: HypothesisWithID,
        hypo2: HypothesisWithID,
        prompt_template: str,
        stage: int,
    ) -> Optional[int]:
        """
        Uses the LLM with a specific prompt to determine the winner between two hypotheses.

        Parameters
        ----------
        hypo1 : HypothesisWithID
            The first hypothesis.
        hypo2 : HypothesisWithID
            The second hypothesis.
        prompt_template : str
            The template for the prompt.
        stage : int
            The stage of the tournament.

        Returns
        -------
        Optional[int]
            1 if hypo1 wins, 2 if hypo2 wins, None if winner cannot be determined.
        """
        # Use PromptTemplate for safer formatting
        prompt_builder = PromptTemplate.from_template(prompt_template)

        # Prepare inputs based on the prompt template structure
        # Assuming both prompts need these core elements
        prompt_input = {
            "goal": self.goal,
            "preferences": self.preferences,
            "notes": self.notes,
            "hypothesis_1": hypo1.content,
            "hypothesis_2": hypo2.content,
            "review_1": hypo1.review,  # Assumes review is available
            "review_2": hypo2.review,
            # Add other fields specific to each prompt if necessary
            "idea_attributes": self.idea_attributes,  # Used in SIMULATED_DEBATE_PROMPT
            # TOURNAMENT_PROMPT specific might just ignore extra keys
        }

        formatted_prompt = prompt_builder.format(**prompt_input)
        response = self.llm.invoke(formatted_prompt)
        response_text = (
            response
            if isinstance(response, str)
            else getattr(response, "content", str(response))
        )

        # print(f"LLM Response for {hypo1.id} vs {hypo2.id}:\n{response_text}")

        # Parse the response to find the winner
        # Look for "better idea: 1", "better idea: 2", "better hypothesis: 1", etc.
        # Also handles simple "<1>" or "<2>" responses.
        winner_str = response_text.split(":")[-1]
        print(f"Winner string: {winner_str}")
        assert ("1" in winner_str) ^ ("2" in winner_str), (
            f"Invalid winner string: {winner_str}"
        )
        winner = 1 if "1" in winner_str else 2

        print(f"LLM determined winner: Hypothesis {winner}")
        return winner

    def run_round_robin_stage(self):
        """
        Runs the round-robin stage of the tournament.
        Every hypothesis competes against every other hypothesis once using TOURNAMENT_PROMPT.
        Updates ELO ratings based on match outcomes.
        """
        hypo_ids = list(self.hypotheses.keys())
        stage = 1
        if len(hypo_ids) < 2:
            print("Not enough hypotheses for round robin stage.")
            return

        # Use itertools.combinations to get unique pairs
        for id1, id2 in tqdm(list(itertools.combinations(hypo_ids, 2))):
            hypo1 = self.hypotheses[id1]
            hypo2 = self.hypotheses[id2]
            rating1 = self.ratings[id1]
            rating2 = self.ratings[id2]

            pair = tuple(sorted((id1, id2)))
            previous_outcome = self.match_history.get(pair, (None, None))
            if previous_outcome[1] != stage:
                # If no history, run the match
                winner = self._determine_winner(
                    hypo1, hypo2, TOURNAMENT_PROMPT, stage=stage
                )

                winner_id = id1 if winner == 1 else id2
                self.match_history[pair] = (winner_id, stage)
                new_rating1, new_rating2 = update_elo(rating1, rating2, winner)

                self.ratings[id1] = new_rating1
                self.ratings[id2] = new_rating2

    def run_bracket_stage(self, k: int = 16) -> Optional[str]:
        """
        Runs the single-elimination bracket stage for the top k hypotheses.
        Uses SIMULATED_DEBATE_PROMPT for matches.
        Does NOT update ELO ratings, only determines a winner.

        Parameters
        ----------
        k : int, optional
            The number of top hypotheses to include in the bracket (must be power of 2).

        Returns
        -------
        Optional[str]
            The ID of the winning hypothesis, or None if the stage cannot run or fails.
        """
        stage = 2
        # Check if k is a power of 2
        if k <= 0 or (k & (k - 1) != 0):
            raise ValueError(f"K must be power of 2. Got {k}.")

        sorted_hypotheses = self.get_sorted_hypotheses()
        if len(sorted_hypotheses) < k:
            print(
                f"Not enough hypotheses ({len(sorted_hypotheses)}) for a Top {k} bracket. Need at least {k}."
            )
            return None

        # Select top k hypothesis IDs
        bracket_contenders_ids = [h_id for h_id, _ in sorted_hypotheses[:k]]

        current_round_ids = bracket_contenders_ids
        round_num = 1

        while len(current_round_ids) > 1:
            next_round_ids = []
            # Create pairings (1 vs k, 2 vs k-1, etc. for the current list)
            num_contenders = len(current_round_ids)
            for i in range(num_contenders // 2):
                id1 = current_round_ids[i]
                id2 = current_round_ids[num_contenders - 1 - i]
                hypo1 = self.hypotheses[id1]
                hypo2 = self.hypotheses[id2]
                rating1 = self.ratings[id1]
                rating2 = self.ratings[id2]

                pair = tuple(sorted((id1, id2)))
                winner_id, previous_stage = self.match_history.get(pair, (None, None))
                if previous_stage != stage:
                    # Pair hasn't played, run the LLM
                    winner = self._determine_winner(
                        hypo1, hypo2, SIMULATED_DEBATE_PROMPT, stage=stage
                    )

                    winner_id = id1 if winner == 1 else id2
                    self.match_history[pair] = (winner_id, stage)
                    new_rating1, new_rating2 = update_elo(rating1, rating2, winner)
                    self.ratings[id1] = new_rating1
                    self.ratings[id2] = new_rating2

                next_round_ids.append(winner_id)

            # Re-sort the winners based on their potentially updated ELO ratings
            next_round_ids.sort(key=lambda h_id: self.ratings[h_id], reverse=True)

            current_round_ids = next_round_ids
            round_num += 1

    def run_tournament(self, k_bracket: int = 16) -> Optional[str]:
        """
        Runs the full two-stage tournament.

        Parameters
        ----------
        k_bracket : int, optional
            The number of top hypotheses for the bracket stage.

        Returns
        -------
        Optional[str]
            The ID of the final winning hypothesis from the bracket stage, or None.
        """
        self.run_round_robin_stage()
        # Print final ELO rankings after round robin
        print("\n--- ELO Rankings after Round Robin Stage ---")
        for i, (h_id, rating) in enumerate(self.get_sorted_hypotheses()):
            print(f"{i + 1}. {h_id}: {rating:.2f}")

        self.run_bracket_stage(k=k_bracket)
