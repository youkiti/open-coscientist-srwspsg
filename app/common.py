import pickle
import os
from typing import Optional

import streamlit as st

# Import the necessary types from the coscientist package
from coscientist.global_state import CoscientistState


def _last_goal_file_path() -> str:
    """Path to the file storing the last confirmed/refined goal (CQ)."""
    base_dir = os.environ.get("COSCIENTIST_DIR", os.path.expanduser("~/.coscientist"))
    try:
        os.makedirs(base_dir, exist_ok=True)
    except Exception:
        # Best-effort; if it fails, the following save may also fail.
        pass
    return os.path.join(base_dir, "last_confirmed_goal.txt")


def save_last_confirmed_goal(goal: str) -> None:
    """Persist the latest confirmed goal for quick test runs."""
    try:
        with open(_last_goal_file_path(), "w", encoding="utf-8") as f:
            f.write(goal or "")
    except Exception:
        # Non-fatal; UI should degrade gracefully if we cannot persist.
        pass


def load_last_confirmed_goal() -> Optional[str]:
    """Load the latest confirmed goal if available."""
    try:
        path = _last_goal_file_path()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                goal = f.read().strip()
                return goal if goal else None
        return None
    except Exception:
        return None

def load_coscientist_state(filepath: str) -> Optional[CoscientistState]:
    """Load a CoscientistState from a pickle file."""
    try:
        with open(filepath, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading state file: {e}")
        return None


def load_coscientist_state_by_goal(goal: str) -> Optional[CoscientistState]:
    """Load the latest CoscientistState for a given research goal."""
    try:
        return CoscientistState.load_latest(goal=goal)
    except Exception as e:
        st.error(f"Error loading state for goal '{goal}': {e}")
        return None


def get_available_states() -> list[str]:
    """Get all available research goals from the goal-based directory structure."""
    try:
        # Use the CoscientistState method to get all available goals
        goals_and_dirs = CoscientistState.list_all_goals()
        # Return just the goal texts (first element of each tuple)
        return [goal for goal, _ in goals_and_dirs]
    except Exception as e:
        st.error(f"Error getting available states: {e}")
        return []
