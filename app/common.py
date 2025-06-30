import os
import pickle
from pathlib import Path
from typing import List, Optional

import streamlit as st

# Import the necessary types from the coscientist package
from coscientist.global_state import CoscientistState


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


def get_available_states() -> List[str]:
    """Get all available research goals from the goal-based directory structure."""
    try:
        # Use the CoscientistState method to get all available goals
        goals_and_dirs = CoscientistState.list_all_goals()
        # Return just the goal texts (first element of each tuple)
        return [goal for goal, _ in goals_and_dirs]
    except Exception as e:
        st.error(f"Error getting available states: {e}")
        return []


def get_available_states_legacy() -> List[str]:
    """Get all available CoscientistState pickle files (legacy method)."""
    pickle_files = []

    # Check the default coscientist directory
    coscientist_dir = os.environ.get(
        "COSCIENTIST_DIR", os.path.expanduser("~/.coscientist")
    )
    if os.path.exists(coscientist_dir):
        for root, dirs, files in os.walk(coscientist_dir):
            for file in files:
                if file.startswith("coscientist_state_") and file.endswith(".pkl"):
                    pickle_files.append(os.path.join(root, file))

    # Also check current directory
    for file in Path(".").glob("*.pkl"):
        if file.name.startswith("coscientist_state_"):
            pickle_files.append(str(file))

    return sorted(pickle_files, key=os.path.getmtime, reverse=True)
