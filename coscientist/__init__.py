"""Open CoScientist Agents - Multi-agent system for AI co-scientist research."""

__version__ = "0.0.1"

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from coscientist.framework import CoscientistConfig, CoscientistFramework
from coscientist.global_state import CoscientistState, CoscientistStateManager
