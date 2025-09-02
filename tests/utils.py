"""
Test utilities and helpers for Coscientist test suite.
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import tempfile
import shutil

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from coscientist.custom_types import ParsedHypothesis, ReviewedHypothesis
from coscientist.global_state import CoscientistState, CoscientistStateManager


class MockLLM(BaseChatModel):
    """Mock LLM for testing without API calls."""
    
    def __init__(self, model_name: str = "mock-llm", **kwargs):
        super().__init__(**kwargs)
        self.model = model_name
        self.response_queue: List[str] = []
        self.call_count = 0
        
    def add_response(self, response: str):
        """Add a response to the queue."""
        self.response_queue.append(response)
        
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a mock response."""
        self.call_count += 1
        
        if self.response_queue:
            content = self.response_queue.pop(0)
        else:
            content = f"Mock response {self.call_count}"
            
        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async version of generate."""
        return self._generate(messages, stop, run_manager, **kwargs)
    
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "mock"


class MockGPTResearcher:
    """Mock GPT Researcher for testing without web access."""
    
    def __init__(self, report: str = "Mock research report"):
        self.report = report
        self.call_count = 0
        
    async def conduct_research(self, query: str) -> str:
        """Mock research method."""
        self.call_count += 1
        await asyncio.sleep(0.01)  # Simulate async work
        return self.report
        
    async def write_report(self, query: str, report_type: str = "research_report") -> str:
        """Mock report writing."""
        self.call_count += 1
        await asyncio.sleep(0.01)
        return self.report


def create_mock_hypothesis(
    id: int = 1,
    hypothesis: str = "Test hypothesis",
    reasoning: str = "Test reasoning",
    confidence: float = 0.8
) -> ParsedHypothesis:
    """Create a mock ParsedHypothesis for testing."""
    return ParsedHypothesis(
        id=id,
        hypothesis=hypothesis,
        reasoning=reasoning,
        confidence=confidence,
        assumptions=["Assumption 1", "Assumption 2"],
        testing_approach="Test approach",
        observables=["Observable 1", "Observable 2"],
        elo_rating=1500.0,
        wins=0,
        losses=0,
        head_to_head_results={}
    )


def create_mock_reviewed_hypothesis(
    hypothesis: ParsedHypothesis,
    review: str = "Test review"
) -> ReviewedHypothesis:
    """Create a mock ReviewedHypothesis for testing."""
    return ReviewedHypothesis(
        hypothesis=hypothesis,
        review=review
    )


def create_test_state(goal: str = "Test goal", use_temp_dir: bool = True) -> CoscientistState:
    """Create a test state with optional temp directory."""
    if use_temp_dir:
        temp_dir = tempfile.mkdtemp(prefix="cosci_test_")
        os.environ["COSCIENTIST_DIR"] = temp_dir
        
    state = CoscientistState(goal=goal)
    return state


def cleanup_test_state(state: CoscientistState):
    """Clean up test state and temp directories."""
    try:
        if "COSCIENTIST_DIR" in os.environ:
            temp_dir = os.environ["COSCIENTIST_DIR"]
            if temp_dir.startswith("/tmp") or "cosci_test" in temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)
            del os.environ["COSCIENTIST_DIR"]
    except Exception:
        pass


class AsyncContextManager:
    """Helper for testing async context managers."""
    
    def __init__(self, return_value=None):
        self.return_value = return_value
        
    async def __aenter__(self):
        return self.return_value
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


def mock_llm_pool(model_type: str = "mock") -> Dict[str, MockLLM]:
    """Create a mock LLM pool for testing."""
    return {
        "gpt-5": MockLLM("gpt-5"),
        "o3": MockLLM("o3"),
        "o4-mini": MockLLM("o4-mini"),
        "claude-opus-4-1-20250805": MockLLM("claude-opus"),
        "claude-sonnet-4": MockLLM("claude-sonnet"),
        "gemini-2.5-pro": MockLLM("gemini-pro"),
        "gemini-2.5-flash": MockLLM("gemini-flash"),
    }


def assert_hypothesis_valid(hypothesis: ParsedHypothesis):
    """Assert that a hypothesis has valid structure."""
    assert hypothesis.id is not None
    assert hypothesis.hypothesis
    assert hypothesis.reasoning
    assert 0 <= hypothesis.confidence <= 1
    assert isinstance(hypothesis.assumptions, list)
    assert isinstance(hypothesis.observables, list)
    assert hypothesis.elo_rating >= 0


def create_mock_api_response(content: str, model: str = "mock") -> Dict[str, Any]:
    """Create a mock API response structure."""
    return {
        "id": "mock-response-id",
        "object": "chat.completion",
        "created": 1234567890,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 200,
            "total_tokens": 300
        }
    }


class MockOpenAIClient:
    """Mock OpenAI client for testing."""
    
    def __init__(self):
        self.responses = MockResponses()
        
    class MockResponses:
        def create(self, **kwargs):
            class MockResponse:
                def __init__(self):
                    self.text = MagicMock()
                    self.text.content = "Mock GPT-5 response"
            return MockResponse()


async def run_async_test(coro):
    """Helper to run async tests."""
    loop = asyncio.get_event_loop()
    if loop.is_running():
        task = asyncio.create_task(coro)
        return await task
    else:
        return await coro


def load_test_fixture(filename: str) -> Any:
    """Load test fixture from JSON file."""
    fixture_path = Path(__file__).parent / "fixtures" / filename
    with open(fixture_path, "r") as f:
        return json.load(f)


def save_test_fixture(data: Any, filename: str):
    """Save test fixture to JSON file."""
    fixture_dir = Path(__file__).parent / "fixtures"
    fixture_dir.mkdir(exist_ok=True)
    
    fixture_path = fixture_dir / filename
    with open(fixture_path, "w") as f:
        json.dump(data, f, indent=2)


def mock_env_vars():
    """Set mock environment variables for testing."""
    mock_vars = {
        "OPENAI_API_KEY": "mock-openai-key",
        "ANTHROPIC_API_KEY": "mock-anthropic-key",
        "GOOGLE_API_KEY": "mock-google-key",
        "TAVILY_API_KEY": "mock-tavily-key",
        "COSCI_RESEARCH_TIMEOUT_SECONDS": "10",
        "COSCI_REPEAT_ACTION_LIMIT": "3",
    }
    
    for key, value in mock_vars.items():
        os.environ[key] = value
        
    return mock_vars


def cleanup_env_vars(mock_vars: Dict[str, str]):
    """Clean up mock environment variables."""
    for key in mock_vars:
        if key in os.environ:
            del os.environ[key]