"""
Pytest configuration and fixtures for Coscientist test suite.
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from tests.utils import (
    MockLLM,
    MockGPTResearcher,
    MockOpenAIClient,
    cleanup_env_vars,
    cleanup_test_state,
    create_test_state,
    mock_env_vars,
    mock_llm_pool,
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_env():
    """Set up and tear down mock environment variables."""
    mock_vars = mock_env_vars()
    yield mock_vars
    cleanup_env_vars(mock_vars)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp(prefix="cosci_test_")
    yield temp_dir
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_state(temp_dir):
    """Create a test CoscientistState with temp directory."""
    os.environ["COSCIENTIST_DIR"] = temp_dir
    state = create_test_state("Test research goal", use_temp_dir=False)
    yield state
    cleanup_test_state(state)


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    return MockLLM("test-llm")


@pytest.fixture
def mock_llm_pools():
    """Mock the LLM pools used by the framework."""
    pools = mock_llm_pool()
    
    with patch("coscientist.framework._SMARTER_LLM_POOL", pools), \
         patch("coscientist.framework._CHEAPER_LLM_POOL", pools):
        yield pools


@pytest.fixture
def mock_gpt_researcher():
    """Create a mock GPT Researcher."""
    return MockGPTResearcher()


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    return MockOpenAIClient()


@pytest.fixture
def mock_openai_responses_client():
    """Mock the OpenAI responses client."""
    mock_client = MagicMock()
    mock_client._generate.return_value = MagicMock(
        generations=[MagicMock(message=MagicMock(content="Mock response"))]
    )
    
    with patch("coscientist.openai_client.create_openai_responses_client", return_value=mock_client):
        yield mock_client


@pytest.fixture
def mock_researcher_config():
    """Mock the researcher configuration."""
    config = {
        "FAST_LLM": "mock:gpt-5",
        "SMART_LLM": "mock:claude-opus-4-1-20250805",
        "STRATEGIC_LLM": "mock:o3",
        "TAVILY_MAX_RESULTS": 5,
        "RESEARCH_MAX_ITERATIONS": 3,
        "EMBEDDING_CHUNK_SIZE": 800,
        "EMBEDDING_KWARGS": {
            "chunk_size": 800
        }
    }
    
    with patch("coscientist.literature_review_agent.load_researcher_config", return_value=config):
        yield config


@pytest.fixture
def sample_hypotheses():
    """Create sample hypotheses for testing."""
    from tests.utils import create_mock_hypothesis
    
    return [
        create_mock_hypothesis(1, "Hypothesis A", "Reasoning A", 0.9),
        create_mock_hypothesis(2, "Hypothesis B", "Reasoning B", 0.8),
        create_mock_hypothesis(3, "Hypothesis C", "Reasoning C", 0.7),
        create_mock_hypothesis(4, "Hypothesis D", "Reasoning D", 0.6),
    ]


@pytest.fixture
def mock_web_fetch():
    """Mock web fetching functionality."""
    async def mock_fetch(url: str):
        return f"Mock content from {url}"
    
    with patch("coscientist.pdf_handler.fetch_url", side_effect=mock_fetch):
        yield mock_fetch


@pytest.fixture
def mock_embeddings():
    """Mock OpenAI embeddings."""
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3] for _ in range(10)]
    
    with patch("langchain_openai.OpenAIEmbeddings", return_value=mock_embeddings):
        yield mock_embeddings


@pytest.fixture(autouse=True)
def reset_singleton_instances():
    """Reset any singleton instances between tests."""
    # Reset any global state that might persist between tests
    yield
    
    # Clean up after test
    if "COSCIENTIST_DIR" in os.environ:
        del os.environ["COSCIENTIST_DIR"]


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit for testing UI components."""
    mock_st = MagicMock()
    mock_st.session_state = {}
    
    with patch("streamlit", mock_st):
        yield mock_st


@pytest.fixture
def capture_logs():
    """Capture log output during tests."""
    import logging
    from io import StringIO
    
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)
    
    logger = logging.getLogger("coscientist")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
    yield log_capture
    
    logger.removeHandler(handler)


@pytest.fixture
def disable_network():
    """Disable network access for tests that should not make external calls."""
    import socket
    
    def block_network(*args, **kwargs):
        raise RuntimeError("Network access not allowed in this test")
    
    original_socket = socket.socket
    socket.socket = block_network
    
    yield
    
    socket.socket = original_socket


# Markers for conditional test execution
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "requires_openai: mark test as requiring OpenAI API key"
    )
    config.addinivalue_line(
        "markers", "requires_anthropic: mark test as requiring Anthropic API key"
    )
    config.addinivalue_line(
        "markers", "requires_google: mark test as requiring Google API key"
    )
    config.addinivalue_line(
        "markers", "requires_tavily: mark test as requiring Tavily API key"
    )


def pytest_runtest_setup(item):
    """Skip tests based on environment variables."""
    markers = list(item.iter_markers())
    
    for marker in markers:
        if marker.name == "requires_openai" and not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not available")
        elif marker.name == "requires_anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("Anthropic API key not available")
        elif marker.name == "requires_google" and not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("Google API key not available")
        elif marker.name == "requires_tavily" and not os.getenv("TAVILY_API_KEY"):
            pytest.skip("Tavily API key not available")