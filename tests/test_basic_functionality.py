"""
Basic functionality tests to verify the test infrastructure works.
"""

import pytest
import asyncio
import json
from pathlib import Path

# Test the test infrastructure itself
def test_basic_python():
    """Test basic Python functionality."""
    assert 1 + 1 == 2
    assert "hello" == "hello"


def test_imports():
    """Test that we can import core modules."""
    from coscientist.framework import CoscientistConfig
    from coscientist.global_state import CoscientistState
    
    assert CoscientistConfig is not None
    assert CoscientistState is not None


def test_config_creation():
    """Test creating a basic configuration."""
    from coscientist.framework import CoscientistConfig
    
    config = CoscientistConfig()
    assert config is not None
    assert hasattr(config, 'debug_mode')


def test_state_creation():
    """Test creating a basic state."""
    from coscientist.global_state import CoscientistState
    import tempfile
    import os
    
    # Use a temporary directory to avoid conflicts
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ["COSCIENTIST_DIR"] = temp_dir
        
        goal = "Test goal"
        state = CoscientistState(goal=goal)
        
        assert state.goal == goal
        assert state.hypotheses == []


@pytest.mark.asyncio
async def test_async_functionality():
    """Test async functionality works in pytest."""
    async def sample_async_function():
        await asyncio.sleep(0.01)  # Very short delay
        return "async_result"
    
    result = await sample_async_function()
    assert result == "async_result"


def test_fixtures_loading():
    """Test that fixture files can be loaded."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    
    # Test literature reviews fixture
    lit_review_file = fixtures_dir / "sample_literature_reviews.json"
    if lit_review_file.exists():
        with open(lit_review_file, 'r') as f:
            data = json.load(f)
        
        assert "alzheimers_disease" in data
        assert "research_findings" in data["alzheimers_disease"]
    
    # Test hypotheses fixture  
    hypotheses_file = fixtures_dir / "sample_hypotheses.json"
    if hypotheses_file.exists():
        with open(hypotheses_file, 'r') as f:
            data = json.load(f)
        
        assert "alzheimers_hypotheses" in data
        assert len(data["alzheimers_hypotheses"]) > 0


def test_literature_review_functions():
    """Test literature review utility functions."""
    from coscientist.literature_review_agent import parse_topic_decomposition
    
    # Test topic parsing
    sample_markdown = """
    Research Topics:
    1. Topic A - Description A
    2. Topic B - Description B  
    3. Topic C - Description C
    """
    
    topics = parse_topic_decomposition(sample_markdown)
    assert len(topics) > 0


def test_custom_types():
    """Test custom type definitions."""
    from coscientist.custom_types import ParsedHypothesis
    
    # Create a sample hypothesis
    hypothesis = ParsedHypothesis(
        id=1,
        hypothesis="Test hypothesis",
        reasoning="Test reasoning",
        confidence=0.8,
        assumptions=["Assumption 1"],
        testing_approach="Test approach",
        observables=["Observable 1"],
        elo_rating=1500.0,
        wins=0,
        losses=0,
        head_to_head_results={}
    )
    
    assert hypothesis.id == 1
    assert hypothesis.hypothesis == "Test hypothesis"
    assert hypothesis.confidence == 0.8
    assert len(hypothesis.assumptions) == 1


def test_common_utilities():
    """Test common utility functions."""
    from coscientist.common import load_prompt
    
    # Test that load_prompt function exists and can be called
    try:
        # This might fail if no prompt files exist, but should not raise import error
        result = load_prompt("test")
        # If it succeeds, result should be a string
        assert isinstance(result, str) or result is None
    except FileNotFoundError:
        # This is expected if no prompt files exist yet
        pass


@pytest.mark.slow
def test_performance_marker():
    """Test that pytest markers work."""
    import time
    start = time.time()
    time.sleep(0.1)  # Short delay to make it "slow"
    elapsed = time.time() - start
    assert elapsed >= 0.1


class TestMockingInfrastructure:
    """Test that our mocking infrastructure works."""
    
    def test_mock_llm(self):
        """Test the MockLLM class."""
        from tests.utils import MockLLM
        
        llm = MockLLM("test-model")
        llm.add_response("Test response")
        
        # Test synchronous generation
        from langchain_core.messages import HumanMessage
        messages = [HumanMessage(content="Test message")]
        result = llm._generate(messages)
        
        assert result.generations[0].message.content == "Test response"
        assert llm.call_count == 1
    
    def test_mock_hypothesis_creation(self):
        """Test creating mock hypotheses."""
        from tests.utils import create_mock_hypothesis
        
        hypothesis = create_mock_hypothesis(
            id=1,
            hypothesis="Test hypothesis",
            reasoning="Test reasoning",
            confidence=0.9
        )
        
        assert hypothesis.id == 1
        assert hypothesis.hypothesis == "Test hypothesis"
        assert hypothesis.confidence == 0.9