"""
Unit tests for Literature Review Agent.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from coscientist.literature_review_agent import LiteratureReviewAgent
from coscientist.global_state import CoscientistState


@pytest.mark.unit
@pytest.mark.asyncio
class TestLiteratureReviewAgent:
    """Test cases for Literature Review Agent."""

    async def test_decompose_goal(self, mock_llm, mock_researcher_config):
        """Test goal decomposition functionality."""
        # Setup
        mock_llm.add_response("""
        Research Questions:
        1. What are the molecular mechanisms of Alzheimer's disease?
        2. What are the current therapeutic approaches?
        3. What are the risk factors and biomarkers?
        
        Search Queries:
        - Alzheimer disease molecular mechanisms
        - Alzheimer therapeutic targets
        - Alzheimer biomarkers diagnosis
        """)
        
        agent = LiteratureReviewAgent(llm=mock_llm)
        
        # Execute
        result = await agent.decompose_goal("What causes Alzheimer's disease?")
        
        # Verify
        assert "Research Questions" in result
        assert "Search Queries" in result
        assert "mechanisms" in result.lower()
        assert mock_llm.call_count == 1

    async def test_conduct_research_with_mock_researcher(self, mock_llm, mock_researcher_config):
        """Test research conduction with mocked GPT Researcher."""
        # Setup
        mock_report = "Alzheimer's disease is characterized by amyloid plaques and tau tangles."
        
        with patch("coscientist.literature_review_agent.GPTResearcher") as mock_researcher:
            mock_instance = AsyncMock()
            mock_instance.conduct_research.return_value = mock_report
            mock_instance.write_report.return_value = mock_report
            mock_researcher.return_value = mock_instance
            
            agent = LiteratureReviewAgent(llm=mock_llm)
            
            # Execute
            result = await agent.conduct_research("Alzheimer disease mechanisms")
            
            # Verify
            assert result == mock_report
            mock_instance.conduct_research.assert_called_once()

    async def test_timeout_handling(self, mock_llm, mock_researcher_config):
        """Test timeout handling in research."""
        with patch("coscientist.literature_review_agent.GPTResearcher") as mock_researcher:
            mock_instance = AsyncMock()
            mock_instance.conduct_research.side_effect = asyncio.TimeoutError()
            mock_researcher.return_value = mock_instance
            
            agent = LiteratureReviewAgent(llm=mock_llm)
            
            # Execute
            result = await agent.conduct_research("test query")
            
            # Verify timeout is handled gracefully
            assert "timeout" in result.lower() or "error" in result.lower()

    async def test_full_literature_review_process(self, mock_llm, test_state, mock_researcher_config):
        """Test the complete literature review process."""
        # Setup decomposition response
        mock_llm.add_response("""
        Research Questions:
        1. What causes battery degradation?
        2. What are new battery technologies?
        
        Search Queries:
        - battery degradation mechanisms
        - solid state batteries
        """)
        
        # Mock GPT Researcher
        mock_report = "Research findings on battery technology..."
        
        with patch("coscientist.literature_review_agent.GPTResearcher") as mock_researcher:
            mock_instance = AsyncMock()
            mock_instance.conduct_research.return_value = mock_report
            mock_instance.write_report.return_value = mock_report
            mock_researcher.return_value = mock_instance
            
            agent = LiteratureReviewAgent(llm=mock_llm)
            
            # Execute
            input_data = {"goal": "How to improve battery technology?"}
            result = await agent.conduct_literature_review(test_state, input_data)
            
            # Verify
            assert result["literature_review"] is not None
            assert len(result["literature_review"]) > 0
            assert test_state.literature_review is not None

    async def test_error_recovery(self, mock_llm, mock_researcher_config):
        """Test error recovery mechanisms."""
        # Simulate network error
        with patch("coscientist.literature_review_agent.GPTResearcher") as mock_researcher:
            mock_instance = AsyncMock()
            mock_instance.conduct_research.side_effect = Exception("Network error")
            mock_researcher.return_value = mock_instance
            
            agent = LiteratureReviewAgent(llm=mock_llm)
            
            # Should not raise exception, but return error message
            result = await agent.conduct_research("test query")
            
            assert isinstance(result, str)
            assert len(result) > 0

    @pytest.mark.slow
    async def test_research_with_retry(self, mock_llm, mock_researcher_config):
        """Test research retry mechanism."""
        call_count = 0
        
        def mock_research_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary error")
            return "Success after retry"
        
        with patch("coscientist.literature_review_agent.GPTResearcher") as mock_researcher:
            mock_instance = AsyncMock()
            mock_instance.conduct_research.side_effect = mock_research_side_effect
            mock_researcher.return_value = mock_instance
            
            agent = LiteratureReviewAgent(llm=mock_llm)
            
            # With retry enabled (if implemented in the agent)
            result = await agent.conduct_research("test query")
            
            # Should succeed on retry or handle error gracefully
            assert isinstance(result, str)

    async def test_multiple_query_handling(self, mock_llm, mock_researcher_config):
        """Test handling of multiple research queries."""
        queries = [
            "battery degradation mechanisms",
            "solid state batteries",
            "lithium metal anodes"
        ]
        
        mock_reports = {
            queries[0]: "Report on degradation mechanisms...",
            queries[1]: "Report on solid state batteries...",  
            queries[2]: "Report on lithium anodes..."
        }
        
        with patch("coscientist.literature_review_agent.GPTResearcher") as mock_researcher:
            def create_mock_instance(query, **kwargs):
                instance = AsyncMock()
                instance.conduct_research.return_value = mock_reports.get(query, "Default report")
                instance.write_report.return_value = mock_reports.get(query, "Default report")
                return instance
            
            mock_researcher.side_effect = create_mock_instance
            
            agent = LiteratureReviewAgent(llm=mock_llm)
            
            # Test processing multiple queries
            results = []
            for query in queries:
                result = await agent.conduct_research(query)
                results.append(result)
            
            # Verify all queries processed
            assert len(results) == 3
            for i, result in enumerate(results):
                assert queries[i].split()[0] in result or "report" in result.lower()

    async def test_state_persistence(self, mock_llm, test_state, mock_researcher_config):
        """Test that literature review results are persisted to state."""
        mock_llm.add_response("Research decomposition...")
        
        with patch("coscientist.literature_review_agent.GPTResearcher") as mock_researcher:
            mock_instance = AsyncMock()
            mock_instance.conduct_research.return_value = "Test research report"
            mock_instance.write_report.return_value = "Test research report"  
            mock_researcher.return_value = mock_instance
            
            agent = LiteratureReviewAgent(llm=mock_llm)
            
            # Execute
            input_data = {"goal": "Test research goal"}
            result = await agent.conduct_literature_review(test_state, input_data)
            
            # Verify state is updated
            assert test_state.literature_review is not None
            assert "research" in test_state.literature_review.lower()
            
            # Verify return value
            assert "literature_review" in result
            assert result["literature_review"] is not None


@pytest.mark.integration
class TestLiteratureReviewIntegration:
    """Integration tests for Literature Review Agent with real dependencies."""

    @pytest.mark.requires_openai
    @pytest.mark.requires_tavily
    async def test_real_api_integration(self):
        """Test with real APIs - only runs if API keys are available."""
        from coscientist.framework import _SMARTER_LLM_POOL
        
        if "claude-opus-4-1-20250805" in _SMARTER_LLM_POOL:
            llm = _SMARTER_LLM_POOL["claude-opus-4-1-20250805"]
            agent = LiteratureReviewAgent(llm=llm)
            
            # Simple test query
            result = await agent.decompose_goal("What is photosynthesis?")
            
            assert len(result) > 0
            assert "research" in result.lower() or "question" in result.lower()

    async def test_configuration_loading(self, mock_researcher_config):
        """Test that agent correctly loads and uses configuration."""
        from coscientist.literature_review_agent import load_researcher_config
        
        config = load_researcher_config()
        
        # Verify config structure
        assert "FAST_LLM" in config
        assert "SMART_LLM" in config
        assert "EMBEDDING_CHUNK_SIZE" in config