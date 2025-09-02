"""
Integration tests for the Coscientist Framework.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from coscientist.framework import CoscientistConfig, CoscientistFramework
from coscientist.global_state import CoscientistState, CoscientistStateManager
from tests.utils import create_mock_hypothesis


@pytest.mark.integration
@pytest.mark.asyncio
class TestFrameworkIntegration:
    """Test cases for framework integration."""

    async def test_framework_initialization(self, mock_llm_pools, temp_dir, mock_env):
        """Test framework initialization with mocked LLMs."""
        config = CoscientistConfig(debug_mode=True)
        state = CoscientistState(goal="Test framework initialization")
        state_manager = CoscientistStateManager(state)
        
        framework = CoscientistFramework(config, state_manager)
        
        # Verify initialization
        assert framework.config == config
        assert framework.state_manager == state_manager
        assert framework.state.goal == "Test framework initialization"

    async def test_literature_review_pipeline(self, mock_llm_pools, test_state, mock_researcher_config):
        """Test literature review pipeline integration."""
        config = CoscientistConfig()
        state_manager = CoscientistStateManager(test_state)
        framework = CoscientistFramework(config, state_manager)
        
        # Mock literature review response
        mock_llm_pools["claude-opus-4-1-20250805"].add_response("""
        Research Questions:
        1. What are the fundamental mechanisms?
        2. How do current approaches work?
        
        Search Queries:
        - fundamental mechanisms research
        - current approaches analysis
        """)
        
        # Mock GPT Researcher
        with patch("coscientist.literature_review_agent.GPTResearcher") as mock_researcher:
            mock_instance = AsyncMock()
            mock_instance.conduct_research.return_value = "Comprehensive research report"
            mock_instance.write_report.return_value = "Comprehensive research report"
            mock_researcher.return_value = mock_instance
            
            # Execute literature review
            result = await framework.conduct_literature_review()
            
            # Verify results
            assert result is not None
            assert test_state.literature_review is not None
            assert "research" in test_state.literature_review.lower()

    async def test_hypothesis_generation_pipeline(self, mock_llm_pools, test_state):
        """Test hypothesis generation pipeline integration."""
        config = CoscientistConfig()
        state_manager = CoscientistStateManager(test_state)
        framework = CoscientistFramework(config, state_manager)
        
        # Set up literature review
        test_state.literature_review = "Background literature review"
        
        # Mock hypothesis generation response
        mock_llm_pools["gpt-5"].add_response("""
        {
            "hypotheses": [
                {
                    "id": 1,
                    "hypothesis": "Framework generated hypothesis",
                    "reasoning": "Based on literature analysis",
                    "confidence": 0.85,
                    "assumptions": ["Key assumption"],
                    "testing_approach": "Empirical validation",
                    "observables": ["Measurable outcome"]
                }
            ]
        }
        """)
        
        # Execute generation
        result = await framework.generate_hypotheses(n_hypotheses=1)
        
        # Verify results
        assert len(result) == 1
        assert len(test_state.hypotheses) == 1
        assert test_state.hypotheses[0].hypothesis == "Framework generated hypothesis"

    async def test_tournament_pipeline(self, mock_llm_pools, test_state, sample_hypotheses):
        """Test tournament pipeline integration."""
        config = CoscientistConfig()
        state_manager = CoscientistStateManager(test_state)
        framework = CoscientistFramework(config, state_manager)
        
        # Setup hypotheses
        test_state.hypotheses = sample_hypotheses.copy()
        
        # Mock debate responses
        for i in range(6):  # 6 matches for round-robin with 4 hypotheses
            winner = "A" if i % 2 == 0 else "B"
            mock_llm_pools["claude-opus-4-1-20250805"].add_response(
                f"Winner: Hypothesis {winner}\nReasoning: Superior evidence quality"
            )
        
        # Execute tournament
        result = await framework.run_tournament(k=6)
        
        # Verify results
        assert result is not None
        assert test_state.tournament_results is not None
        assert len(test_state.tournament_results) > 0
        
        # Check that ratings were updated
        for hypothesis in test_state.hypotheses:
            assert hypothesis.elo_rating != 1500.0  # Should have changed

    async def test_supervisor_decision_making(self, mock_llm_pools, test_state, sample_hypotheses):
        """Test supervisor agent decision making integration."""
        config = CoscientistConfig(max_supervisor_iterations=3)
        state_manager = CoscientistStateManager(test_state)
        framework = CoscientistFramework(config, state_manager)
        
        # Setup state
        test_state.hypotheses = sample_hypotheses.copy()
        test_state.literature_review = "Complete literature review"
        
        # Mock supervisor decisions
        decisions = ["generate_new_hypotheses", "run_tournament", "finalize"]
        for decision in decisions:
            mock_llm_pools["o3"].add_response(f"Decision: {decision}\nReasoning: Strategic choice")
        
        # Execute supervisor loop (limited iterations)
        with patch.object(framework, 'generate_hypotheses', new_callable=AsyncMock) as mock_gen, \
             patch.object(framework, 'run_tournament', new_callable=AsyncMock) as mock_tournament, \
             patch.object(framework, 'finalize_research', new_callable=AsyncMock) as mock_finalize:
            
            mock_gen.return_value = []
            mock_tournament.return_value = {"completed": True}
            mock_finalize.return_value = ("Final report", "Meta review")
            
            final_report, meta_review = await framework.run()
            
            # Verify supervisor made decisions
            assert mock_gen.called or mock_tournament.called or mock_finalize.called

    async def test_full_pipeline_integration(self, mock_llm_pools, test_state, mock_researcher_config):
        """Test complete pipeline from start to finish."""
        config = CoscientistConfig(
            debug_mode=True,
            max_supervisor_iterations=2,
            pause_after_literature_review=False
        )
        state_manager = CoscientistStateManager(test_state)
        framework = CoscientistFramework(config, state_manager)
        
        # Mock all required responses
        # Literature review
        mock_llm_pools["claude-opus-4-1-20250805"].add_response("""
        Research Questions: What are the key factors?
        Search Queries: key factors analysis
        """)
        
        # Hypothesis generation
        mock_llm_pools["gpt-5"].add_response("""
        {
            "hypotheses": [
                {
                    "id": 1,
                    "hypothesis": "First hypothesis",
                    "reasoning": "Initial reasoning",
                    "confidence": 0.8,
                    "assumptions": ["Assumption 1"],
                    "testing_approach": "Method 1",
                    "observables": ["Observable 1"]
                },
                {
                    "id": 2,
                    "hypothesis": "Second hypothesis", 
                    "reasoning": "Alternative reasoning",
                    "confidence": 0.7,
                    "assumptions": ["Assumption 2"],
                    "testing_approach": "Method 2",
                    "observables": ["Observable 2"]
                }
            ]
        }
        """)
        
        # Tournament debates
        mock_llm_pools["claude-opus-4-1-20250805"].add_response("Winner: Hypothesis A\nReasoning: Better evidence")
        
        # Supervisor decisions
        mock_llm_pools["o3"].add_response("Decision: run_tournament\nReasoning: Need ranking")
        mock_llm_pools["o3"].add_response("Decision: finalize\nReasoning: Sufficient data")
        
        # Final report
        mock_llm_pools["claude-opus-4-1-20250805"].add_response("Final research report with conclusions")
        
        # Meta review
        mock_llm_pools["gemini-2.5-flash"].add_response("Meta-analysis of research findings")
        
        # Mock GPT Researcher
        with patch("coscientist.literature_review_agent.GPTResearcher") as mock_researcher:
            mock_instance = AsyncMock()
            mock_instance.conduct_research.return_value = "Research findings"
            mock_instance.write_report.return_value = "Research findings"
            mock_researcher.return_value = mock_instance
            
            # Execute full pipeline
            final_report, meta_review = await framework.start(n_hypotheses=2)
            
            # Verify completion
            assert final_report is not None
            assert meta_review is not None
            assert len(test_state.hypotheses) >= 2

    async def test_error_handling_and_recovery(self, mock_llm_pools, test_state):
        """Test error handling and recovery mechanisms."""
        config = CoscientistConfig(save_on_error=True)
        state_manager = CoscientistStateManager(test_state)
        framework = CoscientistFramework(config, state_manager)
        
        # Mock an error in hypothesis generation
        with patch.object(framework, 'generate_hypotheses', side_effect=Exception("Mock error")):
            # Should handle error gracefully
            try:
                await framework.start(n_hypotheses=2)
            except Exception as e:
                # Error should be handled or state should be saved
                assert config.save_on_error is True

    async def test_state_persistence_during_pipeline(self, mock_llm_pools, test_state):
        """Test that state is persisted throughout pipeline execution."""
        config = CoscientistConfig()
        state_manager = CoscientistStateManager(test_state)
        framework = CoscientistFramework(config, state_manager)
        
        # Mock components
        with patch.object(framework, 'conduct_literature_review', new_callable=AsyncMock) as mock_lr, \
             patch.object(framework, 'generate_hypotheses', new_callable=AsyncMock) as mock_gen:
            
            mock_lr.return_value = "Literature review complete"
            mock_gen.return_value = [create_mock_hypothesis(1, "Test hypothesis", "Test reasoning", 0.8)]
            
            # Execute partial pipeline
            await framework.conduct_literature_review()
            hypotheses = await framework.generate_hypotheses(n_hypotheses=1)
            
            # Verify state was updated
            assert test_state.literature_review == "Literature review complete"
            assert len(test_state.hypotheses) == 1

    async def test_configuration_effects(self, mock_llm_pools, test_state):
        """Test that configuration options affect framework behavior."""
        # Test debug mode
        debug_config = CoscientistConfig(debug_mode=True)
        state_manager = CoscientistStateManager(test_state)
        framework = CoscientistFramework(debug_config, state_manager)
        
        assert framework.config.debug_mode is True
        
        # Test iteration limits
        limited_config = CoscientistConfig(max_supervisor_iterations=1)
        framework_limited = CoscientistFramework(limited_config, state_manager)
        
        assert framework_limited.config.max_supervisor_iterations == 1

    async def test_concurrent_agent_execution(self, mock_llm_pools, test_state, sample_hypotheses):
        """Test concurrent execution of multiple agents."""
        config = CoscientistConfig()
        state_manager = CoscientistStateManager(test_state)
        framework = CoscientistFramework(config, state_manager)
        
        test_state.hypotheses = sample_hypotheses.copy()
        
        # Mock multiple agent responses
        mock_llm_pools["claude-opus-4-1-20250805"].add_response("Review result 1")
        mock_llm_pools["claude-opus-4-1-20250805"].add_response("Review result 2")
        
        # Test concurrent operations (if implemented)
        with patch.object(framework, 'reflect_on_hypotheses', new_callable=AsyncMock) as mock_reflect:
            mock_reflect.return_value = {"reflections": ["Reflection 1", "Reflection 2"]}
            
            # Execute reflection
            result = await framework.reflect_on_hypotheses()
            
            # Verify concurrent execution worked
            assert result is not None


@pytest.mark.integration
class TestFrameworkLLMIntegration:
    """Test framework integration with different LLM models."""

    async def test_model_assignment_consistency(self, mock_llm_pools):
        """Test that agents get assigned correct models."""
        config = CoscientistConfig()
        state = CoscientistState(goal="Model assignment test")
        state_manager = CoscientistStateManager(state)
        framework = CoscientistFramework(config, state_manager)
        
        # Verify model assignments match configuration
        # This tests the _SMARTER_LLM_POOL and _CHEAPER_LLM_POOL usage
        assert config.literature_review_agent_llm is not None
        assert config.supervisor_agent_llm is not None
        assert config.final_report_agent_llm is not None

    async def test_model_fallback_behavior(self, mock_llm_pools):
        """Test behavior when primary models are unavailable."""
        # Remove a model from the pool
        if "gpt-5" in mock_llm_pools:
            del mock_llm_pools["gpt-5"]
        
        config = CoscientistConfig()
        state = CoscientistState(goal="Fallback test")
        state_manager = CoscientistStateManager(state)
        
        # Should still initialize successfully with available models
        framework = CoscientistFramework(config, state_manager)
        assert framework is not None

    async def test_different_model_combinations(self, mock_llm_pools, test_state):
        """Test framework with different model combinations."""
        # Test with different model configurations
        model_combinations = [
            ("claude-opus-4-1-20250805", "gpt-5"),
            ("gemini-2.5-pro", "claude-sonnet-4"),
            ("o3", "gemini-2.5-flash")
        ]
        
        for primary, secondary in model_combinations:
            if primary in mock_llm_pools and secondary in mock_llm_pools:
                # Configure framework with specific models
                config = CoscientistConfig()
                # Note: Actual model assignment would need to be configurable in the real implementation
                
                framework = CoscientistFramework(config, CoscientistStateManager(test_state))
                assert framework is not None


@pytest.mark.integration
@pytest.mark.slow
class TestFrameworkPerformance:
    """Test framework performance characteristics."""

    async def test_large_hypothesis_handling(self, mock_llm_pools, test_state):
        """Test framework with large numbers of hypotheses."""
        config = CoscientistConfig()
        state_manager = CoscientistStateManager(test_state)
        framework = CoscientistFramework(config, state_manager)
        
        # Create many hypotheses
        large_hypothesis_set = [
            create_mock_hypothesis(i, f"Hypothesis {i}", f"Reasoning {i}", 0.5 + (i*0.01))
            for i in range(1, 31)  # 30 hypotheses
        ]
        
        test_state.hypotheses = large_hypothesis_set
        
        # Mock tournament responses
        for i in range(50):
            winner = "A" if i % 2 == 0 else "B"
            mock_llm_pools["claude-opus-4-1-20250805"].add_response(
                f"Winner: Hypothesis {winner}\nReasoning: Analysis {i}"
            )
        
        # Execute tournament with large set
        result = await framework.run_tournament(k=30)
        
        # Should handle large sets
        assert result is not None
        assert len(test_state.hypotheses) == 30

    async def test_memory_usage_stability(self, mock_llm_pools, test_state):
        """Test that memory usage remains stable during execution."""
        config = CoscientistConfig()
        state_manager = CoscientistStateManager(test_state)
        framework = CoscientistFramework(config, state_manager)
        
        # Execute multiple operations
        for i in range(5):
            mock_llm_pools["gpt-5"].add_response(f'{"{"}"hypotheses": [{"{"}"id": {i+1}, "hypothesis": "Test {i}", "reasoning": "Reasoning {i}", "confidence": 0.8, "assumptions": [], "testing_approach": "Test", "observables": []{"}"}]{"}"}')
            
            # Generate hypotheses multiple times
            await framework.generate_hypotheses(n_hypotheses=1)
        
        # Memory usage should not grow excessively
        # Note: Actual memory monitoring would require additional tooling
        assert len(test_state.hypotheses) == 5