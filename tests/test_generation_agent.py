"""
Unit tests for Generation Agent.
"""

import pytest
from unittest.mock import MagicMock, patch

from coscientist.generation_agent import GenerationAgent
from coscientist.custom_types import ParsedHypothesis
from tests.utils import create_mock_hypothesis


@pytest.mark.unit
@pytest.mark.asyncio
class TestGenerationAgent:
    """Test cases for Generation Agent."""

    async def test_generate_hypotheses_basic(self, mock_llm, test_state):
        """Test basic hypothesis generation."""
        # Setup mock response with proper JSON format
        mock_response = """
        {
            "hypotheses": [
                {
                    "id": 1,
                    "hypothesis": "Amyloid beta plaques cause neuronal death",
                    "reasoning": "Amyloid accumulation triggers inflammatory cascades",
                    "confidence": 0.8,
                    "assumptions": ["BBB permeability", "Inflammation is primary"],
                    "testing_approach": "In vivo mouse models with amyloid injections",
                    "observables": ["Neuronal loss", "Inflammatory markers"]
                },
                {
                    "id": 2,
                    "hypothesis": "Tau tangles disrupt microtubule transport",
                    "reasoning": "Hyperphosphorylated tau loses microtubule binding",
                    "confidence": 0.9,
                    "assumptions": ["Tau phosphorylation is causal", "Transport is critical"],
                    "testing_approach": "Cellular transport assays",
                    "observables": ["Transport velocity", "Tau phosphorylation"]
                }
            ]
        }
        """
        mock_llm.add_response(mock_response)
        
        agent = GenerationAgent(llm=mock_llm)
        test_state.literature_review = "Literature review on Alzheimer's disease..."
        
        # Execute
        input_data = {
            "goal": "What causes Alzheimer's disease?",
            "n_hypotheses": 2
        }
        result = await agent.generate_new_hypotheses(test_state, input_data)
        
        # Verify
        assert "hypotheses" in result
        hypotheses = result["hypotheses"]
        assert len(hypotheses) == 2
        
        # Check first hypothesis
        h1 = hypotheses[0]
        assert isinstance(h1, ParsedHypothesis)
        assert h1.hypothesis == "Amyloid beta plaques cause neuronal death"
        assert h1.confidence == 0.8
        assert len(h1.assumptions) == 2
        assert len(h1.observables) == 2
        
        # Verify state is updated
        assert len(test_state.hypotheses) == 2

    async def test_parse_hypothesis_response(self, mock_llm):
        """Test JSON response parsing."""
        agent = GenerationAgent(llm=mock_llm)
        
        json_response = """
        {
            "hypotheses": [
                {
                    "id": 1,
                    "hypothesis": "Test hypothesis",
                    "reasoning": "Test reasoning",
                    "confidence": 0.75,
                    "assumptions": ["Assumption 1", "Assumption 2"],
                    "testing_approach": "Test approach",
                    "observables": ["Observable 1"]
                }
            ]
        }
        """
        
        # Execute
        hypotheses = agent._parse_hypothesis_response(json_response)
        
        # Verify
        assert len(hypotheses) == 1
        h = hypotheses[0]
        assert h.id == 1
        assert h.hypothesis == "Test hypothesis"
        assert h.confidence == 0.75
        assert len(h.assumptions) == 2
        assert len(h.observables) == 1

    async def test_malformed_json_handling(self, mock_llm):
        """Test handling of malformed JSON responses."""
        agent = GenerationAgent(llm=mock_llm)
        
        malformed_json = """
        {
            "hypotheses": [
                {
                    "id": 1,
                    "hypothesis": "Test hypothesis"
                    // Missing closing brace and other fields
        """
        
        # Should handle gracefully
        hypotheses = agent._parse_hypothesis_response(malformed_json)
        assert hypotheses == []

    async def test_missing_required_fields(self, mock_llm):
        """Test handling of responses with missing required fields."""
        agent = GenerationAgent(llm=mock_llm)
        
        incomplete_json = """
        {
            "hypotheses": [
                {
                    "id": 1,
                    "hypothesis": "Test hypothesis"
                    // Missing reasoning, confidence, etc.
                }
            ]
        }
        """
        
        # Should handle gracefully or use defaults
        hypotheses = agent._parse_hypothesis_response(incomplete_json)
        # Implementation might return empty list or hypothesis with defaults
        assert isinstance(hypotheses, list)

    async def test_batch_generation(self, mock_llm, test_state):
        """Test generating larger batches of hypotheses."""
        # Mock response with 5 hypotheses
        hypotheses_data = []
        for i in range(1, 6):
            hypotheses_data.append({
                "id": i,
                "hypothesis": f"Hypothesis {i}",
                "reasoning": f"Reasoning {i}",
                "confidence": 0.5 + (i * 0.1),
                "assumptions": [f"Assumption {i}"],
                "testing_approach": f"Test approach {i}",
                "observables": [f"Observable {i}"]
            })
        
        mock_response = f'{{"hypotheses": {str(hypotheses_data).replace("\'", "\"")}}}' 
        mock_llm.add_response(mock_response)
        
        agent = GenerationAgent(llm=mock_llm)
        test_state.literature_review = "Literature review..."
        
        # Execute
        input_data = {
            "goal": "Test research goal",
            "n_hypotheses": 5
        }
        result = await agent.generate_new_hypotheses(test_state, input_data)
        
        # Verify
        assert len(result["hypotheses"]) == 5
        assert len(test_state.hypotheses) == 5
        
        # Check ascending confidence values
        confidences = [h.confidence for h in result["hypotheses"]]
        assert confidences == [0.6, 0.7, 0.8, 0.9, 1.0]

    async def test_collaborative_generation(self, mock_llm, test_state, sample_hypotheses):
        """Test collaborative hypothesis generation with existing hypotheses."""
        # Setup existing hypotheses
        test_state.hypotheses.extend(sample_hypotheses)
        
        mock_response = """
        {
            "hypotheses": [
                {
                    "id": 5,
                    "hypothesis": "Collaborative hypothesis building on existing work",
                    "reasoning": "Based on previous hypotheses A and B",
                    "confidence": 0.85,
                    "assumptions": ["Previous work is valid"],
                    "testing_approach": "Combined approach from A and B",
                    "observables": ["New observable"]
                }
            ]
        }
        """
        mock_llm.add_response(mock_response)
        
        agent = GenerationAgent(llm=mock_llm)
        test_state.literature_review = "Literature review..."
        
        # Execute
        input_data = {
            "goal": "Test research goal",
            "n_hypotheses": 1,
            "existing_hypotheses": sample_hypotheses
        }
        result = await agent.generate_new_hypotheses(test_state, input_data)
        
        # Verify
        new_hypothesis = result["hypotheses"][0]
        assert "collaborative" in new_hypothesis.hypothesis.lower()
        assert new_hypothesis.id == 5
        assert len(test_state.hypotheses) == 5  # 4 existing + 1 new

    async def test_independent_generation_mode(self, mock_llm, test_state):
        """Test independent hypothesis generation mode."""
        mock_response = """
        {
            "hypotheses": [
                {
                    "id": 1,
                    "hypothesis": "Independent hypothesis with novel approach",
                    "reasoning": "Fresh perspective on the problem",
                    "confidence": 0.7,
                    "assumptions": ["Independent assumption"],
                    "testing_approach": "Novel testing method",
                    "observables": ["Unique observable"]
                }
            ]
        }
        """
        mock_llm.add_response(mock_response)
        
        agent = GenerationAgent(llm=mock_llm)
        test_state.literature_review = "Literature review..."
        
        # Execute with independent mode
        input_data = {
            "goal": "Test research goal",
            "n_hypotheses": 1,
            "generation_mode": "independent"
        }
        result = await agent.generate_new_hypotheses(test_state, input_data)
        
        # Verify
        hypothesis = result["hypotheses"][0]
        assert "independent" in hypothesis.hypothesis.lower()

    async def test_hypothesis_id_assignment(self, mock_llm, test_state, sample_hypotheses):
        """Test that hypothesis IDs are assigned correctly."""
        # Setup existing hypotheses
        test_state.hypotheses.extend(sample_hypotheses)  # IDs 1-4
        
        mock_response = """
        {
            "hypotheses": [
                {
                    "id": 999,
                    "hypothesis": "New hypothesis",
                    "reasoning": "New reasoning", 
                    "confidence": 0.8,
                    "assumptions": ["New assumption"],
                    "testing_approach": "New approach",
                    "observables": ["New observable"]
                }
            ]
        }
        """
        mock_llm.add_response(mock_response)
        
        agent = GenerationAgent(llm=mock_llm)
        test_state.literature_review = "Literature review..."
        
        # Execute
        input_data = {
            "goal": "Test research goal",
            "n_hypotheses": 1
        }
        result = await agent.generate_new_hypotheses(test_state, input_data)
        
        # Verify ID is correctly reassigned
        new_hypothesis = result["hypotheses"][0]
        # Should get ID 5 (next available) regardless of what LLM returned
        expected_id = max(h.id for h in sample_hypotheses) + 1
        assert new_hypothesis.id == expected_id

    async def test_confidence_validation(self, mock_llm, test_state):
        """Test confidence score validation and clamping."""
        mock_response = """
        {
            "hypotheses": [
                {
                    "id": 1,
                    "hypothesis": "High confidence hypothesis",
                    "reasoning": "Very strong reasoning",
                    "confidence": 1.5,
                    "assumptions": ["Strong assumption"],
                    "testing_approach": "Reliable test",
                    "observables": ["Clear observable"]
                },
                {
                    "id": 2,
                    "hypothesis": "Negative confidence hypothesis",
                    "reasoning": "Uncertain reasoning",
                    "confidence": -0.1,
                    "assumptions": ["Weak assumption"],
                    "testing_approach": "Uncertain test",
                    "observables": ["Unclear observable"]
                }
            ]
        }
        """
        mock_llm.add_response(mock_response)
        
        agent = GenerationAgent(llm=mock_llm)
        test_state.literature_review = "Literature review..."
        
        # Execute
        input_data = {
            "goal": "Test research goal",
            "n_hypotheses": 2
        }
        result = await agent.generate_new_hypotheses(test_state, input_data)
        
        # Verify confidence values are clamped to [0, 1]
        confidences = [h.confidence for h in result["hypotheses"]]
        assert all(0 <= c <= 1 for c in confidences)


@pytest.mark.integration
class TestGenerationAgentIntegration:
    """Integration tests for Generation Agent."""

    @pytest.mark.requires_openai
    async def test_real_llm_generation(self):
        """Test hypothesis generation with real LLM."""
        from coscientist.framework import _SMARTER_LLM_POOL
        
        if "gpt-5" in _SMARTER_LLM_POOL:
            llm = _SMARTER_LLM_POOL["gpt-5"]
            agent = GenerationAgent(llm=llm)
            
            # Create minimal test state
            from coscientist.global_state import CoscientistState
            test_goal = "What causes common cold?"
            state = CoscientistState(goal=test_goal)
            state.literature_review = "The common cold is caused by various viruses..."
            
            try:
                # Execute
                input_data = {
                    "goal": test_goal,
                    "n_hypotheses": 2
                }
                result = await agent.generate_new_hypotheses(state, input_data)
                
                # Basic validation
                assert "hypotheses" in result
                assert len(result["hypotheses"]) >= 1
                assert all(isinstance(h, ParsedHypothesis) for h in result["hypotheses"])
                
            finally:
                # Cleanup
                from tests.utils import cleanup_test_state
                cleanup_test_state(state)

    async def test_with_different_llm_models(self, mock_llm_pools):
        """Test generation with different LLM models."""
        from coscientist.generation_agent import GenerationAgent
        
        models_to_test = ["gpt-5", "claude-opus-4-1-20250805", "gemini-2.5-pro"]
        
        for model_name in models_to_test:
            if model_name in mock_llm_pools:
                llm = mock_llm_pools[model_name]
                llm.add_response('{"hypotheses": [{"id": 1, "hypothesis": "Test", "reasoning": "Test", "confidence": 0.8, "assumptions": [], "testing_approach": "Test", "observables": []}]}')
                
                agent = GenerationAgent(llm=llm)
                
                # Should work with any model
                test_state = MagicMock()
                test_state.hypotheses = []
                test_state.literature_review = "Test literature review"
                
                result = await agent.generate_new_hypotheses(test_state, {
                    "goal": "Test goal",
                    "n_hypotheses": 1
                })
                
                assert len(result["hypotheses"]) == 1