"""
Unit tests for Ranking Agent and ELO tournament system.
"""

import pytest
from unittest.mock import MagicMock

from coscientist.ranking_agent import RankingAgent
from coscientist.custom_types import ParsedHypothesis
from tests.utils import create_mock_hypothesis


@pytest.mark.unit
@pytest.mark.asyncio
class TestRankingAgent:
    """Test cases for Ranking Agent."""

    def test_elo_rating_calculation(self):
        """Test ELO rating calculation after matches."""
        agent = RankingAgent(llm=MagicMock())
        
        # Initial ratings
        rating_a = 1500.0
        rating_b = 1500.0
        
        # Test win scenario
        new_a, new_b = agent.calculate_elo_ratings(rating_a, rating_b, winner="A")
        
        # Winner should gain points, loser should lose points
        assert new_a > rating_a
        assert new_b < rating_b
        assert abs(new_a - rating_a) == abs(rating_b - new_b)  # Zero-sum game
        
        # Test lose scenario
        new_a2, new_b2 = agent.calculate_elo_ratings(rating_a, rating_b, winner="B")
        assert new_a2 < rating_a
        assert new_b2 > rating_b

    def test_elo_rating_with_different_initial_ratings(self):
        """Test ELO calculations with different starting ratings."""
        agent = RankingAgent(llm=MagicMock())
        
        # High rated vs low rated
        high_rating = 1800.0
        low_rating = 1200.0
        
        # Low rated wins (upset) - should get more points
        new_low, new_high = agent.calculate_elo_ratings(low_rating, high_rating, winner="A")
        low_gain = new_low - low_rating
        high_loss = high_rating - new_high
        
        # High rated wins (expected) - should get fewer points  
        new_high2, new_low2 = agent.calculate_elo_ratings(high_rating, low_rating, winner="A")
        high_gain = new_high2 - high_rating
        low_loss = low_rating - new_low2
        
        # Upset should result in bigger rating change
        assert low_gain > high_gain
        assert high_loss > low_loss

    async def test_conduct_debate(self, mock_llm):
        """Test debate conduction between two hypotheses."""
        # Setup mock debate response
        mock_debate = """
        Judge Analysis:
        Hypothesis A presents strong evidence for amyloid cascade theory.
        Hypothesis B argues for tau pathology as primary driver.
        
        A's evidence includes clinical trials showing amyloid reduction benefits.
        B's evidence includes autopsy studies showing tau correlates better with symptoms.
        
        Winner: Hypothesis A
        Reasoning: Stronger clinical evidence and therapeutic implications.
        """
        mock_llm.add_response(mock_debate)
        
        agent = RankingAgent(llm=mock_llm)
        
        hyp_a = create_mock_hypothesis(1, "Amyloid causes Alzheimer's", "Amyloid cascade", 0.8)
        hyp_b = create_mock_hypothesis(2, "Tau causes Alzheimer's", "Tau pathology", 0.7)
        
        # Execute
        result = await agent.conduct_debate(hyp_a, hyp_b)
        
        # Verify
        assert result["winner"] in ["A", "B"]
        assert "reasoning" in result
        assert "debate_transcript" in result
        assert len(result["debate_transcript"]) > 0

    async def test_debate_parsing_edge_cases(self, mock_llm):
        """Test debate parsing with various edge cases."""
        # Test ambiguous winner
        mock_debate_ambiguous = """
        Both hypotheses have merit.
        A has good evidence but B also makes valid points.
        It's difficult to determine a clear winner.
        """
        mock_llm.add_response(mock_debate_ambiguous)
        
        agent = RankingAgent(llm=mock_llm)
        
        hyp_a = create_mock_hypothesis(1, "Hypothesis A", "Reasoning A", 0.8)
        hyp_b = create_mock_hypothesis(2, "Hypothesis B", "Reasoning B", 0.8)
        
        # Execute
        result = await agent.conduct_debate(hyp_a, hyp_b)
        
        # Should handle ambiguous case gracefully
        assert result["winner"] in ["A", "B", "tie"]

    async def test_run_tournament(self, mock_llm, test_state, sample_hypotheses):
        """Test running a complete tournament."""
        # Setup mock debate responses
        debate_responses = [
            "Winner: Hypothesis A\nReasoning: Better evidence",
            "Winner: Hypothesis B\nReasoning: Stronger logic", 
            "Winner: Hypothesis A\nReasoning: More comprehensive",
            "Winner: Hypothesis C\nReasoning: Novel approach",
            "Winner: Hypothesis A\nReasoning: Consistent results",
            "Winner: Hypothesis B\nReasoning: Clear methodology"
        ]
        
        for response in debate_responses:
            mock_llm.add_response(response)
        
        agent = RankingAgent(llm=mock_llm)
        test_state.hypotheses = sample_hypotheses.copy()
        
        # Execute tournament
        input_data = {"k": 6}  # 6 matches for 4 hypotheses
        result = await agent.run_tournament(test_state, input_data)
        
        # Verify
        assert "tournament_results" in result
        assert "final_rankings" in result
        
        # Check that ELO ratings were updated
        for hypothesis in test_state.hypotheses:
            assert hypothesis.elo_rating != 1500.0  # Should have changed from default
        
        # Verify win/loss counts
        total_wins = sum(h.wins for h in test_state.hypotheses)
        total_losses = sum(h.losses for h in test_state.hypotheses)
        assert total_wins == total_losses  # Zero-sum game

    def test_tournament_bracket_generation(self, sample_hypotheses):
        """Test tournament bracket generation."""
        agent = RankingAgent(llm=MagicMock())
        
        # Test with 4 hypotheses
        bracket = agent.generate_tournament_bracket(sample_hypotheses, k=6)
        
        # Should generate 6 matchups
        assert len(bracket) == 6
        
        # Each matchup should have two different hypotheses
        for match in bracket:
            assert len(match) == 2
            assert match[0].id != match[1].id

    def test_round_robin_bracket(self, sample_hypotheses):
        """Test round-robin tournament bracket."""
        agent = RankingAgent(llm=MagicMock())
        
        # For 4 hypotheses, round-robin should have 6 matches
        bracket = agent.generate_round_robin_bracket(sample_hypotheses)
        assert len(bracket) == 6
        
        # Each hypothesis should appear in multiple matches
        hypothesis_counts = {}
        for match in bracket:
            for hyp in match:
                hypothesis_counts[hyp.id] = hypothesis_counts.get(hyp.id, 0) + 1
        
        # In round-robin, each hypothesis plays every other once
        for count in hypothesis_counts.values():
            assert count == 3  # Each plays against 3 others

    async def test_ranking_persistence(self, mock_llm, test_state, sample_hypotheses):
        """Test that rankings are persisted correctly in state."""
        mock_llm.add_response("Winner: Hypothesis A\nReasoning: Test")
        
        agent = RankingAgent(llm=mock_llm)
        test_state.hypotheses = sample_hypotheses.copy()
        
        # Store initial ratings
        initial_ratings = {h.id: h.elo_rating for h in test_state.hypotheses}
        
        # Run single match
        input_data = {"k": 1}
        await agent.run_tournament(test_state, input_data)
        
        # Check that state was updated
        for hypothesis in test_state.hypotheses:
            if hypothesis.id in [1, 2]:  # Participants in the match
                assert hypothesis.elo_rating != initial_ratings[hypothesis.id]

    def test_head_to_head_tracking(self, mock_llm, sample_hypotheses):
        """Test head-to-head results tracking."""
        agent = RankingAgent(llm=mock_llm)
        
        hyp_a = sample_hypotheses[0]
        hyp_b = sample_hypotheses[1]
        
        # Simulate A beating B
        agent.update_head_to_head(hyp_a, hyp_b, winner="A")
        
        # Check tracking
        assert hyp_a.head_to_head_results.get(hyp_b.id, {}).get("wins", 0) == 1
        assert hyp_b.head_to_head_results.get(hyp_a.id, {}).get("losses", 0) == 1

    async def test_tournament_with_odd_number_hypotheses(self, mock_llm, test_state):
        """Test tournament handling with odd number of hypotheses."""
        # Create 5 hypotheses
        hypotheses = [create_mock_hypothesis(i, f"Hypothesis {i}", f"Reasoning {i}", 0.8) 
                     for i in range(1, 6)]
        
        test_state.hypotheses = hypotheses
        
        # Mock responses for matches
        for i in range(10):  # More than needed to avoid running out
            mock_llm.add_response("Winner: Hypothesis A\nReasoning: Test")
        
        agent = RankingAgent(llm=mock_llm)
        
        # Execute tournament
        input_data = {"k": 10}  # 10 matches for 5 hypotheses
        result = await agent.run_tournament(test_state, input_data)
        
        # Should handle odd numbers gracefully
        assert "tournament_results" in result
        assert len(test_state.hypotheses) == 5

    def test_ranking_algorithm_consistency(self, sample_hypotheses):
        """Test that ranking algorithm produces consistent results."""
        agent = RankingAgent(llm=MagicMock())
        
        # Set known ELO ratings
        sample_hypotheses[0].elo_rating = 1600.0
        sample_hypotheses[1].elo_rating = 1500.0
        sample_hypotheses[2].elo_rating = 1400.0
        sample_hypotheses[3].elo_rating = 1300.0
        
        # Get rankings
        rankings = agent.get_hypothesis_rankings(sample_hypotheses)
        
        # Should be sorted by ELO rating (descending)
        assert rankings[0].elo_rating >= rankings[1].elo_rating
        assert rankings[1].elo_rating >= rankings[2].elo_rating
        assert rankings[2].elo_rating >= rankings[3].elo_rating

    async def test_debate_quality_metrics(self, mock_llm):
        """Test that debates generate quality metrics."""
        mock_debate = """
        Hypothesis A: Strong evidence from clinical trials
        Hypothesis B: Contradictory evidence from meta-analysis
        
        A provides specific data points and statistical significance.
        B raises valid concerns about study methodology.
        
        Winner: Hypothesis A
        Reasoning: More robust evidence despite methodological concerns.
        Confidence: High
        """
        mock_llm.add_response(mock_debate)
        
        agent = RankingAgent(llm=mock_llm)
        
        hyp_a = create_mock_hypothesis(1, "Clinical trial hypothesis", "RCT evidence", 0.9)
        hyp_b = create_mock_hypothesis(2, "Meta-analysis hypothesis", "Pooled data", 0.8)
        
        # Execute
        result = await agent.conduct_debate(hyp_a, hyp_b)
        
        # Check for quality indicators
        transcript = result["debate_transcript"]
        assert "evidence" in transcript.lower()
        assert "reasoning" in result


@pytest.mark.integration
class TestRankingAgentIntegration:
    """Integration tests for Ranking Agent."""

    @pytest.mark.requires_openai
    async def test_real_llm_debate(self):
        """Test debate with real LLM."""
        from coscientist.framework import _SMARTER_LLM_POOL
        
        if "claude-opus-4-1-20250805" in _SMARTER_LLM_POOL:
            llm = _SMARTER_LLM_POOL["claude-opus-4-1-20250805"]
            agent = RankingAgent(llm=llm)
            
            hyp_a = create_mock_hypothesis(1, "Exercise prevents dementia", "Physical activity increases BDNF", 0.8)
            hyp_b = create_mock_hypothesis(2, "Diet prevents dementia", "Mediterranean diet reduces inflammation", 0.7)
            
            # Execute real debate
            result = await agent.conduct_debate(hyp_a, hyp_b)
            
            # Basic validation
            assert result["winner"] in ["A", "B"]
            assert len(result["reasoning"]) > 10
            assert len(result["debate_transcript"]) > 50

    async def test_tournament_scalability(self, mock_llm):
        """Test tournament with larger number of hypotheses."""
        # Create 20 hypotheses
        large_set = [create_mock_hypothesis(i, f"Hypothesis {i}", f"Reasoning {i}", 0.5 + (i*0.02)) 
                    for i in range(1, 21)]
        
        # Mock many debate responses
        for i in range(50):
            winner = "A" if i % 2 == 0 else "B"
            mock_llm.add_response(f"Winner: Hypothesis {winner}\nReasoning: Test reasoning {i}")
        
        agent = RankingAgent(llm=mock_llm)
        
        # Create test state
        test_state = MagicMock()
        test_state.hypotheses = large_set.copy()
        test_state.tournament_results = []
        
        # Run tournament with many matches
        input_data = {"k": 30}
        result = await agent.run_tournament(test_state, input_data)
        
        # Should handle large tournaments
        assert "tournament_results" in result
        assert len(test_state.hypotheses) == 20