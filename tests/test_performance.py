"""
Performance and load tests for Coscientist framework.
"""

import asyncio
import time
import psutil
import pytest
import gc
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path

from coscientist.framework import CoscientistConfig, CoscientistFramework
from coscientist.global_state import CoscientistState, CoscientistStateManager
from tests.utils import create_mock_hypothesis


@pytest.mark.performance
class TestScalabilityPerformance:
    """Test framework performance with increasing loads."""

    def test_large_hypothesis_set_memory(self, mock_llm_pools):
        """Test memory usage with large hypothesis sets."""
        # Monitor memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Create large hypothesis set (100 hypotheses)
        large_set = []
        for i in range(1, 101):
            hypothesis = create_mock_hypothesis(
                i, 
                f"Hypothesis {i} with detailed description and comprehensive reasoning",
                f"Detailed reasoning for hypothesis {i} including multiple assumptions and complex logic",
                0.5 + (i * 0.005)
            )
            large_set.append(hypothesis)
        
        # Measure memory after creating hypotheses
        after_creation = process.memory_info().rss
        memory_per_hypothesis = (after_creation - initial_memory) / 100
        
        # Memory per hypothesis should be reasonable (< 10KB each)
        assert memory_per_hypothesis < 10 * 1024  # 10KB per hypothesis
        
        # Clean up
        del large_set
        gc.collect()

    @pytest.mark.asyncio
    async def test_tournament_scaling(self, mock_llm_pools, test_state):
        """Test tournament performance with increasing hypothesis counts."""
        from coscientist.ranking_agent import RankingAgent
        
        # Test different tournament sizes
        sizes = [5, 10, 20, 30]
        times = []
        
        for size in sizes:
            # Create hypothesis set of given size
            hypotheses = [
                create_mock_hypothesis(i, f"Hypothesis {i}", f"Reasoning {i}", 0.5)
                for i in range(1, size + 1)
            ]
            test_state.hypotheses = hypotheses
            
            # Mock debate responses
            for j in range(size * 2):  # Enough responses for tournament
                winner = "A" if j % 2 == 0 else "B"
                mock_llm_pools["claude-opus-4-1-20250805"].add_response(
                    f"Winner: Hypothesis {winner}\nReasoning: Test reasoning {j}"
                )
            
            # Time the tournament
            agent = RankingAgent(llm=mock_llm_pools["claude-opus-4-1-20250805"])
            
            start_time = time.time()
            await agent.run_tournament(test_state, {"k": size})
            end_time = time.time()
            
            elapsed = end_time - start_time
            times.append(elapsed)
            
            # Tournament time should scale reasonably (not exponentially)
            # With mocked LLM, should be very fast
            assert elapsed < 1.0  # Should complete within 1 second with mocks
        
        # Verify scaling is reasonable (not exponential growth)
        # Time ratio between largest and smallest should be < 10x
        time_ratio = times[-1] / times[0]
        assert time_ratio < 10

    @pytest.mark.asyncio 
    async def test_concurrent_agent_performance(self, mock_llm_pools, test_state, sample_hypotheses):
        """Test performance of concurrent agent operations."""
        config = CoscientistConfig()
        state_manager = CoscientistStateManager(test_state)
        framework = CoscientistFramework(config, state_manager)
        
        test_state.hypotheses = sample_hypotheses.copy()
        test_state.literature_review = "Background literature"
        
        # Mock responses for concurrent operations
        for i in range(20):
            mock_llm_pools["claude-opus-4-1-20250805"].add_response(f"Response {i}")
            mock_llm_pools["gpt-5"].add_response(f'{"{"}"hypotheses":[{"{"}"id":{i+10},"hypothesis":"H{i}","reasoning":"R{i}","confidence":0.8,"assumptions":[],"testing_approach":"T","observables":[]{"}"}]{"}"}')
        
        # Time concurrent operations
        start_time = time.time()
        
        # Simulate concurrent operations
        tasks = []
        for i in range(5):
            task = framework.generate_hypotheses(n_hypotheses=1)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Concurrent operations should not take excessively long
        assert elapsed < 5.0
        
        # At least some operations should succeed
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) > 0

    def test_state_file_size_scaling(self, temp_dir, mock_env):
        """Test state file size with increasing data."""
        import os
        os.environ["COSCIENTIST_DIR"] = temp_dir
        
        goal = "State size test"
        state = CoscientistState(goal=goal)
        
        # Add increasingly large amounts of data
        data_sizes = []
        file_sizes = []
        
        for i in [10, 50, 100, 200]:
            # Add hypotheses
            for j in range(i - len(state.hypotheses)):
                hypothesis = create_mock_hypothesis(
                    len(state.hypotheses) + 1,
                    f"Hypothesis with significant detail {j}" * 10,  # Long text
                    f"Comprehensive reasoning with multiple points {j}" * 10,
                    0.8
                )
                state.hypotheses.append(hypothesis)
            
            # Add literature review
            state.literature_review = "Literature review content " * i * 100
            
            # Add tournament results
            state.tournament_results.extend([
                {"match": j, "result": f"Detailed result {j}" * 20}
                for j in range(i)
            ])
            
            # Save and measure file size
            checkpoint_path = state.save()
            file_size = checkpoint_path.stat().st_size
            
            data_sizes.append(i)
            file_sizes.append(file_size)
        
        # File size should grow reasonably with data
        # Should not grow exponentially
        size_ratio = file_sizes[-1] / file_sizes[0]
        data_ratio = data_sizes[-1] / data_sizes[0]
        
        # File size growth should be roughly proportional to data growth
        assert size_ratio < data_ratio * 2  # Allow some overhead

    @pytest.mark.asyncio
    async def test_embedding_performance(self, mock_embeddings, sample_hypotheses):
        """Test embedding generation performance."""
        from coscientist.proximity_agent import ProximityAgent
        
        # Test with increasing numbers of hypotheses
        sizes = [10, 20, 50]
        times = []
        
        for size in sizes:
            # Create hypothesis set
            hypotheses = [
                create_mock_hypothesis(i, f"Hypothesis {i}", f"Reasoning {i}", 0.5)
                for i in range(1, size + 1)
            ]
            
            agent = ProximityAgent()
            
            # Time embedding generation
            start_time = time.time()
            await agent.generate_embeddings(hypotheses)
            end_time = time.time()
            
            elapsed = end_time - start_time
            times.append(elapsed)
            
            # Should be fast with mocked embeddings
            assert elapsed < 2.0
        
        # Scaling should be reasonable
        if len(times) > 1:
            time_ratio = times[-1] / times[0]
            assert time_ratio < 10


@pytest.mark.performance
class TestMemoryLeakDetection:
    """Test for memory leaks during long-running operations."""

    @pytest.mark.asyncio
    async def test_repeated_hypothesis_generation(self, mock_llm_pools):
        """Test for memory leaks in repeated hypothesis generation."""
        from coscientist.generation_agent import GenerationAgent
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        agent = GenerationAgent(llm=mock_llm_pools["gpt-5"])
        
        # Perform many hypothesis generations
        for i in range(50):
            mock_llm_pools["gpt-5"].add_response(
                f'{{"hypotheses":[{{"id":{i+1},"hypothesis":"H{i}","reasoning":"R{i}","confidence":0.8,"assumptions":[],"testing_approach":"T","observables":[]}}]}}'
            )
            
            state = MagicMock()
            state.hypotheses = []
            state.literature_review = f"Literature {i}"
            
            await agent.generate_new_hypotheses(state, {"goal": f"Goal {i}", "n_hypotheses": 1})
            
            # Force garbage collection periodically
            if i % 10 == 0:
                gc.collect()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 100MB for 50 operations)
        assert memory_increase < 100 * 1024 * 1024

    @pytest.mark.asyncio
    async def test_repeated_tournaments(self, mock_llm_pools, sample_hypotheses):
        """Test for memory leaks in repeated tournament operations."""
        from coscientist.ranking_agent import RankingAgent
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        agent = RankingAgent(llm=mock_llm_pools["claude-opus-4-1-20250805"])
        
        # Perform many tournaments
        for i in range(20):
            # Mock debate responses
            for j in range(10):
                winner = "A" if j % 2 == 0 else "B"
                mock_llm_pools["claude-opus-4-1-20250805"].add_response(
                    f"Winner: Hypothesis {winner}\nReasoning: Test {i}-{j}"
                )
            
            state = MagicMock()
            state.hypotheses = sample_hypotheses.copy()
            state.tournament_results = []
            
            await agent.run_tournament(state, {"k": 6})
            
            if i % 5 == 0:
                gc.collect()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 50 * 1024 * 1024

    def test_state_loading_memory_efficiency(self, temp_dir, mock_env):
        """Test memory efficiency of state loading/saving."""
        import os
        os.environ["COSCIENTIST_DIR"] = temp_dir
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Create and save large state multiple times
        for i in range(10):
            goal = f"Memory test {i}"
            state = CoscientistState(goal=goal)
            
            # Add significant data
            for j in range(20):
                hypothesis = create_mock_hypothesis(j, f"Hypothesis {j}", f"Reasoning {j}", 0.8)
                state.hypotheses.append(hypothesis)
            
            state.literature_review = "Large literature review " * 1000
            
            # Save and immediately load
            checkpoint_path = state.save()
            loaded_state = CoscientistState.load(checkpoint_path)
            
            assert loaded_state.goal == goal
            
            # Clean up references
            del state
            del loaded_state
            gc.collect()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory should not grow excessively with repeated operations
        assert memory_increase < 100 * 1024 * 1024


@pytest.mark.performance
@pytest.mark.slow
class TestStressTests:
    """Stress tests for extreme conditions."""

    @pytest.mark.asyncio
    async def test_maximum_hypothesis_tournament(self, mock_llm_pools):
        """Test tournament with maximum recommended hypothesis count."""
        from coscientist.ranking_agent import RankingAgent
        
        # Test with 50 hypotheses (upper limit)
        hypotheses = [
            create_mock_hypothesis(i, f"Hypothesis {i}", f"Reasoning {i}", 0.5)
            for i in range(1, 51)
        ]
        
        # Mock many debate responses
        for i in range(200):  # Plenty of responses
            winner = "A" if i % 2 == 0 else "B"
            mock_llm_pools["claude-opus-4-1-20250805"].add_response(
                f"Winner: Hypothesis {winner}\nReasoning: Stress test {i}"
            )
        
        agent = RankingAgent(llm=mock_llm_pools["claude-opus-4-1-20250805"])
        
        state = MagicMock()
        state.hypotheses = hypotheses
        state.tournament_results = []
        
        # Time the large tournament
        start_time = time.time()
        await agent.run_tournament(state, {"k": 100})  # Many matches
        end_time = time.time()
        
        elapsed = end_time - start_time
        
        # Should complete within reasonable time even with large set
        assert elapsed < 10.0  # 10 seconds max with mocks
        
        # All hypotheses should have updated ratings
        for hypothesis in hypotheses:
            assert hypothesis.elo_rating != 1500.0

    @pytest.mark.asyncio
    async def test_deep_supervisor_iterations(self, mock_llm_pools, test_state):
        """Test framework with many supervisor iterations."""
        config = CoscientistConfig(max_supervisor_iterations=20)
        state_manager = CoscientistStateManager(test_state)
        framework = CoscientistFramework(config, state_manager)
        
        # Mock many supervisor decisions
        decisions = ["generate_new_hypotheses", "run_tournament", "reflect"] * 10
        for decision in decisions:
            mock_llm_pools["o3"].add_response(f"Decision: {decision}\nReasoning: Iteration test")
        
        # Mock responses for all operations
        for i in range(50):
            mock_llm_pools["gpt-5"].add_response(f'{{"hypotheses":[{{"id":{i+1},"hypothesis":"H{i}","reasoning":"R{i}","confidence":0.8,"assumptions":[],"testing_approach":"T","observables":[]}}]}}')
            mock_llm_pools["claude-opus-4-1-20250805"].add_response(f"Winner: Hypothesis A\nReasoning: Test {i}")
            mock_llm_pools["claude-opus-4-1-20250805"].add_response(f"Reflection result {i}")
        
        # Add finalization responses
        mock_llm_pools["o3"].add_response("Decision: finalize\nReasoning: Maximum iterations reached")
        mock_llm_pools["claude-opus-4-1-20250805"].add_response("Final report after many iterations")
        mock_llm_pools["gemini-2.5-flash"].add_response("Meta-review of extended process")
        
        test_state.literature_review = "Initial literature"
        
        # Execute with timeout to prevent infinite loops
        start_time = time.time()
        final_report, meta_review = await asyncio.wait_for(
            framework.run(), 
            timeout=30.0
        )
        end_time = time.time()
        
        elapsed = end_time - start_time
        
        # Should complete within timeout
        assert elapsed < 30.0
        assert final_report is not None
        assert meta_review is not None

    def test_extremely_large_literature_review(self, temp_dir, mock_env):
        """Test handling of very large literature reviews."""
        import os
        os.environ["COSCIENTIST_DIR"] = temp_dir
        
        goal = "Large literature test"
        state = CoscientistState(goal=goal)
        
        # Create very large literature review (10MB of text)
        large_text = "This is a comprehensive literature review with detailed findings. " * 100000
        state.literature_review = large_text
        
        # Should be able to save and load large state
        checkpoint_path = state.save()
        assert checkpoint_path.exists()
        
        loaded_state = CoscientistState.load(checkpoint_path)
        assert len(loaded_state.literature_review) == len(large_text)
        
        # File size should be manageable (compressed)
        file_size = checkpoint_path.stat().st_size
        # Pickle compression should reduce size significantly
        assert file_size < len(large_text)  # Should be compressed

    @pytest.mark.asyncio
    async def test_rapid_state_updates(self, temp_dir, mock_env):
        """Test rapid state updates and persistence."""
        import os
        os.environ["COSCIENTIST_DIR"] = temp_dir
        
        goal = "Rapid updates test"
        state = CoscientistState(goal=goal)
        state_manager = CoscientistStateManager(state)
        
        # Perform rapid updates
        start_time = time.time()
        for i in range(100):
            hypothesis = create_mock_hypothesis(i, f"Rapid hypothesis {i}", f"Reasoning {i}", 0.8)
            state.hypotheses.append(hypothesis)
            
            if i % 10 == 0:  # Save every 10 updates
                state_manager.save()
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Rapid updates should complete quickly
        assert elapsed < 5.0
        assert len(state.hypotheses) == 100


@pytest.mark.performance
class TestBenchmarks:
    """Benchmark tests for performance monitoring."""

    @pytest.mark.asyncio
    async def benchmark_hypothesis_generation(self, mock_llm_pools, test_state):
        """Benchmark hypothesis generation performance."""
        from coscientist.generation_agent import GenerationAgent
        
        agent = GenerationAgent(llm=mock_llm_pools["gpt-5"])
        test_state.literature_review = "Background literature"
        
        # Benchmark different batch sizes
        batch_sizes = [1, 5, 10, 20]
        times = {}
        
        for batch_size in batch_sizes:
            # Mock response
            hypotheses_json = {
                "hypotheses": [
                    {
                        "id": i,
                        "hypothesis": f"Benchmark hypothesis {i}",
                        "reasoning": f"Benchmark reasoning {i}",
                        "confidence": 0.8,
                        "assumptions": [f"Assumption {i}"],
                        "testing_approach": f"Test {i}",
                        "observables": [f"Observable {i}"]
                    }
                    for i in range(1, batch_size + 1)
                ]
            }
            
            import json
            mock_llm_pools["gpt-5"].add_response(json.dumps(hypotheses_json))
            
            # Time generation
            start_time = time.time()
            await agent.generate_new_hypotheses(
                test_state, 
                {"goal": "Benchmark goal", "n_hypotheses": batch_size}
            )
            end_time = time.time()
            
            times[batch_size] = end_time - start_time
        
        # Log benchmark results (would be captured by pytest)
        print(f"\nHypothesis Generation Benchmarks:")
        for batch_size, elapsed in times.items():
            print(f"  {batch_size} hypotheses: {elapsed:.3f}s ({elapsed/batch_size:.3f}s per hypothesis)")
        
        # Basic performance assertions
        assert all(t < 2.0 for t in times.values())  # All should be under 2 seconds with mocks

    @pytest.mark.asyncio
    async def benchmark_tournament_performance(self, mock_llm_pools, sample_hypotheses):
        """Benchmark tournament performance across different sizes."""
        from coscientist.ranking_agent import RankingAgent
        
        agent = RankingAgent(llm=mock_llm_pools["claude-opus-4-1-20250805"])
        
        # Test different tournament sizes
        sizes = [4, 8, 16, 32]
        times = {}
        
        for size in sizes:
            # Create hypothesis set
            hypotheses = [
                create_mock_hypothesis(i, f"Tournament hypothesis {i}", f"Reasoning {i}", 0.5)
                for i in range(1, size + 1)
            ]
            
            # Mock debate responses
            num_matches = size * 2  # Rough estimate
            for j in range(num_matches):
                winner = "A" if j % 2 == 0 else "B"
                mock_llm_pools["claude-opus-4-1-20250805"].add_response(
                    f"Winner: Hypothesis {winner}\nReasoning: Benchmark test {j}"
                )
            
            state = MagicMock()
            state.hypotheses = hypotheses
            state.tournament_results = []
            
            # Time tournament
            start_time = time.time()
            await agent.run_tournament(state, {"k": num_matches})
            end_time = time.time()
            
            times[size] = end_time - start_time
        
        # Log benchmark results
        print(f"\nTournament Performance Benchmarks:")
        for size, elapsed in times.items():
            matches_per_second = (size * 2) / elapsed if elapsed > 0 else float('inf')
            print(f"  {size} hypotheses: {elapsed:.3f}s ({matches_per_second:.1f} matches/s)")
        
        # Performance assertions
        assert all(t < 5.0 for t in times.values())  # All should complete within 5 seconds